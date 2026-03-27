"""GBM portfolio environment.

Implements a discrete-time portfolio environment driven by multi-asset
Geometric Brownian Motion (GBM).  Wealth evolves according to the
self-financing portfolio SDE (see Zhou–Li 2000, equation 2.1):

    dx(t) = [r·x(t) + (b − r·1)ᵀ u(t)] dt + u(t)ᵀ σ dW(t)

where u_i(t) is the **dollar amount** invested in risky asset i and
(x − Σu_i) is held in the risk-free asset.

Discretised with an exact log-normal step for the risky assets:

    R_i = exp((μ_i − ½ σ²_ii) Δt + σ_row_i · √Δt · ε),  ε ~ N(0, I)

    x_{t+Δt} = (x_t − 1ᵀu) · exp(r · Δt) + uᵀ R

Observation: ``[wealth]`` — shape ``(1,)`` — the current portfolio wealth.
Time is stored separately in ``PortfolioStep.time`` / ``Batch.time``.

Action: dollar allocation vector u ∈ R^{n_risky}.

Reward: step-wise wealth increment Δx = x_{t+Δt} − x_t, scalar.  The
mean-variance terminal objective is reconstructed by the trainer/algorithm
from the trajectory of (x_t, u_t) pairs.
"""

from __future__ import annotations

import torch

from src.config.schema import EnvConfig
from src.envs.base_env import PortfolioEnv, PortfolioStep


class GBMPortfolioEnv(PortfolioEnv):
    """Discrete-time portfolio environment with exact GBM asset dynamics.

    Args:
        config: Environment configuration containing horizon, n_steps, GBM
            parameters, and asset universe settings.
        device: Device for all tensor operations.
    """

    def __init__(
        self,
        config: EnvConfig,
        device: torch.device | str = "cpu",
    ) -> None:
        self._cfg = config
        self._device = torch.device(device)

        # Parse GBM parameters into tensors
        self._mu = torch.as_tensor(
            config.mu, dtype=torch.float32, device=self._device
        )  # (n_risky,)
        self._sigma = torch.as_tensor(
            config.sigma, dtype=torch.float32, device=self._device
        )  # (n_risky, n_risky)
        self._n_risky: int = self._mu.shape[0]
        self._r: float = config.assets.risk_free_rate
        self._x0: float = config.initial_wealth

        # Pre-compute time-step constants
        self._dt: float = config.horizon / config.n_steps
        self._risk_free_factor: float = float(torch.exp(
            torch.tensor(self._r * self._dt, dtype=torch.float32)
        ))
        # Drift term: (μ − ½ diag(σ σᵀ)) Δt
        cov_diag = (self._sigma @ self._sigma.T).diagonal()  # (n_risky,)
        self._log_drift: torch.Tensor = (self._mu - 0.5 * cov_diag) * self._dt

        # Episode state
        self._step_idx: int = 0
        self._wealth: float = self._x0
        self._gen = torch.Generator(device=self._device)

    # ------------------------------------------------------------------
    # PortfolioEnv interface
    # ------------------------------------------------------------------

    @property
    def obs_dim(self) -> int:
        return 1

    @property
    def action_dim(self) -> int:
        return self._n_risky

    @property
    def horizon(self) -> float:
        return self._cfg.horizon

    @property
    def n_steps(self) -> int:
        return self._cfg.n_steps

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: int | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """Reset the environment to t=0, x=x_0.

        Args:
            seed: RNG seed for reproducibility.

        Returns:
            ``(obs, info)`` where ``obs = tensor([x_0])`` and ``info``
            contains ``{"time": 0.0, "wealth": x_0}``.
        """
        if seed is not None:
            self._gen.manual_seed(seed)
        self._step_idx = 0
        self._wealth = self._x0
        obs = self._make_obs(self._wealth)
        info = {
            "time": torch.tensor(0.0, device=self._device),
            "wealth": torch.tensor(self._wealth, device=self._device),
            "step": 0,
        }
        return obs, info

    def step(self, action: torch.Tensor) -> PortfolioStep:
        """Apply a dollar-allocation action and advance one time step.

        Args:
            action: Dollar allocation vector u, shape ``(n_risky,)``.

        Returns:
            ``PortfolioStep`` with next wealth, reward, done flag, and info.

        Raises:
            RuntimeError: If called before ``reset`` or after episode end.
        """
        if self._step_idx >= self._cfg.n_steps:
            raise RuntimeError(
                "step() called after episode end; call reset() first."
            )

        action = action.to(self._device)

        prev_wealth = self._wealth
        t_current = torch.tensor(
            self._step_idx * self._dt, dtype=torch.float32, device=self._device
        )
        t_next = torch.tensor(
            (self._step_idx + 1) * self._dt, dtype=torch.float32, device=self._device
        )

        # Sample one GBM step for risky assets
        gross_returns = self._sample_gbm_returns()  # (n_risky,)

        # Wealth update: risk-free portion + risky portion
        risk_free_amount = float(prev_wealth) - float(action.sum())
        risky_gain = float((action * gross_returns).sum())
        new_wealth = risk_free_amount * self._risk_free_factor + risky_gain

        self._wealth = new_wealth
        self._step_idx += 1

        done = self._step_idx >= self._cfg.n_steps
        reward = torch.tensor(
            new_wealth - prev_wealth, dtype=torch.float32, device=self._device
        )
        obs = self._make_obs(new_wealth)

        return PortfolioStep(
            obs=obs,
            reward=reward,
            done=done,
            time=t_current,
            next_time=t_next,
            wealth=torch.tensor(new_wealth, dtype=torch.float32, device=self._device),
            info={
                "prev_wealth": torch.tensor(
                    prev_wealth, dtype=torch.float32, device=self._device
                ),
                "gross_returns": gross_returns,
                "step": self._step_idx,
            },
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sample_gbm_returns(self) -> torch.Tensor:
        """Sample exact log-normal gross returns for one time step.

        Returns:
            Tensor of shape ``(n_risky,)`` with R_i = exp(log-drift + diffusion).
        """
        eps = torch.randn(self._n_risky, generator=self._gen, device=self._device)
        diffusion = (self._sigma @ eps) * (self._dt ** 0.5)
        log_returns = self._log_drift + diffusion
        return torch.exp(log_returns)

    def _make_obs(self, wealth: float) -> torch.Tensor:
        """Build the observation tensor from current wealth."""
        return torch.tensor([wealth], dtype=torch.float32, device=self._device)


# ---------------------------------------------------------------------------
# Mean-variance reward utilities
# ---------------------------------------------------------------------------

def compute_mv_terminal_reward(
    terminal_wealth: torch.Tensor,
    target_return: float,
    mv_penalty_coeff: float = 1.0,
) -> torch.Tensor:
    """Compute the mean-variance terminal objective for a batch of episodes.

    Implements the MV auxiliary objective used in the EMV/CTRL papers:

        J_MV = -mv_penalty_coeff * (x_T - w)²

    where w is the target return (Lagrange multiplier).  This is the
    **negative** of the MV penalty so that higher is better (consistent with
    a reward-maximisation framework).

    This function operates on a batch of terminal wealths and is intended
    to be called by trainers/algorithms, not called internally by the env.

    Args:
        terminal_wealth: Terminal wealth values, shape ``(B,)`` or scalar.
        target_return: Target wealth level w in the MV objective.
        mv_penalty_coeff: Coefficient on the squared deviation term.

    Returns:
        Tensor of the same shape as ``terminal_wealth``.
    """
    deviation = terminal_wealth - target_return
    return -mv_penalty_coeff * deviation ** 2
