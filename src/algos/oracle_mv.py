"""Analytic oracle mean-variance benchmark (Zhou–Li 2000).

Implements the closed-form optimal feedback allocation for the multi-asset
Black-Scholes mean-variance problem with known synthetic coefficients.
This policy is a **benchmark only** — it requires known coefficients and is
not a learned policy.

Reference
---------
Zhou, X. Y., & Li, D. (2000). Continuous-time mean-variance portfolio
selection: A stochastic LQ framework. Applied Mathematics and Optimization,
42(1), 19–33.

See also: ``references/portfolio_mv_ctrl_complete_pseudocode.md``, Algorithm A;
and ``references/portfolio_mv_papers_algorithm_summary.md``, Sections 1.5–1.6.

Oracle formula (constant coefficients, Section 1.6 of the algorithm summary)
-----------------------------------------------------------------------------

    ū(t, x) = [σ σᵀ]⁻¹ Bᵀ (γ_embed · exp(−r(T−t)) − x)

where B = b − r·1 is the excess-return row vector, γ_embed is the auxiliary
embedding scalar from the Zhou–Li formulation, and the output ū ∈ R^d is the
vector of optimal dollar allocations in risky assets.

mu convention
-------------
``mu`` in the repo config (``env.mu``) is the **instantaneous drift of the
risky asset price process** b_i, i.e. dP_i = P_i(b_i dt + σ dW).

It is *not* the expected log-return (which would be b_i − ½ σ²_ii).  This
is confirmed by the GBM env step, which computes log-returns as
(mu_i − ½ cov_diag_i)·Δt, treating mu_i as the price-SDE drift b_i.

The excess-return vector used in the oracle formula is therefore
    B = env.mu − r          (no Ito correction needed).

gamma_embed vs target_return
-----------------------------
``gamma_embed`` is the auxiliary embedding scalar γ = λ/(2μ) from the
Zhou–Li classical-embedding construction.  It is **not** the same as
``reward.target_return`` (z).  In practice, it acts as the "virtual terminal
wealth target" for the oracle: when x = γ_embed · exp(−r(T−t)), the optimal
allocation is zero.  The caller is responsible for choosing a meaningful value.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from src.envs.gbm_env import GBMPortfolioEnv


# ---------------------------------------------------------------------------
# Coefficient precomputation
# ---------------------------------------------------------------------------


@dataclass
class OracleCoefficients:
    """Precomputed static quantities for the oracle policy.

    These are derived once from the known synthetic parameters and reused
    on every policy evaluation.
    """

    B: torch.Tensor
    """Excess-return vector b − r·1, shape ``(n_risky,)``."""

    cov: torch.Tensor
    """Covariance matrix σ σᵀ, shape ``(n_risky, n_risky)``."""

    sensitivity: torch.Tensor
    """[σσᵀ]⁻¹ B, shape ``(n_risky,)``.  Computed via a linear solve rather
    than an explicit matrix inversion for numerical stability."""

    r: float
    """Risk-free rate (continuous compounding)."""

    horizon: float
    """Investment horizon T."""

    gamma_embed: float
    """Auxiliary embedding scalar γ = λ/(2μ) from the Zhou–Li formulation."""


def compute_oracle_coefficients(
    mu: list[float] | torch.Tensor,
    sigma: list[list[float]] | torch.Tensor,
    r: float,
    horizon: float,
    gamma_embed: float,
) -> OracleCoefficients:
    """Precompute oracle policy coefficients from known synthetic parameters.

    Args:
        mu: Price-process drift vector b, shape ``(n_risky,)``.  Matches
            ``env.mu`` convention (instantaneous drift, not expected log-return).
        sigma: Volatility factor matrix, shape ``(n_risky, n_risky)``.  Matches
            ``env.sigma`` convention.
        r: Risk-free rate (continuous compounding).
        horizon: Investment horizon T.
        gamma_embed: Auxiliary embedding scalar γ from the Zhou–Li formulation.
            Acts as the virtual terminal wealth target for the oracle.

    Returns:
        ``OracleCoefficients`` with precomputed sensitivity vector and scalars.

    Raises:
        ValueError: If ``sigma @ sigma.T`` is singular (no solution exists for
            the linear system defining the oracle sensitivity vector).
    """
    mu_t = torch.as_tensor(mu, dtype=torch.float64)
    sigma_t = torch.as_tensor(sigma, dtype=torch.float64)

    # Covariance matrix Σ = σ σᵀ, shape (n_risky, n_risky)
    cov = sigma_t @ sigma_t.T

    # Excess-return vector B = b − r·1, shape (n_risky,)
    # env.mu is b (price SDE drift), so no Ito correction needed.
    B = mu_t - r

    # Sensitivity vector [σσᵀ]⁻¹ B via a linear solve (more numerically stable
    # than computing an explicit inverse, especially for ill-conditioned
    # covariance matrices from correlated assets or parameter sweeps).
    try:
        sensitivity = torch.linalg.solve(cov, B)
    except torch.linalg.LinAlgError as exc:
        raise ValueError(
            "Cannot compute oracle coefficients: σ σᵀ is singular.  "
            "Check that the sigma matrix has full rank."
        ) from exc

    return OracleCoefficients(
        B=B,
        cov=cov,
        sensitivity=sensitivity,
        r=r,
        horizon=horizon,
        gamma_embed=gamma_embed,
    )


# ---------------------------------------------------------------------------
# Oracle action computation
# ---------------------------------------------------------------------------


def oracle_action(
    coeffs: OracleCoefficients,
    t: float | torch.Tensor,
    wealth: torch.Tensor,
) -> torch.Tensor:
    """Compute the oracle dollar allocation for the current time and wealth.

    Implements:
        ū(t, x) = sensitivity · (γ_embed · exp(−r(T−t)) − x)

    where ``sensitivity = [σσᵀ]⁻¹ B``.

    Args:
        coeffs: Precomputed oracle coefficients.
        t: Current time, scalar float or tensor.  Supports batched shape ``(B,)``.
        wealth: Current portfolio wealth.  Scalar tensor ``()`` or batched ``(B,)``.

    Returns:
        Dollar allocation vector, shape ``(n_risky,)`` for unbatched input or
        ``(B, n_risky)`` for batched input.
    """
    t_val = torch.as_tensor(t, dtype=wealth.dtype, device=wealth.device)
    sens = coeffs.sensitivity.to(dtype=wealth.dtype, device=wealth.device)

    # Virtual target discounted from T back to t
    r_t = torch.tensor(coeffs.r, dtype=wealth.dtype, device=wealth.device)
    horizon_t = torch.tensor(coeffs.horizon, dtype=wealth.dtype, device=wealth.device)
    gamma_t = torch.tensor(coeffs.gamma_embed, dtype=wealth.dtype, device=wealth.device)

    virtual_target = gamma_t * torch.exp(-r_t * (horizon_t - t_val))

    # gap = virtual_target − wealth: scalar () or batched (B,)
    gap = virtual_target - wealth

    # Multiply: sens (n_risky,) × gap (scalar or B,)
    if gap.dim() == 0:
        # Unbatched: (n_risky,)
        return sens * gap
    else:
        # Batched: (B, n_risky)
        return gap.unsqueeze(-1) * sens.unsqueeze(0)


# ---------------------------------------------------------------------------
# OracleMVPolicy class
# ---------------------------------------------------------------------------


class OracleMVPolicy:
    """Closed-form oracle MV policy for synthetic known-parameter experiments.

    Wraps the Zhou–Li (2000) analytic solution as a callable policy that
    accepts (time, wealth) and returns dollar allocations.  This is a
    **benchmark-only** policy; it is not a learned policy and does not
    implement any training interface.

    Use as an upper-reference comparator for CTRL learning algorithms.

    Example::

        coeffs = compute_oracle_coefficients(mu, sigma, r, T, gamma_embed)
        policy = OracleMVPolicy(coeffs)
        action = policy(t=0.5, wealth=torch.tensor(1.2))
    """

    def __init__(self, coeffs: OracleCoefficients) -> None:
        self.coeffs = coeffs

    @classmethod
    def from_env_params(
        cls,
        mu: list[float] | torch.Tensor,
        sigma: list[list[float]] | torch.Tensor,
        r: float,
        horizon: float,
        gamma_embed: float,
    ) -> "OracleMVPolicy":
        """Construct directly from env config parameters.

        Args:
            mu: Price-process drift vector (``env.mu``).
            sigma: Volatility factor matrix (``env.sigma``).
            r: Risk-free rate (``env.assets.risk_free_rate``).
            horizon: Investment horizon (``env.horizon``).
            gamma_embed: Auxiliary embedding scalar γ.

        Returns:
            Configured ``OracleMVPolicy``.
        """
        coeffs = compute_oracle_coefficients(mu, sigma, r, horizon, gamma_embed)
        return cls(coeffs)

    def __call__(
        self,
        t: float | torch.Tensor,
        wealth: torch.Tensor,
    ) -> torch.Tensor:
        """Return optimal oracle dollar allocation.

        Args:
            t: Current time, scalar or ``(B,)``.
            wealth: Current wealth, scalar tensor or ``(B,)``.

        Returns:
            Dollar allocations, shape ``(n_risky,)`` or ``(B, n_risky)``.
        """
        return oracle_action(self.coeffs, t, wealth)


# ---------------------------------------------------------------------------
# Oracle episode rollout
# ---------------------------------------------------------------------------


def run_oracle_episode(
    policy: OracleMVPolicy,
    env: "GBMPortfolioEnv",
    seed: int | None = None,
) -> dict[str, torch.Tensor]:
    """Roll out the oracle policy for one episode on a GBM environment.

    The oracle is evaluated at each step using the current time and wealth.
    No exploration is added — this is the deterministic analytic policy.

    Args:
        policy: Configured ``OracleMVPolicy``.
        env: A ``GBMPortfolioEnv`` instance (must not be mid-episode).
        seed: Optional seed passed to ``env.reset``.

    Returns:
        Dictionary with:

        - ``"wealth_path"``: shape ``(n_steps + 1,)`` — wealth at each step
          including x_0.
        - ``"actions"``: shape ``(n_steps, n_risky)`` — oracle allocations.
        - ``"terminal_wealth"``: scalar tensor — final wealth.
        - ``"times"``: shape ``(n_steps,)`` — time at each step start.
    """
    obs, info = env.reset(seed=seed)
    wealth_path = [info["wealth"].clone()]
    actions_list: list[torch.Tensor] = []
    times_list: list[torch.Tensor] = []

    done = False
    step_idx = 0
    dt = env.horizon / env.n_steps

    while not done:
        t = torch.tensor(step_idx * dt, dtype=torch.float32, device=obs.device)
        current_wealth = wealth_path[-1]

        action = policy(t=t, wealth=current_wealth)
        action = action.to(dtype=torch.float32, device=obs.device)

        port_step = env.step(action)

        times_list.append(t)
        actions_list.append(action)
        wealth_path.append(port_step.wealth.clone())

        done = port_step.done
        step_idx += 1

    return {
        "wealth_path": torch.stack(wealth_path),        # (n_steps+1,)
        "actions": torch.stack(actions_list),           # (n_steps, n_risky)
        "terminal_wealth": wealth_path[-1],             # scalar
        "times": torch.stack(times_list),               # (n_steps,)
    }
