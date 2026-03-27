"""Abstract base environment interface for portfolio RL.

All concrete environment implementations must satisfy this interface so that
trainers and algorithms remain independent of the specific market dynamics.

Action convention: **dollar allocations** u_i — the dollar amount invested in
each risky asset i.  The remaining wealth (x - sum(u)) is held in the
risk-free asset.

Observation convention: ``obs`` is the base feature vector the model always
receives.  In the simplest GBM case this is ``[wealth]`` (shape ``(1,)``).
Time is passed separately via the ``Batch.time`` and ``Batch.next_time``
fields; models that need time can read it from there.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import torch


@dataclass
class PortfolioStep:
    """Output of a single environment step.

    Attributes:
        obs: Next-state observation tensor, shape ``(obs_dim,)``.
        reward: Scalar tensor; typically the step-wise wealth increment.
        done: ``True`` if the episode has ended.
        time: Current time *before* the step, scalar tensor in ``[0, T]``.
        next_time: Time *after* the step.
        wealth: Portfolio wealth at ``next_time``.
        info: Extra diagnostic information (raw returns, prev wealth, etc.).
    """

    obs: torch.Tensor
    reward: torch.Tensor
    done: bool
    time: torch.Tensor
    next_time: torch.Tensor
    wealth: torch.Tensor
    info: dict = field(default_factory=dict)

    def to_transition(
        self,
        prev_obs: torch.Tensor,
        action: torch.Tensor,
    ) -> "Transition":  # noqa: F821 — imported at call site
        """Build a ``Transition`` from this step and the preceding observation.

        Convenience helper so callers do not have to import the data layer
        just to store environment experience.
        """
        from src.data.types import Transition

        return Transition(
            obs=prev_obs,
            action=action,
            reward=self.reward,
            next_obs=self.obs,
            done=torch.tensor(float(self.done)),
            time=self.time,
            next_time=self.next_time,
        )


class PortfolioEnv(ABC):
    """Abstract base class for portfolio RL environments.

    Subclasses implement the specific market dynamics (GBM, jump-diffusion,
    etc.) while callers interact through this stable interface.
    """

    # ------------------------------------------------------------------
    # Required properties
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def obs_dim(self) -> int:
        """Dimensionality of the observation vector returned by ``reset`` and ``step``."""

    @property
    @abstractmethod
    def action_dim(self) -> int:
        """Dimensionality of the action space (number of risky assets)."""

    @property
    @abstractmethod
    def horizon(self) -> float:
        """Total investment horizon T (in years)."""

    @property
    @abstractmethod
    def n_steps(self) -> int:
        """Number of discrete rebalance steps per episode."""

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    @abstractmethod
    def reset(
        self,
        seed: int | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """Reset the environment to the start of a new episode.

        Args:
            seed: Optional integer seed to reset the RNG for reproducibility.

        Returns:
            Tuple of ``(initial_obs, info)`` where ``initial_obs`` has shape
            ``(obs_dim,)`` and ``info`` contains at least ``{"time": t0,
            "wealth": x0}``.
        """

    @abstractmethod
    def step(self, action: torch.Tensor) -> PortfolioStep:
        """Apply an action and advance the environment by one time step.

        Args:
            action: Dollar allocation vector, shape ``(action_dim,)``.

        Returns:
            A ``PortfolioStep`` with next observation, reward, done flag, and
            diagnostic info.
        """
