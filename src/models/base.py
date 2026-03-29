"""Abstract base classes for actor and critic model interfaces.

ENGINEERING SCAFFOLD
--------------------
These abstract classes define the minimal contract shared by all actor and
critic implementations.  Concrete subclasses document which structural
decisions are theorem-backed versus engineering choices.

Design notes
------------
- Both bases inherit from ``torch.nn.Module`` so parameters are managed by
  standard PyTorch optimisers and ``model.parameters()`` works as expected.
- The method signatures target the CTRL-compatible (t, wealth, w) triplet:
    t      — current time in [0, T]
    wealth — current portfolio wealth  x_t
    w      — outer-loop Lagrange multiplier / target-wealth parameter

  Optional context features can be incorporated by concrete subclasses via
  an extra keyword argument; the abstract interface stays minimal here to
  avoid over-specifying the interface before CTRL algorithm code exists.
"""

from __future__ import annotations

import abc

import torch
import torch.nn as nn



class ActorBase(nn.Module, abc.ABC):
    """Abstract interface for stochastic behavior policies and deterministic
    execution policies.

    Subclasses must implement all four abstract methods.  The intended
    separation of roles is:

    - ``sample``        — stochastic behavior policy π(·|t, x; w); used during
                          training trajectories.
    - ``mean_action``   — deterministic execution policy û(t, x; w); used
                          during evaluation / benchmark comparison.
    - ``log_prob``      — log π(u|t, x; w); required for actor gradient updates.
    - ``entropy``       — H[π(·|t, x)]; required for entropy-regularised
                          objectives.  In the CTRL Gaussian parameterisation the
                          entropy depends only on t (not on wealth or w) because
                          Gaussian entropy depends only on the variance.
    """

    @abc.abstractmethod
    def sample(
        self,
        t: float | torch.Tensor,
        wealth: torch.Tensor,
        w: float | torch.Tensor,
    ) -> torch.Tensor:
        """Sample an action from the stochastic behavior policy π(·|t, x; w).

        Args:
            t: Current time, scalar or ``(B,)``.
            wealth: Current wealth, scalar tensor or ``(B,)``.
            w: Outer-loop Lagrange / target-wealth parameter.

        Returns:
            Sampled dollar allocation, shape ``(n_risky,)`` or ``(B, n_risky)``.
        """

    @abc.abstractmethod
    def mean_action(
        self,
        t: float | torch.Tensor,
        wealth: torch.Tensor,
        w: float | torch.Tensor,
    ) -> torch.Tensor:
        """Return the deterministic execution policy û(t, x; w).

        THEOREM-BACKED requirement: The mean of the exploratory Gaussian policy
        is the deterministic optimal action.  Implementations must not add
        noise here.

        Returns:
            Mean dollar allocation, same shape convention as ``sample``.
        """

    @abc.abstractmethod
    def log_prob(
        self,
        action: torch.Tensor,
        t: float | torch.Tensor,
        wealth: torch.Tensor,
        w: float | torch.Tensor,
    ) -> torch.Tensor:
        """Compute log π(u | t, x; w) for a given action tensor.

        Args:
            action: Dollar allocation(s), shape ``(n_risky,)`` or
                ``(B, n_risky)``.
            t: Current time.
            wealth: Current wealth.
            w: Outer-loop parameter.

        Returns:
            Log-probability scalar for unbatched input, shape ``(B,)`` for
            batched input.
        """

    @abc.abstractmethod
    def entropy(
        self,
        t: float | torch.Tensor,
    ) -> torch.Tensor:
        """Compute the differential entropy H[π(·|t, x)].

        Note: In the CTRL Gaussian parameterisation the entropy depends only
        on t (through the time-varying variance φ₂ e^{φ₃(T-t)}), not on
        wealth or w.

        Returns:
            Scalar entropy value.
        """

    def validate_parameters(self) -> None:
        """Raise ``ValueError`` if any parameter tensor contains non-finite values.

        Checks all parameters registered under ``self.parameters()``.  Call
        this at the start of a training step to fail fast on corrupted model
        state rather than propagating NaN/inf silently through the computation.

        Raises:
            ValueError: If any parameter contains ``inf`` or ``nan``.
        """
        for name, param in self.named_parameters():
            if not torch.isfinite(param).all():
                raise ValueError(
                    f"ActorBase.validate_parameters: non-finite values in "
                    f"parameter '{name}'; shape={tuple(param.shape)}"
                )


class CriticBase(nn.Module, abc.ABC):
    """Abstract interface for critic / value-function modules.

    ENGINEERING SCAFFOLD: Defines the minimal contract for value functions
    that will later be extended with theorem-backed martingale update rules
    in the CTRL algorithm layer (``src/algos/``).

    The forward method mirrors the CTRL parameterisation where the value
    function has explicit dependence on (t, wealth, w).
    """

    @abc.abstractmethod
    def forward(
        self,
        t: float | torch.Tensor,
        wealth: torch.Tensor,
        w: float | torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate J(t, x; w) — the value function at (t, x) under target w.

        Args:
            t: Current time, scalar or ``(B,)``.
            wealth: Current wealth, scalar tensor or ``(B,)``.
            w: Outer-loop Lagrange / target-wealth parameter.

        Returns:
            Scalar value for unbatched input, shape ``(B,)`` for batched input.
        """

    def validate_parameters(self) -> None:
        """Raise ``ValueError`` if any parameter tensor contains non-finite values.

        Checks all parameters registered under ``self.parameters()``.  Call
        this at the start of a training step to fail fast on corrupted model
        state rather than propagating NaN/inf silently through the computation.

        Raises:
            ValueError: If any parameter contains ``inf`` or ``nan``.
        """
        for name, param in self.named_parameters():
            if not torch.isfinite(param).all():
                raise ValueError(
                    f"CriticBase.validate_parameters: non-finite values in "
                    f"parameter '{name}'; shape={tuple(param.shape)}"
                )
