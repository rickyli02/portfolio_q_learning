"""Diagonal Gaussian actor for CTRL-compatible behavior and execution policies.

THEOREM-BACKED STRUCTURE (Huang–Jia–Zhou 2025, companion §4.1)
---------------------------------------------------------------
The stochastic behavior policy is:

    π(·|t, x; w; φ) = N( −φ₁·(x−w),  φ₂·e^{φ₃·(T−t)}·I_d )

with the deterministic execution policy:

    û(t, x; w) = mean = −φ₁·(x−w)

This is the baseline CTRL Gaussian parameterisation.  The mean is linear in
the wealth gap (x − w), which gives a proportional-control structure.

ENGINEERING CHOICES
-------------------
1. φ₁ ∈ R^d (per-asset mean coefficient).  The single-asset CTRL paper uses
   a scalar φ₁; this extends naturally to d assets.  φ₁ > 0 is enforced by
   storing log φ₁ internally.

2. Precision parameterisation (companion §4.2).  The CTRL paper updates in
   φ₂⁻¹, not φ₂ directly, for numerical stability.  This class stores
   log(φ₂⁻¹) = −log(φ₂) so precision is always positive.  Variance is
   reconstructed as φ₂ = exp(−log_phi2_inv).

3. φ₃ is unconstrained (any sign is mathematically valid).

4. Isotropic diagonal covariance: same scalar variance φ₂ e^{φ₃(T-t)} for
   all d assets.  This matches the CTRL baseline; a full diagonal or full
   covariance extension would live in a separate subclass.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from src.models.base import ActorBase


class GaussianActor(ActorBase):
    """Diagonal Gaussian behavior policy and deterministic execution policy.

    Parameters
    ----------
    n_risky : int
        Number of risky assets d.
    horizon : float
        Investment horizon T.
    init_phi1 : float
        Initial value for all φ₁ components (must be > 0).
    init_phi2 : float
        Initial variance level φ₂ (must be > 0).
    init_phi3 : float
        Initial time-decay coefficient φ₃ (unconstrained).
    """

    def __init__(
        self,
        n_risky: int,
        horizon: float,
        init_phi1: float = 1.0,
        init_phi2: float = 0.5,
        init_phi3: float = 0.0,
    ) -> None:
        super().__init__()
        if init_phi1 <= 0:
            raise ValueError(f"init_phi1 must be > 0, got {init_phi1}")
        if init_phi2 <= 0:
            raise ValueError(f"init_phi2 must be > 0, got {init_phi2}")

        self.n_risky = n_risky
        self.horizon = horizon

        # THEOREM-BACKED: φ₁ > 0 — mean coefficient per asset.
        # Store as log φ₁ to ensure positivity under unconstrained optimisation.
        self.log_phi1 = nn.Parameter(
            torch.full((n_risky,), math.log(init_phi1))
        )

        # ENGINEERING: φ₂⁻¹ > 0 — precision (inverse variance level).
        # Store as log φ₂⁻¹ to ensure positivity.
        # Variance: φ₂ = exp(−log_phi2_inv).
        self.log_phi2_inv = nn.Parameter(
            torch.tensor(math.log(1.0 / init_phi2))
        )

        # THEOREM-BACKED: φ₃ — time-decay coefficient (unconstrained).
        self.phi3 = nn.Parameter(torch.tensor(float(init_phi3)))

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------

    @property
    def phi1(self) -> torch.Tensor:
        """Per-asset mean coefficient φ₁ > 0, shape ``(n_risky,)``."""
        return torch.exp(self.log_phi1)

    @property
    def phi2(self) -> torch.Tensor:
        """Variance level φ₂ > 0, scalar."""
        return torch.exp(-self.log_phi2_inv)

    def variance(self, t: float | torch.Tensor) -> torch.Tensor:
        """Scalar variance φ₂·e^{φ₃·(T−t)} at time t.

        THEOREM-BACKED: time-varying Gaussian variance from the CTRL
        parameterisation.

        Args:
            t: Current time, scalar or tensor.

        Returns:
            Scalar variance tensor.
        """
        t_t = torch.as_tensor(t, dtype=self.phi3.dtype, device=self.phi3.device)
        return self.phi2 * torch.exp(self.phi3 * (self.horizon - t_t))

    # ------------------------------------------------------------------
    # ActorBase interface
    # ------------------------------------------------------------------

    def mean_action(
        self,
        t: float | torch.Tensor,
        wealth: torch.Tensor,
        w: float | torch.Tensor,
    ) -> torch.Tensor:
        """Deterministic execution policy û = −φ₁·(x−w).

        THEOREM-BACKED: The mean of the exploratory Gaussian is the
        deterministic optimal action in the CTRL formulation.

        Args:
            t: Unused for mean computation; accepted for interface consistency.
            wealth: Current wealth scalar or ``(B,)``.
            w: Outer-loop target wealth, scalar or compatible tensor.

        Returns:
            Dollar allocations, shape ``(n_risky,)`` or ``(B, n_risky)``.
        """
        w_t = torch.as_tensor(w, dtype=wealth.dtype, device=wealth.device)
        gap = wealth - w_t  # () or (B,)
        phi1 = self.phi1.to(dtype=wealth.dtype, device=wealth.device)

        if gap.dim() == 0:
            return -phi1 * gap  # (n_risky,)
        else:
            return -gap.unsqueeze(-1) * phi1.unsqueeze(0)  # (B, n_risky)

    def sample(
        self,
        t: float | torch.Tensor,
        wealth: torch.Tensor,
        w: float | torch.Tensor,
    ) -> torch.Tensor:
        """Sample from the stochastic behavior policy π(·|t, x; w).

        Returns:
            Sampled dollar allocation, same shape as ``mean_action``.
        """
        mean = self.mean_action(t, wealth, w)
        std = self.variance(t).to(dtype=mean.dtype, device=mean.device).sqrt()
        return mean + std * torch.randn_like(mean)

    def log_prob(
        self,
        action: torch.Tensor,
        t: float | torch.Tensor,
        wealth: torch.Tensor,
        w: float | torch.Tensor,
    ) -> torch.Tensor:
        """Compute log π(u | t, x; w).

        Returns:
            Scalar for unbatched, shape ``(B,)`` for batched.
        """
        mean = self.mean_action(t, wealth, w)
        var = self.variance(t).to(dtype=mean.dtype, device=mean.device)
        dist = torch.distributions.Normal(mean, var.sqrt())
        # Sum log-prob over the asset dimension to get joint log-prob.
        lp = dist.log_prob(action)
        if lp.dim() == 1:
            return lp.sum()   # (n_risky,) → scalar
        return lp.sum(dim=-1)  # (B, n_risky) → (B,)

    def entropy(
        self,
        t: float | torch.Tensor,
    ) -> torch.Tensor:
        """Differential entropy H[π(·|t, x)].

        For a d-dimensional diagonal Gaussian with scalar variance σ²:
            H = 0.5 · d · (1 + log(2π) + log(σ²))

        This depends only on t (through σ²(t) = φ₂ e^{φ₃(T-t)}), not on
        wealth or w.

        Returns:
            Scalar entropy value.
        """
        var = self.variance(t)
        log_var = torch.log(var)
        return 0.5 * self.n_risky * (1.0 + math.log(2.0 * math.pi) + log_var)
