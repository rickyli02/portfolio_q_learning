"""Diagonal Gaussian actor for CTRL-compatible behavior and execution policies.

THEOREM-ALIGNED STRUCTURE (Huang–Jia–Zhou 2025, companion §3.1, §4.1)
-----------------------------------------------------------------------
The paper's multi-asset stochastic behavior policy has the qualitative form:

    π(·|t, x; w; φ) = N( −φ₁·(x−w),  φ₂·e^{φ₃·(T−t)} )

    φ₁ ∈ R^d,  φ₂ ∈ S_{++}^d,  φ₃ ∈ R (treated as fixed in the paper)

with the deterministic execution policy:

    û(t, x; w) = mean = −φ₁·(x−w)

The per-asset vector φ₁ ∈ R^d and the time-varying covariance scaling
φ₂ e^{φ₃(T-t)} are theorem-aligned structure from the 2025 paper §3.6.

REPO SCAFFOLD CHOICES (deviations from the paper's full parameterisation)
--------------------------------------------------------------------------
1. φ₁ ∈ R^d (per-asset mean coefficient): theorem-aligned for multi-asset
   (paper §3.6 states φ₁ ∈ R^d explicitly).  Positivity (φ₁ > 0 per asset)
   is a repo engineering choice enforced by storing log φ₁ internally.
   The paper does not require strict positivity of φ₁ in the theorem itself.

2. Precision parameterisation (companion §4.2).  The CTRL paper updates in
   φ₂⁻¹, not φ₂ directly, for numerical stability.  This class stores
   log(φ₂⁻¹) = −log(φ₂) so precision is always positive.  Variance is
   reconstructed as φ₂ = exp(−log_phi2_inv).

3. φ₃ is stored as a freely learnable nn.Parameter here.  The paper treats
   φ₃ as fixed and often sets φ₃ = θ₃ (the critic decay parameter).
   Making φ₃ learnable is a scaffold choice for this foundation layer;
   the algorithm/trainer layer should enforce or fix φ₃ as appropriate.

4. Isotropic scalar covariance: this implementation uses a scalar φ₂ shared
   across all d assets (isotropic diagonal covariance φ₂ e^{φ₃(T-t)}·I_d).
   The full paper parameterisation uses φ₂ ∈ S_{++}^d (d×d positive-definite
   matrix).  The scalar simplification is acceptable for this foundation block
   but is a deviation from the paper's general multi-asset covariance.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from src.models.base import ActorBase
from src.utils.numerics import warn_if_unstable


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

        # φ₁ ∈ R^d — per-asset mean coefficient (theorem-aligned, paper §3.6).
        # Positivity (φ₁ > 0) is a repo optimization-safety choice; the theorem
        # does not require strict positivity.  Store as log φ₁ to enforce it.
        self.log_phi1 = nn.Parameter(
            torch.full((n_risky,), math.log(init_phi1))
        )

        # φ₂⁻¹ > 0 — precision (inverse variance level).  Scaffold choice:
        # paper uses a full S_{++}^d matrix; this implementation uses a scalar.
        # Store as log φ₂⁻¹ to ensure positivity under unconstrained optimisation.
        # Variance: φ₂ = exp(−log_phi2_inv).
        self.log_phi2_inv = nn.Parameter(
            torch.tensor(math.log(1.0 / init_phi2))
        )

        # φ₃ — time-decay coefficient.  Scaffold choice: stored as a freely
        # learnable parameter here.  The paper treats φ₃ as fixed and often sets
        # φ₃ = θ₃.  The algorithm/trainer layer should fix or couple φ₃ as needed.
        self.phi3 = nn.Parameter(torch.tensor(float(init_phi3)))

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------

    @property
    def phi1(self) -> torch.Tensor:
        """Per-asset mean coefficient φ₁ > 0, shape ``(n_risky,)``."""
        val = torch.exp(self.log_phi1)
        warn_if_unstable(val, "GaussianActor.phi1")
        return val

    @property
    def phi2(self) -> torch.Tensor:
        """Variance level φ₂ > 0, scalar."""
        val = torch.exp(-self.log_phi2_inv)
        warn_if_unstable(val, "GaussianActor.phi2", min_positive=1e-38)
        return val

    def variance(self, t: float | torch.Tensor) -> torch.Tensor:
        """Scalar variance φ₂·e^{φ₃·(T−t)} at time t.

        The time-varying form φ₂ e^{φ₃(T-t)} is theorem-aligned (paper §3.6).
        This implementation uses a scalar φ₂ (isotropic) and a freely learned
        φ₃; both are repo scaffold choices relative to the full paper form.

        Args:
            t: Current time, scalar or tensor.

        Returns:
            Scalar variance tensor.
        """
        t_t = torch.as_tensor(t, dtype=self.phi3.dtype, device=self.phi3.device)
        val = self.phi2 * torch.exp(self.phi3 * (self.horizon - t_t))
        warn_if_unstable(val, "GaussianActor.variance", min_positive=1e-38)
        return val

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

        The mean form −φ₁·(x−w) is theorem-aligned (paper §3.6).
        Used as the deterministic execution policy at evaluation time.

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
