"""Structured quadratic value function for CTRL/EMV-compatible critics.

THEOREM-BACKED STRUCTURE (Huang–Jia–Zhou 2025 / ICAIF 2022, companion §3.1, §4.1)
------------------------------------------------------------------------------------
The structured value function is:

    J(t, x; w) = (x−w)²·e^{−θ₃·(T−t)}  +  θ₂·(t²−T²)  +  θ₁·(t−T)  −  (w−z)²

where z is the configured target terminal wealth (``reward.target_return``).

This parameterisation is:
- quadratic in wealth (captures the MV penalty structure),
- exponentially decaying in time-to-maturity (via θ₃),
- linear additive in polynomial time correction terms (θ₁, θ₂),
- terminal-condition compliant: at t=T the polynomial time corrections θ₁(t−T)
  and θ₂(t²−T²) vanish (both equal 0), while the exponential factor e^{−θ₃·0}=1,
  giving J(T, x; w) = (x−w)² − (w−z)².
  This matches the paper terminal condition v(T,x;w) = (x−w)² − (w−z)².

ENGINEERING NOTES
-----------------
1. θ = (θ₁, θ₂, θ₃) are independent nn.Parameters here.  In the
   theorem-backed CTRL update, θ₃ is linked to φ₃ of the actor (see companion
   §2.2): θ₃ = 2φ₂ = ρ² in the reduced parameterisation.  This constraint is
   NOT enforced in this module — it will be imposed at the algorithm / trainer
   level so the critic stays general as a standalone module.

2. (w−z)² is computed from the passed-in w and the stored z (target_return).
   It does not require a gradient w.r.t. θ and is not a separate parameter.

3. No positivity constraint is enforced on θ₃ in this module.  The CTRL
   algorithm projects iterates back to admissible sets; that logic belongs in
   the algorithm layer.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.base import CriticBase


class QuadraticCritic(CriticBase):
    """Structured quadratic value function J(t, x; w).

    Parameters
    ----------
    horizon : float
        Investment horizon T.
    target_return_z : float
        Target terminal wealth z (``reward.target_return`` in config).
        Used to compute the (w−z)² terminal term.
    init_theta1, init_theta2, init_theta3 : float
        Initial values for the three learnable parameters.
    """

    def __init__(
        self,
        horizon: float,
        target_return_z: float,
        init_theta1: float = 0.0,
        init_theta2: float = 0.0,
        init_theta3: float = 0.5,
    ) -> None:
        super().__init__()
        self.horizon = horizon
        self.z = float(target_return_z)

        # THEOREM-BACKED: three scalar parameters θ = (θ₁, θ₂, θ₃).
        self.theta1 = nn.Parameter(torch.tensor(float(init_theta1)))
        self.theta2 = nn.Parameter(torch.tensor(float(init_theta2)))
        self.theta3 = nn.Parameter(torch.tensor(float(init_theta3)))

    def forward(
        self,
        t: float | torch.Tensor,
        wealth: torch.Tensor,
        w: float | torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate J(t, x; w).

        J(t, x; w) = (x−w)²·e^{−θ₃·(T−t)}
                     + θ₂·(t²−T²)
                     + θ₁·(t−T)
                     − (w−z)²

        Args:
            t: Current time, scalar or ``(B,)``.
            wealth: Current wealth, scalar tensor or ``(B,)``.
            w: Outer-loop Lagrange / target-wealth parameter.

        Returns:
            Value scalar for unbatched, shape ``(B,)`` for batched wealth.
        """
        dtype = self.theta1.dtype
        device = self.theta1.device

        t_t = torch.as_tensor(t, dtype=dtype, device=device)
        w_t = torch.as_tensor(w, dtype=dtype, device=device)
        x_t = wealth.to(dtype=dtype, device=device)

        T = self.horizon
        time_to_go = T - t_t  # scalar or (B,)

        # (x − w)² · e^{−θ₃·(T−t)}
        quad = (x_t - w_t) ** 2 * torch.exp(-self.theta3 * time_to_go)

        # θ₂·(t² − T²) + θ₁·(t − T)
        time_correction = (
            self.theta2 * (t_t ** 2 - T ** 2)
            + self.theta1 * (t_t - T)
        )

        # −(w − z)²  [terminal MV penalty; constant w.r.t. θ]
        terminal_penalty = -(w_t - self.z) ** 2

        return quad + time_correction + terminal_penalty
