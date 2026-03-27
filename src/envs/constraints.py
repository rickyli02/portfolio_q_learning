"""Portfolio allocation constraint utilities.

Provides modular post-processing layers for portfolio actions.  These are
applied *after* the actor produces a raw action and *before* the action is
sent to the environment, so the constraints are decoupled from both the
model and the environment dynamics.

Action convention: signed dollar allocations u_i (positive = long,
negative = short).  The risk-free residual is (x - Σ u_i).

Leverage constraint (this module uses gross exposure / L1 norm):

    gross(u) = Σ_i |u_i|
    u' = u * (x · ℓ) / gross(u)   if gross(u) > x · ℓ
    u' = u                          otherwise

This correctly handles signed (short) allocations.  The paper formula in
Huang–Jia–Zhou 2022 §3.3 uses the signed sum and is equivalent only for
long-only portfolios; the L1 form is the natural generalisation.
"""

from __future__ import annotations

import torch


def apply_leverage_constraint(
    action: torch.Tensor,
    wealth: torch.Tensor | float,
    leverage_cap: float,
) -> torch.Tensor:
    """Rescale dollar allocations to satisfy a gross-exposure leverage cap.

    Uses the L1 norm (gross exposure = Σ|u_i|) so that both long and short
    positions contribute to the leverage measure.  If gross exposure exceeds
    ``wealth * leverage_cap``, the entire allocation vector is scaled
    proportionally (preserving signs and relative sizes) so that gross
    exposure equals the limit.

    Args:
        action: Dollar allocation vector u, shape ``(n_risky,)`` or
            ``(B, n_risky)`` for batched use.
        wealth: Current portfolio wealth, scalar or shape ``(B,)`` for batch.
        leverage_cap: Maximum allowed gross leverage ratio ℓ (e.g. 1.5 for
            150% of wealth in gross risky exposure).

    Returns:
        Leverage-constrained allocation vector of the same shape as ``action``.

    Raises:
        ValueError: If ``leverage_cap`` is not positive.
    """
    if leverage_cap <= 0:
        raise ValueError(f"leverage_cap must be > 0, got {leverage_cap}")

    if isinstance(wealth, (int, float)):
        wealth = torch.tensor(wealth, dtype=action.dtype, device=action.device)

    wealth = wealth.to(action.device)

    # Gross exposure: L1 norm along the last axis
    gross = action.abs().sum(dim=-1, keepdim=True)   # (..., 1)
    limit = wealth.unsqueeze(-1) * leverage_cap if wealth.dim() > 0 else wealth * leverage_cap

    # Scale where gross > limit; leave unchanged otherwise
    exceeds = (gross > limit)
    scale = torch.where(exceeds, limit / gross.clamp(min=1e-8), torch.ones_like(gross))
    return action * scale


def apply_risky_only_projection(
    action: torch.Tensor,
    wealth: torch.Tensor | float,
) -> torch.Tensor:
    """Normalise dollar allocations so gross risky exposure equals wealth.

    Rescales the allocation vector so that Σ|u_i| = wealth, which fully
    invests in risky assets (no risk-free residual in gross terms).  Signs
    are preserved; only the overall scale changes.

    This is a generalisation of the benchmarking convention in
    Huang–Jia–Zhou 2022 §3.4 to signed (long/short) portfolios.  The
    original paper formula ``û = u / Σ u_i * x`` applies only when all
    u_i ≥ 0; this implementation uses gross exposure so it handles short
    positions without catastrophic numerical behaviour.

    Note: This is a benchmarking convention, not a theoretical requirement.
    Only use when explicitly comparing against the paper's reported results.

    Args:
        action: Raw dollar allocation vector, shape ``(n_risky,)`` or
            ``(B, n_risky)``.
        wealth: Current portfolio wealth, scalar or shape ``(B,)``.

    Returns:
        Rescaled allocation where Σ|u_i| = wealth.

    Raises:
        ValueError: If the input action has zero gross exposure (Σ|u_i| = 0),
            because there is no direction to rescale into a risky-only
            portfolio.  Design decision: zero-action inputs must be handled by
            the caller (e.g. by skipping projection when no risky allocation is
            intended).
        ValueError: If ``wealth`` is not positive (scalar check only; tensor
            wealth is the caller's responsibility).
    """
    if isinstance(wealth, (int, float)):
        if wealth <= 0:
            raise ValueError(f"wealth must be > 0 for projection, got {wealth}")
        wealth = torch.tensor(wealth, dtype=action.dtype, device=action.device)

    wealth = wealth.to(action.device)

    gross = action.abs().sum(dim=-1, keepdim=True)  # (..., 1)
    if (gross == 0).any():
        raise ValueError(
            "apply_risky_only_projection requires non-zero gross exposure "
            "(Σ|u_i| > 0).  The input action has zero gross exposure, so "
            "there is no direction to rescale.  Handle zero-action inputs "
            "in the caller before calling this function."
        )
    w = wealth.unsqueeze(-1) if wealth.dim() > 0 else wealth
    return action / gross * w


def clip_action_norm(
    action: torch.Tensor,
    max_norm: float,
) -> torch.Tensor:
    """Clip the L2 norm of a dollar-allocation vector.

    Provides a simple fallback constraint when a more principled leverage cap
    is not configured.

    Args:
        action: Dollar allocation vector, shape ``(n_risky,)`` or
            ``(B, n_risky)``.
        max_norm: Maximum allowed L2 norm.

    Returns:
        Action with L2 norm at most ``max_norm``.
    """
    if max_norm <= 0:
        raise ValueError(f"max_norm must be > 0, got {max_norm}")
    norm = action.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    scale = (max_norm / norm).clamp(max=1.0)
    return action * scale
