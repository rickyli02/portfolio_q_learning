"""Pure tensor utilities for optional-context masking.

Mask convention: a boolean tensor where ``True`` means the feature is
**present** and ``False`` means it is **missing** (should be treated as zero
by the model).

All functions operate on raw tensors and have no side effects; they are
independent of any specific model or environment.
"""

from __future__ import annotations

import torch


def validate_context_pair(
    context: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    """Raise ``ValueError`` if ``context`` and ``mask`` are inconsistent.

    Args:
        context: Feature tensor of shape ``(..., C)``.
        mask: Boolean mask tensor; must have the same shape as ``context``.

    Raises:
        ValueError: If shapes differ or ``mask`` is not a boolean tensor.
    """
    if context.shape != mask.shape:
        raise ValueError(
            f"context shape {tuple(context.shape)} must match "
            f"mask shape {tuple(mask.shape)}"
        )
    if mask.dtype != torch.bool:
        raise ValueError(
            f"mask must have dtype torch.bool, got {mask.dtype}"
        )


def apply_context_mask(context: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Return ``context`` with missing positions zeroed out.

    Positions where ``mask`` is ``False`` are set to zero; present positions
    are returned unchanged.

    Args:
        context: Feature tensor of shape ``(..., C)``.
        mask: Boolean mask of the same shape.

    Returns:
        Tensor of the same shape as ``context``.
    """
    validate_context_pair(context, mask)
    return context * mask.to(context.dtype)


def make_full_mask(
    shape: tuple[int, ...],
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Return an all-``True`` mask (every feature is present).

    Args:
        shape: Desired mask shape, e.g. ``(B, C)`` or ``(C,)``.
        device: Target device.

    Returns:
        Boolean tensor of ones with the given shape.
    """
    return torch.ones(shape, dtype=torch.bool, device=device)


def make_empty_mask(
    shape: tuple[int, ...],
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Return an all-``False`` mask (every feature is missing).

    Args:
        shape: Desired mask shape.
        device: Target device.

    Returns:
        Boolean tensor of zeros with the given shape.
    """
    return torch.zeros(shape, dtype=torch.bool, device=device)


def random_context_dropout(
    context: torch.Tensor,
    mask: torch.Tensor,
    drop_prob: float,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Randomly drop individual context features during training.

    Each feature position that is currently present (``mask == True``) is
    independently set to missing with probability ``drop_prob``.  Positions
    already missing are unaffected.

    Args:
        context: Feature tensor of shape ``(..., C)``.
        mask: Boolean mask of the same shape.
        drop_prob: Probability in ``[0, 1)`` of dropping each present feature.
        generator: Optional ``torch.Generator`` for reproducible dropout.

    Returns:
        Tuple of ``(masked_context, new_mask)`` where ``masked_context`` has
        dropped positions zeroed and ``new_mask`` is the updated boolean mask.

    Raises:
        ValueError: If ``drop_prob`` is not in ``[0, 1)``.
    """
    if not (0.0 <= drop_prob < 1.0):
        raise ValueError(f"drop_prob must be in [0, 1), got {drop_prob}")
    validate_context_pair(context, mask)

    if drop_prob == 0.0:
        return apply_context_mask(context, mask), mask.clone()

    keep_prob = 1.0 - drop_prob
    # Sample a keep/drop decision for each feature position.
    noise = torch.empty_like(context, dtype=torch.float32)
    if generator is not None:
        noise.uniform_(generator=generator)
    else:
        noise.uniform_()
    # A feature is kept only if it was already present AND the random draw
    # is below keep_prob.
    dropout_keep = noise < keep_prob
    new_mask = mask & dropout_keep
    return apply_context_mask(context, new_mask), new_mask
