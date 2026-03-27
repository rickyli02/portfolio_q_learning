"""Base feature utilities and model-input construction.

Provides:
- ``FeatureSpec``: a lightweight descriptor for a named feature group.
- ``build_model_input``: assembles base features and optional context into
  the flat tensor a model receives, always producing the same output shape
  regardless of whether context is present or missing.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from src.features.context_features import ContextBundle


@dataclass
class FeatureSpec:
    """Descriptor for a named feature group.

    Used to document and validate feature dimensions before a model is built.
    Does not hold data; holds only shape and metadata.

    Attributes:
        name: Human-readable identifier for this feature group.
        dim: Number of scalar features in this group.
        optional: ``True`` if this group may be absent at inference time.
    """

    name: str
    dim: int
    optional: bool = False

    def __post_init__(self) -> None:
        if self.dim < 1:
            raise ValueError(f"FeatureSpec '{self.name}' dim must be >= 1, got {self.dim}")


def build_model_input(
    base: torch.Tensor,
    context: ContextBundle | None,
) -> torch.Tensor:
    """Concatenate base features with optional context into a model input.

    The output shape is always ``(..., base_dim + context_dim)`` when context
    is provided, or ``(..., base_dim)`` when it is ``None``.  When context is
    present but partially or fully masked, the missing positions are zeroed
    before concatenation so the model always sees a fixed-width input.

    Args:
        base: Base state tensor of shape ``(B, F)`` or ``(F,)``.
        context: Optional ``ContextBundle`` with values and mask.  If
            ``None``, no context features are appended.

    Returns:
        Concatenated tensor of shape ``(B, F + C)`` or ``(F + C,)`` when
        context is provided; ``(B, F)`` or ``(F,)`` otherwise.

    Raises:
        ValueError: If ``base`` and ``context`` batch dimensions are
            inconsistent.
    """
    if context is None:
        return base

    # Check batch dimension consistency.
    if base.dim() >= 2 and context.values.dim() >= 2:
        if base.shape[0] != context.values.shape[0]:
            raise ValueError(
                f"base batch size {base.shape[0]} does not match "
                f"context batch size {context.values.shape[0]}"
            )

    masked = context.masked_values()
    return torch.cat([base, masked], dim=-1)


def total_input_dim(base_spec: FeatureSpec, context_spec: FeatureSpec | None) -> int:
    """Return the total model input dimension given base and optional context specs.

    Args:
        base_spec: Spec for the base feature group.
        context_spec: Spec for the optional context group, or ``None``.

    Returns:
        ``base_spec.dim`` if no context, ``base_spec.dim + context_spec.dim``
        otherwise.
    """
    if context_spec is None:
        return base_spec.dim
    return base_spec.dim + context_spec.dim
