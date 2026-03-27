"""ContextBundle: a container for optional context features and their mask.

A ``ContextBundle`` pairs a feature tensor with a boolean mask so that
callers never have to manage them separately.  The mask follows the
convention in ``masking.py``: ``True`` = feature present, ``False`` = missing.

Both batched ``(B, C)`` and unbatched ``(C,)`` shapes are supported.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from src.features.masking import (
    apply_context_mask,
    make_empty_mask,
    make_full_mask,
    random_context_dropout,
    validate_context_pair,
)


@dataclass
class ContextBundle:
    """A context feature tensor paired with its boolean presence mask.

    Attributes:
        values: Feature tensor of shape ``(B, C)`` or ``(C,)``.
        mask: Boolean mask of the same shape; ``True`` = feature present.
    """

    values: torch.Tensor
    mask: torch.Tensor

    def __post_init__(self) -> None:
        validate_context_pair(self.values, self.mask)

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def all_present(cls, values: torch.Tensor) -> "ContextBundle":
        """Create a bundle where every feature is marked as present.

        Args:
            values: Context feature tensor.

        Returns:
            ``ContextBundle`` with an all-``True`` mask.
        """
        mask = make_full_mask(tuple(values.shape), device=values.device)
        return cls(values=values, mask=mask)

    @classmethod
    def all_missing(
        cls,
        shape: tuple[int, ...],
        device: torch.device | str = "cpu",
    ) -> "ContextBundle":
        """Create a zero-valued bundle where every feature is marked missing.

        Useful as a placeholder when no context is available at inference.

        Args:
            shape: Desired tensor shape, e.g. ``(B, C)`` or ``(C,)``.
            device: Target device.

        Returns:
            ``ContextBundle`` with zero values and an all-``False`` mask.
        """
        values = torch.zeros(shape, dtype=torch.float32, device=device)
        mask = make_empty_mask(shape, device=device)
        return cls(values=values, mask=mask)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @property
    def dim(self) -> int:
        """Size of the context feature dimension (last axis)."""
        return self.values.shape[-1]

    @property
    def batch_size(self) -> int | None:
        """Batch size B, or ``None`` if the bundle is unbatched ``(C,)``."""
        if self.values.dim() >= 2:
            return self.values.shape[0]
        return None

    def masked_values(self) -> torch.Tensor:
        """Return the context tensor with missing positions zeroed out."""
        return apply_context_mask(self.values, self.mask)

    def with_dropout(
        self,
        drop_prob: float,
        generator: torch.Generator | None = None,
    ) -> "ContextBundle":
        """Return a new bundle with random per-feature dropout applied.

        Only features that are currently present can be dropped.

        Args:
            drop_prob: Probability of dropping each present feature.
            generator: Optional RNG for reproducibility.

        Returns:
            New ``ContextBundle`` with updated values and mask.
        """
        new_values, new_mask = random_context_dropout(
            self.values, self.mask, drop_prob, generator=generator
        )
        return ContextBundle(values=new_values, mask=new_mask)

    def to(self, device: torch.device | str) -> "ContextBundle":
        """Return a copy of this bundle moved to ``device``."""
        return ContextBundle(
            values=self.values.to(device),
            mask=self.mask.to(device),
        )

    def is_fully_missing(self) -> bool:
        """Return ``True`` if no features are present."""
        return not self.mask.any().item()

    def is_fully_present(self) -> bool:
        """Return ``True`` if all features are present."""
        return self.mask.all().item()
