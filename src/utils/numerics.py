"""Lightweight numerical-stability diagnostics — Phase 17A.

Provides warning helpers for detecting obviously unstable tensor values
during forward passes.  These helpers warn but never mutate or clamp
the tensor; callers remain responsible for handling or propagating the
unstable values.
"""

from __future__ import annotations

import warnings

import torch


def warn_if_unstable(
    tensor: torch.Tensor,
    name: str,
    threshold: float = 1e6,
) -> None:
    """Issue a warning if *tensor* contains large, infinite, or NaN values.

    Checks two conditions, in order:

    1. Non-finite values (``inf`` or ``nan``) — always warned regardless of
       *threshold*, because they indicate a computation has already broken down.
    2. Large-but-finite absolute values (``|v| > threshold``) — warned when
       the maximum absolute value exceeds *threshold*, indicating potential
       numerical instability before overflow.

    The tensor is never mutated or clamped.  This function is a diagnostic
    only and has no effect on the computation graph.

    Args:
        tensor: Tensor to inspect.
        name:   Human-readable name included in the warning message.
        threshold: Absolute-value threshold for large-finite warnings.
            Default 1e6 follows the project roadmap convention.
    """
    with torch.no_grad():
        if not torch.isfinite(tensor).all():
            n_inf = torch.isinf(tensor).sum().item()
            n_nan = torch.isnan(tensor).sum().item()
            warnings.warn(
                f"warn_if_unstable: '{name}' contains non-finite values "
                f"(inf={n_inf}, nan={n_nan}); shape={tuple(tensor.shape)}",
                stacklevel=2,
            )
        elif tensor.abs().max().item() > threshold:
            max_abs = tensor.abs().max().item()
            warnings.warn(
                f"warn_if_unstable: '{name}' has large absolute value "
                f"max_abs={max_abs:.3e} > threshold={threshold:.3e}; "
                f"shape={tuple(tensor.shape)}",
                stacklevel=2,
            )
