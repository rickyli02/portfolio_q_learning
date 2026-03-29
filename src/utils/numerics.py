"""Lightweight numerical-stability diagnostics — Phase 17A/17B.

Provides warning helpers for detecting obviously unstable tensor values
during forward passes.  These helpers warn but never mutate or clamp
the tensor; callers remain responsible for handling or propagating the
unstable values.
"""

from __future__ import annotations

import math
import warnings

import torch


def warn_if_unstable(
    tensor: torch.Tensor,
    name: str,
    threshold: float = 1e6,
    min_positive: float | None = None,
) -> None:
    """Issue a warning if *tensor* contains unstable values.

    Checks three conditions, in order:

    1. Non-finite values (``inf`` or ``nan``) — always warned regardless of
       *threshold*, because they indicate a computation has already broken down.
    2. Large-but-finite absolute values (``|v| > threshold``) — warned when
       the maximum absolute value exceeds *threshold*, indicating potential
       numerical instability before overflow.
    3. Near-zero / underflow values (``min(v) <= min_positive``) — warned
       only when *min_positive* is provided and any value in the tensor is at
       or below that threshold.  Use this for quantities that must be strictly
       positive (e.g. variance, precision) where float32 underflow to zero
       would silently poison downstream ``log(v)`` computations.

    The tensor is never mutated or clamped.  This function is a diagnostic
    only and has no effect on the computation graph.

    Args:
        tensor:       Tensor to inspect.
        name:         Human-readable name included in the warning message.
        threshold:    Absolute-value threshold for large-finite warnings.
            Default 1e6 follows the project roadmap convention.
        min_positive: Optional lower-bound for underflow detection.  When
            provided, warns if ``tensor.min() <= min_positive``.  A value of
            ``1e-38`` catches float32 underflow to zero (float32 smallest
            normal ≈ 1.18e-38) while allowing ordinary small values.
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
        elif min_positive is not None and tensor.min().item() <= min_positive:
            min_val = tensor.min().item()
            warnings.warn(
                f"warn_if_unstable: '{name}' has near-zero or underflow value "
                f"min={min_val:.3e} <= min_positive={min_positive:.3e}; "
                f"shape={tuple(tensor.shape)}",
                stacklevel=2,
            )


def warn_if_ill_conditioned(
    matrix: torch.Tensor,
    name: str,
    threshold: float = 1e8,
) -> None:
    """Warn if a square matrix is ill-conditioned but still invertible.

    Computes the 2-norm condition number via ``torch.linalg.cond`` and warns
    when it exceeds *threshold*.  A warning here means the linear system
    ``matrix @ x = b`` can technically be solved but the solution may carry
    large relative errors due to the amplification of floating-point noise.

    Does nothing if the condition number cannot be computed (e.g. the matrix
    is not square or linalg.cond raises).  The matrix is never mutated.

    Args:
        matrix:    Square 2-D tensor to inspect.
        name:      Human-readable name for the warning message.
        threshold: Condition-number threshold above which a warning is issued.
            Default 1e8 is conservative for float64 (machine epsilon ≈ 2e-16)
            and reliably catches near-singular cases before solution quality
            degrades noticeably.
    """
    with torch.no_grad():
        try:
            cond = torch.linalg.cond(matrix).item()
        except Exception:
            return
        if not math.isfinite(cond):
            warnings.warn(
                f"warn_if_ill_conditioned: '{name}' has non-finite condition number "
                f"(cond={cond}); matrix may be singular; shape={tuple(matrix.shape)}",
                stacklevel=2,
            )
        elif cond > threshold:
            warnings.warn(
                f"warn_if_ill_conditioned: '{name}' is ill-conditioned "
                f"(cond={cond:.3e} > threshold={threshold:.3e}); "
                f"linear-solve accuracy may be reduced; shape={tuple(matrix.shape)}",
                stacklevel=2,
            )
