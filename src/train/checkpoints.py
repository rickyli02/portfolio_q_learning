"""Checkpoint file IO helpers — Phase 13C.

Provides the narrowest useful file-based save/load boundary on top of the
approved in-memory ``CTRLCheckpointPayload`` from Phase 13B.

Uses PyTorch-native serialization (``torch.save`` / ``torch.load``) so the
on-disk format is identical to the approved in-memory payload shape.

SCOPE BOUNDARY
--------------
The following are NOT implemented here:
- checkpoint naming policies or directory management
- logging / metadata sidecars
- config-dispatch wiring
- auto-save cadence or callback systems
- experiment registry or run management
- cross-version compatibility machinery

These belong in future bounded tasks.
"""

from __future__ import annotations

from pathlib import Path

import torch

from src.train.ctrl_state import CTRLCheckpointPayload


def save_checkpoint(payload: CTRLCheckpointPayload, path: Path | str) -> None:
    """Save a ``CTRLCheckpointPayload`` to disk using PyTorch serialization.

    REPO ENGINEERING (Phase 13C): thin wrapper around ``torch.save``.  The
    on-disk object is the payload dataclass itself; no metadata sidecars are
    written.

    Args:
        payload: In-memory checkpoint payload to persist.
        path:    Destination file path.  Parent directory must exist.

    Raises:
        TypeError: if ``payload`` is not a ``CTRLCheckpointPayload``.
    """
    if not isinstance(payload, CTRLCheckpointPayload):
        raise TypeError(
            f"expected CTRLCheckpointPayload, got {type(payload).__name__}"
        )
    torch.save(payload, Path(path))


def load_checkpoint(path: Path | str) -> CTRLCheckpointPayload:
    """Load a ``CTRLCheckpointPayload`` from a file path.

    REPO ENGINEERING (Phase 13C): thin wrapper around ``torch.load``.
    Validates that the loaded object is a ``CTRLCheckpointPayload`` before
    returning it.

    Args:
        path: Source file path produced by ``save_checkpoint``.

    Returns:
        ``CTRLCheckpointPayload`` loaded from disk.

    Raises:
        FileNotFoundError: if ``path`` does not exist.
        ValueError:        if the file exists but does not contain a
                           ``CTRLCheckpointPayload``.
    """
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"checkpoint path does not exist: {resolved}")
    obj = torch.load(resolved, weights_only=False)
    if not isinstance(obj, CTRLCheckpointPayload):
        raise ValueError(
            f"expected CTRLCheckpointPayload at {resolved}, "
            f"got {type(obj).__name__}"
        )
    return obj
