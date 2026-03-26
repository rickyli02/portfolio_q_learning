"""Checkpoint save and load utilities."""
from pathlib import Path

import torch


def save_checkpoint(state: dict, path: Path) -> None:
    """Save a state dict to disk, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(
    path: Path,
    map_location: str | torch.device = "cpu",
    weights_only: bool = False,
) -> dict:
    """Load and return a state dict from disk.

    Args:
        path: Path to the checkpoint file.
        map_location: Device to map tensors onto when loading.
        weights_only: If True, restricts unpickling to tensor-only payloads
            (safer but incompatible with checkpoints containing optimizer
            state, RNG state, or config objects).  Defaults to False so
            richer checkpoints load correctly; set to True when loading
            untrusted files.
    """
    return torch.load(path, map_location=map_location, weights_only=weights_only)
