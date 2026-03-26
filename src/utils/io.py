"""Checkpoint save and load utilities."""
from pathlib import Path

import torch


def save_checkpoint(state: dict, path: Path) -> None:
    """Save a state dict to disk, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(
    path: Path, map_location: str | torch.device = "cpu"
) -> dict:
    """Load and return a state dict from disk."""
    return torch.load(path, map_location=map_location, weights_only=True)
