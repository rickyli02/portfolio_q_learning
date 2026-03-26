"""Unit tests for src/utils/io.py."""
import tempfile
from pathlib import Path

import torch

from src.utils.io import load_checkpoint, save_checkpoint


def test_save_and_load_roundtrip():
    state = {"weights": torch.randn(4, 4), "step": 10}
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "checkpoints" / "ckpt.pt"
        save_checkpoint(state, path)
        assert path.exists()
        loaded = load_checkpoint(path)
        assert torch.allclose(state["weights"], loaded["weights"])
        assert loaded["step"] == 10
