"""Unit tests for src/utils/seed.py."""
import torch

from src.utils.seed import set_seed


def test_set_seed_reproducible():
    set_seed(42)
    a = torch.randn(10)
    set_seed(42)
    b = torch.randn(10)
    assert torch.allclose(a, b)


def test_different_seeds_differ():
    set_seed(0)
    a = torch.randn(10)
    set_seed(1)
    b = torch.randn(10)
    assert not torch.allclose(a, b)
