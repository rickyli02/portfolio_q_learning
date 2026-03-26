"""Unit tests for src/utils/device.py."""
import torch

from src.utils.device import get_device


def test_get_device_returns_torch_device():
    d = get_device()
    assert isinstance(d, torch.device)


def test_get_device_explicit_cpu():
    d = get_device("cpu")
    assert d == torch.device("cpu")
