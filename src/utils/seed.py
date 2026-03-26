"""Deterministic seeding utilities."""
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic_cudnn: bool = True) -> None:
    """Set random seeds for Python, NumPy, and PyTorch for reproducibility.

    Args:
        seed: Integer seed value.
        deterministic_cudnn: If True, sets cuDNN to deterministic mode and
            disables benchmark mode.  This ensures bit-for-bit reproducibility
            on GPU at the cost of some runtime performance.  Non-determinism
            can still arise from certain PyTorch operations not covered by
            these flags; see PyTorch reproducibility docs for details.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
