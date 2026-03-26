"""Device selection utilities."""
import torch


def get_device(device: str | None = None) -> torch.device:
    """Return a torch.device.

    Priority when device is None: CUDA > MPS > CPU.
    """
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
