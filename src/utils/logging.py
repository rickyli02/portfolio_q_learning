"""Training step and experiment logging utilities."""
import logging
import sys
from pathlib import Path


def get_logger(name: str, log_file: Path | None = None) -> logging.Logger:
    """Return a configured logger writing to stdout and optionally a file."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def log_step(logger: logging.Logger, step: int, **metrics: float) -> None:
    """Log a single training step with arbitrary named metrics."""
    parts = [f"step={step}"] + [f"{k}={v:.4f}" for k, v in metrics.items()]
    logger.info("  ".join(parts))
