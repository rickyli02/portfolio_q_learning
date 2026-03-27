"""Trainer layer for portfolio RL.

Phase 9A: single-trajectory CTRL update step (``ctrl_train_step``).

Future phases will add multi-episode training loops, outer-loop ``w``
scheduling, checkpoint and logging infrastructure.  Those are not present yet.
"""

from src.train.ctrl_trainer import CTRLStepResult, ctrl_train_step

__all__ = [
    "CTRLStepResult",
    "ctrl_train_step",
]
