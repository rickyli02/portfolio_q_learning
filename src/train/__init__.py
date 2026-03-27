"""Trainer layer for portfolio RL.

Current primitives:
- Phase 9A: single-trajectory CTRL update step (``ctrl_train_step``).
- Phase 9B: fixed-length CTRL trainer run (``ctrl_train_run``).
- Phase 10A: outer-loop w (Lagrange multiplier) update primitive (``ctrl_w_update``).
- Phase 10B: single outer iteration — inner run + w update (``ctrl_outer_iter``).

Future phases will add repeated outer-loop scheduling, checkpoint and logging
infrastructure, and config-dispatch wiring.  Those are not present yet.
"""

from src.train.ctrl_outer_iter import CTRLOuterIterResult, ctrl_outer_iter
from src.train.ctrl_runner import CTRLRunResult, ctrl_train_run
from src.train.ctrl_trainer import CTRLStepResult, ctrl_train_step
from src.train.w_update import CTRLWUpdateResult, ctrl_w_update

__all__ = [
    "CTRLOuterIterResult",
    "CTRLRunResult",
    "CTRLStepResult",
    "CTRLWUpdateResult",
    "ctrl_outer_iter",
    "ctrl_train_run",
    "ctrl_train_step",
    "ctrl_w_update",
]
