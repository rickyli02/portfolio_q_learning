"""Trainer layer for portfolio RL.

Current primitives:
- Phase 9A: single-trajectory CTRL update step (``ctrl_train_step``).
- Phase 9B: fixed-length CTRL trainer run (``ctrl_train_run``).
- Phase 10A: outer-loop w (Lagrange multiplier) update primitive (``ctrl_w_update``).
- Phase 10B: single outer iteration — inner run + w update (``ctrl_outer_iter``).
- Phase 10C: fixed-length outer-loop schedule (``ctrl_outer_loop``).
- Phase 11A/11B: stateful trainer shell with persistent current_w and validation boundary (``CTRLTrainerState``).
- Phase 12A: read-only trainer snapshot and scalar-summary layer (``CTRLTrainerSnapshot``).

Future phases will add checkpoint and logging infrastructure, config-dispatch
wiring, and full offline/online trainer classes.  Those are not present yet.
"""

from src.train.ctrl_outer_iter import CTRLOuterIterResult, ctrl_outer_iter
from src.train.ctrl_outer_loop import CTRLOuterLoopResult, ctrl_outer_loop
from src.train.ctrl_runner import CTRLRunResult, ctrl_train_run
from src.train.ctrl_state import CTRLTrainerSnapshot, CTRLTrainerState
from src.train.ctrl_trainer import CTRLStepResult, ctrl_train_step
from src.train.w_update import CTRLWUpdateResult, ctrl_w_update

__all__ = [
    "CTRLOuterIterResult",
    "CTRLOuterLoopResult",
    "CTRLRunResult",
    "CTRLStepResult",
    "CTRLTrainerSnapshot",
    "CTRLTrainerState",
    "CTRLWUpdateResult",
    "ctrl_outer_iter",
    "ctrl_outer_loop",
    "ctrl_train_run",
    "ctrl_train_step",
    "ctrl_w_update",
]
