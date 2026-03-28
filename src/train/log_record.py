"""In-memory trainer logging record — Phase 14A.

Provides a typed scalar log-record boundary oriented toward future
logging / plotting consumers.  All fields are plain Python scalars so
records are serialization-friendly without any tensor dependencies.

SCOPE BOUNDARY
--------------
The following are NOT implemented here:
- writing logs to disk (CSV / JSONL / structured files)
- config-dispatch wiring
- callback / progress systems
- checkpoint metadata sidecars
- plotting or DataFrame output

These belong in future bounded tasks.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.train.ctrl_state import CTRLTrainerSnapshot


@dataclass(frozen=True)
class CTRLLogRecord:
    """Typed scalar log record for one trainer snapshot — Phase 14A.

    All fields are plain Python scalars (``float``, ``int``, or ``None``).
    No tensor objects are stored so records are safe to serialise with any
    standard format (JSON, CSV, etc.) in future logging phases.

    Attributes:
        current_w:            Lagrange multiplier at snapshot time.
        target_return_z:      Target terminal wealth z.
        w_step_size:          Outer-loop step size a_w.
        last_terminal_wealth: Terminal portfolio wealth from the most recent
                              inner run's final step.  ``None`` if no run has
                              been executed before the snapshot.
        last_w_prev:          Lagrange multiplier before the most recent w
                              update.  ``None`` if no run has been executed.
        last_n_updates:       Total inner actor/critic steps in the most
                              recent call.  ``None`` if no run has been
                              executed.
    """

    current_w: float
    target_return_z: float
    w_step_size: float
    last_terminal_wealth: float | None
    last_w_prev: float | None
    last_n_updates: int | None


def record_from_snapshot(snap: CTRLTrainerSnapshot) -> CTRLLogRecord:
    """Derive a ``CTRLLogRecord`` from a single trainer snapshot.

    REPO ENGINEERING (Phase 14A): one-to-one field mapping; no computation
    is performed.  The resulting record is a stable, serialization-friendly
    view of the snapshot's scalar state.

    Args:
        snap: ``CTRLTrainerSnapshot`` produced by ``CTRLTrainerState.snapshot()``.

    Returns:
        ``CTRLLogRecord`` with the same scalar field values.
    """
    return CTRLLogRecord(
        current_w=snap.current_w,
        target_return_z=snap.target_return_z,
        w_step_size=snap.w_step_size,
        last_terminal_wealth=snap.last_terminal_wealth,
        last_w_prev=snap.last_w_prev,
        last_n_updates=snap.last_n_updates,
    )


def records_from_history(
    history: tuple[CTRLTrainerSnapshot, ...],
) -> list[CTRLLogRecord]:
    """Derive a list of ``CTRLLogRecord`` from a trainer history sequence.

    REPO ENGINEERING (Phase 14A): maps ``record_from_snapshot`` over the
    history tuple in call order.  Returns an empty list when history is
    empty.

    Args:
        history: Tuple of ``CTRLTrainerSnapshot`` entries from
                 ``CTRLTrainerState.history``.

    Returns:
        List of ``CTRLLogRecord`` in the same order as ``history``.
    """
    return [record_from_snapshot(snap) for snap in history]
