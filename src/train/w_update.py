"""CTRL outer-loop w (Lagrange multiplier / target-wealth) update primitive.

THEOREM-ALIGNED (Huang–Jia–Zhou 2025, §3.5, §3.8)
--------------------------------------------------
The outer-loop Lagrange multiplier update takes the form:

    w_{n+1} = Proj_{[w_min, w_max]}(w_n - a_w · (x_T - z))

where:
- x_T is the terminal portfolio wealth from the most recent trajectory,
- z   is the target terminal wealth (``reward.target_return`` in config),
- a_w is a positive outer-loop step size,
- Proj_{[w_min, w_max]} is optional projection onto a simple closed interval.

The signal x_T - z is the same quantity as ``compute_w_update_target`` in
``src/algos/ctrl.py``, computed here from plain Python floats so the update
primitive is decoupled from tensor computation graphs.

SCOPE BOUNDARY
--------------
The following are NOT implemented here:
- repeated w-update schedules across epochs (adaptive a_w, annealing, etc.)
- joint trainer-loop orchestration
- batch or averaged w updates across multiple trajectories
- checkpoint / logging infrastructure
- config-dispatch wiring
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CTRLWUpdateResult:
    """Typed output of one outer-loop w update step.

    THEOREM-ALIGNED: stores the raw signal, the unprojected candidate, and
    the final (optionally projected) w so callers can inspect every stage of
    the update.

    Attributes:
        w_prev:      Lagrange multiplier before this update.
        signal:      Raw update signal x_T - z (Huang–Jia–Zhou 2025, §3.5).
        w_next_raw:  Unprojected candidate w_prev - step_size · signal.
        w_next:      Final w after optional projection onto [w_min, w_max].
                     Equals ``w_next_raw`` when no bounds are supplied.
    """

    w_prev: float
    signal: float
    w_next_raw: float
    w_next: float


def ctrl_w_update(
    w: float,
    terminal_wealth: float,
    target_return_z: float,
    step_size: float,
    w_min: float | None = None,
    w_max: float | None = None,
) -> CTRLWUpdateResult:
    """Perform one outer-loop w update step.

    THEOREM-ALIGNED: implements the discrete Lagrange multiplier update:

        signal      = x_T - z
        w_next_raw  = w - step_size · signal
        w_next      = Proj_{[w_min, w_max]}(w_next_raw)   (if bounds given)

    (Huang–Jia–Zhou 2025, §3.5, §3.8)

    REPO ENGINEERING: ``terminal_wealth`` should be sourced from the final
    step of the most recent trainer run, e.g.:

        run_result.final_step.terminal_wealth

    Operates on plain Python floats; no tensor graphs are involved.

    Args:
        w:                Current outer-loop Lagrange multiplier / target wealth.
        terminal_wealth:  Terminal portfolio wealth x_T from the last trajectory.
                          Typically ``run_result.final_step.terminal_wealth``.
        target_return_z:  Target terminal wealth z (``reward.target_return``).
        step_size:        Positive outer-loop step size a_w.  Must be > 0.
        w_min:            Optional lower bound for interval projection.
                          If ``None``, no lower clamp is applied.
        w_max:            Optional upper bound for interval projection.
                          If ``None``, no upper clamp is applied.

    Returns:
        ``CTRLWUpdateResult`` with ``w_prev``, ``signal``, ``w_next_raw``,
        and ``w_next``.

    Raises:
        ValueError: if ``step_size <= 0``.
    """
    if step_size <= 0.0:
        raise ValueError(f"step_size must be > 0, got {step_size}")
    if w_min is not None and w_max is not None and w_min > w_max:
        raise ValueError(
            f"w_min must be <= w_max when both are provided, got w_min={w_min}, w_max={w_max}"
        )

    signal: float = terminal_wealth - target_return_z
    w_next_raw: float = w - step_size * signal

    w_next: float = w_next_raw
    if w_min is not None:
        w_next = max(w_min, w_next)
    if w_max is not None:
        w_next = min(w_max, w_next)

    return CTRLWUpdateResult(
        w_prev=w,
        signal=signal,
        w_next_raw=w_next_raw,
        w_next=w_next,
    )
