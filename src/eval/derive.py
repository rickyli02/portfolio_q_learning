"""Pure derivation helpers from record types to scalar eval objects — Phase 15I.

Provides helpers that derive ``CTRLEvalSummary`` and ``CTRLEvalAggregate``
directly from already-evaluated ``CTRLEvalRecord`` / ``CTRLEvalRecordSet``
objects without re-running any evaluation rollout.

SCOPE BOUNDARY
--------------
The following are NOT implemented here:
- file IO changes
- re-running evaluation rollouts inside derivation helpers
- plotting or report generation
- benchmark comparison tables
- broader backtesting framework
- trainer integration
- stochastic evaluation

These belong in future bounded tasks.
"""

from __future__ import annotations

from src.eval.aggregate import CTRLEvalAggregate
from src.eval.record import CTRLEvalRecord
from src.eval.record_set import CTRLEvalRecordSet
from src.eval.summary import CTRLEvalSummary


def summary_from_record(record: CTRLEvalRecord) -> CTRLEvalSummary:
    """Derive a ``CTRLEvalSummary`` from an already-evaluated ``CTRLEvalRecord``.

    Extracts the plain scalar fields already stored in ``record``; no rollout
    is re-executed.

    Args:
        record: A fully populated ``CTRLEvalRecord``.

    Returns:
        ``CTRLEvalSummary`` with scalar fields drawn from ``record``.
    """
    return CTRLEvalSummary(
        terminal_wealth=record.terminal_wealth,
        initial_wealth=record.initial_wealth,
        target_return_z=record.target_return_z,
        terminal_gap=record.terminal_gap,
        n_steps=record.n_steps,
        min_wealth=record.min_wealth,
        max_wealth=record.max_wealth,
    )


def aggregate_from_record_set(
    record_set: CTRLEvalRecordSet,
) -> tuple[list[CTRLEvalSummary], CTRLEvalAggregate]:
    """Derive per-seed summaries and an aggregate from a ``CTRLEvalRecordSet``.

    Derives one ``CTRLEvalSummary`` per record in seed order, then computes
    scalar aggregates using the same math as ``eval_aggregate``.  No rollout
    is re-executed.

    When no record in the set carries target information
    (``target_return_z is None``), the aggregate target-related fields
    (``mean_terminal_gap``, ``target_hit_rate``) are ``None``.

    Args:
        record_set: A fully populated ``CTRLEvalRecordSet``.

    Returns:
        A tuple ``(summaries, aggregate)`` where ``summaries`` is a list of
        ``CTRLEvalSummary`` in seed order and ``aggregate`` is the
        ``CTRLEvalAggregate`` computed from those summaries.
    """
    summaries = [summary_from_record(r) for r in record_set.records]

    n = len(summaries)
    terminal_wealths = [s.terminal_wealth for s in summaries]

    mean_terminal_wealth = sum(terminal_wealths) / n
    min_terminal_wealth = min(terminal_wealths)
    max_terminal_wealth = max(terminal_wealths)

    # Use target info from the first record; all records in a set share the
    # same target_return_z (set via eval_record_set).
    has_target = summaries[0].target_return_z is not None

    if has_target:
        gaps = [s.terminal_gap for s in summaries]
        mean_terminal_gap: float | None = sum(gaps) / n  # type: ignore[arg-type]
        target_return_z = summaries[0].target_return_z
        hits = sum(1 for tw in terminal_wealths if tw >= target_return_z)  # type: ignore[operator]
        target_hit_rate: float | None = hits / n
    else:
        mean_terminal_gap = None
        target_hit_rate = None

    aggregate = CTRLEvalAggregate(
        n_episodes=n,
        mean_terminal_wealth=mean_terminal_wealth,
        min_terminal_wealth=min_terminal_wealth,
        max_terminal_wealth=max_terminal_wealth,
        mean_terminal_gap=mean_terminal_gap,
        target_hit_rate=target_hit_rate,
    )
    return summaries, aggregate
