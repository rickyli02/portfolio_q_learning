"""Multi-seed deterministic evaluation record set — Phase 15G.

Provides a typed collection of per-seed ``CTRLEvalRecord`` objects and a
helper that runs deterministic evaluation across an explicit sequence of seeds.

SCOPE BOUNDARY
--------------
The following are NOT implemented here:
- file IO for record sets
- plotting or report generation
- scalar aggregate changes
- benchmark comparison tables
- broader backtesting framework
- stochastic evaluation
- trainer integration

These belong in future bounded tasks.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.envs.base_env import PortfolioEnv
from src.eval.record import CTRLEvalRecord, eval_record
from src.models.base import ActorBase


@dataclass
class CTRLEvalRecordSet:
    """Typed collection of per-seed deterministic evaluation records — Phase 15G.

    Attributes:
        seeds:   Explicit seeds used for evaluation, in evaluation order.
        records: Per-seed ``CTRLEvalRecord`` objects in the same order as
                 ``seeds``.
    """

    seeds: list[int]
    records: list[CTRLEvalRecord]


def eval_record_set(
    actor: ActorBase,
    env: PortfolioEnv,
    w: float,
    seeds: list[int],
    target_return_z: float | None = None,
) -> CTRLEvalRecordSet:
    """Run deterministic evaluation for each seed and return a typed record set.

    Evaluates exactly one deterministic episode per seed using ``eval_record``,
    preserving seed order exactly.

    Args:
        actor:           Policy satisfying ``ActorBase``; only ``mean_action``
                         is used (deterministic execution policy).
        env:             Portfolio environment.
        w:               Outer-loop target-wealth Lagrange parameter.
        seeds:           Non-empty sequence of RNG seeds to evaluate.
        target_return_z: Optional target terminal wealth z, forwarded to each
                         ``eval_record`` call.

    Returns:
        ``CTRLEvalRecordSet`` with ``seeds`` and ``records`` in seed order.

    Raises:
        ValueError: If ``seeds`` is empty.
    """
    if not seeds:
        raise ValueError("seeds must be non-empty")

    records = [
        eval_record(actor, env, w, target_return_z=target_return_z, seed=s)
        for s in seeds
    ]
    return CTRLEvalRecordSet(seeds=list(seeds), records=records)
