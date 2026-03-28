"""Multi-episode deterministic evaluation aggregate — Phase 15C.

Provides a typed aggregate summary of multiple deterministic evaluation
episodes and a helper that runs one episode per seed and returns both
the per-episode summaries and the aggregate.

SCOPE BOUNDARY
--------------
The following are NOT implemented here:
- stochastic rollout evaluation
- file IO for aggregates
- benchmark comparison tables
- plotting or report generation
- config-dispatch wiring
- broader backtesting framework

These belong in future bounded tasks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from src.envs.base_env import PortfolioEnv
from src.eval.summary import CTRLEvalSummary, eval_summary
from src.models.base import ActorBase


@dataclass(frozen=True)
class CTRLEvalAggregate:
    """Typed scalar aggregate over multiple deterministic evaluation episodes.

    All fields are plain Python scalars so the aggregate is inspection-friendly
    and serialization-safe without tensor dependencies.

    Attributes:
        n_episodes:          Number of episodes evaluated.
        mean_terminal_wealth: Mean terminal wealth across episodes.
        min_terminal_wealth:  Minimum terminal wealth across episodes.
        max_terminal_wealth:  Maximum terminal wealth across episodes.
        mean_terminal_gap:   Mean of (x_T − z) across episodes, or ``None``
                             when ``target_return_z`` was not provided.
        target_hit_rate:     Fraction of episodes where x_T ≥ target_return_z,
                             or ``None`` when ``target_return_z`` was not
                             provided.
    """

    n_episodes: int
    mean_terminal_wealth: float
    min_terminal_wealth: float
    max_terminal_wealth: float
    mean_terminal_gap: float | None
    target_hit_rate: float | None


def eval_aggregate(
    actor: ActorBase,
    env: PortfolioEnv,
    w: float,
    seeds: Sequence[int],
    target_return_z: float | None = None,
) -> tuple[list[CTRLEvalSummary], CTRLEvalAggregate]:
    """Run deterministic evaluation across explicit seeds and aggregate results.

    Calls ``eval_summary`` once per seed to collect per-episode summaries,
    then computes scalar aggregates.

    Args:
        actor:          Policy satisfying ``ActorBase``; only ``mean_action``
                        is used (deterministic execution policy).
        env:            Portfolio environment.
        w:              Outer-loop target-wealth Lagrange parameter.
        seeds:          Non-empty sequence of integer RNG seeds.  One episode
                        is evaluated per seed.
        target_return_z: Optional target terminal wealth z.  When provided,
                        ``mean_terminal_gap`` and ``target_hit_rate`` are
                        populated; otherwise both are ``None``.

    Returns:
        A tuple ``(summaries, aggregate)`` where ``summaries`` is a list of
        ``CTRLEvalSummary`` in seed order and ``aggregate`` is the
        ``CTRLEvalAggregate`` computed from those summaries.

    Raises:
        ValueError: If ``seeds`` is empty.
    """
    if len(seeds) == 0:
        raise ValueError("seeds must be non-empty")

    summaries = [
        eval_summary(actor, env, w, target_return_z=target_return_z, seed=s)
        for s in seeds
    ]

    n = len(summaries)
    terminal_wealths = [s.terminal_wealth for s in summaries]

    mean_terminal_wealth = sum(terminal_wealths) / n
    min_terminal_wealth = min(terminal_wealths)
    max_terminal_wealth = max(terminal_wealths)

    if target_return_z is not None:
        gaps = [s.terminal_gap for s in summaries]  # all float, not None
        mean_terminal_gap: float | None = sum(gaps) / n  # type: ignore[arg-type]
        hits = sum(1 for tw in terminal_wealths if tw >= target_return_z)
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
