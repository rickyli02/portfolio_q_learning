"""Deterministic evaluation summary — Phase 15A.

Provides a typed scalar summary of one deterministic execution-policy
evaluation episode, and a helper that wraps ``evaluate_ctrl_deterministic``
and extracts the summary fields.

SCOPE BOUNDARY
--------------
The following are NOT implemented here:
- multi-episode backtesting or averaging
- benchmark comparison tables
- plotting or report generation
- config-dispatch wiring
- trainer integration beyond calling the evaluation helper

These belong in future bounded tasks.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.algos.ctrl import evaluate_ctrl_deterministic
from src.envs.base_env import PortfolioEnv
from src.models.base import ActorBase


@dataclass(frozen=True)
class CTRLEvalSummary:
    """Typed scalar summary of one deterministic evaluation episode — Phase 15A.

    All fields are plain Python scalars so the summary is inspection-friendly
    and serialization-safe without tensor dependencies.

    Attributes:
        terminal_wealth:  Final portfolio wealth x_T.
        initial_wealth:   Portfolio wealth at t=0.
        target_return_z:  Target terminal wealth z, or ``None`` if not provided.
        terminal_gap:     x_T − z (signed gap to target), or ``None`` when
                          ``target_return_z`` is ``None``.
        n_steps:          Number of environment steps in the episode.
        min_wealth:       Minimum wealth along the full path (including t=0 and T).
        max_wealth:       Maximum wealth along the full path (including t=0 and T).
    """

    terminal_wealth: float
    initial_wealth: float
    target_return_z: float | None
    terminal_gap: float | None
    n_steps: int
    min_wealth: float
    max_wealth: float


def eval_summary(
    actor: ActorBase,
    env: PortfolioEnv,
    w: float,
    target_return_z: float | None = None,
    seed: int | None = None,
) -> CTRLEvalSummary:
    """Run one deterministic evaluation episode and return a scalar summary.

    Calls ``evaluate_ctrl_deterministic`` then extracts plain scalar fields
    into a ``CTRLEvalSummary``.  No tensor objects are retained.

    Args:
        actor:          Policy satisfying ``ActorBase``; only ``mean_action``
                        is used (deterministic execution policy).
        env:            Portfolio environment.
        w:              Outer-loop target-wealth Lagrange parameter.
        target_return_z: Optional target terminal wealth z.  When provided,
                        ``terminal_gap = x_T - z`` is populated; otherwise
                        ``terminal_gap`` is ``None``.
        seed:           Optional RNG seed for the environment reset.

    Returns:
        ``CTRLEvalSummary`` with scalar fields derived from the episode.
    """
    result = evaluate_ctrl_deterministic(actor, env, w, seed=seed)

    terminal_wealth = float(result.terminal_wealth)
    initial_wealth = float(result.wealth_path[0])
    n_steps = int(result.times.shape[0])
    min_wealth = float(result.wealth_path.min())
    max_wealth = float(result.wealth_path.max())

    terminal_gap: float | None
    if target_return_z is not None:
        terminal_gap = terminal_wealth - float(target_return_z)
    else:
        terminal_gap = None

    return CTRLEvalSummary(
        terminal_wealth=terminal_wealth,
        initial_wealth=initial_wealth,
        target_return_z=target_return_z,
        terminal_gap=terminal_gap,
        n_steps=n_steps,
        min_wealth=min_wealth,
        max_wealth=max_wealth,
    )
