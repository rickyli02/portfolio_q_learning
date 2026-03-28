"""Deterministic evaluation trajectory record — Phase 15E.

Provides a typed path-level wrapper around one deterministic execution-policy
evaluation episode and a helper that wraps ``evaluate_ctrl_deterministic``
and populates all record fields.

SCOPE BOUNDARY
--------------
The following are NOT implemented here:
- plotting or report generation
- aggregate / summary file IO changes
- backtesting framework
- stochastic behavior-policy evaluation
- config-dispatch wiring
- trainer integration

These belong in future bounded tasks.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from src.algos.ctrl import evaluate_ctrl_deterministic
from src.envs.base_env import PortfolioEnv
from src.models.base import ActorBase


@dataclass
class CTRLEvalRecord:
    """Typed path-level record for one deterministic evaluation episode — Phase 15E.

    Combines the full tensor path data from the evaluation rollout with
    pre-extracted scalar summaries for inspection and future plotting use.

    Attributes:
        times:           Step start-times t_0…t_{n-1}, shape ``(n_steps,)``.
        wealth_path:     Wealth at each time including t=0 and t=T,
                         shape ``(n_steps + 1,)``.
        actions:         Deterministic mean actions, shape ``(n_steps, n_risky)``.
        terminal_wealth: Final portfolio wealth x_T (plain float).
        initial_wealth:  Portfolio wealth at t=0 (plain float).
        n_steps:         Number of environment steps.
        min_wealth:      Minimum wealth along the full path.
        max_wealth:      Maximum wealth along the full path.
        target_return_z: Target terminal wealth z, or ``None`` if not provided.
        terminal_gap:    x_T − z (signed gap to target), or ``None`` when
                         ``target_return_z`` is ``None``.
    """

    times: torch.Tensor           # (n_steps,)
    wealth_path: torch.Tensor     # (n_steps + 1,)
    actions: torch.Tensor         # (n_steps, n_risky)
    terminal_wealth: float
    initial_wealth: float
    n_steps: int
    min_wealth: float
    max_wealth: float
    target_return_z: float | None
    terminal_gap: float | None


def eval_record(
    actor: ActorBase,
    env: PortfolioEnv,
    w: float,
    target_return_z: float | None = None,
    seed: int | None = None,
) -> CTRLEvalRecord:
    """Run one deterministic evaluation episode and return a typed path record.

    Wraps ``evaluate_ctrl_deterministic`` and extracts both tensor path data
    and plain scalar summaries into a ``CTRLEvalRecord``.

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
        ``CTRLEvalRecord`` with tensor path fields and pre-extracted scalars.
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

    return CTRLEvalRecord(
        times=result.times,
        wealth_path=result.wealth_path,
        actions=result.actions,
        terminal_wealth=terminal_wealth,
        initial_wealth=initial_wealth,
        n_steps=n_steps,
        min_wealth=min_wealth,
        max_wealth=max_wealth,
        target_return_z=target_return_z,
        terminal_gap=terminal_gap,
    )
