"""CTRL fixed-length outer-loop schedule helper.

REPO ENGINEERING NOTES
----------------------
This module extends the approved Phase 10B single outer iteration into a
fixed-length repeated outer-loop schedule:

    for j in range(n_outer_iters):
        run ctrl_outer_iter at current w
        thread w_next into the next iteration as the new w

This is the narrowest useful composition that produces an evolving ``w``
sequence from the already-approved Phase 10B primitive.

Seed scheduling: when ``base_seed`` is provided, outer iteration ``j`` uses
base seed ``base_seed + j * n_updates``.  This ensures inner step seeds are
non-overlapping across outer iterations (inner step ``k`` of outer iter ``j``
receives seed ``base_seed + j * n_updates + k``).

SCOPE BOUNDARY
--------------
The following are NOT implemented here:
- adaptive or learned w schedules
- convergence checks / early stopping
- checkpoint or logging infrastructure
- config-dispatch wiring
- offline / online trainer classes
- callback / progress infrastructure

These belong in future bounded tasks.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from src.envs.base_env import PortfolioEnv
from src.models.base import ActorBase, CriticBase
from src.train.ctrl_outer_iter import CTRLOuterIterResult, ctrl_outer_iter


@dataclass
class CTRLOuterLoopResult:
    """Typed output of a fixed-length CTRL outer-loop schedule.

    REPO ENGINEERING: stores per-iteration results and a direct reference to
    the final iteration for convenience.

    Attributes:
        iters:         Per-outer-iteration results, length ``n_outer_iters``.
        final_iter:    Alias for ``iters[-1]``; the result of the last iteration.
        w_init:        Initial Lagrange multiplier supplied to the first iteration.
        w_final:       Final Lagrange multiplier after the last w update;
                       equals ``iters[-1].w_next``.
        n_outer_iters: Number of outer iterations executed; equals ``len(iters)``.
    """

    iters: list[CTRLOuterIterResult]
    final_iter: CTRLOuterIterResult
    w_init: float
    w_final: float
    n_outer_iters: int


def ctrl_outer_loop(
    actor: ActorBase,
    critic: CriticBase,
    env: PortfolioEnv,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    w_init: float,
    target_return_z: float,
    w_step_size: float,
    n_outer_iters: int,
    n_updates: int,
    entropy_temp: float,
    base_seed: int | None = None,
    w_min: float | None = None,
    w_max: float | None = None,
) -> CTRLOuterLoopResult:
    """Execute a fixed-length sequence of CTRL outer iterations.

    THEOREM-ALIGNED: repeatedly applies the slow Lagrange-multiplier update
    loop (Huang–Jia–Zhou 2025, §3.5, §3.8) for ``n_outer_iters`` steps,
    threading ``w_next`` from each iteration into the next as the new ``w``.

    Seed scheduling: outer iteration ``j`` receives
    ``base_seed + j * n_updates`` as its ``base_seed``, so inner step ``k``
    of iteration ``j`` uses seed ``base_seed + j * n_updates + k``.  Seeds
    are non-overlapping across iterations.

    Args:
        actor:            Stochastic behavior policy (``ActorBase``).
        critic:           Value function (``CriticBase``).
        env:              Portfolio environment.
        actor_optimizer:  PyTorch optimiser bound to actor parameters.
        critic_optimizer: PyTorch optimiser bound to critic parameters.
        w_init:           Initial Lagrange multiplier / target-wealth parameter.
        target_return_z:  Target terminal wealth z for w-update signals.
        w_step_size:      Positive outer-loop step size a_w.
        n_outer_iters:    Number of outer iterations to execute.  Must be >= 1.
        n_updates:        Number of inner actor/critic steps per outer iteration.
        entropy_temp:     Entropy regularisation temperature γ.
        base_seed:        Optional base seed for deterministic scheduling.
        w_min:            Optional lower bound for w projection.
        w_max:            Optional upper bound for w projection.

    Returns:
        ``CTRLOuterLoopResult`` with per-iteration history, final iteration,
        ``w_init``, ``w_final``, and ``n_outer_iters``.

    Raises:
        ValueError: if ``n_outer_iters < 1``.
        ValueError: propagated from ``ctrl_outer_iter`` for invalid
                    ``w_step_size``, ``n_updates``, or bound order.
    """
    if n_outer_iters < 1:
        raise ValueError(f"n_outer_iters must be >= 1, got {n_outer_iters}")

    current_w: float = w_init
    iters: list[CTRLOuterIterResult] = []

    for j in range(n_outer_iters):
        iter_seed = base_seed + j * n_updates if base_seed is not None else None
        iter_result = ctrl_outer_iter(
            actor,
            critic,
            env,
            actor_optimizer,
            critic_optimizer,
            w=current_w,
            target_return_z=target_return_z,
            w_step_size=w_step_size,
            n_updates=n_updates,
            entropy_temp=entropy_temp,
            base_seed=iter_seed,
            w_min=w_min,
            w_max=w_max,
        )
        iters.append(iter_result)
        current_w = iter_result.w_next

    return CTRLOuterLoopResult(
        iters=iters,
        final_iter=iters[-1],
        w_init=w_init,
        w_final=iters[-1].w_next,
        n_outer_iters=n_outer_iters,
    )
