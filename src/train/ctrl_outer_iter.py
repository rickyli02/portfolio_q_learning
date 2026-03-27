"""CTRL single outer-iteration helper.

REPO ENGINEERING NOTES
----------------------
This module composes the approved Phase 9B fixed-length trainer run with the
approved Phase 10A single w-update primitive into one typed outer iteration:

    1. Run ``ctrl_train_run`` for ``n_updates`` inner steps at constant ``w``.
    2. Run ``ctrl_w_update`` using the final trajectory's terminal wealth.
    3. Return a typed result containing both sub-results and the new ``w``.

This represents exactly one outer iteration of the slow Lagrange-multiplier
loop described in Huang–Jia–Zhou 2025 (§3.5, §3.8).

SCOPE BOUNDARY
--------------
The following are NOT implemented here:
- repeated outer-loop scheduling or epoch loops
- adaptive w step-size schedules
- convergence checks or early stopping
- checkpoint or logging infrastructure
- config-dispatch wiring
- offline / online trainer classes
- callback or progress infrastructure

These belong in future bounded tasks.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from src.envs.base_env import PortfolioEnv
from src.models.base import ActorBase, CriticBase
from src.train.ctrl_runner import CTRLRunResult, ctrl_train_run
from src.train.w_update import CTRLWUpdateResult, ctrl_w_update


@dataclass
class CTRLOuterIterResult:
    """Typed output of one CTRL outer iteration.

    REPO ENGINEERING: stores both sub-results in full so callers can inspect
    all scalar diagnostics from the inner run and the w-update step.

    Attributes:
        run_result:       Full result from the inner fixed-length trainer run
                          (``CTRLRunResult`` from ``ctrl_train_run``).
        w_update_result:  Full result from the outer w-update step
                          (``CTRLWUpdateResult`` from ``ctrl_w_update``).
        w_prev:           Lagrange multiplier used for the inner run (input ``w``).
        w_next:           Updated Lagrange multiplier after projection;
                          equals ``w_update_result.w_next``.
    """

    run_result: CTRLRunResult
    w_update_result: CTRLWUpdateResult
    w_prev: float
    w_next: float


def ctrl_outer_iter(
    actor: ActorBase,
    critic: CriticBase,
    env: PortfolioEnv,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    w: float,
    target_return_z: float,
    w_step_size: float,
    n_updates: int,
    entropy_temp: float,
    base_seed: int | None = None,
    w_min: float | None = None,
    w_max: float | None = None,
) -> CTRLOuterIterResult:
    """Execute one CTRL outer iteration: inner run then w update.

    THEOREM-ALIGNED: implements one step of the slow outer Lagrange-multiplier
    loop (Huang–Jia–Zhou 2025, §3.5, §3.8):

    1. Run ``ctrl_train_run`` for ``n_updates`` inner actor/critic steps at
       constant ``w``, yielding a ``CTRLRunResult``.
    2. Run ``ctrl_w_update`` using ``run_result.final_step.terminal_wealth``
       as x_T to compute the new ``w``.

    The updated ``w_next`` should be fed as ``w`` to the next outer iteration
    by the caller.  This function does not manage the outer schedule.

    Args:
        actor:            Stochastic behavior policy (``ActorBase``).
        critic:           Value function (``CriticBase``).
        env:              Portfolio environment.
        actor_optimizer:  PyTorch optimiser bound to actor parameters.
        critic_optimizer: PyTorch optimiser bound to critic parameters.
        w:                Current Lagrange multiplier / target-wealth parameter.
        target_return_z:  Target terminal wealth z for the w-update signal.
        w_step_size:      Positive outer-loop step size a_w.  Must be > 0.
        n_updates:        Number of inner actor/critic update steps.  Must be >= 1.
        entropy_temp:     Entropy regularisation temperature γ for inner steps.
        base_seed:        Optional base seed for the inner run; step k uses
                          ``base_seed + k``.
        w_min:            Optional lower bound for w projection.
        w_max:            Optional upper bound for w projection.

    Returns:
        ``CTRLOuterIterResult`` with ``run_result``, ``w_update_result``,
        ``w_prev``, and ``w_next``.

    Raises:
        ValueError: propagated from ``ctrl_train_run`` if ``n_updates < 1``.
        ValueError: propagated from ``ctrl_w_update`` if ``w_step_size <= 0``
                    or if ``w_min > w_max``.
    """
    # 1. Inner fixed-length trainer run at constant w.
    run_result = ctrl_train_run(
        actor,
        critic,
        env,
        actor_optimizer,
        critic_optimizer,
        w=w,
        entropy_temp=entropy_temp,
        n_updates=n_updates,
        base_seed=base_seed,
    )

    # 2. Outer w update using terminal wealth from the final inner step.
    w_update_result = ctrl_w_update(
        w=w,
        terminal_wealth=run_result.final_step.terminal_wealth,
        target_return_z=target_return_z,
        step_size=w_step_size,
        w_min=w_min,
        w_max=w_max,
    )

    return CTRLOuterIterResult(
        run_result=run_result,
        w_update_result=w_update_result,
        w_prev=w,
        w_next=w_update_result.w_next,
    )
