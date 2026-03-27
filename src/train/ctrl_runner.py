"""CTRL fixed-length trainer run helper.

REPO ENGINEERING NOTES
----------------------
This module wraps the approved ``ctrl_train_step`` (Phase 9A) into a
fixed-length repeated-update runner.  It is the narrowest useful extension
of the single-step trainer: same models, same optimizer, same ``w`` across
all ``n_updates`` steps.

Seed scheduling: when a ``base_seed`` is provided, step k receives seed
``base_seed + k`` so the trajectory collection is deterministic and
reproducible across runs with identical model state.

SCOPE BOUNDARY
--------------
The following are NOT implemented here:
- outer-loop ``w`` updates or scheduling
- replay-buffer or batch training
- TD(λ) traces
- checkpoint or logging infrastructure
- config-dispatch wiring
- early stopping, callbacks, or metrics sinks
- full offline / online trainer classes

These belong in future bounded tasks.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from src.envs.base_env import PortfolioEnv
from src.models.base import ActorBase, CriticBase
from src.train.ctrl_trainer import CTRLStepResult, ctrl_train_step


@dataclass
class CTRLRunResult:
    """Scalar outputs from a fixed-length CTRL trainer run.

    REPO ENGINEERING: stores per-step result objects and a direct reference
    to the final step for convenience.  All tensor graph references are
    released inside ``ctrl_train_step`` before each result is appended.

    Attributes:
        steps:      Per-update results, length ``n_updates``.  Indexed as
                    ``result.steps[k]`` for the k-th update.
        final_step: Alias for ``steps[-1]``; the result of the last update.
        n_updates:  Number of update steps executed; equals ``len(steps)``.
    """

    steps: list[CTRLStepResult]
    final_step: CTRLStepResult
    n_updates: int


def ctrl_train_run(
    actor: ActorBase,
    critic: CriticBase,
    env: PortfolioEnv,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    w: float,
    entropy_temp: float,
    n_updates: int,
    base_seed: int | None = None,
) -> CTRLRunResult:
    """Execute a fixed-length sequence of CTRL single-trajectory update steps.

    REPO ENGINEERING: calls ``ctrl_train_step`` ``n_updates`` times with the
    same actor, critic, env, optimizers, ``w``, and ``entropy_temp``.  Model
    and optimizer state accumulate across steps (parameters are updated in
    place by each step's optimizer calls).

    Seed scheduling: if ``base_seed`` is provided, step k receives seed
    ``base_seed + k``, making the trajectory sequence fully reproducible
    given the same initial model state.  If ``base_seed`` is ``None``, each
    step uses a fresh unseeded trajectory (stochastic).

    Args:
        actor:            Stochastic behavior policy (``ActorBase``).
        critic:           Value function (``CriticBase``).
        env:              Portfolio environment.
        actor_optimizer:  PyTorch optimiser bound to actor parameters.
        critic_optimizer: PyTorch optimiser bound to critic parameters.
        w:                Outer-loop Lagrange / target-wealth parameter;
                          held fixed for the entire run.
        entropy_temp:     Entropy regularisation temperature γ; held fixed.
        n_updates:        Number of update steps to execute.  Must be >= 1.
        base_seed:        Optional base seed for deterministic trajectory
                          scheduling.  Step k uses seed ``base_seed + k``.

    Returns:
        ``CTRLRunResult`` with per-step history and final-step summary.

    Raises:
        ValueError: if ``n_updates < 1``.
    """
    if n_updates < 1:
        raise ValueError(f"n_updates must be >= 1, got {n_updates}")

    steps: list[CTRLStepResult] = []
    for k in range(n_updates):
        seed = base_seed + k if base_seed is not None else None
        step_result = ctrl_train_step(
            actor,
            critic,
            env,
            actor_optimizer,
            critic_optimizer,
            w=w,
            entropy_temp=entropy_temp,
            seed=seed,
        )
        steps.append(step_result)

    return CTRLRunResult(
        steps=steps,
        final_step=steps[-1],
        n_updates=n_updates,
    )
