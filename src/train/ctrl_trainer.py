"""CTRL single-trajectory trainer step.

REPO ENGINEERING NOTES
----------------------
This module provides the narrowest useful integration slice of the CTRL
training loop: one stochastic trajectory → one critic gradient step → one
actor gradient step → one typed result.  It orchestrates the approved
mathematical primitives from ``src/algos/ctrl.py`` through standard PyTorch
optimisers without introducing any training-loop scaffolding.

SCOPE BOUNDARY
--------------
The following are NOT implemented here:
- multi-episode training loops
- outer-loop ``w`` update scheduling or projection
- replay-buffer or batch training
- TD(λ) traces
- checkpoint or logging infrastructure
- config-dispatch wiring
- evaluation / plotting scripts

These belong in future bounded tasks.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from src.algos.ctrl import (
    aggregate_trajectory_stats,
    collect_ctrl_trajectory,
    compute_ctrl_actor_loss,
    compute_ctrl_critic_loss,
    compute_martingale_residuals,
    evaluate_critic_on_trajectory,
    reeval_ctrl_trajectory,
)
from src.envs.base_env import PortfolioEnv
from src.models.base import ActorBase, CriticBase


@dataclass
class CTRLStepResult:
    """Scalar outputs from one CTRL trainer step.

    REPO ENGINEERING: stores only detached scalar diagnostics; all tensor
    graph references are released after the step function returns.

    Attributes:
        critic_loss:    Scalar critic surrogate loss value (detached float).
        actor_loss:     Scalar actor surrogate loss value (detached float).
        terminal_wealth: Terminal portfolio wealth x_T from the trajectory.
        sum_log_prob:   Σ_k log π(u_k | t_k, x_k; w) over the trajectory
                        (diagnostic; higher magnitude → more off-mode exploration).
        mean_entropy:   Mean H[π(·|t_k)] over steps (exploration level diagnostic).
        n_steps:        Number of steps in the trajectory.
    """

    critic_loss: float
    actor_loss: float
    terminal_wealth: float
    sum_log_prob: float
    mean_entropy: float
    n_steps: int


def ctrl_train_step(
    actor: ActorBase,
    critic: CriticBase,
    env: PortfolioEnv,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    w: float,
    entropy_temp: float,
    seed: int | None = None,
) -> CTRLStepResult:
    """Perform one CTRL single-trajectory update step.

    REPO ENGINEERING: orchestrates the Phase 8A–8C primitives in order:

    1. Derive ``dt`` from ``env.horizon`` and ``env.n_steps`` (not from caller).
    2. Collect one stochastic trajectory under the current actor policy.
    3. Compute detached Phase 8A primitives (critic eval, martingale residuals,
       trajectory stats).
    4. Gradient-tracked Phase 8B re-evaluation on the stored trajectory.
    5. Critic gradient step: assemble L_critic, backward, step.
    6. Actor gradient step: assemble L_actor, backward, step.
    7. Return a ``CTRLStepResult`` with detached scalar diagnostics.

    GRAPH ISOLATION NOTE: the critic and actor gradient graphs are independent
    because ``reeval_ctrl_trajectory`` computes separate forward passes
    through critic and actor parameters using only detached trajectory inputs.
    ``critic_loss.backward()`` frees only the critic-parameter path; the
    log_probs graph for the actor is unaffected.  No ``retain_graph=True``
    is required.

    Args:
        actor:            Stochastic behavior policy (``ActorBase``).
        critic:           Value function (``CriticBase``).
        env:              Portfolio environment; must expose ``horizon`` and
                          ``n_steps`` properties.
        actor_optimizer:  PyTorch optimiser bound to actor parameters.
        critic_optimizer: PyTorch optimiser bound to critic parameters.
        w:                Outer-loop Lagrange / target-wealth parameter for
                          this step.  Updated by the outer loop (not here).
        entropy_temp:     Entropy regularisation temperature γ.
        seed:             Optional RNG seed for reproducible trajectory collection.

    Returns:
        ``CTRLStepResult`` with scalar diagnostics from this step.
    """
    # 0. Fail fast on non-finite model parameters before any computation.
    actor.validate_parameters()
    critic.validate_parameters()

    # 1. Derive dt from env (not caller-supplied) to prevent mismatches.
    dt: float = env.horizon / env.n_steps

    # 2. Collect one stochastic trajectory.
    traj = collect_ctrl_trajectory(actor, env, w=w, seed=seed)

    # 3. Phase 8A detached primitives.
    critic_eval = evaluate_critic_on_trajectory(critic, traj, dt=dt)
    residuals = compute_martingale_residuals(critic_eval, traj, entropy_temp=entropy_temp)
    stats = aggregate_trajectory_stats(traj)

    # 4. Phase 8B gradient-tracked re-evaluation.
    #    Inputs (t_k, x_k, u_k) are detached stored tensors; grad flows only
    #    through actor and critic parameters respectively.
    critic_optimizer.zero_grad()
    actor_optimizer.zero_grad()
    ge = reeval_ctrl_trajectory(actor, critic, traj, dt=dt)

    # 5. Critic gradient step.
    critic_loss = compute_ctrl_critic_loss(ge, residuals)
    critic_loss.backward()   # frees critic-param graph; actor graph intact
    critic_optimizer.step()

    # 6. Actor gradient step.
    actor_loss = compute_ctrl_actor_loss(ge, residuals)
    actor_loss.backward()
    actor_optimizer.step()

    return CTRLStepResult(
        critic_loss=critic_loss.item(),
        actor_loss=actor_loss.item(),
        terminal_wealth=traj.terminal_wealth.item(),
        sum_log_prob=stats.sum_log_prob.item(),
        mean_entropy=stats.mean_entropy.item(),
        n_steps=stats.n_steps,
    )
