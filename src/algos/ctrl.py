"""CTRL baseline trajectory and policy-evaluation scaffolding.

THEOREM-ALIGNED SCAFFOLDING (Huang–Jia–Zhou 2025, companion §3.1, §4.1)
------------------------------------------------------------------------
This module provides the data-flow boundary for the CTRL baseline algorithm:

1. ``CTRLTrajectory`` — container for a single stochastic behavior-policy
   rollout, holding exactly the quantities required by the theorem-aligned
   critic (martingale) and actor (policy-gradient) update steps.

2. ``collect_ctrl_trajectory`` — stochastic behavior-policy rollout using
   the approved GaussianActor and GBMPortfolioEnv interfaces.

3. ``evaluate_ctrl_deterministic`` — deterministic execution-policy
   evaluation using the actor mean action; used for comparison against
   the oracle benchmark and for reporting terminal-wealth statistics.

SCOPE BOUNDARY
--------------
This module is scaffolding only.  The following are NOT implemented here:
- actor parameter updates,
- critic (θ) parameter updates,
- outer-loop w (Lagrange multiplier) updates,
- training loops,
- config dispatch.

These belong in the trainer layer (src/train/) once the algorithm-layer
update equations are assigned in a future bounded task.

REPO ENGINEERING NOTES
----------------------
- Time at step k is computed as k * dt before calling env.step(), matching
  the GBMPortfolioEnv convention (step.time = step_idx * dt).
- Log-probs and entropy are collected under torch.no_grad() and stored
  detached; gradient computation happens during the update step using the
  re-evaluated distribution, not the stored values.
- ``w`` (outer-loop Lagrange multiplier / target wealth) is passed
  explicitly to all helpers; it is not stored in the environment because
  it is updated on a different outer schedule.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from src.envs.base_env import PortfolioEnv
from src.models.base import ActorBase


@dataclass
class CTRLTrajectory:
    """Container for one stochastic behavior-policy rollout.

    THEOREM-ALIGNED: stores exactly the fields required for the CTRL
    martingale critic loss and actor policy-gradient update.

    Attributes:
        times: Step start-times t_0 … t_{n-1}, shape ``(n_steps,)``.
        wealth_path: Wealth at each time including t=0 and t=T,
            shape ``(n_steps + 1,)``.
        actions: Sampled dollar allocations, shape ``(n_steps, n_risky)``.
        log_probs: Log π(u_k | t_k, x_k; w) for each sampled action,
            shape ``(n_steps,)``.
        entropy_terms: H[π(·|t_k)] at each step, shape ``(n_steps,)``.
        terminal_wealth: Final portfolio wealth x_T, scalar tensor.
        w: Outer-loop target-wealth parameter used during collection.
    """

    times: torch.Tensor            # (n_steps,)
    wealth_path: torch.Tensor      # (n_steps + 1,)
    actions: torch.Tensor          # (n_steps, n_risky)
    log_probs: torch.Tensor        # (n_steps,)
    entropy_terms: torch.Tensor    # (n_steps,)
    terminal_wealth: torch.Tensor  # scalar
    w: float


@dataclass
class CTRLEvalResult:
    """Output of one deterministic execution-policy evaluation episode.

    REPO ENGINEERING: stores the deterministic path for comparison against
    oracle benchmarks and for terminal-wealth reporting.

    Attributes:
        times: Step start-times, shape ``(n_steps,)``.
        wealth_path: Wealth at each time, shape ``(n_steps + 1,)``.
        actions: Deterministic mean actions, shape ``(n_steps, n_risky)``.
        terminal_wealth: Final portfolio wealth x_T, scalar tensor.
        w: Outer-loop target-wealth parameter used during evaluation.
    """

    times: torch.Tensor            # (n_steps,)
    wealth_path: torch.Tensor      # (n_steps + 1,)
    actions: torch.Tensor          # (n_steps, n_risky)
    terminal_wealth: torch.Tensor  # scalar
    w: float


def collect_ctrl_trajectory(
    actor: ActorBase,
    env: PortfolioEnv,
    w: float,
    seed: int | None = None,
) -> CTRLTrajectory:
    """Collect one stochastic behavior-policy trajectory.

    THEOREM-ALIGNED: samples actions from the exploratory Gaussian
    π(·|t, x; w; φ) at each step and records log-probability and entropy,
    as required by the CTRL actor update.

    REPO ENGINEERING: time at step k is computed as k * dt to avoid
    depending on env internal state between action sampling and stepping.
    Log-probs and entropy are collected detached; gradients are recomputed
    from stored (t, x, u) tuples during the update step.

    Args:
        actor: Stochastic behavior policy satisfying ``ActorBase``.
        env: Portfolio environment (e.g. ``GBMPortfolioEnv``).
        w: Outer-loop Lagrange / target-wealth parameter.
        seed: Optional RNG seed for the environment reset.

    Returns:
        ``CTRLTrajectory`` populated from one full episode.
    """
    # Seed both env RNG and PyTorch global RNG so actor.sample() is also
    # reproducible.  torch.manual_seed affects torch.randn_like in sample().
    if seed is not None:
        torch.manual_seed(seed)
    obs, info = env.reset(seed=seed)
    current_wealth: torch.Tensor = info["wealth"].to(dtype=torch.float32)
    device = current_wealth.device
    dt = env.horizon / env.n_steps

    times_list: list[torch.Tensor] = []
    wealth_list: list[torch.Tensor] = [current_wealth.clone()]
    action_list: list[torch.Tensor] = []
    log_prob_list: list[torch.Tensor] = []
    entropy_list: list[torch.Tensor] = []

    for k in range(env.n_steps):
        t_k = torch.tensor(k * dt, dtype=torch.float32, device=device)

        with torch.no_grad():
            action_k = actor.sample(t_k, current_wealth, w)
            lp_k = actor.log_prob(action_k, t_k, current_wealth, w)
            h_k = actor.entropy(t_k)

        step = env.step(action_k.detach())
        next_wealth = step.wealth.to(dtype=torch.float32, device=device)

        times_list.append(t_k)
        action_list.append(action_k.detach())
        log_prob_list.append(lp_k.detach())
        entropy_list.append(h_k.detach())
        wealth_list.append(next_wealth)
        current_wealth = next_wealth

    times = torch.stack(times_list)            # (n_steps,)
    wealth_path = torch.stack(wealth_list)     # (n_steps + 1,)
    actions = torch.stack(action_list)         # (n_steps, n_risky)
    log_probs = torch.stack(log_prob_list)     # (n_steps,)
    entropy_terms = torch.stack(entropy_list)  # (n_steps,)

    return CTRLTrajectory(
        times=times,
        wealth_path=wealth_path,
        actions=actions,
        log_probs=log_probs,
        entropy_terms=entropy_terms,
        terminal_wealth=wealth_path[-1],
        w=w,
    )


def evaluate_ctrl_deterministic(
    actor: ActorBase,
    env: PortfolioEnv,
    w: float,
    seed: int | None = None,
) -> CTRLEvalResult:
    """Run one deterministic execution-policy evaluation episode.

    THEOREM-ALIGNED: uses the actor mean action û = −φ₁·(x−w), which is
    the deterministic execution policy for comparison against the oracle
    benchmark and for out-of-training terminal-wealth reporting.

    REPO ENGINEERING: no stochastic noise is added; this is the pure
    deterministic path for evaluation purposes only.

    Args:
        actor: Policy satisfying ``ActorBase``; only ``mean_action`` is used.
        env: Portfolio environment.
        w: Outer-loop target-wealth parameter.
        seed: Optional RNG seed for the environment reset.

    Returns:
        ``CTRLEvalResult`` with the deterministic wealth path and actions.
    """
    obs, info = env.reset(seed=seed)
    current_wealth: torch.Tensor = info["wealth"].to(dtype=torch.float32)
    device = current_wealth.device
    dt = env.horizon / env.n_steps

    times_list: list[torch.Tensor] = []
    wealth_list: list[torch.Tensor] = [current_wealth.clone()]
    action_list: list[torch.Tensor] = []

    for k in range(env.n_steps):
        t_k = torch.tensor(k * dt, dtype=torch.float32, device=device)

        with torch.no_grad():
            action_k = actor.mean_action(t_k, current_wealth, w)

        step = env.step(action_k.detach())
        next_wealth = step.wealth.to(dtype=torch.float32, device=device)

        times_list.append(t_k)
        action_list.append(action_k.detach())
        wealth_list.append(next_wealth)
        current_wealth = next_wealth

    times = torch.stack(times_list)         # (n_steps,)
    wealth_path = torch.stack(wealth_list)  # (n_steps + 1,)
    actions = torch.stack(action_list)      # (n_steps, n_risky)

    return CTRLEvalResult(
        times=times,
        wealth_path=wealth_path,
        actions=actions,
        terminal_wealth=wealth_path[-1],
        w=w,
    )
