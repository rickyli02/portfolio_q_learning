"""CTRL baseline trajectory, policy-evaluation, and objective primitives.

THEOREM-ALIGNED SCAFFOLDING (Huang–Jia–Zhou 2025, companion §3.1, §4.1)
------------------------------------------------------------------------
This module provides the data-flow boundary for the CTRL baseline algorithm:

Rollout / evaluation (Phase 7A)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1. ``CTRLTrajectory`` — container for a single stochastic behavior-policy
   rollout, holding exactly the quantities required by the theorem-aligned
   critic (martingale) and actor (policy-gradient) update steps.

2. ``collect_ctrl_trajectory`` — stochastic behavior-policy rollout using
   the approved GaussianActor and GBMPortfolioEnv interfaces.

3. ``evaluate_ctrl_deterministic`` — deterministic execution-policy
   evaluation using the actor mean action; used for comparison against
   the oracle benchmark and for reporting terminal-wealth statistics.

Objective and loss primitives (Phase 8A)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
4. ``CTRLCriticEval`` — container for *detached* critic J values along a
   trajectory; diagnostic inputs to the martingale critic loss.

5. ``CTRLMartingaleResiduals`` — one-step martingale residuals
   δ_k = J(t_{k+1}, x_{k+1}) - J(t_k, x_k) + γ·log π_k·dt; stored
   detached (companion §4.5).

6. ``CTRLTrajectoryStats`` — aggregated log-prob and entropy statistics
   for outer-loop diagnostics.

7. ``evaluate_critic_on_trajectory`` — detached critic evaluation along a
   collected trajectory (uses ``torch.no_grad()``).

8. ``compute_martingale_residuals`` — computes detached δ_k.

9. ``aggregate_trajectory_stats`` — aggregates log-prob and entropy terms.

10. ``compute_terminal_mv_objective`` — terminal MV objective (x_T−w)²−(w−z)².

11. ``compute_w_update_target`` — outer-loop Lagrange signal x_T − z.

Gradient-tracked re-evaluation (Phase 8B)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
12. ``CTRLGradEval`` — container for gradient-tracked actor and critic
    re-evaluations on a stored trajectory; preserves grad paths for trainer
    use.  Distinct from Phase 8A detached helpers.

13. ``reeval_ctrl_trajectory`` — re-evaluates actor log-probs, entropy, and
    critic J values with gradient tracking enabled so the trainer can call
    ``.backward()`` on them.

SCOPE BOUNDARY
--------------
This module provides pure computation primitives only.  The following are
NOT implemented here:
- actor parameter updates (optimizer steps, gradient computation),
- critic (θ) parameter updates,
- outer-loop w update schedules or projections,
- training / episode loops,
- config dispatch.

These belong in the trainer layer (src/train/) once assigned in a future
bounded task.

REPO ENGINEERING NOTES
----------------------
- Time at step k is computed as k * dt before calling env.step(), matching
  the GBMPortfolioEnv convention (step.time = step_idx * dt).
- Log-probs and entropy are collected under torch.no_grad() and stored
  detached; gradient computation happens during the update step using the
  re-evaluated distribution, not the stored values.
- Critic values in CTRLCriticEval are also stored detached; re-evaluation
  with grad tracking is the trainer's responsibility.
- ``w`` (outer-loop Lagrange multiplier / target wealth) is passed
  explicitly to all helpers; it is not stored in the environment because
  it is updated on a different outer schedule.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from src.envs.base_env import PortfolioEnv
from src.models.base import ActorBase, CriticBase


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
    _, info = env.reset(seed=seed)
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
    _, info = env.reset(seed=seed)
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


# ---------------------------------------------------------------------------
# Phase 8A: objective and loss primitives
# ---------------------------------------------------------------------------


@dataclass
class CTRLCriticEval:
    """Critic J values evaluated at each step of a trajectory.

    THEOREM-ALIGNED: J(t_k, x_k; w; θ) and J(t_{k+1}, x_{k+1}; w; θ) at
    each step are the direct inputs to the CTRL martingale critic loss and
    actor gradient (Huang–Jia–Zhou 2025, §3.3, §3.8 and companion §4.5).

    Attributes:
        J_at_steps: J(t_k, x_k; w; θ) for k=0..n-1, shape ``(n_steps,)``.
        J_at_next:  J(t_{k+1}, x_{k+1}; w; θ) for k=0..n-1, shape ``(n_steps,)``.
        dt:         Step size (horizon / n_steps) used to construct t_{k+1}.
        w:          Outer-loop target-wealth parameter.
    """

    J_at_steps: torch.Tensor   # (n_steps,)
    J_at_next: torch.Tensor    # (n_steps,)
    dt: float
    w: float


@dataclass
class CTRLMartingaleResiduals:
    """One-step martingale residuals for CTRL critic and actor updates.

    THEOREM-ALIGNED: δ_k = J(t_{k+1}, x_{k+1}) - J(t_k, x_k) + γ·log π_k·dt
    is the discrete approximation of dJ + γ·log π·dt from the martingale PE
    condition (Huang–Jia–Zhou 2025, §3.3).  These residuals are the shared
    building block for both the critic update (weighted by ∂J/∂θ) and the
    actor update (weighted by ∂log π/∂φ).

    Gradient computation for actual parameter updates is left to the trainer;
    these residuals are stored detached.

    Attributes:
        residuals:    δ_k for k=0..n-1, shape ``(n_steps,)``.
        entropy_temp: Entropy regularization temperature γ used.
        dt:           Step size used.
    """

    residuals: torch.Tensor   # (n_steps,)
    entropy_temp: float
    dt: float


@dataclass
class CTRLTrajectoryStats:
    """Aggregated log-prob and entropy statistics from a CTRLTrajectory.

    REPO ENGINEERING: convenience aggregation of the log-prob and entropy
    fields already stored on CTRLTrajectory.  Useful for outer-loop
    diagnostics and monitoring exploration levels.  No theorem-specific
    formula is applied; these are raw trajectory summaries.

    Attributes:
        sum_log_prob:  Σ_k log π(u_k|t_k, x_k; w), scalar.
        mean_log_prob: Mean log π over steps, scalar.
        sum_entropy:   Σ_k H[π(·|t_k)], scalar.
        mean_entropy:  Mean H over steps, scalar.
        n_steps:       Number of steps in the trajectory.
    """

    sum_log_prob: torch.Tensor   # scalar
    mean_log_prob: torch.Tensor  # scalar
    sum_entropy: torch.Tensor    # scalar
    mean_entropy: torch.Tensor   # scalar
    n_steps: int


def evaluate_critic_on_trajectory(
    critic: CriticBase,
    traj: CTRLTrajectory,
    dt: float,
) -> CTRLCriticEval:
    """Evaluate the critic J along a collected trajectory.

    THEOREM-ALIGNED: evaluates J(t_k, x_k; w; θ) and J(t_{k+1}, x_{k+1}; w; θ)
    at each step.  These paired values are required by the CTRL martingale
    critic loss and actor gradient (Huang–Jia–Zhou 2025, §3.3 and companion §4.5).

    REPO ENGINEERING: calls ``critic.forward`` with batched ``(t, wealth, w)``
    tensors constructed from the trajectory's stored times and wealth path.
    Operates under ``torch.no_grad()``; re-evaluation with gradient tracking
    is the trainer layer's responsibility.

    Args:
        critic: Value function satisfying ``CriticBase``.
        traj: Collected stochastic trajectory from ``collect_ctrl_trajectory``.
        dt: Step size (horizon / n_steps).  Must match the spacing used when
            collecting the trajectory.

    Returns:
        ``CTRLCriticEval`` with ``J_at_steps``, ``J_at_next``, ``dt``, and ``w``.
    """
    t_curr = traj.times                   # (n_steps,)
    t_next = t_curr + dt                  # (n_steps,)
    x_curr = traj.wealth_path[:-1]        # (n_steps,)
    x_next = traj.wealth_path[1:]         # (n_steps,)

    with torch.no_grad():
        J_steps = critic(t_curr, x_curr, traj.w).detach()  # (n_steps,)
        J_next = critic(t_next, x_next, traj.w).detach()   # (n_steps,)

    return CTRLCriticEval(
        J_at_steps=J_steps,
        J_at_next=J_next,
        dt=dt,
        w=traj.w,
    )


def compute_martingale_residuals(
    critic_eval: CTRLCriticEval,
    traj: CTRLTrajectory,
    entropy_temp: float,
) -> CTRLMartingaleResiduals:
    """Compute one-step martingale residuals δ_k.

    THEOREM-ALIGNED: implements the discrete approximation of dJ + γ·log π·dt
    from the martingale PE condition (Huang–Jia–Zhou 2025, §3.3):

        δ_k = J(t_{k+1}, x_{k+1}; w) - J(t_k, x_k; w) + γ·log π(u_k|...)·dt

    These residuals are the inputs to both the CTRL critic update (weighted by
    ∂J/∂θ) and the actor update (weighted by ∂log π/∂φ₁ or ∂log π/∂φ₂⁻¹).
    They are stored detached; gradient-tracked re-evaluation of critic and actor
    values is the trainer layer's responsibility.

    Args:
        critic_eval: Pre-evaluated J values from ``evaluate_critic_on_trajectory``.
        traj: Trajectory whose ``log_probs`` correspond to the same episode as
            ``critic_eval`` (same actor, same rollout).
        entropy_temp: Entropy regularization temperature γ.

    Returns:
        ``CTRLMartingaleResiduals`` with residuals tensor and metadata.
    """
    dJ = critic_eval.J_at_next - critic_eval.J_at_steps          # (n_steps,)
    entropy_penalty = entropy_temp * traj.log_probs * critic_eval.dt  # (n_steps,)
    residuals = (dJ + entropy_penalty).detach()

    return CTRLMartingaleResiduals(
        residuals=residuals,
        entropy_temp=entropy_temp,
        dt=critic_eval.dt,
    )


def aggregate_trajectory_stats(traj: CTRLTrajectory) -> CTRLTrajectoryStats:
    """Aggregate log-prob and entropy statistics from a trajectory.

    REPO ENGINEERING: convenience aggregation of the log-prob and entropy
    fields already stored on ``CTRLTrajectory``.  No theorem-specific formula
    is applied; these are raw trajectory summaries for diagnostics.

    Args:
        traj: Collected trajectory from ``collect_ctrl_trajectory``.

    Returns:
        ``CTRLTrajectoryStats`` with summed and mean log-prob and entropy.
    """
    return CTRLTrajectoryStats(
        sum_log_prob=traj.log_probs.sum(),
        mean_log_prob=traj.log_probs.mean(),
        sum_entropy=traj.entropy_terms.sum(),
        mean_entropy=traj.entropy_terms.mean(),
        n_steps=traj.times.shape[0],
    )


def compute_terminal_mv_objective(
    traj: CTRLTrajectory,
    target_return_z: float,
) -> torch.Tensor:
    """Compute the terminal mean-variance objective tied to the outer-loop w.

    THEOREM-ALIGNED: returns the paper-aligned terminal critic condition

        J(T, x_T; w) = (x_T - w)² - (w - z)²

    from the QuadraticCritic terminal condition (Huang–Jia–Zhou 2025, §3.6
    and companion §3.1).  At t = T the time-correction terms θ₂(t²−T²) and
    θ₁(t−T) vanish and the exponential factor e^{−θ₃·0} = 1, leaving only
    the quadratic wealth deviation minus the (w−z)² penalty.

    This quantity is the paper-aligned terminal objective tied to the current
    outer-loop w stored in the trajectory.

    Note: ``compute_w_update_target`` returns the *outer-loop multiplier signal*
    x_T − z used in the Lagrange update step; this function returns the
    *terminal MV objective* (x_T − w)² − (w − z)², which is a distinct quantity.

    Args:
        traj: Trajectory whose ``terminal_wealth`` is x_T and ``w`` is the
            outer-loop target used during collection.
        target_return_z: Target terminal wealth z (``reward.target_return``
            in config).

    Returns:
        Scalar tensor (x_T - w)² - (w - z)².
    """
    dtype = traj.terminal_wealth.dtype
    device = traj.terminal_wealth.device
    w_t = torch.as_tensor(traj.w, dtype=dtype, device=device)
    z_t = torch.as_tensor(target_return_z, dtype=dtype, device=device)
    return (traj.terminal_wealth - w_t) ** 2 - (w_t - z_t) ** 2


def compute_w_update_target(
    traj: CTRLTrajectory,
    target_return_z: float,
) -> torch.Tensor:
    """Compute the outer-loop Lagrange multiplier update signal x_T - z.

    THEOREM-ALIGNED: returns x_T - z, the raw terminal quantity from the
    Lagrange multiplier update step (Huang–Jia–Zhou 2025, §3.5, §3.8):

        w_{n+1} = Proj(w_n - a_{w,n} · (x_T - z))

    This is distinct from the terminal MV objective (x_T − w)² − (w − z)²
    returned by ``compute_terminal_mv_objective``.  The projection operator
    and step-size schedule ``a_{w,n}`` are left to the trainer layer.

    Args:
        traj: Trajectory whose ``terminal_wealth`` is x_T.
        target_return_z: Target terminal wealth z (``reward.target_return``
            in config).

    Returns:
        Scalar tensor x_T - z.
    """
    z = torch.as_tensor(
        target_return_z,
        dtype=traj.terminal_wealth.dtype,
        device=traj.terminal_wealth.device,
    )
    return traj.terminal_wealth - z


# ---------------------------------------------------------------------------
# Phase 8B: gradient-tracked re-evaluation
# ---------------------------------------------------------------------------


@dataclass
class CTRLGradEval:
    """Gradient-tracked actor and critic re-evaluation on a stored trajectory.

    THEOREM-ALIGNED: preserves gradient paths w.r.t. actor parameters φ
    and critic parameters θ so the trainer can compute:

    - actor mean gradient:       ∂log π(u_k|t_k, x_k; w; φ)/∂φ₁
    - actor covariance gradient: ∂log π(u_k|...)/∂φ₂⁻¹
    - critic gradient:           ∂J(t_k, x_k; w; θ)/∂θ

    (Huang–Jia–Zhou 2025, §3.8 and companion §4.5)

    REPO ENGINEERING: this container is intentionally distinct from
    ``CTRLCriticEval`` (Phase 8A, detached) and ``CTRLTrajectoryStats``
    (Phase 8A, detached diagnostics).  The stored tensors have ``requires_grad``
    tied to the live actor / critic parameter graph; do NOT call ``.detach()``
    on them before passing to trainer loss computations.

    Attributes:
        log_probs:     log π(u_k|t_k,x_k;w;φ) recomputed with grad, ``(n_steps,)``.
        entropy_terms: H[π(·|t_k;φ)] recomputed with grad, ``(n_steps,)``.
        j_at_steps:    J(t_k,x_k;w;θ) recomputed with grad, ``(n_steps,)``.
        j_at_next:     J(t_{k+1},x_{k+1};w;θ) recomputed with grad, ``(n_steps,)``.
        dt:            Step size used to construct t_{k+1} = t_k + dt.
        w:             Outer-loop target-wealth parameter.
    """

    log_probs: torch.Tensor      # (n_steps,) — grad w.r.t. actor φ
    entropy_terms: torch.Tensor  # (n_steps,) — grad w.r.t. actor φ
    j_at_steps: torch.Tensor     # (n_steps,) — grad w.r.t. critic θ
    j_at_next: torch.Tensor      # (n_steps,) — grad w.r.t. critic θ
    dt: float
    w: float


def reeval_ctrl_trajectory(
    actor: ActorBase,
    critic: CriticBase,
    traj: CTRLTrajectory,
    dt: float,
) -> CTRLGradEval:
    """Re-evaluate actor and critic on a stored trajectory with gradient tracking.

    THEOREM-ALIGNED: re-computes log π(u_k|t_k,x_k;w;φ), H[π(·|t_k;φ)],
    J(t_k,x_k;w;θ), and J(t_{k+1},x_{k+1};w;θ) using the stored trajectory
    state ``(t_k, x_k, u_k)`` but with live model parameters.  The outputs
    have gradient paths w.r.t. actor and critic parameters so that:

    - critic loss:  Σ_k ∂J/∂θ · δ_k  can be differentiated through J_at_steps/J_at_next
    - actor loss:   Σ_k ∂log π/∂φ · δ_k  can be differentiated through log_probs

    (Huang–Jia–Zhou 2025, §3.8 and companion §4.5)

    REPO ENGINEERING:
    - The stored trajectory actions, times, and wealth path are detached inputs;
      gradient flows only through actor / critic parameters, not through the
      trajectory state (consistent with the off-policy re-evaluation pattern).
    - Step-by-step loop is used rather than a fully-batched call to remain safe
      with the current actor API, which handles scalar ``t`` per step.
    - Contrast with ``evaluate_critic_on_trajectory`` (Phase 8A), which wraps
      everything in ``torch.no_grad()`` and returns detached tensors.

    Args:
        actor:  Stochastic behavior policy satisfying ``ActorBase``.
        critic: Value function satisfying ``CriticBase``.
        traj:   Stored trajectory from ``collect_ctrl_trajectory``.
        dt:     Step size (horizon / n_steps); must match the trajectory spacing.

    Returns:
        ``CTRLGradEval`` with gradient-tracked log-probs, entropy, and J values.
    """
    n_steps = traj.times.shape[0]

    log_prob_list: list[torch.Tensor] = []
    entropy_list: list[torch.Tensor] = []
    j_steps_list: list[torch.Tensor] = []
    j_next_list: list[torch.Tensor] = []

    for k in range(n_steps):
        t_k = traj.times[k]
        x_k = traj.wealth_path[k]
        x_k1 = traj.wealth_path[k + 1]
        u_k = traj.actions[k]

        # Actor re-evaluation: gradient flows through φ₁, φ₂, φ₃
        lp_k = actor.log_prob(u_k, t_k, x_k, traj.w)
        h_k = actor.entropy(t_k)

        # Critic re-evaluation: gradient flows through θ₁, θ₂, θ₃
        j_k = critic(t_k, x_k, traj.w)
        j_k1 = critic(t_k + dt, x_k1, traj.w)

        log_prob_list.append(lp_k)
        entropy_list.append(h_k)
        j_steps_list.append(j_k)
        j_next_list.append(j_k1)

    return CTRLGradEval(
        log_probs=torch.stack(log_prob_list),     # (n_steps,)
        entropy_terms=torch.stack(entropy_list),  # (n_steps,)
        j_at_steps=torch.stack(j_steps_list),     # (n_steps,)
        j_at_next=torch.stack(j_next_list),       # (n_steps,)
        dt=dt,
        w=traj.w,
    )
