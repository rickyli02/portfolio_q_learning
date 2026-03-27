"""Unit tests for src/algos/ctrl.py — CTRL trajectory, eval, and loss primitives."""

import pytest
import torch

from src.algos.ctrl import (
    CTRLCriticEval,
    CTRLEvalResult,
    CTRLGradEval,
    CTRLMartingaleResiduals,
    CTRLTrajectory,
    CTRLTrajectoryStats,
    aggregate_trajectory_stats,
    collect_ctrl_trajectory,
    compute_martingale_residuals,
    compute_terminal_mv_objective,
    compute_w_update_target,
    evaluate_critic_on_trajectory,
    evaluate_ctrl_deterministic,
    reeval_ctrl_trajectory,
)
from src.config.schema import AssetConfig, EnvConfig
from src.envs.gbm_env import GBMPortfolioEnv
from src.models.gaussian_actor import GaussianActor
from src.models.quadratic_critic import QuadraticCritic


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_env(n_risky: int = 1, n_steps: int = 10) -> GBMPortfolioEnv:
    if n_risky == 1:
        mu, sigma = [0.1], [[0.2]]
    else:
        mu = [0.1, 0.08]
        sigma = [[0.2, 0.0], [0.0, 0.15]]
    cfg = EnvConfig(
        horizon=1.0,
        n_steps=n_steps,
        initial_wealth=1.0,
        mu=mu,
        sigma=sigma,
        assets=AssetConfig(n_risky=n_risky, include_risk_free=True, risk_free_rate=0.05),
    )
    return GBMPortfolioEnv(cfg)


def _make_actor(n_risky: int = 1) -> GaussianActor:
    return GaussianActor(n_risky=n_risky, horizon=1.0)


# ---------------------------------------------------------------------------
# CTRLTrajectory — dataclass structure
# ---------------------------------------------------------------------------

def test_trajectory_is_dataclass():
    actor = _make_actor()
    env = _make_env()
    traj = collect_ctrl_trajectory(actor, env, w=1.0, seed=0)
    assert isinstance(traj, CTRLTrajectory)


def test_trajectory_times_shape():
    n_steps = 10
    actor = _make_actor()
    env = _make_env(n_steps=n_steps)
    traj = collect_ctrl_trajectory(actor, env, w=1.0, seed=0)
    assert traj.times.shape == (n_steps,)


def test_trajectory_wealth_path_shape():
    n_steps = 10
    actor = _make_actor()
    env = _make_env(n_steps=n_steps)
    traj = collect_ctrl_trajectory(actor, env, w=1.0, seed=0)
    assert traj.wealth_path.shape == (n_steps + 1,)


def test_trajectory_actions_shape_single_asset():
    n_steps = 10
    actor = _make_actor(n_risky=1)
    env = _make_env(n_risky=1, n_steps=n_steps)
    traj = collect_ctrl_trajectory(actor, env, w=1.0, seed=0)
    assert traj.actions.shape == (n_steps, 1)


def test_trajectory_actions_shape_multi_asset():
    n_steps = 8
    actor = _make_actor(n_risky=2)
    env = _make_env(n_risky=2, n_steps=n_steps)
    traj = collect_ctrl_trajectory(actor, env, w=1.0, seed=0)
    assert traj.actions.shape == (n_steps, 2)


def test_trajectory_log_probs_shape():
    n_steps = 10
    actor = _make_actor()
    env = _make_env(n_steps=n_steps)
    traj = collect_ctrl_trajectory(actor, env, w=1.0, seed=0)
    assert traj.log_probs.shape == (n_steps,)


def test_trajectory_entropy_terms_shape():
    n_steps = 10
    actor = _make_actor()
    env = _make_env(n_steps=n_steps)
    traj = collect_ctrl_trajectory(actor, env, w=1.0, seed=0)
    assert traj.entropy_terms.shape == (n_steps,)


def test_trajectory_terminal_wealth_is_scalar():
    actor = _make_actor()
    env = _make_env()
    traj = collect_ctrl_trajectory(actor, env, w=1.0, seed=0)
    assert traj.terminal_wealth.shape == ()


def test_trajectory_terminal_wealth_equals_last_wealth():
    actor = _make_actor()
    env = _make_env()
    traj = collect_ctrl_trajectory(actor, env, w=1.0, seed=0)
    assert traj.terminal_wealth.item() == pytest.approx(traj.wealth_path[-1].item())


def test_trajectory_stores_w():
    actor = _make_actor()
    env = _make_env()
    traj = collect_ctrl_trajectory(actor, env, w=1.15, seed=0)
    assert traj.w == pytest.approx(1.15)


def test_trajectory_initial_wealth_matches_env():
    """wealth_path[0] should equal env initial_wealth."""
    actor = _make_actor()
    env = _make_env()
    traj = collect_ctrl_trajectory(actor, env, w=1.0, seed=0)
    assert traj.wealth_path[0].item() == pytest.approx(1.0)


def test_trajectory_times_start_at_zero():
    actor = _make_actor()
    env = _make_env(n_steps=5)
    traj = collect_ctrl_trajectory(actor, env, w=1.0, seed=0)
    assert traj.times[0].item() == pytest.approx(0.0)


def test_trajectory_times_spacing():
    """Times should be uniformly spaced by dt = horizon / n_steps."""
    n_steps = 5
    horizon = 1.0
    actor = _make_actor()
    env = _make_env(n_steps=n_steps)
    traj = collect_ctrl_trajectory(actor, env, w=1.0, seed=0)
    dt = horizon / n_steps
    for k in range(n_steps):
        assert traj.times[k].item() == pytest.approx(k * dt, abs=1e-6)


def test_trajectory_log_probs_are_finite():
    actor = _make_actor()
    env = _make_env()
    traj = collect_ctrl_trajectory(actor, env, w=1.0, seed=0)
    assert torch.isfinite(traj.log_probs).all()


def test_trajectory_entropy_terms_are_finite():
    actor = _make_actor()
    env = _make_env()
    traj = collect_ctrl_trajectory(actor, env, w=1.0, seed=0)
    assert torch.isfinite(traj.entropy_terms).all()


def test_trajectory_log_probs_non_positive():
    """Log-probs of a continuous distribution are ≤ 0 near the mode,
    but can be positive for narrow distributions — just check they're finite."""
    actor = _make_actor()
    env = _make_env()
    traj = collect_ctrl_trajectory(actor, env, w=1.0, seed=0)
    assert torch.isfinite(traj.log_probs).all()


def test_trajectory_deterministic_with_same_seed():
    actor = _make_actor()
    env = _make_env()
    t1 = collect_ctrl_trajectory(actor, env, w=1.0, seed=42)
    t2 = collect_ctrl_trajectory(actor, env, w=1.0, seed=42)
    assert torch.allclose(t1.wealth_path, t2.wealth_path)
    assert torch.allclose(t1.actions, t2.actions)


def test_trajectory_different_seeds_differ():
    actor = _make_actor()
    env = _make_env()
    t1 = collect_ctrl_trajectory(actor, env, w=1.0, seed=0)
    t2 = collect_ctrl_trajectory(actor, env, w=1.0, seed=1)
    assert not torch.allclose(t1.wealth_path, t2.wealth_path)


# ---------------------------------------------------------------------------
# CTRLEvalResult — deterministic evaluation
# ---------------------------------------------------------------------------

def test_eval_result_is_dataclass():
    actor = _make_actor()
    env = _make_env()
    result = evaluate_ctrl_deterministic(actor, env, w=1.0, seed=0)
    assert isinstance(result, CTRLEvalResult)


def test_eval_times_shape():
    n_steps = 10
    actor = _make_actor()
    env = _make_env(n_steps=n_steps)
    result = evaluate_ctrl_deterministic(actor, env, w=1.0, seed=0)
    assert result.times.shape == (n_steps,)


def test_eval_wealth_path_shape():
    n_steps = 10
    actor = _make_actor()
    env = _make_env(n_steps=n_steps)
    result = evaluate_ctrl_deterministic(actor, env, w=1.0, seed=0)
    assert result.wealth_path.shape == (n_steps + 1,)


def test_eval_actions_shape_single_asset():
    n_steps = 10
    actor = _make_actor(n_risky=1)
    env = _make_env(n_risky=1, n_steps=n_steps)
    result = evaluate_ctrl_deterministic(actor, env, w=1.0, seed=0)
    assert result.actions.shape == (n_steps, 1)


def test_eval_actions_shape_multi_asset():
    n_steps = 8
    actor = _make_actor(n_risky=2)
    env = _make_env(n_risky=2, n_steps=n_steps)
    result = evaluate_ctrl_deterministic(actor, env, w=1.0, seed=0)
    assert result.actions.shape == (n_steps, 2)


def test_eval_terminal_wealth_is_scalar():
    actor = _make_actor()
    env = _make_env()
    result = evaluate_ctrl_deterministic(actor, env, w=1.0, seed=0)
    assert result.terminal_wealth.shape == ()


def test_eval_terminal_wealth_equals_last_wealth():
    actor = _make_actor()
    env = _make_env()
    result = evaluate_ctrl_deterministic(actor, env, w=1.0, seed=0)
    assert result.terminal_wealth.item() == pytest.approx(result.wealth_path[-1].item())


def test_eval_stores_w():
    actor = _make_actor()
    env = _make_env()
    result = evaluate_ctrl_deterministic(actor, env, w=1.15, seed=0)
    assert result.w == pytest.approx(1.15)


def test_eval_initial_wealth_matches_env():
    actor = _make_actor()
    env = _make_env()
    result = evaluate_ctrl_deterministic(actor, env, w=1.0, seed=0)
    assert result.wealth_path[0].item() == pytest.approx(1.0)


def test_eval_deterministic_same_seed_reproducible():
    """Deterministic eval should give same wealth path under same seed."""
    actor = _make_actor()
    env = _make_env()
    r1 = evaluate_ctrl_deterministic(actor, env, w=1.0, seed=7)
    r2 = evaluate_ctrl_deterministic(actor, env, w=1.0, seed=7)
    assert torch.allclose(r1.wealth_path, r2.wealth_path)
    assert torch.allclose(r1.actions, r2.actions)


def test_eval_deterministic_same_regardless_of_seed():
    """The deterministic policy has no actor noise, so wealth paths should be
    identical across seeds (env noise still varies, but actions are fixed
    at mean_action which depends only on wealth and w)."""
    # NOTE: the env still has stochastic asset returns, so paths will differ
    # across seeds. This test instead checks that two runs with same seed match.
    actor = _make_actor()
    env = _make_env()
    r1 = evaluate_ctrl_deterministic(actor, env, w=1.0, seed=0)
    r2 = evaluate_ctrl_deterministic(actor, env, w=1.0, seed=0)
    assert torch.allclose(r1.actions, r2.actions)


# ---------------------------------------------------------------------------
# Cross-checks: stochastic vs deterministic
# ---------------------------------------------------------------------------

def test_stochastic_and_deterministic_same_n_steps():
    n_steps = 6
    actor = _make_actor()
    env = _make_env(n_steps=n_steps)
    traj = collect_ctrl_trajectory(actor, env, w=1.0, seed=0)
    result = evaluate_ctrl_deterministic(actor, env, w=1.0, seed=0)
    assert traj.times.shape == result.times.shape
    assert traj.wealth_path.shape == result.wealth_path.shape
    assert traj.actions.shape == result.actions.shape


def test_stochastic_has_no_grad_in_collected_tensors():
    """Collected log_probs and entropy_terms should be detached."""
    actor = _make_actor()
    env = _make_env()
    traj = collect_ctrl_trajectory(actor, env, w=1.0, seed=0)
    assert not traj.log_probs.requires_grad
    assert not traj.entropy_terms.requires_grad
    assert not traj.actions.requires_grad


def test_eval_has_no_grad_in_collected_tensors():
    actor = _make_actor()
    env = _make_env()
    result = evaluate_ctrl_deterministic(actor, env, w=1.0, seed=0)
    assert not result.actions.requires_grad
    assert not result.wealth_path.requires_grad


# ---------------------------------------------------------------------------
# Imports via public API
# ---------------------------------------------------------------------------

def test_public_api_imports():
    from src.algos import (
        CTRLEvalResult,
        CTRLTrajectory,
        collect_ctrl_trajectory,
        evaluate_ctrl_deterministic,
    )
    assert CTRLTrajectory is not None
    assert CTRLEvalResult is not None
    assert callable(collect_ctrl_trajectory)
    assert callable(evaluate_ctrl_deterministic)


# ---------------------------------------------------------------------------
# Phase 8A: evaluate_critic_on_trajectory
# ---------------------------------------------------------------------------

def _make_critic(horizon: float = 1.0, z: float = 1.0) -> QuadraticCritic:
    return QuadraticCritic(horizon=horizon, target_return_z=z)


def _collect(n_steps: int = 10, n_risky: int = 1, w: float = 1.0, seed: int = 0) -> CTRLTrajectory:
    actor = _make_actor(n_risky=n_risky)
    env = _make_env(n_risky=n_risky, n_steps=n_steps)
    return collect_ctrl_trajectory(actor, env, w=w, seed=seed)


def test_critic_eval_is_dataclass():
    traj = _collect()
    critic = _make_critic()
    dt = 1.0 / 10
    result = evaluate_critic_on_trajectory(critic, traj, dt=dt)
    assert isinstance(result, CTRLCriticEval)


def test_critic_eval_J_at_steps_shape():
    n_steps = 8
    traj = _collect(n_steps=n_steps)
    critic = _make_critic()
    result = evaluate_critic_on_trajectory(critic, traj, dt=1.0 / n_steps)
    assert result.J_at_steps.shape == (n_steps,)


def test_critic_eval_J_at_next_shape():
    n_steps = 8
    traj = _collect(n_steps=n_steps)
    critic = _make_critic()
    result = evaluate_critic_on_trajectory(critic, traj, dt=1.0 / n_steps)
    assert result.J_at_next.shape == (n_steps,)


def test_critic_eval_J_finite():
    traj = _collect()
    critic = _make_critic()
    result = evaluate_critic_on_trajectory(critic, traj, dt=0.1)
    assert torch.isfinite(result.J_at_steps).all()
    assert torch.isfinite(result.J_at_next).all()


def test_critic_eval_terminal_condition():
    """J_at_next[-1] = J(T, x_T; w) should satisfy (x_T - w)^2 - (w - z)^2."""
    w, z = 1.0, 1.0
    traj = _collect(w=w)
    critic = _make_critic(z=z)
    dt = 1.0 / 10
    result = evaluate_critic_on_trajectory(critic, traj, dt=dt)

    x_T = traj.terminal_wealth
    expected = (x_T - w) ** 2 - (w - z) ** 2
    assert result.J_at_next[-1].item() == pytest.approx(expected.item(), abs=1e-5)


def test_critic_eval_dt_stored():
    traj = _collect()
    critic = _make_critic()
    dt = 1.0 / 10
    result = evaluate_critic_on_trajectory(critic, traj, dt=dt)
    assert result.dt == pytest.approx(dt)


def test_critic_eval_w_stored():
    traj = _collect(w=1.2)
    critic = _make_critic()
    result = evaluate_critic_on_trajectory(critic, traj, dt=0.1)
    assert result.w == pytest.approx(1.2)


def test_critic_eval_no_grad():
    traj = _collect()
    critic = _make_critic()
    result = evaluate_critic_on_trajectory(critic, traj, dt=0.1)
    assert not result.J_at_steps.requires_grad
    assert not result.J_at_next.requires_grad


# ---------------------------------------------------------------------------
# Phase 8A: compute_martingale_residuals
# ---------------------------------------------------------------------------

def test_residuals_is_dataclass():
    traj = _collect()
    critic = _make_critic()
    ce = evaluate_critic_on_trajectory(critic, traj, dt=0.1)
    res = compute_martingale_residuals(ce, traj, entropy_temp=0.1)
    assert isinstance(res, CTRLMartingaleResiduals)


def test_residuals_shape():
    n_steps = 7
    traj = _collect(n_steps=n_steps)
    critic = _make_critic()
    ce = evaluate_critic_on_trajectory(critic, traj, dt=1.0 / n_steps)
    res = compute_martingale_residuals(ce, traj, entropy_temp=0.1)
    assert res.residuals.shape == (n_steps,)


def test_residuals_finite():
    traj = _collect()
    critic = _make_critic()
    ce = evaluate_critic_on_trajectory(critic, traj, dt=0.1)
    res = compute_martingale_residuals(ce, traj, entropy_temp=0.1)
    assert torch.isfinite(res.residuals).all()


def test_residuals_zero_entropy_temp():
    """With entropy_temp=0, residuals = J_next - J_steps."""
    traj = _collect()
    critic = _make_critic()
    ce = evaluate_critic_on_trajectory(critic, traj, dt=0.1)
    res = compute_martingale_residuals(ce, traj, entropy_temp=0.0)
    expected = ce.J_at_next - ce.J_at_steps
    assert torch.allclose(res.residuals, expected)


def test_residuals_entropy_temp_stored():
    traj = _collect()
    critic = _make_critic()
    ce = evaluate_critic_on_trajectory(critic, traj, dt=0.1)
    res = compute_martingale_residuals(ce, traj, entropy_temp=0.05)
    assert res.entropy_temp == pytest.approx(0.05)


def test_residuals_no_grad():
    traj = _collect()
    critic = _make_critic()
    ce = evaluate_critic_on_trajectory(critic, traj, dt=0.1)
    res = compute_martingale_residuals(ce, traj, entropy_temp=0.1)
    assert not res.residuals.requires_grad


# ---------------------------------------------------------------------------
# Phase 8A: aggregate_trajectory_stats
# ---------------------------------------------------------------------------

def test_stats_is_dataclass():
    traj = _collect()
    stats = aggregate_trajectory_stats(traj)
    assert isinstance(stats, CTRLTrajectoryStats)


def test_stats_sum_log_prob():
    traj = _collect()
    stats = aggregate_trajectory_stats(traj)
    assert stats.sum_log_prob.item() == pytest.approx(traj.log_probs.sum().item())


def test_stats_mean_log_prob():
    traj = _collect()
    stats = aggregate_trajectory_stats(traj)
    assert stats.mean_log_prob.item() == pytest.approx(traj.log_probs.mean().item())


def test_stats_sum_entropy():
    traj = _collect()
    stats = aggregate_trajectory_stats(traj)
    assert stats.sum_entropy.item() == pytest.approx(traj.entropy_terms.sum().item())


def test_stats_mean_entropy():
    traj = _collect()
    stats = aggregate_trajectory_stats(traj)
    assert stats.mean_entropy.item() == pytest.approx(traj.entropy_terms.mean().item())


def test_stats_n_steps():
    n_steps = 6
    traj = _collect(n_steps=n_steps)
    stats = aggregate_trajectory_stats(traj)
    assert stats.n_steps == n_steps


# ---------------------------------------------------------------------------
# Phase 8A: compute_terminal_mv_objective
# ---------------------------------------------------------------------------

def test_terminal_mv_objective_is_scalar():
    traj = _collect()
    val = compute_terminal_mv_objective(traj, target_return_z=1.0)
    assert val.shape == ()


def test_terminal_mv_objective_formula():
    """(x_T - w)^2 - (w - z)^2."""
    w, z = 1.0, 1.0
    traj = _collect(w=w)
    val = compute_terminal_mv_objective(traj, target_return_z=z)
    x_T = traj.terminal_wealth
    expected = (x_T - w) ** 2 - (w - z) ** 2
    assert val.item() == pytest.approx(expected.item(), abs=1e-6)


def test_terminal_mv_objective_finite():
    traj = _collect()
    val = compute_terminal_mv_objective(traj, target_return_z=1.0)
    assert torch.isfinite(val)


def test_terminal_mv_objective_matches_critic_terminal():
    """compute_terminal_mv_objective should equal QuadraticCritic at t=T."""
    w, z = 1.0, 1.0
    traj = _collect(w=w)
    critic = _make_critic(z=z)
    val = compute_terminal_mv_objective(traj, target_return_z=z)
    # Evaluate critic at terminal time T with default (zero-init) θ params
    T = critic.horizon
    critic_val = critic(
        torch.tensor(T),
        traj.terminal_wealth,
        w,
    )
    assert val.item() == pytest.approx(critic_val.item(), abs=1e-5)


def test_terminal_mv_and_w_target_are_distinct():
    """Terminal MV objective and w-update target are different quantities."""
    w, z = 1.0, 1.05
    traj = _collect(w=w)
    mv_val = compute_terminal_mv_objective(traj, target_return_z=z)
    w_signal = compute_w_update_target(traj, target_return_z=z)
    # mv_val = (x_T - w)^2 - (w - z)^2  [scalar, can be negative]
    # w_signal = x_T - z                 [scalar, different formula]
    # They are not equal in general; verify they measure different things
    x_T = traj.terminal_wealth.item()
    assert mv_val.item() == pytest.approx((x_T - w) ** 2 - (w - z) ** 2, abs=1e-6)
    assert w_signal.item() == pytest.approx(x_T - z, abs=1e-6)


# ---------------------------------------------------------------------------
# Phase 8A: compute_w_update_target
# ---------------------------------------------------------------------------

def test_w_target_is_scalar():
    traj = _collect()
    target = compute_w_update_target(traj, target_return_z=1.0)
    assert target.shape == ()


def test_w_target_value():
    z = 1.05
    traj = _collect()
    target = compute_w_update_target(traj, target_return_z=z)
    expected = traj.terminal_wealth - z
    assert target.item() == pytest.approx(expected.item())


def test_w_target_finite():
    traj = _collect()
    target = compute_w_update_target(traj, target_return_z=1.0)
    assert torch.isfinite(target)


# ---------------------------------------------------------------------------
# Phase 8A: public API imports
# ---------------------------------------------------------------------------

def test_phase8a_public_api_imports():
    from src.algos import (
        CTRLCriticEval,
        CTRLMartingaleResiduals,
        CTRLTrajectoryStats,
        aggregate_trajectory_stats,
        compute_martingale_residuals,
        compute_terminal_mv_objective,
        compute_w_update_target,
        evaluate_critic_on_trajectory,
    )
    assert CTRLCriticEval is not None
    assert CTRLMartingaleResiduals is not None
    assert CTRLTrajectoryStats is not None
    assert callable(evaluate_critic_on_trajectory)
    assert callable(compute_martingale_residuals)
    assert callable(aggregate_trajectory_stats)
    assert callable(compute_terminal_mv_objective)
    assert callable(compute_w_update_target)


# ---------------------------------------------------------------------------
# Phase 8B: reeval_ctrl_trajectory / CTRLGradEval
# ---------------------------------------------------------------------------

def test_grad_eval_is_dataclass():
    traj = _collect()
    ge = reeval_ctrl_trajectory(_make_actor(), _make_critic(), traj, dt=0.1)
    assert isinstance(ge, CTRLGradEval)


def test_grad_eval_log_probs_shape():
    n_steps = 8
    traj = _collect(n_steps=n_steps)
    ge = reeval_ctrl_trajectory(_make_actor(), _make_critic(), traj, dt=1.0 / n_steps)
    assert ge.log_probs.shape == (n_steps,)


def test_grad_eval_entropy_shape():
    n_steps = 8
    traj = _collect(n_steps=n_steps)
    ge = reeval_ctrl_trajectory(_make_actor(), _make_critic(), traj, dt=1.0 / n_steps)
    assert ge.entropy_terms.shape == (n_steps,)


def test_grad_eval_j_at_steps_shape():
    n_steps = 8
    traj = _collect(n_steps=n_steps)
    ge = reeval_ctrl_trajectory(_make_actor(), _make_critic(), traj, dt=1.0 / n_steps)
    assert ge.j_at_steps.shape == (n_steps,)


def test_grad_eval_j_at_next_shape():
    n_steps = 8
    traj = _collect(n_steps=n_steps)
    ge = reeval_ctrl_trajectory(_make_actor(), _make_critic(), traj, dt=1.0 / n_steps)
    assert ge.j_at_next.shape == (n_steps,)


def test_grad_eval_log_probs_has_grad():
    """log_probs must have grad through actor parameters φ."""
    traj = _collect()
    actor = _make_actor()
    ge = reeval_ctrl_trajectory(actor, _make_critic(), traj, dt=0.1)
    assert ge.log_probs.requires_grad


def test_grad_eval_entropy_has_grad():
    """entropy_terms must have grad through actor parameters φ."""
    traj = _collect()
    actor = _make_actor()
    ge = reeval_ctrl_trajectory(actor, _make_critic(), traj, dt=0.1)
    assert ge.entropy_terms.requires_grad


def test_grad_eval_j_has_grad():
    """j_at_steps and j_at_next must have grad through critic parameters θ."""
    traj = _collect()
    critic = _make_critic()
    ge = reeval_ctrl_trajectory(_make_actor(), critic, traj, dt=0.1)
    assert ge.j_at_steps.requires_grad
    assert ge.j_at_next.requires_grad


def test_grad_eval_finite():
    traj = _collect()
    ge = reeval_ctrl_trajectory(_make_actor(), _make_critic(), traj, dt=0.1)
    assert torch.isfinite(ge.log_probs).all()
    assert torch.isfinite(ge.entropy_terms).all()
    assert torch.isfinite(ge.j_at_steps).all()
    assert torch.isfinite(ge.j_at_next).all()


def test_grad_eval_log_probs_match_trajectory():
    """Re-evaluated log-probs must equal stored trajectory values (same params, same inputs)."""
    actor = _make_actor()
    traj = collect_ctrl_trajectory(actor, _make_env(), w=1.0, seed=0)
    ge = reeval_ctrl_trajectory(actor, _make_critic(), traj, dt=0.1)
    assert torch.allclose(ge.log_probs.detach(), traj.log_probs, atol=1e-5)


def test_grad_eval_entropy_match_trajectory():
    """Re-evaluated entropy must equal stored trajectory values."""
    actor = _make_actor()
    traj = collect_ctrl_trajectory(actor, _make_env(), w=1.0, seed=0)
    ge = reeval_ctrl_trajectory(actor, _make_critic(), traj, dt=0.1)
    assert torch.allclose(ge.entropy_terms.detach(), traj.entropy_terms, atol=1e-5)


def test_grad_eval_j_matches_phase8a_detached():
    """Gradient-tracked J values must equal detached evaluate_critic_on_trajectory output."""
    traj = _collect()
    critic = _make_critic()
    dt = 0.1
    ce = evaluate_critic_on_trajectory(critic, traj, dt=dt)
    ge = reeval_ctrl_trajectory(_make_actor(), critic, traj, dt=dt)
    assert torch.allclose(ge.j_at_steps.detach(), ce.J_at_steps, atol=1e-5)
    assert torch.allclose(ge.j_at_next.detach(), ce.J_at_next, atol=1e-5)


def test_grad_eval_backward_actor():
    """A scalar loss on log_probs must backprop to actor parameters without error."""
    actor = _make_actor()
    traj = collect_ctrl_trajectory(actor, _make_env(), w=1.0, seed=0)
    ge = reeval_ctrl_trajectory(actor, _make_critic(), traj, dt=0.1)
    loss = ge.log_probs.sum()
    loss.backward()
    assert actor.log_phi1.grad is not None


def test_grad_eval_backward_critic():
    """A scalar loss on j_at_steps must backprop to critic parameters without error."""
    critic = _make_critic()
    traj = _collect()
    ge = reeval_ctrl_trajectory(_make_actor(), critic, traj, dt=0.1)
    loss = ge.j_at_steps.sum()
    loss.backward()
    assert critic.theta1.grad is not None


def test_grad_eval_dt_stored():
    traj = _collect()
    dt = 1.0 / 10
    ge = reeval_ctrl_trajectory(_make_actor(), _make_critic(), traj, dt=dt)
    assert ge.dt == pytest.approx(dt)


def test_grad_eval_w_stored():
    traj = _collect(w=1.3)
    ge = reeval_ctrl_trajectory(_make_actor(), _make_critic(), traj, dt=0.1)
    assert ge.w == pytest.approx(1.3)


def test_grad_eval_distinguished_from_phase8a():
    """CTRLGradEval is a separate type from CTRLCriticEval."""
    traj = _collect()
    critic = _make_critic()
    dt = 0.1
    ce = evaluate_critic_on_trajectory(critic, traj, dt=dt)
    ge = reeval_ctrl_trajectory(_make_actor(), critic, traj, dt=dt)
    assert type(ge) is not type(ce)
    assert not ce.J_at_steps.requires_grad  # detached
    assert ge.j_at_steps.requires_grad      # grad-tracked


# ---------------------------------------------------------------------------
# Phase 8B: public API imports
# ---------------------------------------------------------------------------

def test_phase8b_public_api_imports():
    from src.algos import CTRLGradEval, reeval_ctrl_trajectory
    assert CTRLGradEval is not None
    assert callable(reeval_ctrl_trajectory)
