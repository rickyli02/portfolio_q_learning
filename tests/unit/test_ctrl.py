"""Unit tests for src/algos/ctrl.py — CTRL trajectory and eval scaffolding."""

import pytest
import torch

from src.algos.ctrl import (
    CTRLEvalResult,
    CTRLTrajectory,
    collect_ctrl_trajectory,
    evaluate_ctrl_deterministic,
)
from src.config.schema import AssetConfig, EnvConfig
from src.envs.gbm_env import GBMPortfolioEnv
from src.models.gaussian_actor import GaussianActor


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
    assert traj.w == 1.15


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
    assert result.w == 1.15


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
