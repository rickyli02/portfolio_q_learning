"""Unit tests for src/envs/: GBMPortfolioEnv, PortfolioStep, and constraints."""

import pytest
import torch

from src.config.schema import EnvConfig, AssetConfig
from src.envs.base_env import PortfolioStep
from src.envs.gbm_env import GBMPortfolioEnv, compute_mv_terminal_reward
from src.envs.constraints import (
    apply_leverage_constraint,
    apply_risky_only_projection,
    clip_action_norm,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_env(n_risky: int = 1, n_steps: int = 20, seed: int = 0) -> GBMPortfolioEnv:
    """Create a small single- or multi-asset GBM env."""
    if n_risky == 1:
        mu = [0.1]
        sigma = [[0.2]]
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
    env = GBMPortfolioEnv(cfg)
    env.reset(seed=seed)
    return env


# ---------------------------------------------------------------------------
# GBMPortfolioEnv — properties
# ---------------------------------------------------------------------------

def test_obs_dim_single_asset():
    env = _make_env(n_risky=1)
    assert env.obs_dim == 1


def test_action_dim_single_asset():
    env = _make_env(n_risky=1)
    assert env.action_dim == 1


def test_action_dim_multi_asset():
    env = _make_env(n_risky=2)
    assert env.action_dim == 2


def test_horizon_and_n_steps():
    env = _make_env(n_steps=50)
    assert env.horizon == pytest.approx(1.0)
    assert env.n_steps == 50


# ---------------------------------------------------------------------------
# GBMPortfolioEnv — reset
# ---------------------------------------------------------------------------

def test_reset_returns_obs_shape():
    env = _make_env()
    obs, info = env.reset(seed=0)
    assert obs.shape == (1,)


def test_reset_initial_wealth_matches_config():
    env = _make_env()
    obs, info = env.reset()
    assert obs[0].item() == pytest.approx(1.0)
    assert info["wealth"].item() == pytest.approx(1.0)


def test_reset_time_is_zero():
    env = _make_env()
    _, info = env.reset()
    assert info["time"].item() == pytest.approx(0.0)


def test_reset_reproducible_with_same_seed():
    env = _make_env()
    env.reset(seed=42)
    action = torch.tensor([0.5])
    step1 = env.step(action)

    env.reset(seed=42)
    step2 = env.step(action)
    assert torch.allclose(step1.obs, step2.obs)


def test_reset_different_seeds_differ():
    env = _make_env()
    env.reset(seed=0)
    step0 = env.step(torch.tensor([0.5]))

    env.reset(seed=99)
    step99 = env.step(torch.tensor([0.5]))
    # Almost surely different
    assert not torch.allclose(step0.wealth, step99.wealth)


# ---------------------------------------------------------------------------
# GBMPortfolioEnv — step
# ---------------------------------------------------------------------------

def test_step_obs_shape():
    env = _make_env()
    env.reset(seed=0)
    step = env.step(torch.tensor([0.5]))
    assert step.obs.shape == (1,)


def test_step_reward_is_wealth_increment():
    env = _make_env()
    env.reset(seed=0)
    _, info = env.reset(seed=0)
    prev_wealth = info["wealth"].item()
    step = env.step(torch.tensor([0.5]))
    expected = step.wealth.item() - prev_wealth
    assert step.reward.item() == pytest.approx(expected, abs=1e-5)


def test_step_done_only_at_last_step():
    env = _make_env(n_steps=5)
    env.reset(seed=0)
    steps = [env.step(torch.tensor([0.2])) for _ in range(5)]
    assert all(not s.done for s in steps[:-1])
    assert steps[-1].done


def test_step_time_advances():
    env = _make_env(n_steps=4)
    env.reset(seed=0)
    step = env.step(torch.tensor([0.3]))
    dt = 1.0 / 4
    assert step.time.item() == pytest.approx(0.0)
    assert step.next_time.item() == pytest.approx(dt, abs=1e-6)


def test_step_after_done_raises():
    env = _make_env(n_steps=2)
    env.reset(seed=0)
    env.step(torch.tensor([0.2]))
    env.step(torch.tensor([0.2]))
    with pytest.raises(RuntimeError, match="episode end"):
        env.step(torch.tensor([0.2]))


def test_full_episode_runs():
    env = _make_env(n_steps=10)
    env.reset(seed=1)
    for _ in range(10):
        env.step(torch.tensor([0.5]))  # should not raise


def test_zero_action_risk_free_only():
    """With zero risky allocation, wealth should grow at the risk-free rate."""
    env = _make_env(n_steps=1)
    env.reset(seed=0)
    step = env.step(torch.tensor([0.0]))
    dt = 1.0
    r = 0.05
    expected = 1.0 * (torch.exp(torch.tensor(r * dt)).item())
    assert step.wealth.item() == pytest.approx(expected, rel=1e-5)


def test_multi_asset_step_obs_shape():
    env = _make_env(n_risky=2)
    env.reset(seed=0)
    step = env.step(torch.tensor([0.3, 0.2]))
    assert step.obs.shape == (1,)


def test_step_to_transition():
    env = _make_env(n_steps=3)
    obs0, _ = env.reset(seed=0)
    action = torch.tensor([0.4])
    step = env.step(action)
    t = step.to_transition(prev_obs=obs0, action=action)
    assert t.obs.shape == obs0.shape
    assert t.next_obs.shape == step.obs.shape
    assert t.done.item() == pytest.approx(float(step.done))


# ---------------------------------------------------------------------------
# compute_mv_terminal_reward
# ---------------------------------------------------------------------------

def test_mv_reward_at_target_is_zero():
    wealth = torch.tensor([1.0, 1.0])
    r = compute_mv_terminal_reward(wealth, target_return=1.0)
    assert torch.allclose(r, torch.zeros(2))


def test_mv_reward_below_target_is_negative():
    wealth = torch.tensor([0.8])
    r = compute_mv_terminal_reward(wealth, target_return=1.0)
    assert r.item() < 0


def test_mv_reward_shape():
    wealth = torch.randn(8).abs() + 0.5
    r = compute_mv_terminal_reward(wealth, target_return=1.0, mv_penalty_coeff=2.0)
    assert r.shape == (8,)


# ---------------------------------------------------------------------------
# constraints — apply_leverage_constraint
# ---------------------------------------------------------------------------

def test_leverage_within_cap_unchanged():
    action = torch.tensor([0.3, 0.2])
    wealth = torch.tensor(1.0)
    out = apply_leverage_constraint(action, wealth, leverage_cap=1.5)
    assert torch.allclose(out, action)


def test_leverage_exceeding_cap_scaled():
    action = torch.tensor([1.0, 1.0])  # gross = 2.0, cap = 1.0 * 1.5 = 1.5
    wealth = torch.tensor(1.0)
    out = apply_leverage_constraint(action, wealth, leverage_cap=1.5)
    assert out.abs().sum().item() == pytest.approx(1.5, rel=1e-5)
    # Proportions and signs preserved
    assert torch.allclose(out[0] / out[1], action[0] / action[1])


def test_leverage_zero_total_no_crash():
    action = torch.zeros(2)
    wealth = torch.tensor(1.0)
    out = apply_leverage_constraint(action, wealth, leverage_cap=1.5)
    assert torch.all(out == 0)


def test_leverage_invalid_cap_raises():
    with pytest.raises(ValueError, match="leverage_cap"):
        apply_leverage_constraint(torch.tensor([0.5]), torch.tensor(1.0), leverage_cap=0.0)


def test_leverage_batched():
    action = torch.tensor([[1.0, 1.0], [0.3, 0.2]])  # (2, 2)
    wealth = torch.tensor([1.0, 1.0])
    out = apply_leverage_constraint(action, wealth, leverage_cap=1.5)
    assert out.shape == (2, 2)
    # First row: gross=2.0 > 1.5, should be scaled
    assert out[0].abs().sum().item() == pytest.approx(1.5, rel=1e-4)
    # Second row: gross=0.5 <= 1.5, unchanged
    assert torch.allclose(out[1], action[1])


def test_leverage_large_short_is_constrained():
    """Large short position must be scaled down, not pass through silently."""
    action = torch.tensor([-10.0, 0.0])  # gross = 10.0
    wealth = torch.tensor(1.0)
    out = apply_leverage_constraint(action, wealth, leverage_cap=1.5)
    # Gross exposure should be exactly at the cap
    assert out.abs().sum().item() == pytest.approx(1.5, rel=1e-5)
    # Sign preserved
    assert out[0].item() < 0


def test_leverage_mixed_long_short_gross_at_cap():
    """Mixed long/short: gross = |1.0| + |-0.8| = 1.8 > cap 1.5."""
    action = torch.tensor([1.0, -0.8])
    wealth = torch.tensor(1.0)
    out = apply_leverage_constraint(action, wealth, leverage_cap=1.5)
    assert out.abs().sum().item() == pytest.approx(1.5, rel=1e-5)
    # Signs preserved
    assert out[0].item() > 0
    assert out[1].item() < 0


def test_leverage_mixed_within_cap_unchanged():
    """Mixed long/short within cap should be unchanged."""
    action = torch.tensor([0.5, -0.3])  # gross = 0.8 < 1.5
    wealth = torch.tensor(1.0)
    out = apply_leverage_constraint(action, wealth, leverage_cap=1.5)
    assert torch.allclose(out, action)


# ---------------------------------------------------------------------------
# constraints — apply_risky_only_projection
# ---------------------------------------------------------------------------

def test_risky_only_gross_equals_wealth():
    """Gross exposure (Σ|u_i|) should equal wealth after projection."""
    action = torch.tensor([0.4, 0.3])
    wealth = torch.tensor(1.0)
    out = apply_risky_only_projection(action, wealth)
    assert out.abs().sum().item() == pytest.approx(1.0, rel=1e-5)


def test_risky_only_proportions_preserved():
    action = torch.tensor([0.4, 0.2])
    wealth = torch.tensor(1.0)
    out = apply_risky_only_projection(action, wealth)
    assert (out[0] / out[1]).item() == pytest.approx(2.0, rel=1e-5)


def test_risky_only_short_position_gross_equals_wealth():
    """Signed allocation: gross should still equal wealth; sign preserved."""
    action = torch.tensor([-1.0, 0.5])  # gross = 1.5
    wealth = torch.tensor(1.0)
    out = apply_risky_only_projection(action, wealth)
    assert out.abs().sum().item() == pytest.approx(1.0, rel=1e-5)
    assert out[0].item() < 0  # sign preserved
    assert out[1].item() > 0


def test_risky_only_sign_ratio_preserved():
    """Ratio of allocations preserved with mixed sign."""
    action = torch.tensor([-2.0, 1.0])  # gross = 3.0
    wealth = torch.tensor(1.5)
    out = apply_risky_only_projection(action, wealth)
    assert out.abs().sum().item() == pytest.approx(1.5, rel=1e-5)
    # Ratio |u0|/|u1| = 2 preserved
    assert (out[0].abs() / out[1].abs()).item() == pytest.approx(2.0, rel=1e-5)


def test_risky_only_zero_gross_raises():
    """Zero-gross input must raise ValueError (design decision: no direction to rescale)."""
    with pytest.raises(ValueError, match="zero gross exposure"):
        apply_risky_only_projection(torch.zeros(2), torch.tensor(1.0))


# ---------------------------------------------------------------------------
# constraints — clip_action_norm
# ---------------------------------------------------------------------------

def test_clip_norm_below_max_unchanged():
    action = torch.tensor([0.3, 0.4])  # norm = 0.5
    out = clip_action_norm(action, max_norm=1.0)
    assert torch.allclose(out, action)


def test_clip_norm_above_max_clipped():
    action = torch.tensor([3.0, 4.0])  # norm = 5.0
    out = clip_action_norm(action, max_norm=1.0)
    assert out.norm().item() == pytest.approx(1.0, rel=1e-5)


def test_clip_norm_direction_preserved():
    action = torch.tensor([3.0, 4.0])
    out = clip_action_norm(action, max_norm=1.0)
    orig_unit = action / action.norm()
    out_unit = out / out.norm()
    assert torch.allclose(orig_unit, out_unit, atol=1e-6)


def test_clip_norm_invalid_raises():
    with pytest.raises(ValueError, match="max_norm"):
        clip_action_norm(torch.tensor([1.0]), max_norm=0.0)
