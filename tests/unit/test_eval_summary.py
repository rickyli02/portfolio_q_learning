"""Unit tests for src/eval/summary.py — Phase 15A deterministic evaluation summary."""

import dataclasses

import pytest
import torch

from src.config.schema import AssetConfig, EnvConfig
from src.envs.gbm_env import GBMPortfolioEnv
from src.eval.summary import CTRLEvalSummary, eval_summary
from src.models.gaussian_actor import GaussianActor
from src.models.quadratic_critic import QuadraticCritic


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_env(n_steps: int = 5) -> GBMPortfolioEnv:
    cfg = EnvConfig(
        horizon=1.0,
        n_steps=n_steps,
        initial_wealth=1.0,
        mu=[0.1],
        sigma=[[0.2]],
        assets=AssetConfig(n_risky=1, include_risk_free=True, risk_free_rate=0.05),
    )
    return GBMPortfolioEnv(cfg)


def _make_actor() -> GaussianActor:
    return GaussianActor(n_risky=1, horizon=1.0)


# ===========================================================================
# Phase 15A — deterministic evaluation summary foundation
# ===========================================================================


# --- summary is returned and is the right type ---

def test_eval_summary_returns_eval_summary_instance():
    """eval_summary returns a CTRLEvalSummary instance."""
    actor = _make_actor()
    env = _make_env()
    result = eval_summary(actor, env, w=1.0, seed=42)
    assert isinstance(result, CTRLEvalSummary)


# --- field population ---

def test_eval_summary_n_steps_matches_env():
    """n_steps in summary matches the environment's n_steps."""
    actor = _make_actor()
    env = _make_env(n_steps=7)
    result = eval_summary(actor, env, w=1.0, seed=0)
    assert result.n_steps == 7


def test_eval_summary_initial_wealth_matches_env_initial_wealth():
    """initial_wealth equals the env's configured initial wealth."""
    actor = _make_actor()
    env = _make_env()
    result = eval_summary(actor, env, w=1.0, seed=0)
    assert result.initial_wealth == pytest.approx(1.0, rel=1e-5)


def test_eval_summary_terminal_wealth_is_positive():
    """terminal_wealth is a positive finite scalar."""
    actor = _make_actor()
    env = _make_env()
    result = eval_summary(actor, env, w=1.0, seed=0)
    assert result.terminal_wealth > 0


def test_eval_summary_min_leq_initial_leq_max():
    """min_wealth <= initial_wealth <= max_wealth."""
    actor = _make_actor()
    env = _make_env()
    result = eval_summary(actor, env, w=1.0, seed=0)
    assert result.min_wealth <= result.initial_wealth + 1e-6
    assert result.initial_wealth <= result.max_wealth + 1e-6


def test_eval_summary_min_leq_terminal_leq_max():
    """min_wealth <= terminal_wealth <= max_wealth."""
    actor = _make_actor()
    env = _make_env()
    result = eval_summary(actor, env, w=1.0, seed=0)
    assert result.min_wealth <= result.terminal_wealth + 1e-6
    assert result.terminal_wealth <= result.max_wealth + 1e-6


# --- target_return_z and terminal_gap ---

def test_eval_summary_with_z_populates_terminal_gap():
    """When target_return_z is provided, terminal_gap is x_T - z."""
    actor = _make_actor()
    env = _make_env()
    z = 1.1
    result = eval_summary(actor, env, w=1.0, target_return_z=z, seed=0)
    assert result.target_return_z == pytest.approx(z)
    assert result.terminal_gap == pytest.approx(result.terminal_wealth - z)


def test_eval_summary_without_z_terminal_gap_is_none():
    """When target_return_z is omitted, terminal_gap and target_return_z are None."""
    actor = _make_actor()
    env = _make_env()
    result = eval_summary(actor, env, w=1.0, seed=0)
    assert result.target_return_z is None
    assert result.terminal_gap is None


def test_eval_summary_terminal_gap_is_positive_when_wealth_above_target():
    """terminal_gap is strictly positive when terminal_wealth exceeds z."""
    actor = _make_actor()
    env = _make_env()
    # z very small so terminal_wealth > z is virtually certain from any seed
    result = eval_summary(actor, env, w=1.0, target_return_z=0.01, seed=0)
    assert result.terminal_gap is not None
    assert result.terminal_gap > 0


# --- scalar types (no tensors) ---

def test_eval_summary_all_fields_are_plain_scalars():
    """All fields in CTRLEvalSummary are plain Python scalars (float/int/None)."""
    actor = _make_actor()
    env = _make_env()
    result = eval_summary(actor, env, w=1.0, target_return_z=1.1, seed=0)
    assert isinstance(result.terminal_wealth, float)
    assert isinstance(result.initial_wealth, float)
    assert isinstance(result.target_return_z, float)
    assert isinstance(result.terminal_gap, float)
    assert isinstance(result.n_steps, int)
    assert isinstance(result.min_wealth, float)
    assert isinstance(result.max_wealth, float)
    # Must not be torch tensors
    assert not isinstance(result.terminal_wealth, torch.Tensor)
    assert not isinstance(result.n_steps, torch.Tensor)


# --- frozen dataclass ---

def test_eval_summary_is_frozen():
    """CTRLEvalSummary is a frozen dataclass; field assignment raises."""
    actor = _make_actor()
    env = _make_env()
    result = eval_summary(actor, env, w=1.0, seed=0)
    with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
        result.terminal_wealth = 99.9  # type: ignore[misc]


# --- reproducibility ---

def test_eval_summary_reproducible_with_same_seed():
    """Two calls with the same seed produce identical terminal_wealth."""
    actor = _make_actor()
    env = _make_env()
    r1 = eval_summary(actor, env, w=1.0, seed=7)
    r2 = eval_summary(actor, env, w=1.0, seed=7)
    assert r1.terminal_wealth == pytest.approx(r2.terminal_wealth)
    assert r1.min_wealth == pytest.approx(r2.min_wealth)
    assert r1.max_wealth == pytest.approx(r2.max_wealth)


# --- direct extraction cross-check against evaluate_ctrl_deterministic ---

def test_eval_summary_fields_match_raw_deterministic_result():
    """eval_summary fields exactly match the underlying evaluate_ctrl_deterministic output."""
    from src.algos.ctrl import evaluate_ctrl_deterministic
    actor = _make_actor()
    env = _make_env(n_steps=5)
    w = 1.0
    seed = 42
    raw = evaluate_ctrl_deterministic(actor, env, w=w, seed=seed)
    summary = eval_summary(actor, env, w=w, seed=seed)

    assert summary.terminal_wealth == pytest.approx(float(raw.terminal_wealth))
    assert summary.initial_wealth == pytest.approx(float(raw.wealth_path[0]))
    assert summary.n_steps == int(raw.times.shape[0])
    assert summary.min_wealth == pytest.approx(float(raw.wealth_path.min()))
    assert summary.max_wealth == pytest.approx(float(raw.wealth_path.max()))


# --- public API exports ---

def test_phase15a_public_api_imports():
    """CTRLEvalSummary and eval_summary are exported from src.eval."""
    from src.eval import CTRLEvalSummary as _S, eval_summary as _f
    assert _S is CTRLEvalSummary
    assert callable(_f)
