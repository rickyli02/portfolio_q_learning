"""Unit tests for CTRLEvalScalarBundle / bundle_from_record_set — Phase 15J."""

from __future__ import annotations

import pytest

from src.config.schema import AssetConfig, EnvConfig
from src.envs.gbm_env import GBMPortfolioEnv
from src.eval import (
    CTRLEvalAggregate,
    CTRLEvalScalarBundle,
    CTRLEvalSummary,
    bundle_from_record_set,
    eval_record_set,
    eval_summary,
)
from src.models.gaussian_actor import GaussianActor


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


def _manual_aggregate(summaries: list[CTRLEvalSummary], z: float | None) -> dict:
    """Compute expected aggregate fields manually from a summary list."""
    tws = [s.terminal_wealth for s in summaries]
    n = len(tws)
    result: dict = {
        "n_episodes": n,
        "mean_terminal_wealth": sum(tws) / n,
        "min_terminal_wealth": min(tws),
        "max_terminal_wealth": max(tws),
        "mean_terminal_gap": None,
        "target_hit_rate": None,
    }
    if z is not None:
        gaps = [s.terminal_gap for s in summaries]
        result["mean_terminal_gap"] = sum(gaps) / n  # type: ignore[arg-type]
        result["target_hit_rate"] = sum(1 for tw in tws if tw >= z) / n
    return result


# ===========================================================================
# Public API exports
# ===========================================================================


def test_public_exports_include_bundle():
    import src.eval as ev

    assert hasattr(ev, "CTRLEvalScalarBundle")
    assert hasattr(ev, "bundle_from_record_set")
    assert "CTRLEvalScalarBundle" in ev.__all__
    assert "bundle_from_record_set" in ev.__all__


# ===========================================================================
# Return type
# ===========================================================================


def test_bundle_from_record_set_returns_correct_type():
    actor = _make_actor()
    env = _make_env()
    rs = eval_record_set(actor, env, w=1.0, seeds=[0, 1])
    bundle = bundle_from_record_set(rs)
    assert isinstance(bundle, CTRLEvalScalarBundle)


def test_bundle_fields_have_correct_types():
    actor = _make_actor()
    env = _make_env()
    rs = eval_record_set(actor, env, w=1.0, seeds=[0, 1])
    bundle = bundle_from_record_set(rs)
    assert isinstance(bundle.seeds, list)
    assert isinstance(bundle.summaries, list)
    assert isinstance(bundle.aggregate, CTRLEvalAggregate)
    assert all(isinstance(s, CTRLEvalSummary) for s in bundle.summaries)


# ===========================================================================
# Exact seed preservation
# ===========================================================================


def test_bundle_seeds_preserved_exactly():
    actor = _make_actor()
    env = _make_env()
    seeds = [7, 3, 42]
    rs = eval_record_set(actor, env, w=1.0, seeds=seeds)
    bundle = bundle_from_record_set(rs)
    assert bundle.seeds == seeds


def test_bundle_seeds_count_matches_summaries_count():
    actor = _make_actor()
    env = _make_env()
    seeds = [0, 1, 2, 3, 4]
    rs = eval_record_set(actor, env, w=1.0, seeds=seeds)
    bundle = bundle_from_record_set(rs)
    assert len(bundle.summaries) == len(seeds)


# ===========================================================================
# Exact summary derivation — no target
# ===========================================================================


def test_bundle_summaries_match_independent_eval_summary_no_target():
    """Per-seed summaries match independently constructed eval_summary calls."""
    actor = _make_actor()
    env = _make_env()
    seeds = [1, 5, 11]
    w = 1.0

    rs = eval_record_set(actor, env, w=w, seeds=seeds)
    bundle = bundle_from_record_set(rs)
    ref = [eval_summary(actor, env, w=w, seed=s) for s in seeds]

    assert len(bundle.summaries) == len(ref)
    for i, (got, exp) in enumerate(zip(bundle.summaries, ref)):
        assert got.terminal_wealth == pytest.approx(exp.terminal_wealth, rel=1e-6), f"terminal_wealth at {i}"
        assert got.initial_wealth == pytest.approx(exp.initial_wealth, rel=1e-6), f"initial_wealth at {i}"
        assert got.n_steps == exp.n_steps, f"n_steps at {i}"
        assert got.min_wealth == pytest.approx(exp.min_wealth, rel=1e-6), f"min_wealth at {i}"
        assert got.max_wealth == pytest.approx(exp.max_wealth, rel=1e-6), f"max_wealth at {i}"
        assert got.target_return_z == exp.target_return_z, f"target_return_z at {i}"
        assert got.terminal_gap == exp.terminal_gap, f"terminal_gap at {i}"


# ===========================================================================
# Exact summary derivation — with target
# ===========================================================================


def test_bundle_summaries_match_independent_eval_summary_with_target():
    """Per-seed summaries with target match independently constructed eval_summary calls."""
    actor = _make_actor()
    env = _make_env()
    seeds = [2, 8, 20]
    w = 0.5
    z = 1.1

    rs = eval_record_set(actor, env, w=w, seeds=seeds, target_return_z=z)
    bundle = bundle_from_record_set(rs)
    ref = [eval_summary(actor, env, w=w, target_return_z=z, seed=s) for s in seeds]

    for i, (got, exp) in enumerate(zip(bundle.summaries, ref)):
        assert got.terminal_wealth == pytest.approx(exp.terminal_wealth, rel=1e-6), f"terminal_wealth at {i}"
        assert got.initial_wealth == pytest.approx(exp.initial_wealth, rel=1e-6), f"initial_wealth at {i}"
        assert got.n_steps == exp.n_steps, f"n_steps at {i}"
        assert got.min_wealth == pytest.approx(exp.min_wealth, rel=1e-6), f"min_wealth at {i}"
        assert got.max_wealth == pytest.approx(exp.max_wealth, rel=1e-6), f"max_wealth at {i}"
        assert got.target_return_z == pytest.approx(exp.target_return_z, rel=1e-6), f"target_return_z at {i}"  # type: ignore[arg-type]
        assert got.terminal_gap == pytest.approx(exp.terminal_gap, rel=1e-6), f"terminal_gap at {i}"  # type: ignore[arg-type]


# ===========================================================================
# Exact aggregate derivation — no target
# ===========================================================================


def test_bundle_aggregate_matches_manual_aggregate_no_target():
    """Aggregate fields match manually computed values (no target)."""
    actor = _make_actor()
    env = _make_env()
    seeds = [0, 1, 2, 3]
    w = 1.0

    ref_summaries = [eval_summary(actor, env, w=w, seed=s) for s in seeds]
    expected = _manual_aggregate(ref_summaries, z=None)

    rs = eval_record_set(actor, env, w=w, seeds=seeds)
    bundle = bundle_from_record_set(rs)
    agg = bundle.aggregate

    assert agg.n_episodes == expected["n_episodes"]
    assert agg.mean_terminal_wealth == pytest.approx(expected["mean_terminal_wealth"], rel=1e-6)
    assert agg.min_terminal_wealth == pytest.approx(expected["min_terminal_wealth"], rel=1e-6)
    assert agg.max_terminal_wealth == pytest.approx(expected["max_terminal_wealth"], rel=1e-6)
    assert agg.mean_terminal_gap is None
    assert agg.target_hit_rate is None


# ===========================================================================
# Exact aggregate derivation — with target
# ===========================================================================


def test_bundle_aggregate_matches_manual_aggregate_with_target():
    """Aggregate fields match manually computed values (with target)."""
    actor = _make_actor()
    env = _make_env()
    seeds = [0, 5, 10, 15]
    w = 1.0
    z = 1.1

    ref_summaries = [eval_summary(actor, env, w=w, target_return_z=z, seed=s) for s in seeds]
    expected = _manual_aggregate(ref_summaries, z=z)

    rs = eval_record_set(actor, env, w=w, seeds=seeds, target_return_z=z)
    bundle = bundle_from_record_set(rs)
    agg = bundle.aggregate

    assert agg.n_episodes == expected["n_episodes"]
    assert agg.mean_terminal_wealth == pytest.approx(expected["mean_terminal_wealth"], rel=1e-6)
    assert agg.min_terminal_wealth == pytest.approx(expected["min_terminal_wealth"], rel=1e-6)
    assert agg.max_terminal_wealth == pytest.approx(expected["max_terminal_wealth"], rel=1e-6)
    assert agg.mean_terminal_gap == pytest.approx(expected["mean_terminal_gap"], rel=1e-6)
    assert agg.target_hit_rate == pytest.approx(expected["target_hit_rate"], rel=1e-6)


# ===========================================================================
# Target-related fields None when no target
# ===========================================================================


def test_bundle_target_fields_none_when_no_target():
    actor = _make_actor()
    env = _make_env()
    rs = eval_record_set(actor, env, w=1.0, seeds=[0, 1, 2])
    bundle = bundle_from_record_set(rs)
    assert bundle.aggregate.mean_terminal_gap is None
    assert bundle.aggregate.target_hit_rate is None
    for s in bundle.summaries:
        assert s.target_return_z is None
        assert s.terminal_gap is None


# ===========================================================================
# Scalar aggregate field types
# ===========================================================================


def test_bundle_aggregate_scalar_types_with_target():
    actor = _make_actor()
    env = _make_env()
    rs = eval_record_set(actor, env, w=1.0, seeds=[0, 1], target_return_z=1.1)
    bundle = bundle_from_record_set(rs)
    agg = bundle.aggregate
    assert isinstance(agg.n_episodes, int)
    assert isinstance(agg.mean_terminal_wealth, float)
    assert isinstance(agg.min_terminal_wealth, float)
    assert isinstance(agg.max_terminal_wealth, float)
    assert isinstance(agg.mean_terminal_gap, float)
    assert isinstance(agg.target_hit_rate, float)
