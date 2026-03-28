"""Unit tests for src/eval/derive.py — Phase 15I pure derivation helpers."""

from __future__ import annotations

import pytest

from src.config.schema import AssetConfig, EnvConfig
from src.envs.gbm_env import GBMPortfolioEnv
from src.eval import (
    CTRLEvalAggregate,
    CTRLEvalSummary,
    aggregate_from_record_set,
    eval_aggregate,
    eval_record,
    eval_record_set,
    eval_summary,
    summary_from_record,
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


# ===========================================================================
# Public API exports
# ===========================================================================


def test_public_exports_include_derive_helpers():
    import src.eval as ev

    assert hasattr(ev, "summary_from_record")
    assert hasattr(ev, "aggregate_from_record_set")
    assert "summary_from_record" in ev.__all__
    assert "aggregate_from_record_set" in ev.__all__


# ===========================================================================
# Return types
# ===========================================================================


def test_summary_from_record_returns_ctrlevalsummary():
    actor = _make_actor()
    env = _make_env()
    record = eval_record(actor, env, w=1.0, seed=0)
    result = summary_from_record(record)
    assert isinstance(result, CTRLEvalSummary)


def test_aggregate_from_record_set_returns_correct_types():
    actor = _make_actor()
    env = _make_env()
    rs = eval_record_set(actor, env, w=1.0, seeds=[0, 1])
    summaries, aggregate = aggregate_from_record_set(rs)
    assert isinstance(summaries, list)
    assert all(isinstance(s, CTRLEvalSummary) for s in summaries)
    assert isinstance(aggregate, CTRLEvalAggregate)


# ===========================================================================
# summary_from_record — exact field derivation
# ===========================================================================


def test_summary_from_record_matches_independent_eval_summary():
    """summary_from_record matches eval_summary field-by-field for same seed."""
    actor = _make_actor()
    env = _make_env(n_steps=7)
    seed = 13
    w = 1.0

    record = eval_record(actor, env, w=w, seed=seed)
    derived = summary_from_record(record)
    reference = eval_summary(actor, env, w=w, seed=seed)

    assert derived.terminal_wealth == pytest.approx(reference.terminal_wealth, rel=1e-6)
    assert derived.initial_wealth == pytest.approx(reference.initial_wealth, rel=1e-6)
    assert derived.n_steps == reference.n_steps
    assert derived.min_wealth == pytest.approx(reference.min_wealth, rel=1e-6)
    assert derived.max_wealth == pytest.approx(reference.max_wealth, rel=1e-6)
    assert derived.target_return_z == reference.target_return_z
    assert derived.terminal_gap == reference.terminal_gap


def test_summary_from_record_with_target_return_z_matches_eval_summary():
    """summary_from_record with target_return_z matches eval_summary field-by-field."""
    actor = _make_actor()
    env = _make_env()
    seed = 5
    w = 0.5
    z = 1.2

    record = eval_record(actor, env, w=w, target_return_z=z, seed=seed)
    derived = summary_from_record(record)
    reference = eval_summary(actor, env, w=w, target_return_z=z, seed=seed)

    assert derived.terminal_wealth == pytest.approx(reference.terminal_wealth, rel=1e-6)
    assert derived.initial_wealth == pytest.approx(reference.initial_wealth, rel=1e-6)
    assert derived.n_steps == reference.n_steps
    assert derived.min_wealth == pytest.approx(reference.min_wealth, rel=1e-6)
    assert derived.max_wealth == pytest.approx(reference.max_wealth, rel=1e-6)
    assert derived.target_return_z == pytest.approx(reference.target_return_z, rel=1e-6)  # type: ignore[arg-type]
    assert derived.terminal_gap == pytest.approx(reference.terminal_gap, rel=1e-6)  # type: ignore[arg-type]


def test_summary_from_record_no_target_return_z_fields_are_none():
    """When record has no target, derived summary target fields are None."""
    actor = _make_actor()
    env = _make_env()
    record = eval_record(actor, env, w=1.0, seed=0)
    derived = summary_from_record(record)
    assert derived.target_return_z is None
    assert derived.terminal_gap is None


def test_summary_from_record_scalar_types():
    """All summary fields from derivation are plain Python scalars."""
    actor = _make_actor()
    env = _make_env()
    record = eval_record(actor, env, w=1.0, target_return_z=1.1, seed=0)
    derived = summary_from_record(record)
    assert isinstance(derived.terminal_wealth, float)
    assert isinstance(derived.initial_wealth, float)
    assert isinstance(derived.n_steps, int)
    assert isinstance(derived.min_wealth, float)
    assert isinstance(derived.max_wealth, float)
    assert isinstance(derived.target_return_z, float)
    assert isinstance(derived.terminal_gap, float)


# ===========================================================================
# aggregate_from_record_set — seed order preserved
# ===========================================================================


def test_aggregate_from_record_set_summaries_in_seed_order():
    """Derived summaries match per-seed eval_summary calls in exact seed order."""
    actor = _make_actor()
    env = _make_env()
    seeds = [3, 9, 17]
    w = 1.0

    rs = eval_record_set(actor, env, w=w, seeds=seeds)
    summaries, _ = aggregate_from_record_set(rs)

    reference_summaries = [eval_summary(actor, env, w=w, seed=s) for s in seeds]

    assert len(summaries) == len(seeds)
    for i, (got, ref) in enumerate(zip(summaries, reference_summaries)):
        assert got.terminal_wealth == pytest.approx(ref.terminal_wealth, rel=1e-6), f"terminal_wealth mismatch at index {i}"
        assert got.initial_wealth == pytest.approx(ref.initial_wealth, rel=1e-6), f"initial_wealth at {i}"
        assert got.n_steps == ref.n_steps, f"n_steps at {i}"
        assert got.min_wealth == pytest.approx(ref.min_wealth, rel=1e-6), f"min_wealth at {i}"
        assert got.max_wealth == pytest.approx(ref.max_wealth, rel=1e-6), f"max_wealth at {i}"
        assert got.target_return_z == ref.target_return_z, f"target_return_z at {i}"
        assert got.terminal_gap == ref.terminal_gap, f"terminal_gap at {i}"


# ===========================================================================
# aggregate_from_record_set — exact aggregate derivation, no target
# ===========================================================================


def test_aggregate_from_record_set_matches_eval_aggregate_no_target():
    """aggregate_from_record_set aggregate matches eval_aggregate field-by-field (no target)."""
    actor = _make_actor()
    env = _make_env()
    seeds = [0, 1, 2, 3]
    w = 1.0

    rs = eval_record_set(actor, env, w=w, seeds=seeds)
    _, derived_agg = aggregate_from_record_set(rs)
    _, reference_agg = eval_aggregate(actor, env, w=w, seeds=seeds)

    assert derived_agg.n_episodes == reference_agg.n_episodes
    assert derived_agg.mean_terminal_wealth == pytest.approx(reference_agg.mean_terminal_wealth, rel=1e-6)
    assert derived_agg.min_terminal_wealth == pytest.approx(reference_agg.min_terminal_wealth, rel=1e-6)
    assert derived_agg.max_terminal_wealth == pytest.approx(reference_agg.max_terminal_wealth, rel=1e-6)
    assert derived_agg.mean_terminal_gap is None
    assert derived_agg.target_hit_rate is None


# ===========================================================================
# aggregate_from_record_set — exact aggregate derivation, with target
# ===========================================================================


def test_aggregate_from_record_set_matches_eval_aggregate_with_target():
    """aggregate_from_record_set aggregate matches eval_aggregate field-by-field (with target)."""
    actor = _make_actor()
    env = _make_env()
    seeds = [0, 5, 10, 15]
    w = 1.0
    z = 1.1

    rs = eval_record_set(actor, env, w=w, seeds=seeds, target_return_z=z)
    _, derived_agg = aggregate_from_record_set(rs)
    _, reference_agg = eval_aggregate(actor, env, w=w, seeds=seeds, target_return_z=z)

    assert derived_agg.n_episodes == reference_agg.n_episodes
    assert derived_agg.mean_terminal_wealth == pytest.approx(reference_agg.mean_terminal_wealth, rel=1e-6)
    assert derived_agg.min_terminal_wealth == pytest.approx(reference_agg.min_terminal_wealth, rel=1e-6)
    assert derived_agg.max_terminal_wealth == pytest.approx(reference_agg.max_terminal_wealth, rel=1e-6)
    assert derived_agg.mean_terminal_gap == pytest.approx(reference_agg.mean_terminal_gap, rel=1e-6)  # type: ignore[arg-type]
    assert derived_agg.target_hit_rate == pytest.approx(reference_agg.target_hit_rate, rel=1e-6)  # type: ignore[arg-type]


def test_aggregate_from_record_set_target_related_fields_none_when_no_target():
    """mean_terminal_gap and target_hit_rate are None when records have no target."""
    actor = _make_actor()
    env = _make_env()
    rs = eval_record_set(actor, env, w=1.0, seeds=[0, 1, 2])
    _, agg = aggregate_from_record_set(rs)
    assert agg.mean_terminal_gap is None
    assert agg.target_hit_rate is None


def test_aggregate_from_record_set_target_fields_present_when_target_provided():
    """mean_terminal_gap and target_hit_rate are float when target is provided."""
    actor = _make_actor()
    env = _make_env()
    rs = eval_record_set(actor, env, w=1.0, seeds=[0, 1, 2], target_return_z=1.1)
    _, agg = aggregate_from_record_set(rs)
    assert isinstance(agg.mean_terminal_gap, float)
    assert isinstance(agg.target_hit_rate, float)


def test_aggregate_from_record_set_n_episodes_matches_seed_count():
    """n_episodes in derived aggregate equals the number of seeds."""
    actor = _make_actor()
    env = _make_env()
    seeds = [0, 1, 2, 3, 4]
    rs = eval_record_set(actor, env, w=1.0, seeds=seeds)
    _, agg = aggregate_from_record_set(rs)
    assert agg.n_episodes == len(seeds)


def test_aggregate_from_record_set_scalar_types():
    """All aggregate fields are plain Python scalars."""
    actor = _make_actor()
    env = _make_env()
    rs = eval_record_set(actor, env, w=1.0, seeds=[0, 1], target_return_z=1.1)
    _, agg = aggregate_from_record_set(rs)
    assert isinstance(agg.n_episodes, int)
    assert isinstance(agg.mean_terminal_wealth, float)
    assert isinstance(agg.min_terminal_wealth, float)
    assert isinstance(agg.max_terminal_wealth, float)
    assert isinstance(agg.mean_terminal_gap, float)
    assert isinstance(agg.target_hit_rate, float)
