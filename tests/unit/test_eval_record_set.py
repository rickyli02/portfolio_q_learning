"""Unit tests for src/eval/record_set.py — Phase 15G multi-seed record set."""

from __future__ import annotations

import pytest
import torch

from src.config.schema import AssetConfig, EnvConfig
from src.envs.gbm_env import GBMPortfolioEnv
from src.eval import CTRLEvalRecord, CTRLEvalRecordSet, eval_record, eval_record_set
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


def _tensors_close(a: torch.Tensor, b: torch.Tensor) -> bool:
    return torch.allclose(a.float(), b.float(), atol=1e-6)


def _records_equal(a: CTRLEvalRecord, b: CTRLEvalRecord) -> bool:
    return (
        _tensors_close(a.times, b.times)
        and _tensors_close(a.wealth_path, b.wealth_path)
        and _tensors_close(a.actions, b.actions)
        and a.terminal_wealth == pytest.approx(b.terminal_wealth, rel=1e-6)
        and a.initial_wealth == pytest.approx(b.initial_wealth, rel=1e-6)
        and a.n_steps == b.n_steps
        and a.min_wealth == pytest.approx(b.min_wealth, rel=1e-6)
        and a.max_wealth == pytest.approx(b.max_wealth, rel=1e-6)
        and a.target_return_z == b.target_return_z
        and a.terminal_gap == b.terminal_gap
    )


# ===========================================================================
# Public API exports
# ===========================================================================


def test_public_exports_include_record_set():
    """CTRLEvalRecordSet and eval_record_set are exported from src.eval."""
    import src.eval as ev

    assert hasattr(ev, "CTRLEvalRecordSet")
    assert hasattr(ev, "eval_record_set")
    assert "CTRLEvalRecordSet" in ev.__all__
    assert "eval_record_set" in ev.__all__


# ===========================================================================
# Return type
# ===========================================================================


def test_eval_record_set_returns_correct_type():
    """eval_record_set returns a CTRLEvalRecordSet instance."""
    actor = _make_actor()
    env = _make_env()
    result = eval_record_set(actor, env, w=1.0, seeds=[0])
    assert isinstance(result, CTRLEvalRecordSet)


# ===========================================================================
# Seeds preserved exactly
# ===========================================================================


def test_seeds_preserved_in_result():
    """Seeds list stored in the returned record set matches the input exactly."""
    actor = _make_actor()
    env = _make_env()
    seeds = [3, 7, 42]
    result = eval_record_set(actor, env, w=1.0, seeds=seeds)
    assert result.seeds == seeds


def test_seeds_order_preserved():
    """Seeds are stored in exactly the order they were supplied."""
    actor = _make_actor()
    env = _make_env()
    seeds = [99, 1, 50, 0]
    result = eval_record_set(actor, env, w=1.0, seeds=seeds)
    assert result.seeds == [99, 1, 50, 0]


# ===========================================================================
# Record count matches seed count
# ===========================================================================


def test_record_count_matches_seed_count():
    """Number of records equals number of seeds."""
    actor = _make_actor()
    env = _make_env()
    seeds = [0, 1, 2, 3, 4]
    result = eval_record_set(actor, env, w=1.0, seeds=seeds)
    assert len(result.records) == len(seeds)


def test_single_seed_gives_one_record():
    """A single seed produces exactly one record."""
    actor = _make_actor()
    env = _make_env()
    result = eval_record_set(actor, env, w=1.0, seeds=[17])
    assert len(result.records) == 1


# ===========================================================================
# Per-seed records match independent eval_record calls
# ===========================================================================


def test_per_seed_records_match_independent_eval_record_calls():
    """Each record in the set exactly matches an independent eval_record call."""
    actor = _make_actor()
    env = _make_env()
    seeds = [0, 5, 13]
    w = 1.0

    result = eval_record_set(actor, env, w=w, seeds=seeds)
    expected = [eval_record(actor, env, w=w, seed=s) for s in seeds]

    assert len(result.records) == len(expected)
    for i, (got, exp) in enumerate(zip(result.records, expected)):
        assert _tensors_close(got.times, exp.times), f"times mismatch at index {i}"
        assert _tensors_close(got.wealth_path, exp.wealth_path), f"wealth_path mismatch at index {i}"
        assert _tensors_close(got.actions, exp.actions), f"actions mismatch at index {i}"
        assert got.terminal_wealth == pytest.approx(exp.terminal_wealth, rel=1e-6), f"terminal_wealth mismatch at index {i}"
        assert got.initial_wealth == pytest.approx(exp.initial_wealth, rel=1e-6), f"initial_wealth mismatch at index {i}"
        assert got.n_steps == exp.n_steps, f"n_steps mismatch at index {i}"
        assert got.min_wealth == pytest.approx(exp.min_wealth, rel=1e-6), f"min_wealth mismatch at index {i}"
        assert got.max_wealth == pytest.approx(exp.max_wealth, rel=1e-6), f"max_wealth mismatch at index {i}"
        assert got.target_return_z == exp.target_return_z, f"target_return_z mismatch at index {i}"
        assert got.terminal_gap == exp.terminal_gap, f"terminal_gap mismatch at index {i}"


def test_records_are_ctrlevalrecord_instances():
    """Each record in the set is a CTRLEvalRecord."""
    actor = _make_actor()
    env = _make_env()
    result = eval_record_set(actor, env, w=1.0, seeds=[0, 1])
    for rec in result.records:
        assert isinstance(rec, CTRLEvalRecord)


# ===========================================================================
# target_return_z forwarded correctly
# ===========================================================================


def test_target_return_z_forwarded_to_all_records():
    """target_return_z is present in every record when provided."""
    actor = _make_actor()
    env = _make_env()
    z = 1.3
    result = eval_record_set(actor, env, w=1.0, seeds=[0, 1, 2], target_return_z=z)
    for rec in result.records:
        assert rec.target_return_z == pytest.approx(z, rel=1e-6)
        assert rec.terminal_gap is not None


def test_target_return_z_none_propagates_to_all_records():
    """When target_return_z is omitted, all records have target_return_z=None."""
    actor = _make_actor()
    env = _make_env()
    result = eval_record_set(actor, env, w=1.0, seeds=[0, 1, 2])
    for rec in result.records:
        assert rec.target_return_z is None
        assert rec.terminal_gap is None


def test_target_return_z_records_match_independent_calls():
    """Records with target_return_z match independent eval_record calls field-by-field."""
    actor = _make_actor()
    env = _make_env()
    seeds = [2, 8]
    w = 0.5
    z = 1.2

    result = eval_record_set(actor, env, w=w, seeds=seeds, target_return_z=z)
    expected = [eval_record(actor, env, w=w, target_return_z=z, seed=s) for s in seeds]

    for i, (got, exp) in enumerate(zip(result.records, expected)):
        assert _tensors_close(got.times, exp.times), f"times mismatch at index {i}"
        assert _tensors_close(got.wealth_path, exp.wealth_path), f"wealth_path mismatch at index {i}"
        assert _tensors_close(got.actions, exp.actions), f"actions mismatch at index {i}"
        assert got.terminal_wealth == pytest.approx(exp.terminal_wealth, rel=1e-6), f"terminal_wealth mismatch at index {i}"
        assert got.initial_wealth == pytest.approx(exp.initial_wealth, rel=1e-6), f"initial_wealth mismatch at index {i}"
        assert got.n_steps == exp.n_steps, f"n_steps mismatch at index {i}"
        assert got.min_wealth == pytest.approx(exp.min_wealth, rel=1e-6), f"min_wealth mismatch at index {i}"
        assert got.max_wealth == pytest.approx(exp.max_wealth, rel=1e-6), f"max_wealth mismatch at index {i}"
        assert got.target_return_z == pytest.approx(exp.target_return_z, rel=1e-6), f"target_return_z mismatch at index {i}"
        assert got.terminal_gap == pytest.approx(exp.terminal_gap, rel=1e-6), f"terminal_gap mismatch at index {i}"


# ===========================================================================
# Empty seeds rejected
# ===========================================================================


def test_empty_seeds_raises_value_error():
    """Passing an empty seeds list raises ValueError."""
    actor = _make_actor()
    env = _make_env()
    with pytest.raises(ValueError, match="seeds must be non-empty"):
        eval_record_set(actor, env, w=1.0, seeds=[])
