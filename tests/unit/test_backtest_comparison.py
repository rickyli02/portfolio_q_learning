"""Unit tests for src/backtest/comparison.py — Phase 16A CTRL-vs-oracle comparison."""

from __future__ import annotations

import pytest

from src.algos.oracle_mv import OracleMVPolicy, run_oracle_episode
from src.backtest import (
    CTRLOracleBacktestComparison,
    CTRLOracleComparisonSummary,
    run_ctrl_oracle_comparison,
)
from src.config.schema import AssetConfig, EnvConfig
from src.envs.gbm_env import GBMPortfolioEnv
from src.eval import (
    CTRLEvalScalarBundle,
    eval_record_set,
    eval_summary,
)
from src.eval.derive import bundle_from_record_set
from src.models.gaussian_actor import GaussianActor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


MU = [0.1]
SIGMA = [[0.2]]
R = 0.05
HORIZON = 1.0
N_STEPS = 5
GAMMA_EMBED = 1.5


def _make_env() -> GBMPortfolioEnv:
    cfg = EnvConfig(
        horizon=HORIZON,
        n_steps=N_STEPS,
        initial_wealth=1.0,
        mu=MU,
        sigma=SIGMA,
        assets=AssetConfig(n_risky=1, include_risk_free=True, risk_free_rate=R),
    )
    return GBMPortfolioEnv(cfg)


def _make_actor() -> GaussianActor:
    return GaussianActor(n_risky=1, horizon=HORIZON)


def _make_oracle() -> OracleMVPolicy:
    return OracleMVPolicy.from_env_params(
        mu=MU, sigma=SIGMA, r=R, horizon=HORIZON, gamma_embed=GAMMA_EMBED
    )


def _manual_deltas(ctrl_tws: list[float], oracle_tws: list[float]) -> dict:
    deltas = [c - o for c, o in zip(ctrl_tws, oracle_tws)]
    n = len(deltas)
    return {
        "mean": sum(deltas) / n,
        "min": min(deltas),
        "max": max(deltas),
        "win_rate": sum(1 for d in deltas if d > 0) / n,
    }


# ===========================================================================
# Public API exports
# ===========================================================================


def test_public_exports_include_comparison_types():
    import src.backtest as bt

    assert hasattr(bt, "CTRLOracleBacktestComparison")
    assert hasattr(bt, "CTRLOracleComparisonSummary")
    assert hasattr(bt, "run_ctrl_oracle_comparison")
    assert "CTRLOracleBacktestComparison" in bt.__all__
    assert "CTRLOracleComparisonSummary" in bt.__all__
    assert "run_ctrl_oracle_comparison" in bt.__all__


# ===========================================================================
# Return types
# ===========================================================================


def test_run_ctrl_oracle_comparison_returns_correct_type():
    actor = _make_actor()
    env = _make_env()
    oracle = _make_oracle()
    result = run_ctrl_oracle_comparison(actor, env, w=1.0, oracle_policy=oracle, seeds=[0, 1])
    assert isinstance(result, CTRLOracleBacktestComparison)
    assert isinstance(result.ctrl_bundle, CTRLEvalScalarBundle)
    assert isinstance(result.oracle_bundle, CTRLEvalScalarBundle)
    assert isinstance(result.comparison, CTRLOracleComparisonSummary)


# ===========================================================================
# Seed preservation
# ===========================================================================


def test_seeds_preserved_exactly():
    actor = _make_actor()
    env = _make_env()
    oracle = _make_oracle()
    seeds = [7, 3, 42]
    result = run_ctrl_oracle_comparison(actor, env, w=1.0, oracle_policy=oracle, seeds=seeds)
    assert result.seeds == seeds
    assert result.ctrl_bundle.seeds == seeds
    assert result.oracle_bundle.seeds == seeds


def test_seed_count_matches_bundle_summary_count():
    actor = _make_actor()
    env = _make_env()
    oracle = _make_oracle()
    seeds = [0, 1, 2, 3]
    result = run_ctrl_oracle_comparison(actor, env, w=1.0, oracle_policy=oracle, seeds=seeds)
    assert len(result.ctrl_bundle.summaries) == len(seeds)
    assert len(result.oracle_bundle.summaries) == len(seeds)


# ===========================================================================
# CTRL bundle contents — compared to independent eval_summary calls
# ===========================================================================


def test_ctrl_bundle_summaries_match_independent_eval_summary():
    """CTRL per-seed summaries match independent eval_summary calls field-by-field."""
    actor = _make_actor()
    env = _make_env()
    oracle = _make_oracle()
    seeds = [1, 5, 11]
    w = 1.0

    result = run_ctrl_oracle_comparison(actor, env, w=w, oracle_policy=oracle, seeds=seeds)
    ref = [eval_summary(actor, env, w=w, seed=s) for s in seeds]

    for i, (got, exp) in enumerate(zip(result.ctrl_bundle.summaries, ref)):
        assert got.terminal_wealth == pytest.approx(exp.terminal_wealth, rel=1e-6), f"tw at {i}"
        assert got.initial_wealth == pytest.approx(exp.initial_wealth, rel=1e-6), f"iw at {i}"
        assert got.n_steps == exp.n_steps, f"n_steps at {i}"
        assert got.min_wealth == pytest.approx(exp.min_wealth, rel=1e-6), f"min_w at {i}"
        assert got.max_wealth == pytest.approx(exp.max_wealth, rel=1e-6), f"max_w at {i}"
        assert got.target_return_z == exp.target_return_z, f"z at {i}"
        assert got.terminal_gap == exp.terminal_gap, f"gap at {i}"


def test_ctrl_bundle_summaries_with_target_match_eval_summary():
    """CTRL summaries with target_return_z match independent eval_summary."""
    actor = _make_actor()
    env = _make_env()
    oracle = _make_oracle()
    seeds = [2, 8]
    w = 0.5
    z = 1.1

    result = run_ctrl_oracle_comparison(actor, env, w=w, oracle_policy=oracle, seeds=seeds, target_return_z=z)
    ref = [eval_summary(actor, env, w=w, target_return_z=z, seed=s) for s in seeds]

    for i, (got, exp) in enumerate(zip(result.ctrl_bundle.summaries, ref)):
        assert got.terminal_wealth == pytest.approx(exp.terminal_wealth, rel=1e-6), f"tw at {i}"
        assert got.target_return_z == pytest.approx(exp.target_return_z, rel=1e-6), f"z at {i}"  # type: ignore[arg-type]
        assert got.terminal_gap == pytest.approx(exp.terminal_gap, rel=1e-6), f"gap at {i}"  # type: ignore[arg-type]


# ===========================================================================
# Oracle bundle contents — compared to independent run_oracle_episode calls
# ===========================================================================


def test_oracle_bundle_summaries_match_independent_oracle_rollouts():
    """Oracle per-seed summaries match independently run oracle episodes."""
    actor = _make_actor()
    env = _make_env()
    oracle = _make_oracle()
    seeds = [0, 3, 7]

    result = run_ctrl_oracle_comparison(actor, env, w=1.0, oracle_policy=oracle, seeds=seeds)

    for i, (got, seed) in enumerate(zip(result.oracle_bundle.summaries, seeds)):
        ref = run_oracle_episode(oracle, env, seed=seed)
        ref_tw = float(ref["wealth_path"][-1])
        ref_iw = float(ref["wealth_path"][0])
        ref_min = float(ref["wealth_path"].min())
        ref_max = float(ref["wealth_path"].max())
        ref_nsteps = int(ref["times"].shape[0])

        assert got.terminal_wealth == pytest.approx(ref_tw, rel=1e-6), f"oracle tw at {i}"
        assert got.initial_wealth == pytest.approx(ref_iw, rel=1e-6), f"oracle iw at {i}"
        assert got.n_steps == ref_nsteps, f"oracle n_steps at {i}"
        assert got.min_wealth == pytest.approx(ref_min, rel=1e-6), f"oracle min_w at {i}"
        assert got.max_wealth == pytest.approx(ref_max, rel=1e-6), f"oracle max_w at {i}"
        assert got.target_return_z is None
        assert got.terminal_gap is None


def test_oracle_bundle_with_target_terminal_gap_is_correct():
    """Oracle terminal_gap = oracle_tw - z for each seed."""
    actor = _make_actor()
    env = _make_env()
    oracle = _make_oracle()
    seeds = [0, 1]
    z = 1.2

    result = run_ctrl_oracle_comparison(actor, env, w=1.0, oracle_policy=oracle, seeds=seeds, target_return_z=z)

    for i, (got, seed) in enumerate(zip(result.oracle_bundle.summaries, seeds)):
        ref = run_oracle_episode(oracle, env, seed=seed)
        ref_tw = float(ref["wealth_path"][-1])
        expected_gap = ref_tw - z
        assert got.target_return_z == pytest.approx(z, rel=1e-6), f"z at {i}"
        assert got.terminal_gap == pytest.approx(expected_gap, rel=1e-6), f"gap at {i}"


# ===========================================================================
# Comparison summary — exact from manually computed deltas
# ===========================================================================


def test_comparison_summary_fields_match_manual_deltas():
    """Comparison summary fields match manually computed terminal-wealth deltas."""
    actor = _make_actor()
    env = _make_env()
    oracle = _make_oracle()
    seeds = [0, 1, 2, 3]
    w = 1.0

    result = run_ctrl_oracle_comparison(actor, env, w=w, oracle_policy=oracle, seeds=seeds)

    ctrl_tws = [s.terminal_wealth for s in result.ctrl_bundle.summaries]
    oracle_tws = [s.terminal_wealth for s in result.oracle_bundle.summaries]
    expected = _manual_deltas(ctrl_tws, oracle_tws)

    cmp = result.comparison
    assert cmp.mean_terminal_wealth_delta == pytest.approx(expected["mean"], rel=1e-6)
    assert cmp.min_terminal_wealth_delta == pytest.approx(expected["min"], rel=1e-6)
    assert cmp.max_terminal_wealth_delta == pytest.approx(expected["max"], rel=1e-6)
    assert cmp.ctrl_win_rate == pytest.approx(expected["win_rate"], rel=1e-6)


def test_comparison_summary_win_rate_correctness():
    """ctrl_win_rate equals fraction of seeds where ctrl_tw > oracle_tw."""
    actor = _make_actor()
    env = _make_env()
    oracle = _make_oracle()
    seeds = [0, 1, 2, 3, 4]
    w = 1.0

    result = run_ctrl_oracle_comparison(actor, env, w=w, oracle_policy=oracle, seeds=seeds)

    ctrl_tws = [s.terminal_wealth for s in result.ctrl_bundle.summaries]
    oracle_tws = [s.terminal_wealth for s in result.oracle_bundle.summaries]
    expected_rate = sum(1 for c, o in zip(ctrl_tws, oracle_tws) if c > o) / len(seeds)

    assert result.comparison.ctrl_win_rate == pytest.approx(expected_rate, rel=1e-6)


# ===========================================================================
# no-target behavior
# ===========================================================================


def test_no_target_null_fields_in_both_bundles():
    """Without target, all target-related fields are None in both bundles."""
    actor = _make_actor()
    env = _make_env()
    oracle = _make_oracle()
    result = run_ctrl_oracle_comparison(actor, env, w=1.0, oracle_policy=oracle, seeds=[0, 1])

    for s in result.ctrl_bundle.summaries:
        assert s.target_return_z is None
        assert s.terminal_gap is None
    for s in result.oracle_bundle.summaries:
        assert s.target_return_z is None
        assert s.terminal_gap is None
    assert result.ctrl_bundle.aggregate.mean_terminal_gap is None
    assert result.ctrl_bundle.aggregate.target_hit_rate is None
    assert result.oracle_bundle.aggregate.mean_terminal_gap is None
    assert result.oracle_bundle.aggregate.target_hit_rate is None


# ===========================================================================
# with-target behavior
# ===========================================================================


def test_with_target_fields_populated_in_both_bundles():
    """With target, target-related fields are float in both bundles."""
    actor = _make_actor()
    env = _make_env()
    oracle = _make_oracle()
    result = run_ctrl_oracle_comparison(
        actor, env, w=1.0, oracle_policy=oracle, seeds=[0, 1], target_return_z=1.1
    )

    for s in result.ctrl_bundle.summaries:
        assert isinstance(s.target_return_z, float)
        assert isinstance(s.terminal_gap, float)
    for s in result.oracle_bundle.summaries:
        assert isinstance(s.target_return_z, float)
        assert isinstance(s.terminal_gap, float)
    assert isinstance(result.ctrl_bundle.aggregate.mean_terminal_gap, float)
    assert isinstance(result.oracle_bundle.aggregate.mean_terminal_gap, float)


# ===========================================================================
# Comparison summary scalar types
# ===========================================================================


def test_comparison_summary_scalar_types():
    actor = _make_actor()
    env = _make_env()
    oracle = _make_oracle()
    result = run_ctrl_oracle_comparison(actor, env, w=1.0, oracle_policy=oracle, seeds=[0, 1])
    cmp = result.comparison
    assert isinstance(cmp.mean_terminal_wealth_delta, float)
    assert isinstance(cmp.min_terminal_wealth_delta, float)
    assert isinstance(cmp.max_terminal_wealth_delta, float)
    assert isinstance(cmp.ctrl_win_rate, float)
