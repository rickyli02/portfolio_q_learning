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


# ===========================================================================
# Phase 15B — evaluation summary file IO foundation
# ===========================================================================

from src.eval.io import load_eval_summaries, save_eval_summaries


def _make_summary(
    terminal_wealth: float = 1.05,
    initial_wealth: float = 1.0,
    target_return_z: float | None = 1.1,
    terminal_gap: float | None = -0.05,
    n_steps: int = 5,
    min_wealth: float = 0.95,
    max_wealth: float = 1.08,
) -> CTRLEvalSummary:
    return CTRLEvalSummary(
        terminal_wealth=terminal_wealth,
        initial_wealth=initial_wealth,
        target_return_z=target_return_z,
        terminal_gap=terminal_gap,
        n_steps=n_steps,
        min_wealth=min_wealth,
        max_wealth=max_wealth,
    )


# --- single-record exact roundtrip ---

def test_save_load_single_summary_exact_roundtrip(tmp_path):
    """Save one CTRLEvalSummary and load it back; every field matches exactly."""
    s = _make_summary()
    p = tmp_path / "eval.jsonl"
    save_eval_summaries([s], p)
    loaded = load_eval_summaries(p)
    assert len(loaded) == 1
    r = loaded[0]
    assert r.terminal_wealth == pytest.approx(s.terminal_wealth)
    assert r.initial_wealth == pytest.approx(s.initial_wealth)
    assert r.target_return_z == pytest.approx(s.target_return_z)
    assert r.terminal_gap == pytest.approx(s.terminal_gap)
    assert r.n_steps == s.n_steps
    assert r.min_wealth == pytest.approx(s.min_wealth)
    assert r.max_wealth == pytest.approx(s.max_wealth)


# --- None optional fields roundtrip ---

def test_save_load_none_optional_fields_roundtrip(tmp_path):
    """target_return_z=None and terminal_gap=None survive the roundtrip."""
    s = _make_summary(target_return_z=None, terminal_gap=None)
    p = tmp_path / "eval_none.jsonl"
    save_eval_summaries([s], p)
    loaded = load_eval_summaries(p)
    assert loaded[0].target_return_z is None
    assert loaded[0].terminal_gap is None


# --- multiple-record roundtrip preserving order ---

def test_save_load_multiple_summaries_preserves_order(tmp_path):
    """Multiple summaries are saved and loaded in their original order."""
    summaries = [_make_summary(terminal_wealth=float(i)) for i in range(5)]
    p = tmp_path / "eval_multi.jsonl"
    save_eval_summaries(summaries, p)
    loaded = load_eval_summaries(p)
    assert len(loaded) == 5
    for i, r in enumerate(loaded):
        assert r.terminal_wealth == pytest.approx(float(i))


# --- empty list roundtrip ---

def test_save_load_empty_list_summaries(tmp_path):
    """Saving an empty list produces a file that loads back as an empty list."""
    p = tmp_path / "eval_empty.jsonl"
    save_eval_summaries([], p)
    loaded = load_eval_summaries(p)
    assert loaded == []


# --- loaded entries are CTRLEvalSummary instances ---

def test_loaded_entries_are_eval_summary_instances(tmp_path):
    """Each entry returned by load_eval_summaries is a CTRLEvalSummary."""
    p = tmp_path / "eval_type.jsonl"
    save_eval_summaries([_make_summary(), _make_summary(terminal_wealth=2.0)], p)
    loaded = load_eval_summaries(p)
    assert all(isinstance(r, CTRLEvalSummary) for r in loaded)


# --- overwrite semantics ---

def test_save_eval_summaries_overwrites_existing_file(tmp_path):
    """save_eval_summaries overwrites an existing file rather than appending."""
    p = tmp_path / "eval_overwrite.jsonl"
    save_eval_summaries([_make_summary(terminal_wealth=1.0)], p)
    save_eval_summaries([_make_summary(terminal_wealth=9.9)], p)
    loaded = load_eval_summaries(p)
    assert len(loaded) == 1
    assert loaded[0].terminal_wealth == pytest.approx(9.9)


# --- error cases ---

def test_load_eval_summaries_nonexistent_path_raises(tmp_path):
    """load_eval_summaries raises FileNotFoundError for a nonexistent path."""
    with pytest.raises(FileNotFoundError):
        load_eval_summaries(tmp_path / "does_not_exist.jsonl")


def test_load_eval_summaries_malformed_json_raises(tmp_path):
    """load_eval_summaries raises ValueError for invalid JSON on a line."""
    p = tmp_path / "bad.jsonl"
    p.write_text("not valid json\n")
    with pytest.raises(ValueError, match="Malformed JSON"):
        load_eval_summaries(p)


def test_load_eval_summaries_missing_required_field_raises(tmp_path):
    """load_eval_summaries raises ValueError when a required field is absent."""
    import json as _json
    p = tmp_path / "missing.jsonl"
    # omit terminal_wealth
    p.write_text(
        _json.dumps({
            "initial_wealth": 1.0, "n_steps": 5,
            "min_wealth": 0.9, "max_wealth": 1.1,
        }) + "\n"
    )
    with pytest.raises(ValueError, match="Missing required field"):
        load_eval_summaries(p)


def test_load_eval_summaries_wrong_type_required_float_raises(tmp_path):
    """load_eval_summaries raises ValueError for a string in a required float field."""
    import json as _json
    p = tmp_path / "bad_type.jsonl"
    p.write_text(
        _json.dumps({
            "terminal_wealth": "oops", "initial_wealth": 1.0,
            "n_steps": 5, "min_wealth": 0.9, "max_wealth": 1.1,
        }) + "\n"
    )
    with pytest.raises(ValueError, match="terminal_wealth"):
        load_eval_summaries(p)


def test_load_eval_summaries_float_in_n_steps_raises(tmp_path):
    """load_eval_summaries raises ValueError for a float in the n_steps integer field."""
    import json as _json
    p = tmp_path / "bad_int.jsonl"
    p.write_text(
        _json.dumps({
            "terminal_wealth": 1.05, "initial_wealth": 1.0,
            "n_steps": 5.5, "min_wealth": 0.9, "max_wealth": 1.1,
        }) + "\n"
    )
    with pytest.raises(ValueError, match="n_steps"):
        load_eval_summaries(p)


def test_load_eval_summaries_wrong_type_optional_float_raises(tmp_path):
    """load_eval_summaries raises ValueError for a string in an optional float field."""
    import json as _json
    p = tmp_path / "bad_opt.jsonl"
    p.write_text(
        _json.dumps({
            "terminal_wealth": 1.05, "initial_wealth": 1.0,
            "n_steps": 5, "min_wealth": 0.9, "max_wealth": 1.1,
            "target_return_z": "bad",
        }) + "\n"
    )
    with pytest.raises(ValueError, match="target_return_z"):
        load_eval_summaries(p)


# --- integration: roundtrip from eval_summary output ---

def test_save_load_summaries_from_eval_summary_output(tmp_path):
    """Summaries produced by eval_summary survive the file roundtrip with exact fields."""
    actor = _make_actor()
    env = _make_env(n_steps=5)
    orig = eval_summary(actor, env, w=1.0, target_return_z=1.1, seed=3)
    p = tmp_path / "live_eval.jsonl"
    save_eval_summaries([orig], p)
    loaded = load_eval_summaries(p)
    assert len(loaded) == 1
    r = loaded[0]
    assert r.terminal_wealth == pytest.approx(orig.terminal_wealth)
    assert r.initial_wealth == pytest.approx(orig.initial_wealth)
    assert r.target_return_z == pytest.approx(orig.target_return_z)
    assert r.terminal_gap == pytest.approx(orig.terminal_gap)
    assert r.n_steps == orig.n_steps
    assert r.min_wealth == pytest.approx(orig.min_wealth)
    assert r.max_wealth == pytest.approx(orig.max_wealth)


# --- public API export ---

def test_phase15b_public_api_imports():
    """save_eval_summaries and load_eval_summaries are exported from src.eval."""
    from src.eval import load_eval_summaries as _load, save_eval_summaries as _save
    assert callable(_save)
    assert callable(_load)


# ===========================================================================
# Phase 15C — multi-episode deterministic evaluation aggregate foundation
# ===========================================================================

from src.eval.aggregate import CTRLEvalAggregate, eval_aggregate


# --- basic contract ---

def test_eval_aggregate_returns_correct_types():
    """eval_aggregate returns (list[CTRLEvalSummary], CTRLEvalAggregate)."""
    actor = _make_actor()
    env = _make_env()
    summaries, agg = eval_aggregate(actor, env, w=1.0, seeds=[0, 1, 2])
    assert isinstance(summaries, list)
    assert all(isinstance(s, CTRLEvalSummary) for s in summaries)
    assert isinstance(agg, CTRLEvalAggregate)


def test_eval_aggregate_n_episodes_matches_seeds():
    """n_episodes equals the number of provided seeds."""
    actor = _make_actor()
    env = _make_env()
    seeds = [0, 1, 2, 3, 4]
    summaries, agg = eval_aggregate(actor, env, w=1.0, seeds=seeds)
    assert agg.n_episodes == len(seeds)
    assert len(summaries) == len(seeds)


def test_eval_aggregate_summaries_in_seed_order():
    """Per-episode summaries match independent eval_summary calls in full, in seed order."""
    actor = _make_actor()
    env = _make_env()
    seeds = [7, 42, 13]
    z = 1.05
    summaries, _ = eval_aggregate(actor, env, w=1.0, seeds=seeds, target_return_z=z)
    for i, s in enumerate(seeds):
        expected = eval_summary(actor, env, w=1.0, target_return_z=z, seed=s)
        assert summaries[i].terminal_wealth == pytest.approx(expected.terminal_wealth)
        assert summaries[i].initial_wealth == pytest.approx(expected.initial_wealth)
        assert summaries[i].n_steps == expected.n_steps
        assert summaries[i].min_wealth == pytest.approx(expected.min_wealth)
        assert summaries[i].max_wealth == pytest.approx(expected.max_wealth)
        assert summaries[i].target_return_z == pytest.approx(expected.target_return_z)
        assert summaries[i].terminal_gap == pytest.approx(expected.terminal_gap)


# --- exact aggregation cross-check ---

def test_eval_aggregate_fields_match_manual_aggregation():
    """Aggregate fields match aggregates computed from independent eval_summary calls."""
    actor = _make_actor()
    env = _make_env()
    seeds = [0, 1, 2, 3]
    z = 1.05
    # Build expected values from INDEPENDENT eval_summary calls, not from helper's summaries
    indep = [eval_summary(actor, env, w=1.0, target_return_z=z, seed=s) for s in seeds]
    _, agg = eval_aggregate(actor, env, w=1.0, seeds=seeds, target_return_z=z)

    terminal_wealths = [s.terminal_wealth for s in indep]
    expected_mean = sum(terminal_wealths) / len(terminal_wealths)
    expected_min = min(terminal_wealths)
    expected_max = max(terminal_wealths)
    expected_gap_mean = sum(s.terminal_gap for s in indep) / len(indep)
    expected_hit_rate = sum(1 for tw in terminal_wealths if tw >= z) / len(terminal_wealths)

    assert agg.n_episodes == len(seeds)
    assert agg.mean_terminal_wealth == pytest.approx(expected_mean)
    assert agg.min_terminal_wealth == pytest.approx(expected_min)
    assert agg.max_terminal_wealth == pytest.approx(expected_max)
    assert agg.mean_terminal_gap == pytest.approx(expected_gap_mean)
    assert agg.target_hit_rate == pytest.approx(expected_hit_rate)


# --- target_return_z provided ---

def test_eval_aggregate_with_z_populates_gap_and_hit_rate():
    """mean_terminal_gap and target_hit_rate are floats when target_return_z is given."""
    actor = _make_actor()
    env = _make_env()
    _, agg = eval_aggregate(actor, env, w=1.0, seeds=[0, 1, 2], target_return_z=1.1)
    assert isinstance(agg.mean_terminal_gap, float)
    assert isinstance(agg.target_hit_rate, float)
    assert 0.0 <= agg.target_hit_rate <= 1.0


def test_eval_aggregate_target_hit_rate_all_above_very_low_target():
    """target_hit_rate is 1.0 when z is far below any achievable terminal wealth."""
    actor = _make_actor()
    env = _make_env()
    _, agg = eval_aggregate(actor, env, w=1.0, seeds=[0, 1, 2, 3], target_return_z=0.001)
    assert agg.target_hit_rate == pytest.approx(1.0)


def test_eval_aggregate_target_hit_rate_none_above_very_high_target():
    """target_hit_rate is 0.0 when z is far above any achievable terminal wealth."""
    actor = _make_actor()
    env = _make_env()
    _, agg = eval_aggregate(actor, env, w=1.0, seeds=[0, 1, 2, 3], target_return_z=1e9)
    assert agg.target_hit_rate == pytest.approx(0.0)


# --- target_return_z omitted ---

def test_eval_aggregate_without_z_gap_and_hit_rate_are_none():
    """mean_terminal_gap and target_hit_rate are None when target_return_z is omitted."""
    actor = _make_actor()
    env = _make_env()
    _, agg = eval_aggregate(actor, env, w=1.0, seeds=[0, 1, 2])
    assert agg.mean_terminal_gap is None
    assert agg.target_hit_rate is None


# --- aggregate fields are plain scalars ---

def test_eval_aggregate_fields_are_plain_scalars():
    """All non-None fields in CTRLEvalAggregate are plain Python scalars."""
    actor = _make_actor()
    env = _make_env()
    _, agg = eval_aggregate(actor, env, w=1.0, seeds=[0, 1], target_return_z=1.0)
    assert isinstance(agg.n_episodes, int)
    assert isinstance(agg.mean_terminal_wealth, float)
    assert isinstance(agg.min_terminal_wealth, float)
    assert isinstance(agg.max_terminal_wealth, float)
    assert isinstance(agg.mean_terminal_gap, float)
    assert isinstance(agg.target_hit_rate, float)


# --- frozen dataclass ---

def test_eval_aggregate_is_frozen():
    """CTRLEvalAggregate is a frozen dataclass; field assignment raises."""
    import dataclasses as _dc
    actor = _make_actor()
    env = _make_env()
    _, agg = eval_aggregate(actor, env, w=1.0, seeds=[0])
    with pytest.raises((_dc.FrozenInstanceError, AttributeError)):
        agg.n_episodes = 99  # type: ignore[misc]


# --- empty seeds raises ---

def test_eval_aggregate_empty_seeds_raises():
    """eval_aggregate raises ValueError for an empty seeds sequence."""
    actor = _make_actor()
    env = _make_env()
    with pytest.raises(ValueError):
        eval_aggregate(actor, env, w=1.0, seeds=[])


# --- public API export ---

def test_phase15c_public_api_imports():
    """CTRLEvalAggregate and eval_aggregate are exported from src.eval."""
    from src.eval import CTRLEvalAggregate as _A, eval_aggregate as _f
    assert _A is CTRLEvalAggregate
    assert callable(_f)
