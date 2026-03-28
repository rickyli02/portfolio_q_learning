"""Unit tests for src/eval/bundle_io.py — Phase 15K scalar-bundle file IO."""

from __future__ import annotations

import json

import pytest

from src.config.schema import AssetConfig, EnvConfig
from src.envs.gbm_env import GBMPortfolioEnv
from src.eval import (
    CTRLEvalAggregate,
    CTRLEvalScalarBundle,
    CTRLEvalSummary,
    bundle_from_record_set,
    eval_record_set,
    load_eval_bundles,
    save_eval_bundles,
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


def _synthetic_summary(
    tw: float = 1.3,
    iw: float = 1.0,
    n_steps: int = 5,
    min_w: float = 0.9,
    max_w: float = 1.4,
    target_return_z: float | None = None,
    terminal_gap: float | None = None,
) -> CTRLEvalSummary:
    return CTRLEvalSummary(
        terminal_wealth=tw,
        initial_wealth=iw,
        target_return_z=target_return_z,
        terminal_gap=terminal_gap,
        n_steps=n_steps,
        min_wealth=min_w,
        max_wealth=max_w,
    )


def _synthetic_bundle(
    seeds: list[int] | None = None,
    target_return_z: float | None = None,
) -> CTRLEvalScalarBundle:
    if seeds is None:
        seeds = [0, 1, 2]
    summaries = [
        _synthetic_summary(
            tw=1.0 + i * 0.1,
            target_return_z=target_return_z,
            terminal_gap=(1.0 + i * 0.1 - target_return_z) if target_return_z is not None else None,
        )
        for i in range(len(seeds))
    ]
    tws = [s.terminal_wealth for s in summaries]
    n = len(summaries)
    has_target = target_return_z is not None
    aggregate = CTRLEvalAggregate(
        n_episodes=n,
        mean_terminal_wealth=sum(tws) / n,
        min_terminal_wealth=min(tws),
        max_terminal_wealth=max(tws),
        mean_terminal_gap=(sum(s.terminal_gap for s in summaries) / n) if has_target else None,  # type: ignore[misc]
        target_hit_rate=(sum(1 for tw in tws if tw >= target_return_z) / n) if has_target else None,
    )
    return CTRLEvalScalarBundle(seeds=list(seeds), summaries=summaries, aggregate=aggregate)


def _assert_summaries_equal(got: CTRLEvalSummary, exp: CTRLEvalSummary, label: str = "") -> None:
    pfx = f"{label}: " if label else ""
    assert got.terminal_wealth == pytest.approx(exp.terminal_wealth, rel=1e-6), f"{pfx}terminal_wealth"
    assert got.initial_wealth == pytest.approx(exp.initial_wealth, rel=1e-6), f"{pfx}initial_wealth"
    assert got.n_steps == exp.n_steps, f"{pfx}n_steps"
    assert got.min_wealth == pytest.approx(exp.min_wealth, rel=1e-6), f"{pfx}min_wealth"
    assert got.max_wealth == pytest.approx(exp.max_wealth, rel=1e-6), f"{pfx}max_wealth"
    assert got.target_return_z == exp.target_return_z, f"{pfx}target_return_z"
    assert got.terminal_gap == exp.terminal_gap, f"{pfx}terminal_gap"


def _assert_aggregates_equal(got: CTRLEvalAggregate, exp: CTRLEvalAggregate, label: str = "") -> None:
    pfx = f"{label}: " if label else ""
    assert got.n_episodes == exp.n_episodes, f"{pfx}n_episodes"
    assert got.mean_terminal_wealth == pytest.approx(exp.mean_terminal_wealth, rel=1e-6), f"{pfx}mean_terminal_wealth"
    assert got.min_terminal_wealth == pytest.approx(exp.min_terminal_wealth, rel=1e-6), f"{pfx}min_terminal_wealth"
    assert got.max_terminal_wealth == pytest.approx(exp.max_terminal_wealth, rel=1e-6), f"{pfx}max_terminal_wealth"
    assert got.mean_terminal_gap == exp.mean_terminal_gap, f"{pfx}mean_terminal_gap"
    assert got.target_hit_rate == exp.target_hit_rate, f"{pfx}target_hit_rate"


# ===========================================================================
# Public API exports
# ===========================================================================


def test_public_exports_include_bundle_io():
    import src.eval as ev

    assert hasattr(ev, "save_eval_bundles")
    assert hasattr(ev, "load_eval_bundles")
    assert "save_eval_bundles" in ev.__all__
    assert "load_eval_bundles" in ev.__all__


# ===========================================================================
# Roundtrip — single bundle, all fields
# ===========================================================================


def test_roundtrip_single_bundle_all_fields(tmp_path):
    """Exact field-by-field roundtrip for a single bundle with target."""
    b = _synthetic_bundle(seeds=[7, 42], target_return_z=1.1)
    p = tmp_path / "bundles.jsonl"
    save_eval_bundles([b], p)
    loaded = load_eval_bundles(p)

    assert len(loaded) == 1
    out = loaded[0]
    assert out.seeds == b.seeds
    assert len(out.summaries) == len(b.summaries)
    for i, (got, exp) in enumerate(zip(out.summaries, b.summaries)):
        _assert_summaries_equal(got, exp, label=f"summary[{i}]")
    _assert_aggregates_equal(out.aggregate, b.aggregate)


def test_roundtrip_single_bundle_no_target(tmp_path):
    """Exact roundtrip for a bundle with no target (nullable fields are None)."""
    b = _synthetic_bundle(seeds=[0, 1, 2], target_return_z=None)
    p = tmp_path / "no_target.jsonl"
    save_eval_bundles([b], p)
    loaded = load_eval_bundles(p)

    out = loaded[0]
    assert out.seeds == b.seeds
    for s in out.summaries:
        assert s.target_return_z is None
        assert s.terminal_gap is None
    assert out.aggregate.mean_terminal_gap is None
    assert out.aggregate.target_hit_rate is None


# ===========================================================================
# Roundtrip — multiple bundles, order preserved
# ===========================================================================


def test_roundtrip_multiple_bundles_order_preserved(tmp_path):
    """Three bundles roundtrip in exact order."""
    bundles = [
        _synthetic_bundle(seeds=[0, 1]),
        _synthetic_bundle(seeds=[10, 20, 30], target_return_z=1.2),
        _synthetic_bundle(seeds=[99]),
    ]
    p = tmp_path / "multi.jsonl"
    save_eval_bundles(bundles, p)
    loaded = load_eval_bundles(p)

    assert len(loaded) == 3
    for bi, (out, exp) in enumerate(zip(loaded, bundles)):
        assert out.seeds == exp.seeds, f"seeds mismatch at bundle {bi}"
        for i, (gs, es) in enumerate(zip(out.summaries, exp.summaries)):
            _assert_summaries_equal(gs, es, label=f"bundle[{bi}] summary[{i}]")
        _assert_aggregates_equal(out.aggregate, exp.aggregate, label=f"bundle[{bi}]")


# ===========================================================================
# Live integration roundtrip
# ===========================================================================


def test_live_roundtrip_from_bundle_from_record_set(tmp_path):
    """Live bundle_from_record_set result survives save/load roundtrip."""
    actor = _make_actor()
    env = _make_env(n_steps=5)
    seeds = [0, 3, 7]
    rs = eval_record_set(actor, env, w=1.0, seeds=seeds, target_return_z=1.1)
    b = bundle_from_record_set(rs)

    p = tmp_path / "live.jsonl"
    save_eval_bundles([b], p)
    loaded = load_eval_bundles(p)

    assert len(loaded) == 1
    out = loaded[0]
    assert out.seeds == seeds
    for i, (gs, es) in enumerate(zip(out.summaries, b.summaries)):
        _assert_summaries_equal(gs, es, label=f"live summary[{i}]")
    _assert_aggregates_equal(out.aggregate, b.aggregate, label="live aggregate")


# ===========================================================================
# Error: nonexistent file
# ===========================================================================


def test_load_nonexistent_file_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        load_eval_bundles(tmp_path / "missing.jsonl")


# ===========================================================================
# Error: malformed / invalid JSON
# ===========================================================================


def test_load_invalid_json_raises_value_error(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text("not-json\n")
    with pytest.raises(ValueError, match="Malformed JSON"):
        load_eval_bundles(p)


def test_load_json_array_top_level_raises_value_error(tmp_path):
    p = tmp_path / "array.jsonl"
    p.write_text(json.dumps([1, 2]) + "\n")
    with pytest.raises(ValueError, match="Expected a JSON object"):
        load_eval_bundles(p)


def test_load_json_scalar_top_level_raises_value_error(tmp_path):
    p = tmp_path / "scalar.jsonl"
    p.write_text("42\n")
    with pytest.raises(ValueError, match="Expected a JSON object"):
        load_eval_bundles(p)


# ---------------------------------------------------------------------------
# Canonical valid object builder
# ---------------------------------------------------------------------------


def _base_valid_obj(n_seeds: int = 2) -> dict:
    summary = {
        "terminal_wealth": 1.3,
        "initial_wealth": 1.0,
        "target_return_z": None,
        "terminal_gap": None,
        "n_steps": 5,
        "min_wealth": 0.9,
        "max_wealth": 1.4,
    }
    return {
        "seeds": list(range(n_seeds)),
        "summaries": [dict(summary) for _ in range(n_seeds)],
        "aggregate": {
            "n_episodes": n_seeds,
            "mean_terminal_wealth": 1.3,
            "min_terminal_wealth": 1.2,
            "max_terminal_wealth": 1.4,
            "mean_terminal_gap": None,
            "target_hit_rate": None,
        },
    }


# ===========================================================================
# Error: missing required top-level fields
# ===========================================================================


@pytest.mark.parametrize("missing_field", ["seeds", "summaries", "aggregate"])
def test_load_missing_top_level_field_raises_value_error(tmp_path, missing_field):
    obj = _base_valid_obj()
    del obj[missing_field]
    p = tmp_path / "missing.jsonl"
    p.write_text(json.dumps(obj) + "\n")
    with pytest.raises(ValueError, match=missing_field):
        load_eval_bundles(p)


# ===========================================================================
# Error: wrong scalar/list types at bundle level
# ===========================================================================


def test_load_seeds_not_list_raises_value_error(tmp_path):
    obj = _base_valid_obj()
    obj["seeds"] = "bad"
    p = tmp_path / "bad_seeds.jsonl"
    p.write_text(json.dumps(obj) + "\n")
    with pytest.raises(ValueError, match="'seeds'"):
        load_eval_bundles(p)


def test_load_seeds_contains_float_raises_value_error(tmp_path):
    obj = _base_valid_obj()
    obj["seeds"] = [0, 1.5]
    p = tmp_path / "float_seed.jsonl"
    p.write_text(json.dumps(obj) + "\n")
    with pytest.raises(ValueError, match="'seeds"):
        load_eval_bundles(p)


def test_load_summaries_not_list_raises_value_error(tmp_path):
    obj = _base_valid_obj()
    obj["summaries"] = "bad"
    p = tmp_path / "bad_summaries.jsonl"
    p.write_text(json.dumps(obj) + "\n")
    with pytest.raises(ValueError, match="'summaries'"):
        load_eval_bundles(p)


# ===========================================================================
# Error: wrong types inside embedded summary
# ===========================================================================


def test_load_summary_missing_field_raises_value_error(tmp_path):
    obj = _base_valid_obj(n_seeds=1)
    del obj["summaries"][0]["terminal_wealth"]
    p = tmp_path / "missing_sum_field.jsonl"
    p.write_text(json.dumps(obj) + "\n")
    with pytest.raises(ValueError, match="terminal_wealth"):
        load_eval_bundles(p)


def test_load_summary_wrong_scalar_type_raises_value_error(tmp_path):
    obj = _base_valid_obj(n_seeds=1)
    obj["summaries"][0]["n_steps"] = 5.5  # float not int
    p = tmp_path / "bad_sum_scalar.jsonl"
    p.write_text(json.dumps(obj) + "\n")
    with pytest.raises(ValueError):
        load_eval_bundles(p)


def test_load_summary_terminal_wealth_string_raises_value_error(tmp_path):
    obj = _base_valid_obj(n_seeds=1)
    obj["summaries"][0]["terminal_wealth"] = "bad"
    p = tmp_path / "bad_sum_tw.jsonl"
    p.write_text(json.dumps(obj) + "\n")
    with pytest.raises(ValueError):
        load_eval_bundles(p)


# ===========================================================================
# Error: wrong types inside embedded aggregate
# ===========================================================================


def test_load_aggregate_missing_field_raises_value_error(tmp_path):
    obj = _base_valid_obj()
    del obj["aggregate"]["n_episodes"]
    p = tmp_path / "missing_agg_field.jsonl"
    p.write_text(json.dumps(obj) + "\n")
    with pytest.raises(ValueError, match="n_episodes"):
        load_eval_bundles(p)


def test_load_aggregate_wrong_scalar_type_raises_value_error(tmp_path):
    obj = _base_valid_obj()
    obj["aggregate"]["n_episodes"] = 2.0  # float not int
    p = tmp_path / "bad_agg_scalar.jsonl"
    p.write_text(json.dumps(obj) + "\n")
    with pytest.raises(ValueError):
        load_eval_bundles(p)


# ===========================================================================
# Error: bundle-level consistency mismatches
# ===========================================================================


def test_load_seeds_summaries_length_mismatch_raises_value_error(tmp_path):
    obj = _base_valid_obj(n_seeds=2)
    obj["seeds"] = [0, 1, 2]  # 3 seeds, 2 summaries
    p = tmp_path / "len_mismatch.jsonl"
    p.write_text(json.dumps(obj) + "\n")
    with pytest.raises(ValueError, match=r"len\(seeds\)"):
        load_eval_bundles(p)


def test_load_aggregate_n_episodes_mismatch_raises_value_error(tmp_path):
    obj = _base_valid_obj(n_seeds=2)
    obj["aggregate"]["n_episodes"] = 5  # 5 != 2 summaries
    p = tmp_path / "n_ep_mismatch.jsonl"
    p.write_text(json.dumps(obj) + "\n")
    with pytest.raises(ValueError, match="n_episodes"):
        load_eval_bundles(p)
