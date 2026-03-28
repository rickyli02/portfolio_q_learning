"""Unit tests for src/eval/record_set_io.py — Phase 15H record-set file IO."""

from __future__ import annotations

import json

import pytest
import torch

from src.config.schema import AssetConfig, EnvConfig
from src.envs.gbm_env import GBMPortfolioEnv
from src.eval import (
    CTRLEvalRecord,
    CTRLEvalRecordSet,
    eval_record_set,
    load_eval_record_sets,
    save_eval_record_sets,
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


def _tensors_close(a: torch.Tensor, b: torch.Tensor) -> bool:
    return torch.allclose(a.float(), b.float(), atol=1e-6)


def _synthetic_record(
    n_steps: int = 3,
    n_risky: int = 1,
    target_return_z: float | None = None,
    terminal_gap: float | None = None,
    wealth_end: float = 1.5,
) -> CTRLEvalRecord:
    times = torch.arange(n_steps, dtype=torch.float32)
    wealth_path = torch.linspace(1.0, wealth_end, n_steps + 1, dtype=torch.float32)
    actions = torch.ones(n_steps, n_risky, dtype=torch.float32) * 0.4
    return CTRLEvalRecord(
        times=times,
        wealth_path=wealth_path,
        actions=actions,
        terminal_wealth=float(wealth_path[-1]),
        initial_wealth=float(wealth_path[0]),
        n_steps=n_steps,
        min_wealth=float(wealth_path.min()),
        max_wealth=float(wealth_path.max()),
        target_return_z=target_return_z,
        terminal_gap=terminal_gap,
    )


def _synthetic_record_set(
    seeds: list[int] | None = None,
    n_steps: int = 3,
    n_risky: int = 1,
    target_return_z: float | None = None,
) -> CTRLEvalRecordSet:
    if seeds is None:
        seeds = [0, 1, 2]
    records = [
        _synthetic_record(
            n_steps=n_steps,
            n_risky=n_risky,
            target_return_z=target_return_z,
            terminal_gap=(1.5 - target_return_z) if target_return_z is not None else None,
            wealth_end=1.5 + i * 0.1,
        )
        for i in range(len(seeds))
    ]
    return CTRLEvalRecordSet(seeds=list(seeds), records=records)


def _assert_records_equal(
    got: CTRLEvalRecord, exp: CTRLEvalRecord, label: str = ""
) -> None:
    pfx = f"{label}: " if label else ""
    assert _tensors_close(got.times, exp.times), f"{pfx}times mismatch"
    assert _tensors_close(got.wealth_path, exp.wealth_path), f"{pfx}wealth_path mismatch"
    assert _tensors_close(got.actions, exp.actions), f"{pfx}actions mismatch"
    assert got.terminal_wealth == pytest.approx(exp.terminal_wealth, rel=1e-6), f"{pfx}terminal_wealth"
    assert got.initial_wealth == pytest.approx(exp.initial_wealth, rel=1e-6), f"{pfx}initial_wealth"
    assert got.n_steps == exp.n_steps, f"{pfx}n_steps"
    assert got.min_wealth == pytest.approx(exp.min_wealth, rel=1e-6), f"{pfx}min_wealth"
    assert got.max_wealth == pytest.approx(exp.max_wealth, rel=1e-6), f"{pfx}max_wealth"
    assert got.target_return_z == exp.target_return_z, f"{pfx}target_return_z"
    assert got.terminal_gap == exp.terminal_gap, f"{pfx}terminal_gap"


# ===========================================================================
# Public API exports
# ===========================================================================


def test_public_exports_include_record_set_io():
    import src.eval as ev

    assert hasattr(ev, "save_eval_record_sets")
    assert hasattr(ev, "load_eval_record_sets")
    assert "save_eval_record_sets" in ev.__all__
    assert "load_eval_record_sets" in ev.__all__


# ===========================================================================
# Roundtrip — single record set, all fields
# ===========================================================================


def test_roundtrip_single_record_set_all_fields(tmp_path):
    """Exact roundtrip for a single record set: seeds and every per-record field."""
    rs = _synthetic_record_set(seeds=[7, 42], target_return_z=1.3)
    p = tmp_path / "sets.jsonl"
    save_eval_record_sets([rs], p)
    loaded = load_eval_record_sets(p)

    assert len(loaded) == 1
    out = loaded[0]

    assert out.seeds == rs.seeds
    assert len(out.records) == len(rs.records)
    for i, (got, exp) in enumerate(zip(out.records, rs.records)):
        _assert_records_equal(got, exp, label=f"record[{i}]")


# ===========================================================================
# Roundtrip — multiple record sets, order preserved
# ===========================================================================


def test_roundtrip_multiple_record_sets_order_preserved(tmp_path):
    """Three record sets roundtrip in exact order."""
    sets = [
        _synthetic_record_set(seeds=[0, 1]),
        _synthetic_record_set(seeds=[10, 20, 30], n_steps=4),
        _synthetic_record_set(seeds=[99], target_return_z=1.2),
    ]
    p = tmp_path / "multi.jsonl"
    save_eval_record_sets(sets, p)
    loaded = load_eval_record_sets(p)

    assert len(loaded) == 3
    for rs_idx, (out_rs, exp_rs) in enumerate(zip(loaded, sets)):
        assert out_rs.seeds == exp_rs.seeds, f"seeds mismatch at set {rs_idx}"
        assert len(out_rs.records) == len(exp_rs.records)
        for rec_idx, (got, exp) in enumerate(zip(out_rs.records, exp_rs.records)):
            _assert_records_equal(got, exp, label=f"set[{rs_idx}] record[{rec_idx}]")


# ===========================================================================
# Nullable optional fields inside embedded records
# ===========================================================================


def test_roundtrip_nullable_fields_none(tmp_path):
    """target_return_z=None and terminal_gap=None inside embedded records roundtrip."""
    rs = _synthetic_record_set(seeds=[0, 1], target_return_z=None)
    p = tmp_path / "none_fields.jsonl"
    save_eval_record_sets([rs], p)
    loaded = load_eval_record_sets(p)

    for rec in loaded[0].records:
        assert rec.target_return_z is None
        assert rec.terminal_gap is None


def test_roundtrip_nullable_fields_present(tmp_path):
    """target_return_z and terminal_gap values inside embedded records roundtrip."""
    rs = _synthetic_record_set(seeds=[3, 5], target_return_z=1.4)
    p = tmp_path / "with_z.jsonl"
    save_eval_record_sets([rs], p)
    loaded = load_eval_record_sets(p)

    for got, exp in zip(loaded[0].records, rs.records):
        assert got.target_return_z == pytest.approx(exp.target_return_z, rel=1e-6)  # type: ignore[arg-type]
        assert got.terminal_gap == pytest.approx(exp.terminal_gap, rel=1e-6)  # type: ignore[arg-type]


# ===========================================================================
# Tensor dtype
# ===========================================================================


def test_loaded_embedded_tensors_are_float32(tmp_path):
    """Restored tensor fields in embedded records have dtype torch.float32."""
    rs = _synthetic_record_set(seeds=[0])
    p = tmp_path / "dtype.jsonl"
    save_eval_record_sets([rs], p)
    loaded = load_eval_record_sets(p)

    rec = loaded[0].records[0]
    assert rec.times.dtype == torch.float32
    assert rec.wealth_path.dtype == torch.float32
    assert rec.actions.dtype == torch.float32


# ===========================================================================
# Live integration roundtrip from eval_record_set
# ===========================================================================


def test_live_roundtrip_from_eval_record_set(tmp_path):
    """Live eval_record_set result survives save/load roundtrip intact."""
    actor = _make_actor()
    env = _make_env(n_steps=5)
    seeds = [0, 3, 7]
    rs = eval_record_set(actor, env, w=1.0, seeds=seeds, target_return_z=1.2)

    p = tmp_path / "live.jsonl"
    save_eval_record_sets([rs], p)
    loaded = load_eval_record_sets(p)

    assert len(loaded) == 1
    out = loaded[0]
    assert out.seeds == seeds
    assert len(out.records) == len(seeds)
    for i, (got, exp) in enumerate(zip(out.records, rs.records)):
        _assert_records_equal(got, exp, label=f"live record[{i}]")


# ===========================================================================
# Error: nonexistent file
# ===========================================================================


def test_load_nonexistent_file_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        load_eval_record_sets(tmp_path / "missing.jsonl")


# ===========================================================================
# Error: malformed / invalid JSON
# ===========================================================================


def test_load_invalid_json_raises_value_error(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text("not-json\n")
    with pytest.raises(ValueError, match="Malformed JSON"):
        load_eval_record_sets(p)


def test_load_json_array_top_level_raises_value_error(tmp_path):
    p = tmp_path / "array.jsonl"
    p.write_text(json.dumps([1, 2, 3]) + "\n")
    with pytest.raises(ValueError, match="Expected a JSON object"):
        load_eval_record_sets(p)


def test_load_json_scalar_top_level_raises_value_error(tmp_path):
    p = tmp_path / "scalar.jsonl"
    p.write_text("42\n")
    with pytest.raises(ValueError, match="Expected a JSON object"):
        load_eval_record_sets(p)


# ===========================================================================
# Error: missing required set-level fields
# ===========================================================================


def _base_valid_set_obj(n_steps: int = 3, n_seeds: int = 2) -> dict:
    rec = {
        "times": list(range(n_steps)),
        "wealth_path": [1.0 + i * 0.1 for i in range(n_steps + 1)],
        "actions": [[0.4]] * n_steps,
        "terminal_wealth": 1.3,
        "initial_wealth": 1.0,
        "n_steps": n_steps,
        "min_wealth": 1.0,
        "max_wealth": 1.3,
        "target_return_z": None,
        "terminal_gap": None,
    }
    return {
        "seeds": list(range(n_seeds)),
        "records": [rec] * n_seeds,
    }


@pytest.mark.parametrize("missing_field", ["seeds", "records"])
def test_load_missing_set_level_field_raises_value_error(tmp_path, missing_field):
    obj = _base_valid_set_obj()
    del obj[missing_field]
    p = tmp_path / "missing.jsonl"
    p.write_text(json.dumps(obj) + "\n")
    with pytest.raises(ValueError, match=missing_field):
        load_eval_record_sets(p)


# ===========================================================================
# Error: wrong scalar/list types at set level
# ===========================================================================


def test_load_seeds_not_list_raises_value_error(tmp_path):
    obj = _base_valid_set_obj()
    obj["seeds"] = "not-a-list"
    p = tmp_path / "bad_seeds.jsonl"
    p.write_text(json.dumps(obj) + "\n")
    with pytest.raises(ValueError, match="'seeds'"):
        load_eval_record_sets(p)


def test_load_seeds_contains_float_raises_value_error(tmp_path):
    obj = _base_valid_set_obj()
    obj["seeds"] = [0, 1.5]
    p = tmp_path / "float_seed.jsonl"
    p.write_text(json.dumps(obj) + "\n")
    with pytest.raises(ValueError, match="'seeds"):
        load_eval_record_sets(p)


def test_load_records_not_list_raises_value_error(tmp_path):
    obj = _base_valid_set_obj()
    obj["records"] = "not-a-list"
    p = tmp_path / "bad_records.jsonl"
    p.write_text(json.dumps(obj) + "\n")
    with pytest.raises(ValueError, match="'records'"):
        load_eval_record_sets(p)


# ===========================================================================
# Error: len(seeds) != len(records)
# ===========================================================================


def test_load_seeds_records_length_mismatch_raises_value_error(tmp_path):
    obj = _base_valid_set_obj(n_seeds=2)
    obj["seeds"] = [0, 1, 2]  # 3 seeds, 2 records
    p = tmp_path / "len_mismatch.jsonl"
    p.write_text(json.dumps(obj) + "\n")
    with pytest.raises(ValueError, match=r"len\(seeds\)"):
        load_eval_record_sets(p)


# ===========================================================================
# Error: wrong type inside embedded record
# ===========================================================================


def test_load_embedded_record_missing_field_raises_value_error(tmp_path):
    obj = _base_valid_set_obj(n_seeds=1)
    del obj["records"][0]["n_steps"]
    p = tmp_path / "missing_rec_field.jsonl"
    p.write_text(json.dumps(obj) + "\n")
    with pytest.raises(ValueError, match="n_steps"):
        load_eval_record_sets(p)


def test_load_embedded_record_wrong_scalar_type_raises_value_error(tmp_path):
    obj = _base_valid_set_obj(n_seeds=1)
    obj["records"][0]["terminal_wealth"] = "bad"
    p = tmp_path / "bad_scalar_rec.jsonl"
    p.write_text(json.dumps(obj) + "\n")
    with pytest.raises(ValueError):
        load_eval_record_sets(p)


def test_load_embedded_record_times_not_list_raises_value_error(tmp_path):
    obj = _base_valid_set_obj(n_seeds=1)
    obj["records"][0]["times"] = "not-a-list"
    p = tmp_path / "bad_times.jsonl"
    p.write_text(json.dumps(obj) + "\n")
    with pytest.raises(ValueError, match="times"):
        load_eval_record_sets(p)


# ===========================================================================
# Error: inconsistent embedded record path lengths
# ===========================================================================


def test_load_embedded_record_times_wrong_length_raises_value_error(tmp_path):
    obj = _base_valid_set_obj(n_steps=3, n_seeds=1)
    obj["records"][0]["times"] = [0.0, 1.0]  # length 2, not 3
    p = tmp_path / "bad_times_len.jsonl"
    p.write_text(json.dumps(obj) + "\n")
    with pytest.raises(ValueError, match=r"len\(times\)"):
        load_eval_record_sets(p)


def test_load_embedded_record_wealth_path_wrong_length_raises_value_error(tmp_path):
    obj = _base_valid_set_obj(n_steps=3, n_seeds=1)
    obj["records"][0]["wealth_path"] = [1.0, 1.1, 1.2]  # length 3, not 4
    p = tmp_path / "bad_wp_len.jsonl"
    p.write_text(json.dumps(obj) + "\n")
    with pytest.raises(ValueError, match=r"len\(wealth_path\)"):
        load_eval_record_sets(p)


def test_load_embedded_record_actions_wrong_length_raises_value_error(tmp_path):
    obj = _base_valid_set_obj(n_steps=3, n_seeds=1)
    obj["records"][0]["actions"] = [[0.4], [0.4]]  # length 2, not 3
    p = tmp_path / "bad_act_len.jsonl"
    p.write_text(json.dumps(obj) + "\n")
    with pytest.raises(ValueError, match=r"len\(actions\)"):
        load_eval_record_sets(p)


def test_load_embedded_record_inconsistent_action_rows_raises_value_error(tmp_path):
    obj = _base_valid_set_obj(n_steps=3, n_seeds=1)
    obj["records"][0]["actions"] = [[0.4], [0.4, 0.2], [0.4]]
    p = tmp_path / "bad_act_width.jsonl"
    p.write_text(json.dumps(obj) + "\n")
    with pytest.raises(ValueError, match="inconsistent action rows"):
        load_eval_record_sets(p)
