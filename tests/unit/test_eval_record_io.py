"""Unit tests for src/eval/record_io.py — Phase 15F record file IO."""

from __future__ import annotations

import json

import pytest
import torch

from src.config.schema import AssetConfig, EnvConfig
from src.envs.gbm_env import GBMPortfolioEnv
from src.eval import (
    CTRLEvalRecord,
    eval_record,
    load_eval_records,
    save_eval_records,
)
from src.eval.record_io import load_eval_records, save_eval_records
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


def _make_record(
    n_steps: int = 5,
    target_return_z: float | None = None,
    seed: int = 0,
) -> CTRLEvalRecord:
    actor = _make_actor()
    env = _make_env(n_steps=n_steps)
    return eval_record(actor, env, w=1.0, target_return_z=target_return_z, seed=seed)


def _synthetic_record(
    n_steps: int = 3,
    n_risky: int = 2,
    target_return_z: float | None = None,
    terminal_gap: float | None = None,
) -> CTRLEvalRecord:
    """Build a deterministic synthetic record without running a live episode."""
    times = torch.arange(n_steps, dtype=torch.float32)
    wealth_path = torch.linspace(1.0, 1.5, n_steps + 1, dtype=torch.float32)
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


def _tensors_close(a: torch.Tensor, b: torch.Tensor) -> bool:
    return torch.allclose(a.float(), b.float(), atol=1e-6)


# ===========================================================================
# Public API exports
# ===========================================================================


def test_public_exports_include_record_io():
    """save_eval_records and load_eval_records are exported from src.eval."""
    import src.eval as ev

    assert hasattr(ev, "save_eval_records")
    assert hasattr(ev, "load_eval_records")
    assert "save_eval_records" in ev.__all__
    assert "load_eval_records" in ev.__all__


# ===========================================================================
# Roundtrip — single record
# ===========================================================================


def test_roundtrip_single_record_all_fields(tmp_path):
    """Exact field-by-field roundtrip for a single record."""
    rec = _synthetic_record(n_steps=4, n_risky=2, target_return_z=1.3, terminal_gap=0.1)
    p = tmp_path / "records.jsonl"
    save_eval_records([rec], p)
    loaded = load_eval_records(p)

    assert len(loaded) == 1
    out = loaded[0]

    assert _tensors_close(out.times, rec.times)
    assert _tensors_close(out.wealth_path, rec.wealth_path)
    assert _tensors_close(out.actions, rec.actions)
    assert out.terminal_wealth == pytest.approx(rec.terminal_wealth, rel=1e-6)
    assert out.initial_wealth == pytest.approx(rec.initial_wealth, rel=1e-6)
    assert out.n_steps == rec.n_steps
    assert out.min_wealth == pytest.approx(rec.min_wealth, rel=1e-6)
    assert out.max_wealth == pytest.approx(rec.max_wealth, rel=1e-6)
    assert out.target_return_z == pytest.approx(rec.target_return_z, rel=1e-6)
    assert out.terminal_gap == pytest.approx(rec.terminal_gap, rel=1e-6)


# ===========================================================================
# Roundtrip — multiple records, order preservation
# ===========================================================================


def test_roundtrip_multiple_records_order_preserved(tmp_path):
    """Save three records and verify exact order is preserved on load."""
    recs = [
        _synthetic_record(n_steps=2, n_risky=1, target_return_z=1.0, terminal_gap=0.05),
        _synthetic_record(n_steps=3, n_risky=1, target_return_z=None, terminal_gap=None),
        _synthetic_record(n_steps=5, n_risky=2, target_return_z=1.2, terminal_gap=-0.1),
    ]
    p = tmp_path / "multi.jsonl"
    save_eval_records(recs, p)
    loaded = load_eval_records(p)

    assert len(loaded) == 3
    for orig, out in zip(recs, loaded):
        assert _tensors_close(out.times, orig.times)
        assert _tensors_close(out.wealth_path, orig.wealth_path)
        assert _tensors_close(out.actions, orig.actions)
        assert out.n_steps == orig.n_steps
        assert out.terminal_wealth == pytest.approx(orig.terminal_wealth, rel=1e-6)
        assert out.initial_wealth == pytest.approx(orig.initial_wealth, rel=1e-6)
        assert out.min_wealth == pytest.approx(orig.min_wealth, rel=1e-6)
        assert out.max_wealth == pytest.approx(orig.max_wealth, rel=1e-6)


def test_roundtrip_multiple_records_n_steps_differ(tmp_path):
    """Records with different n_steps roundtrip independently."""
    recs = [_synthetic_record(n_steps=n) for n in [2, 7, 4]]
    p = tmp_path / "diff_steps.jsonl"
    save_eval_records(recs, p)
    loaded = load_eval_records(p)

    assert [r.n_steps for r in loaded] == [2, 7, 4]


# ===========================================================================
# Nullable optional fields
# ===========================================================================


def test_roundtrip_target_return_z_none(tmp_path):
    """target_return_z=None and terminal_gap=None roundtrip as None."""
    rec = _synthetic_record(n_steps=3, target_return_z=None, terminal_gap=None)
    p = tmp_path / "none_fields.jsonl"
    save_eval_records([rec], p)
    loaded = load_eval_records(p)

    assert loaded[0].target_return_z is None
    assert loaded[0].terminal_gap is None


def test_roundtrip_target_return_z_present(tmp_path):
    """target_return_z and terminal_gap are preserved exactly when set."""
    rec = _synthetic_record(n_steps=3, target_return_z=1.5, terminal_gap=-0.05)
    p = tmp_path / "with_z.jsonl"
    save_eval_records([rec], p)
    loaded = load_eval_records(p)

    assert loaded[0].target_return_z == pytest.approx(1.5, rel=1e-6)
    assert loaded[0].terminal_gap == pytest.approx(-0.05, rel=1e-6)


# ===========================================================================
# Tensor dtype
# ===========================================================================


def test_loaded_tensors_are_float32(tmp_path):
    """Restored tensor fields have dtype torch.float32."""
    rec = _synthetic_record()
    p = tmp_path / "dtype.jsonl"
    save_eval_records([rec], p)
    loaded = load_eval_records(p)

    out = loaded[0]
    assert out.times.dtype == torch.float32
    assert out.wealth_path.dtype == torch.float32
    assert out.actions.dtype == torch.float32


# ===========================================================================
# Live integration roundtrip from eval_record
# ===========================================================================


def test_live_roundtrip_from_eval_record(tmp_path):
    """Live eval_record result survives save/load roundtrip intact."""
    rec = _make_record(n_steps=5, target_return_z=1.2, seed=7)
    p = tmp_path / "live.jsonl"
    save_eval_records([rec], p)
    loaded = load_eval_records(p)

    assert len(loaded) == 1
    out = loaded[0]
    assert out.n_steps == rec.n_steps
    assert _tensors_close(out.times, rec.times)
    assert _tensors_close(out.wealth_path, rec.wealth_path)
    assert _tensors_close(out.actions, rec.actions)
    assert out.terminal_wealth == pytest.approx(rec.terminal_wealth, rel=1e-6)
    assert out.target_return_z == pytest.approx(rec.target_return_z, rel=1e-6)  # type: ignore[arg-type]
    assert out.terminal_gap == pytest.approx(rec.terminal_gap, rel=1e-6)  # type: ignore[arg-type]


# ===========================================================================
# Error: nonexistent file
# ===========================================================================


def test_load_nonexistent_file_raises_file_not_found(tmp_path):
    """Loading from a nonexistent path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="not found"):
        load_eval_records(tmp_path / "missing.jsonl")


# ===========================================================================
# Error: malformed / invalid JSON
# ===========================================================================


def test_load_invalid_json_raises_value_error(tmp_path):
    """A line with invalid JSON raises ValueError."""
    p = tmp_path / "bad.jsonl"
    p.write_text("not-json\n")
    with pytest.raises(ValueError, match="Malformed JSON"):
        load_eval_records(p)


def test_load_json_array_top_level_raises_value_error(tmp_path):
    """A line that is a JSON array (not an object) raises ValueError."""
    p = tmp_path / "array.jsonl"
    p.write_text(json.dumps([1, 2, 3]) + "\n")
    with pytest.raises(ValueError, match="Expected a JSON object"):
        load_eval_records(p)


def test_load_json_scalar_top_level_raises_value_error(tmp_path):
    """A line that is a bare JSON scalar raises ValueError."""
    p = tmp_path / "scalar.jsonl"
    p.write_text("42\n")
    with pytest.raises(ValueError, match="Expected a JSON object"):
        load_eval_records(p)


# ===========================================================================
# Error: missing required fields
# ===========================================================================


def _base_valid_obj(n_steps: int = 3, n_risky: int = 1) -> dict:
    return {
        "times": [0.0, 1.0, 2.0][:n_steps],
        "wealth_path": [1.0, 1.1, 1.2, 1.3][: n_steps + 1],
        "actions": [[0.4]] * n_steps,
        "terminal_wealth": 1.3,
        "initial_wealth": 1.0,
        "n_steps": n_steps,
        "min_wealth": 1.0,
        "max_wealth": 1.3,
        "target_return_z": None,
        "terminal_gap": None,
    }


@pytest.mark.parametrize(
    "missing_field",
    [
        "times",
        "wealth_path",
        "actions",
        "terminal_wealth",
        "initial_wealth",
        "n_steps",
        "min_wealth",
        "max_wealth",
    ],
)
def test_load_missing_required_field_raises_value_error(tmp_path, missing_field):
    """Omitting any required field raises ValueError."""
    obj = _base_valid_obj()
    del obj[missing_field]
    p = tmp_path / "missing.jsonl"
    p.write_text(json.dumps(obj) + "\n")
    with pytest.raises(ValueError, match=missing_field):
        load_eval_records(p)


# ===========================================================================
# Error: wrong scalar types
# ===========================================================================


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("terminal_wealth", "not-a-number"),
        ("initial_wealth", True),
        ("min_wealth", [1.0]),
        ("max_wealth", {"v": 1.0}),
        ("n_steps", 3.5),
        ("n_steps", True),
    ],
)
def test_load_wrong_scalar_type_raises_value_error(tmp_path, field, bad_value):
    """A required scalar field with the wrong type raises ValueError."""
    obj = _base_valid_obj()
    obj[field] = bad_value
    p = tmp_path / "bad_scalar.jsonl"
    p.write_text(json.dumps(obj) + "\n")
    with pytest.raises(ValueError):
        load_eval_records(p)


# ===========================================================================
# Error: wrong list / tensor field types
# ===========================================================================


def test_load_times_not_list_raises_value_error(tmp_path):
    """times field that is not a list raises ValueError."""
    obj = _base_valid_obj()
    obj["times"] = "not-a-list"
    p = tmp_path / "bad_times.jsonl"
    p.write_text(json.dumps(obj) + "\n")
    with pytest.raises(ValueError, match="'times'"):
        load_eval_records(p)


def test_load_wealth_path_contains_string_raises_value_error(tmp_path):
    """wealth_path containing a string element raises ValueError."""
    obj = _base_valid_obj()
    obj["wealth_path"] = [1.0, "x", 1.2, 1.3]
    p = tmp_path / "bad_wp.jsonl"
    p.write_text(json.dumps(obj) + "\n")
    with pytest.raises(ValueError, match="'wealth_path"):
        load_eval_records(p)


def test_load_actions_not_list_of_lists_raises_value_error(tmp_path):
    """actions field that is a flat list (not list of lists) raises ValueError."""
    obj = _base_valid_obj()
    obj["actions"] = [0.4, 0.4, 0.4]  # flat list, not list-of-lists
    p = tmp_path / "flat_actions.jsonl"
    p.write_text(json.dumps(obj) + "\n")
    with pytest.raises(ValueError, match="'actions"):
        load_eval_records(p)


# ===========================================================================
# Error: inconsistent path lengths
# ===========================================================================


def test_load_times_wrong_length_raises_value_error(tmp_path):
    """len(times) != n_steps raises ValueError."""
    obj = _base_valid_obj(n_steps=3)
    obj["times"] = [0.0, 1.0]  # length 2, not 3
    p = tmp_path / "bad_times_len.jsonl"
    p.write_text(json.dumps(obj) + "\n")
    with pytest.raises(ValueError, match=r"len\(times\)"):
        load_eval_records(p)


def test_load_wealth_path_wrong_length_raises_value_error(tmp_path):
    """len(wealth_path) != n_steps+1 raises ValueError."""
    obj = _base_valid_obj(n_steps=3)
    obj["wealth_path"] = [1.0, 1.1, 1.2]  # length 3, not 4
    p = tmp_path / "bad_wp_len.jsonl"
    p.write_text(json.dumps(obj) + "\n")
    with pytest.raises(ValueError, match=r"len\(wealth_path\)"):
        load_eval_records(p)


def test_load_actions_wrong_length_raises_value_error(tmp_path):
    """len(actions) != n_steps raises ValueError."""
    obj = _base_valid_obj(n_steps=3)
    obj["actions"] = [[0.4], [0.4]]  # length 2, not 3
    p = tmp_path / "bad_act_len.jsonl"
    p.write_text(json.dumps(obj) + "\n")
    with pytest.raises(ValueError, match=r"len\(actions\)"):
        load_eval_records(p)


def test_load_actions_inconsistent_inner_width_raises_value_error(tmp_path):
    """actions rows with inconsistent inner width raise ValueError."""
    obj = _base_valid_obj(n_steps=3)
    obj["actions"] = [[0.4], [0.4, 0.2], [0.4]]  # row 1 has width 2
    p = tmp_path / "bad_act_width.jsonl"
    p.write_text(json.dumps(obj) + "\n")
    with pytest.raises(ValueError, match="inconsistent action rows"):
        load_eval_records(p)
