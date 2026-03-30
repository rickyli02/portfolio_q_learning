"""Tests for run-artifact persistence helpers — Phase 20C.

Coverage:
- save_experiment_report writes a file with expected scalar fields
- load_experiment_report round-trips back to the original report
- save_experiment_config writes a file with effective config values
- script success path still works without output_dir (existing behavior)
- script writes report.json and resolved_config.yaml when output_dir is given
- saved report.json content reflects config-derived scalars, not placeholder values
- saved resolved_config.yaml content reflects effective config fields
- public exports available from src.backtest
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import tempfile
from pathlib import Path

import pytest
import yaml

from src.backtest.experiment_io import (
    load_experiment_report,
    save_experiment_config,
    save_experiment_report,
)
from src.backtest.train_compare_report import CTRLTrainCompareReport
from src.config.schema import AlgorithmConfig, EnvConfig, EvalConfig, ExperimentConfig, OptimConfig, RewardConfig

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SCRIPT = _REPO_ROOT / "scripts" / "run_config_experiment.py"
_TINY_CFG = _REPO_ROOT / "configs" / "experiments" / "ctrl_baseline_tiny.yaml"


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _tiny_report() -> CTRLTrainCompareReport:
    return CTRLTrainCompareReport(
        post_training_w=0.99,
        target_return_z=1.0,
        last_n_updates=4,
        last_terminal_wealth=1.05,
        n_eval_seeds=3,
        ctrl_mean_terminal_wealth=1.03,
        oracle_mean_terminal_wealth=1.04,
        mean_terminal_wealth_delta=-0.01,
        ctrl_win_rate=0.333,
    )


def _tiny_cfg() -> ExperimentConfig:
    cfg = ExperimentConfig()
    cfg.env = EnvConfig(env_type="gbm", horizon=1.0, n_steps=5, mu=[0.08], sigma=[[0.20]])
    cfg.env.assets.n_risky = 1
    cfg.reward = RewardConfig(target_return=1.0, entropy_temp=0.1)
    cfg.optim = OptimConfig(n_epochs=2, n_steps_per_epoch=1)
    cfg.eval = EvalConfig(n_eval_episodes=2)
    cfg.algo = AlgorithmConfig(algo_type="ctrl_baseline", oracle_gamma_embed=1.0)
    return cfg


# ---------------------------------------------------------------------------
# save_experiment_report / load_experiment_report
# ---------------------------------------------------------------------------


def test_save_experiment_report_creates_file():
    report = _tiny_report()
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "report.json"
        save_experiment_report(report, p)
        assert p.exists()


def test_save_experiment_report_file_is_valid_json():
    report = _tiny_report()
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "report.json"
        save_experiment_report(report, p)
        data = json.loads(p.read_text())
    assert isinstance(data, dict)


def test_save_experiment_report_contains_all_fields():
    report = _tiny_report()
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "report.json"
        save_experiment_report(report, p)
        data = json.loads(p.read_text())
    expected_fields = {
        "post_training_w", "target_return_z", "last_n_updates",
        "last_terminal_wealth", "n_eval_seeds", "ctrl_mean_terminal_wealth",
        "oracle_mean_terminal_wealth", "mean_terminal_wealth_delta", "ctrl_win_rate",
    }
    assert expected_fields <= data.keys()


def test_save_experiment_report_values_match_report():
    report = _tiny_report()
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "report.json"
        save_experiment_report(report, p)
        data = json.loads(p.read_text())
    assert data["post_training_w"] == pytest.approx(0.99)
    assert data["n_eval_seeds"] == 3
    assert data["ctrl_win_rate"] == pytest.approx(0.333)


def test_load_experiment_report_round_trips():
    report = _tiny_report()
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "report.json"
        save_experiment_report(report, p)
        loaded = load_experiment_report(p)
    assert loaded.post_training_w == pytest.approx(report.post_training_w)
    assert loaded.n_eval_seeds == report.n_eval_seeds
    assert loaded.ctrl_win_rate == pytest.approx(report.ctrl_win_rate)
    assert loaded.last_n_updates == report.last_n_updates


def test_load_experiment_report_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_experiment_report(Path("/nonexistent/report.json"))


# ---------------------------------------------------------------------------
# save_experiment_config
# ---------------------------------------------------------------------------


def test_save_experiment_config_creates_file():
    cfg = _tiny_cfg()
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "resolved_config.yaml"
        save_experiment_config(cfg, p)
        assert p.exists()


def test_save_experiment_config_file_is_valid_yaml():
    cfg = _tiny_cfg()
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "resolved_config.yaml"
        save_experiment_config(cfg, p)
        data = yaml.safe_load(p.read_text())
    assert isinstance(data, dict)


def test_save_experiment_config_contains_seed():
    cfg = _tiny_cfg()
    cfg.seed = 77
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "resolved_config.yaml"
        save_experiment_config(cfg, p)
        data = yaml.safe_load(p.read_text())
    assert data["seed"] == 77


def test_save_experiment_config_contains_algo_type():
    cfg = _tiny_cfg()
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "resolved_config.yaml"
        save_experiment_config(cfg, p)
        data = yaml.safe_load(p.read_text())
    assert data["algo"]["algo_type"] == "ctrl_baseline"


def test_save_experiment_config_contains_oracle_gamma_embed():
    cfg = _tiny_cfg()
    cfg.algo.oracle_gamma_embed = 1.5
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "resolved_config.yaml"
        save_experiment_config(cfg, p)
        data = yaml.safe_load(p.read_text())
    assert data["algo"]["oracle_gamma_embed"] == pytest.approx(1.5)


def test_save_experiment_config_contains_env_params():
    cfg = _tiny_cfg()
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "resolved_config.yaml"
        save_experiment_config(cfg, p)
        data = yaml.safe_load(p.read_text())
    assert data["env"]["env_type"] == "gbm"
    assert data["env"]["mu"] == [0.08]


# ---------------------------------------------------------------------------
# Script artifact path
# ---------------------------------------------------------------------------


def _load_script():
    spec = importlib.util.spec_from_file_location("run_config_experiment", _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run(argv: list[str]) -> tuple[int, str, str]:
    mod = _load_script()
    out_buf = io.StringIO()
    err_buf = io.StringIO()
    with contextlib.redirect_stdout(out_buf), contextlib.redirect_stderr(err_buf):
        rc = mod.main(argv)
    return rc, out_buf.getvalue(), err_buf.getvalue()


def test_script_without_output_dir_still_returns_zero():
    rc, _, _ = _run([str(_TINY_CFG)])
    assert rc == 0


def test_script_with_output_dir_returns_zero():
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp) / "run_out"
        rc, _, _ = _run([str(_TINY_CFG), str(out_dir)])
    assert rc == 0


def test_script_with_output_dir_creates_report_json():
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp) / "run_out"
        _run([str(_TINY_CFG), str(out_dir)])
        assert (out_dir / "report.json").exists()


def test_script_with_output_dir_creates_resolved_config_yaml():
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp) / "run_out"
        _run([str(_TINY_CFG), str(out_dir)])
        assert (out_dir / "resolved_config.yaml").exists()


def test_script_report_json_contains_expected_fields():
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp) / "run_out"
        _run([str(_TINY_CFG), str(out_dir)])
        data = json.loads((out_dir / "report.json").read_text())
    assert "post_training_w" in data
    assert "ctrl_mean_terminal_wealth" in data
    assert "ctrl_win_rate" in data


def test_script_resolved_config_yaml_reflects_config_values():
    """resolved_config.yaml contains the effective seed and algo_type from the YAML."""
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp) / "run_out"
        _run([str(_TINY_CFG), str(out_dir)])
        data = yaml.safe_load((out_dir / "resolved_config.yaml").read_text())
    # tiny config has seed=42 and algo_type=ctrl_baseline
    assert data["seed"] == 42
    assert data["algo"]["algo_type"] == "ctrl_baseline"


def test_script_stdout_mentions_artifacts_when_output_dir_given():
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp) / "run_out"
        _, out, _ = _run([str(_TINY_CFG), str(out_dir)])
    assert "report.json" in out
    assert "resolved_config.yaml" in out


# ---------------------------------------------------------------------------
# Invalid output destination — failure path
# ---------------------------------------------------------------------------


def test_script_returns_nonzero_for_invalid_output_dir():
    """Script returns nonzero when output_dir cannot be created (parent is a file)."""
    with tempfile.TemporaryDirectory() as tmp:
        # Create a regular file at the path that output_dir would use as a parent.
        # mkdir(..., parents=True) will raise OSError when a path component is a file.
        blocker = Path(tmp) / "blocker"
        blocker.write_text("not a directory")
        invalid_out = blocker / "run_out"  # parent is a file, not a dir
        rc, _, _ = _run([str(_TINY_CFG), str(invalid_out)])
    assert rc != 0


def test_script_stderr_describes_invalid_output_dir_error():
    """stderr contains a clear error message when output_dir creation fails."""
    with tempfile.TemporaryDirectory() as tmp:
        blocker = Path(tmp) / "blocker"
        blocker.write_text("not a directory")
        invalid_out = blocker / "run_out"
        _, _, err = _run([str(_TINY_CFG), str(invalid_out)])
    assert "error" in err.lower()


# ---------------------------------------------------------------------------
# Public export tests
# ---------------------------------------------------------------------------


def test_persistence_helpers_exported_from_backtest():
    from src.backtest import (
        load_experiment_report as _le,
        save_experiment_config as _sc,
        save_experiment_report as _sr,
    )
    assert _le is load_experiment_report
    assert _sc is save_experiment_config
    assert _sr is save_experiment_report
