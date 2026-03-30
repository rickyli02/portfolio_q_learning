"""Tests for the saved-artifact inspection script — Phase 21A.

Coverage:
- script returns success on real saved artifacts produced by the Phase 20C path
- stdout contains stable summary markers
- stdout contains report-derived scalar values
- stdout contains config-derived identifying fields (seed, env_type, algo_type)
- missing report path fails clearly (nonzero rc + stderr error text)
- missing config path fails clearly (nonzero rc + stderr error text)
- malformed report.json fails clearly
- wrong argument count fails with usage text in stderr
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

from src.backtest.experiment_io import save_experiment_config, save_experiment_report
from src.backtest.train_compare_report import CTRLTrainCompareReport
from src.config.schema import AlgorithmConfig, EnvConfig, EvalConfig, ExperimentConfig, OptimConfig, RewardConfig

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SCRIPT = _REPO_ROOT / "scripts" / "run_inspect_artifacts.py"
_RUN_SCRIPT = _REPO_ROOT / "scripts" / "run_config_experiment.py"
_TINY_CFG = _REPO_ROOT / "configs" / "experiments" / "ctrl_baseline_tiny.yaml"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_script(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run_inspect(argv: list[str]) -> tuple[int, str, str]:
    mod = _load_script(_SCRIPT, "run_inspect_artifacts")
    out_buf = io.StringIO()
    err_buf = io.StringIO()
    with contextlib.redirect_stdout(out_buf), contextlib.redirect_stderr(err_buf):
        rc = mod.main(argv)
    return rc, out_buf.getvalue(), err_buf.getvalue()


def _run_experiment(argv: list[str]) -> tuple[int, str, str]:
    mod = _load_script(_RUN_SCRIPT, "run_config_experiment")
    out_buf = io.StringIO()
    err_buf = io.StringIO()
    with contextlib.redirect_stdout(out_buf), contextlib.redirect_stderr(err_buf):
        rc = mod.main(argv)
    return rc, out_buf.getvalue(), err_buf.getvalue()


def _make_artifacts(tmp: str) -> tuple[Path, Path]:
    """Run the experiment script to produce real Phase 20C artifacts."""
    out_dir = Path(tmp) / "run_out"
    rc, _, _ = _run_experiment([str(_TINY_CFG), str(out_dir)])
    assert rc == 0, "Artifact production run failed"
    return out_dir / "report.json", out_dir / "resolved_config.yaml"


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
    cfg.seed = 55
    cfg.env = EnvConfig(env_type="gbm", horizon=1.0, n_steps=5, mu=[0.08], sigma=[[0.20]])
    cfg.env.assets.n_risky = 1
    cfg.reward = RewardConfig(target_return=1.0, entropy_temp=0.1)
    cfg.optim = OptimConfig(n_epochs=2, n_steps_per_epoch=1)
    cfg.eval = EvalConfig(n_eval_episodes=2)
    cfg.algo = AlgorithmConfig(algo_type="ctrl_baseline", oracle_gamma_embed=1.0)
    return cfg


# ---------------------------------------------------------------------------
# Script exists
# ---------------------------------------------------------------------------


def test_inspect_script_exists():
    assert _SCRIPT.exists(), f"script not found at {_SCRIPT}"


# ---------------------------------------------------------------------------
# Success path on real Phase 20C artifacts
# ---------------------------------------------------------------------------


def test_inspect_returns_zero_on_real_artifacts():
    with tempfile.TemporaryDirectory() as tmp:
        report_p, cfg_p = _make_artifacts(tmp)
        rc, _, _ = _run_inspect([str(report_p), str(cfg_p)])
    assert rc == 0


def test_inspect_stdout_contains_summary_header():
    with tempfile.TemporaryDirectory() as tmp:
        report_p, cfg_p = _make_artifacts(tmp)
        _, out, _ = _run_inspect([str(report_p), str(cfg_p)])
    assert "--- artifact inspection summary ---" in out


def test_inspect_stdout_contains_end_marker():
    with tempfile.TemporaryDirectory() as tmp:
        report_p, cfg_p = _make_artifacts(tmp)
        _, out, _ = _run_inspect([str(report_p), str(cfg_p)])
    assert "--- end ---" in out


def test_inspect_stdout_contains_report_scalars():
    with tempfile.TemporaryDirectory() as tmp:
        report_p, cfg_p = _make_artifacts(tmp)
        _, out, _ = _run_inspect([str(report_p), str(cfg_p)])
    assert "post_training_w" in out
    assert "ctrl_mean_tw" in out
    assert "oracle_mean_tw" in out
    assert "mean_tw_delta" in out
    assert "ctrl_win_rate" in out


def test_inspect_stdout_contains_config_fields():
    """Output includes identifying config fields from the resolved_config.yaml."""
    with tempfile.TemporaryDirectory() as tmp:
        report_p, cfg_p = _make_artifacts(tmp)
        _, out, _ = _run_inspect([str(report_p), str(cfg_p)])
    assert "seed" in out
    assert "env_type" in out
    assert "algo_type" in out


def test_inspect_stdout_reflects_config_seed():
    """The printed seed value matches the value in the tiny config (42)."""
    with tempfile.TemporaryDirectory() as tmp:
        report_p, cfg_p = _make_artifacts(tmp)
        _, out, _ = _run_inspect([str(report_p), str(cfg_p)])
    assert "42" in out  # tiny config seed


def test_inspect_stdout_reflects_algo_type():
    """The printed algo_type matches the value in the tiny config."""
    with tempfile.TemporaryDirectory() as tmp:
        report_p, cfg_p = _make_artifacts(tmp)
        _, out, _ = _run_inspect([str(report_p), str(cfg_p)])
    assert "ctrl_baseline" in out


def test_inspect_echoes_artifact_paths():
    """Both artifact paths are echoed in the output."""
    with tempfile.TemporaryDirectory() as tmp:
        report_p, cfg_p = _make_artifacts(tmp)
        _, out, _ = _run_inspect([str(report_p), str(cfg_p)])
    assert str(report_p) in out
    assert str(cfg_p) in out


# ---------------------------------------------------------------------------
# Failure paths
# ---------------------------------------------------------------------------


def test_inspect_returns_nonzero_for_missing_report():
    with tempfile.TemporaryDirectory() as tmp:
        cfg_p = Path(tmp) / "resolved_config.yaml"
        save_experiment_config(_tiny_cfg(), cfg_p)
        rc, _, _ = _run_inspect(["/nonexistent/report.json", str(cfg_p)])
    assert rc != 0


def test_inspect_stderr_describes_missing_report():
    with tempfile.TemporaryDirectory() as tmp:
        cfg_p = Path(tmp) / "resolved_config.yaml"
        save_experiment_config(_tiny_cfg(), cfg_p)
        _, _, err = _run_inspect(["/nonexistent/report.json", str(cfg_p)])
    assert "error" in err.lower()


def test_inspect_returns_nonzero_for_missing_config():
    with tempfile.TemporaryDirectory() as tmp:
        report_p = Path(tmp) / "report.json"
        save_experiment_report(_tiny_report(), report_p)
        rc, _, _ = _run_inspect([str(report_p), "/nonexistent/config.yaml"])
    assert rc != 0


def test_inspect_stderr_describes_missing_config():
    with tempfile.TemporaryDirectory() as tmp:
        report_p = Path(tmp) / "report.json"
        save_experiment_report(_tiny_report(), report_p)
        _, _, err = _run_inspect([str(report_p), "/nonexistent/config.yaml"])
    assert "error" in err.lower()


def test_inspect_returns_nonzero_for_malformed_report():
    with tempfile.TemporaryDirectory() as tmp:
        report_p = Path(tmp) / "report.json"
        cfg_p = Path(tmp) / "resolved_config.yaml"
        report_p.write_text("not valid json {{{")
        save_experiment_config(_tiny_cfg(), cfg_p)
        rc, _, _ = _run_inspect([str(report_p), str(cfg_p)])
    assert rc != 0


def test_inspect_stderr_describes_malformed_report():
    with tempfile.TemporaryDirectory() as tmp:
        report_p = Path(tmp) / "report.json"
        cfg_p = Path(tmp) / "resolved_config.yaml"
        report_p.write_text("not valid json {{{")
        save_experiment_config(_tiny_cfg(), cfg_p)
        _, _, err = _run_inspect([str(report_p), str(cfg_p)])
    assert "error" in err.lower()


def test_inspect_returns_nonzero_for_no_args():
    rc, _, _ = _run_inspect([])
    assert rc != 0


def test_inspect_usage_text_for_no_args():
    _, _, err = _run_inspect([])
    assert "Usage" in err or "usage" in err.lower()


def test_inspect_returns_nonzero_for_one_arg():
    rc, _, _ = _run_inspect(["/some/report.json"])
    assert rc != 0


def test_inspect_returns_nonzero_for_too_many_args():
    rc, _, _ = _run_inspect(["/a", "/b", "/c"])
    assert rc != 0


def test_inspect_returns_nonzero_for_malformed_config():
    """Malformed resolved_config.yaml produces nonzero rc (not an uncaught parser error)."""
    with tempfile.TemporaryDirectory() as tmp:
        report_p = Path(tmp) / "report.json"
        cfg_p = Path(tmp) / "resolved_config.yaml"
        save_experiment_report(_tiny_report(), report_p)
        cfg_p.write_text("seed: [1")  # invalid YAML
        rc, _, _ = _run_inspect([str(report_p), str(cfg_p)])
    assert rc != 0


def test_inspect_stderr_describes_malformed_config():
    """stderr contains a clear error message for malformed config YAML."""
    with tempfile.TemporaryDirectory() as tmp:
        report_p = Path(tmp) / "report.json"
        cfg_p = Path(tmp) / "resolved_config.yaml"
        save_experiment_report(_tiny_report(), report_p)
        cfg_p.write_text("seed: [1")  # invalid YAML
        _, _, err = _run_inspect([str(report_p), str(cfg_p)])
    assert "error" in err.lower()
