"""Tests for the saved-artifact plotting script — Phase 21B.

Coverage:
- script returns success on real saved artifacts produced by the Phase 20C path
- output image file is created and non-empty
- stdout echoes artifact paths and config-derived identifiers (seed, env_type, algo_type)
- stdout reflects report-derived scalar values (ctrl_mean_tw, oracle_mean_tw)
- missing report path fails clearly (nonzero rc + stderr error text)
- missing config path fails clearly (nonzero rc + stderr error text)
- malformed report.json fails clearly
- malformed resolved_config.yaml fails clearly
- wrong argument count fails with usage text in stderr
- invalid output destination fails clearly
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import tempfile
from pathlib import Path

import pytest

from src.backtest.experiment_io import save_experiment_config, save_experiment_report
from src.backtest.train_compare_report import CTRLTrainCompareReport
from src.config.schema import AlgorithmConfig, EnvConfig, EvalConfig, ExperimentConfig, OptimConfig, RewardConfig

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SCRIPT = _REPO_ROOT / "scripts" / "run_plot_artifacts.py"
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


def _run_plot(argv: list[str]) -> tuple[int, str, str]:
    mod = _load_script(_SCRIPT, "run_plot_artifacts")
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


def test_plot_script_exists():
    assert _SCRIPT.exists(), f"script not found at {_SCRIPT}"


# ---------------------------------------------------------------------------
# Success path on real Phase 20C artifacts
# ---------------------------------------------------------------------------


def test_plot_returns_zero_on_real_artifacts():
    with tempfile.TemporaryDirectory() as tmp:
        report_p, cfg_p = _make_artifacts(tmp)
        out_img = Path(tmp) / "comparison.png"
        rc, _, _ = _run_plot([str(report_p), str(cfg_p), str(out_img)])
    assert rc == 0


def test_plot_creates_output_image():
    with tempfile.TemporaryDirectory() as tmp:
        report_p, cfg_p = _make_artifacts(tmp)
        out_img = Path(tmp) / "comparison.png"
        _run_plot([str(report_p), str(cfg_p), str(out_img)])
        assert out_img.exists()


def test_plot_output_image_is_nonempty():
    with tempfile.TemporaryDirectory() as tmp:
        report_p, cfg_p = _make_artifacts(tmp)
        out_img = Path(tmp) / "comparison.png"
        _run_plot([str(report_p), str(cfg_p), str(out_img)])
        assert out_img.stat().st_size > 0


def test_plot_stdout_echoes_output_path():
    with tempfile.TemporaryDirectory() as tmp:
        report_p, cfg_p = _make_artifacts(tmp)
        out_img = Path(tmp) / "comparison.png"
        _, out, _ = _run_plot([str(report_p), str(cfg_p), str(out_img)])
    assert str(out_img) in out


def test_plot_stdout_contains_seed():
    """stdout includes the seed read from the config (42 from tiny config)."""
    with tempfile.TemporaryDirectory() as tmp:
        report_p, cfg_p = _make_artifacts(tmp)
        out_img = Path(tmp) / "comparison.png"
        _, out, _ = _run_plot([str(report_p), str(cfg_p), str(out_img)])
    assert "seed=42" in out


def test_plot_stdout_contains_env_type():
    with tempfile.TemporaryDirectory() as tmp:
        report_p, cfg_p = _make_artifacts(tmp)
        out_img = Path(tmp) / "comparison.png"
        _, out, _ = _run_plot([str(report_p), str(cfg_p), str(out_img)])
    assert "env_type=gbm" in out


def test_plot_stdout_contains_algo_type():
    with tempfile.TemporaryDirectory() as tmp:
        report_p, cfg_p = _make_artifacts(tmp)
        out_img = Path(tmp) / "comparison.png"
        _, out, _ = _run_plot([str(report_p), str(cfg_p), str(out_img)])
    assert "algo_type=ctrl_baseline" in out


def test_plot_stdout_contains_ctrl_mean_tw():
    with tempfile.TemporaryDirectory() as tmp:
        report_p, cfg_p = _make_artifacts(tmp)
        out_img = Path(tmp) / "comparison.png"
        _, out, _ = _run_plot([str(report_p), str(cfg_p), str(out_img)])
    assert "ctrl_mean_tw" in out


def test_plot_stdout_contains_oracle_mean_tw():
    with tempfile.TemporaryDirectory() as tmp:
        report_p, cfg_p = _make_artifacts(tmp)
        out_img = Path(tmp) / "comparison.png"
        _, out, _ = _run_plot([str(report_p), str(cfg_p), str(out_img)])
    assert "oracle_mean_tw" in out


def test_plot_success_stderr_is_empty():
    """Successful runs should produce no Python-level warning noise on stderr."""
    with tempfile.TemporaryDirectory() as tmp:
        report_p, cfg_p = _make_artifacts(tmp)
        out_img = Path(tmp) / "comparison.png"
        _, _, err = _run_plot([str(report_p), str(cfg_p), str(out_img)])
    assert err == ""


# ---------------------------------------------------------------------------
# Failure paths — missing artifacts
# ---------------------------------------------------------------------------


def test_plot_returns_nonzero_for_missing_report():
    with tempfile.TemporaryDirectory() as tmp:
        cfg_p = Path(tmp) / "resolved_config.yaml"
        save_experiment_config(_tiny_cfg(), cfg_p)
        out_img = Path(tmp) / "out.png"
        rc, _, _ = _run_plot(["/nonexistent/report.json", str(cfg_p), str(out_img)])
    assert rc != 0


def test_plot_stderr_describes_missing_report():
    with tempfile.TemporaryDirectory() as tmp:
        cfg_p = Path(tmp) / "resolved_config.yaml"
        save_experiment_config(_tiny_cfg(), cfg_p)
        out_img = Path(tmp) / "out.png"
        _, _, err = _run_plot(["/nonexistent/report.json", str(cfg_p), str(out_img)])
    assert "error" in err.lower()


def test_plot_returns_nonzero_for_missing_config():
    with tempfile.TemporaryDirectory() as tmp:
        report_p = Path(tmp) / "report.json"
        save_experiment_report(_tiny_report(), report_p)
        out_img = Path(tmp) / "out.png"
        rc, _, _ = _run_plot([str(report_p), "/nonexistent/config.yaml", str(out_img)])
    assert rc != 0


def test_plot_stderr_describes_missing_config():
    with tempfile.TemporaryDirectory() as tmp:
        report_p = Path(tmp) / "report.json"
        save_experiment_report(_tiny_report(), report_p)
        out_img = Path(tmp) / "out.png"
        _, _, err = _run_plot([str(report_p), "/nonexistent/config.yaml", str(out_img)])
    assert "error" in err.lower()


# ---------------------------------------------------------------------------
# Failure paths — malformed artifacts
# ---------------------------------------------------------------------------


def test_plot_returns_nonzero_for_malformed_report():
    with tempfile.TemporaryDirectory() as tmp:
        report_p = Path(tmp) / "report.json"
        cfg_p = Path(tmp) / "resolved_config.yaml"
        report_p.write_text("not valid json {{{")
        save_experiment_config(_tiny_cfg(), cfg_p)
        out_img = Path(tmp) / "out.png"
        rc, _, _ = _run_plot([str(report_p), str(cfg_p), str(out_img)])
    assert rc != 0


def test_plot_stderr_describes_malformed_report():
    with tempfile.TemporaryDirectory() as tmp:
        report_p = Path(tmp) / "report.json"
        cfg_p = Path(tmp) / "resolved_config.yaml"
        report_p.write_text("not valid json {{{")
        save_experiment_config(_tiny_cfg(), cfg_p)
        out_img = Path(tmp) / "out.png"
        _, _, err = _run_plot([str(report_p), str(cfg_p), str(out_img)])
    assert "error" in err.lower()


def test_plot_returns_nonzero_for_malformed_config():
    with tempfile.TemporaryDirectory() as tmp:
        report_p = Path(tmp) / "report.json"
        cfg_p = Path(tmp) / "resolved_config.yaml"
        save_experiment_report(_tiny_report(), report_p)
        cfg_p.write_text("seed: [1")  # invalid YAML
        out_img = Path(tmp) / "out.png"
        rc, _, _ = _run_plot([str(report_p), str(cfg_p), str(out_img)])
    assert rc != 0


def test_plot_stderr_describes_malformed_config():
    with tempfile.TemporaryDirectory() as tmp:
        report_p = Path(tmp) / "report.json"
        cfg_p = Path(tmp) / "resolved_config.yaml"
        save_experiment_report(_tiny_report(), report_p)
        cfg_p.write_text("seed: [1")  # invalid YAML
        out_img = Path(tmp) / "out.png"
        _, _, err = _run_plot([str(report_p), str(cfg_p), str(out_img)])
    assert "error" in err.lower()


# ---------------------------------------------------------------------------
# Failure paths — invalid output destination
# ---------------------------------------------------------------------------


def test_plot_returns_nonzero_for_invalid_output_destination():
    """Script returns nonzero when output image path cannot be written."""
    with tempfile.TemporaryDirectory() as tmp:
        report_p, cfg_p = _make_artifacts(tmp)
        # Create a file where the parent directory would need to be
        blocker = Path(tmp) / "blocker"
        blocker.write_text("not a directory")
        invalid_out = blocker / "comparison.png"
        rc, _, _ = _run_plot([str(report_p), str(cfg_p), str(invalid_out)])
    assert rc != 0


def test_plot_stderr_describes_invalid_output_destination():
    with tempfile.TemporaryDirectory() as tmp:
        report_p, cfg_p = _make_artifacts(tmp)
        blocker = Path(tmp) / "blocker"
        blocker.write_text("not a directory")
        invalid_out = blocker / "comparison.png"
        _, _, err = _run_plot([str(report_p), str(cfg_p), str(invalid_out)])
    assert "error" in err.lower()


# ---------------------------------------------------------------------------
# Failure paths — wrong argument count
# ---------------------------------------------------------------------------


def test_plot_returns_nonzero_for_no_args():
    rc, _, _ = _run_plot([])
    assert rc != 0


def test_plot_usage_text_for_no_args():
    _, _, err = _run_plot([])
    assert "Usage" in err or "usage" in err.lower()


def test_plot_returns_nonzero_for_one_arg():
    rc, _, _ = _run_plot(["/some/report.json"])
    assert rc != 0


def test_plot_returns_nonzero_for_two_args():
    rc, _, _ = _run_plot(["/some/report.json", "/some/config.yaml"])
    assert rc != 0


def test_plot_returns_nonzero_for_too_many_args():
    rc, _, _ = _run_plot(["/a", "/b", "/c", "/d"])
    assert rc != 0
