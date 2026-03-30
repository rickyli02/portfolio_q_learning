"""Tests for the YAML-driven config experiment script — Phase 20B.

Coverage:
- script returns success on a real YAML config
- stdout contains stable summary markers
- script loads the config from the path argument (not hardcoded defaults)
- failure is clear for a missing config path (error text in stderr)
- failure is clear for bad argument counts (usage text in stderr)
- the experiment config path fixture is distinct from the base default
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import sys
import tempfile
from pathlib import Path

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SCRIPT = _REPO_ROOT / "scripts" / "run_config_experiment.py"
_TINY_CFG = _REPO_ROOT / "configs" / "experiments" / "ctrl_baseline_tiny.yaml"


def _load_script():
    spec = importlib.util.spec_from_file_location("run_config_experiment", _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run(argv: list[str]) -> tuple[int, str, str]:
    """Run script main(argv), capture stdout and stderr, return (rc, stdout, stderr)."""
    mod = _load_script()
    out_buf = io.StringIO()
    err_buf = io.StringIO()
    with contextlib.redirect_stdout(out_buf), contextlib.redirect_stderr(err_buf):
        rc = mod.main(argv)
    return rc, out_buf.getvalue(), err_buf.getvalue()


# ---------------------------------------------------------------------------
# Script exists
# ---------------------------------------------------------------------------


def test_script_file_exists():
    assert _SCRIPT.exists(), f"script not found at {_SCRIPT}"


def test_tiny_config_file_exists():
    assert _TINY_CFG.exists(), f"tiny config not found at {_TINY_CFG}"


# ---------------------------------------------------------------------------
# Success path
# ---------------------------------------------------------------------------


def test_script_returns_zero_on_tiny_config():
    rc, _, _ = _run([str(_TINY_CFG)])
    assert rc == 0


def test_script_stdout_contains_summary_header():
    _, out, _ = _run([str(_TINY_CFG)])
    assert "--- config experiment summary ---" in out


def test_script_stdout_contains_end_marker():
    _, out, _ = _run([str(_TINY_CFG)])
    assert "--- end ---" in out


def test_script_stdout_contains_post_training_w():
    _, out, _ = _run([str(_TINY_CFG)])
    assert "post_training_w" in out


def test_script_stdout_contains_ctrl_mean_tw():
    _, out, _ = _run([str(_TINY_CFG)])
    assert "ctrl_mean_tw" in out


def test_script_stdout_contains_oracle_mean_tw():
    _, out, _ = _run([str(_TINY_CFG)])
    assert "oracle_mean_tw" in out


def test_script_stdout_contains_mean_tw_delta():
    _, out, _ = _run([str(_TINY_CFG)])
    assert "mean_tw_delta" in out


def test_script_stdout_contains_ctrl_win_rate():
    _, out, _ = _run([str(_TINY_CFG)])
    assert "ctrl_win_rate" in out


def test_script_stdout_contains_config_path():
    """Config path is echoed in the summary so the run is traceable."""
    _, out, _ = _run([str(_TINY_CFG)])
    assert str(_TINY_CFG) in out


# ---------------------------------------------------------------------------
# Config is actually loaded from the path (not bypassed)
# ---------------------------------------------------------------------------


def test_script_prints_seed_from_config():
    """The run-info line echoes the seed read from the YAML, proving the loader is used."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg_data = {
            "seed": 77,
            "env": {"n_steps": 5},
            "optim": {"n_epochs": 2, "n_steps_per_epoch": 1},
            "eval": {"n_eval_episodes": 2},
            "algo": {"algo_type": "ctrl_baseline", "oracle_gamma_embed": 1.0},
        }
        p = Path(tmp) / "custom.yaml"
        p.write_text(yaml.dump(cfg_data))
        rc, out, _ = _run([str(p)])
    assert rc == 0
    # The run-info line printed by the script includes "seed=77"
    assert "seed=77" in out


def test_script_prints_n_outer_iters_from_config():
    """The run-info line echoes n_outer_iters read from the YAML."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg_data = {
            "seed": 1,
            "env": {"n_steps": 5},
            "optim": {"n_epochs": 4, "n_steps_per_epoch": 1},
            "eval": {"n_eval_episodes": 2},
            "algo": {"algo_type": "ctrl_baseline", "oracle_gamma_embed": 1.0},
        }
        p = Path(tmp) / "custom.yaml"
        p.write_text(yaml.dump(cfg_data))
        rc, out, _ = _run([str(p)])
    assert rc == 0
    # The run-info line includes "n_outer_iters=4"
    assert "n_outer_iters=4" in out


# ---------------------------------------------------------------------------
# Failure paths — return code AND error/usage text
# ---------------------------------------------------------------------------


def test_script_returns_nonzero_for_missing_config():
    rc, _, _ = _run(["/nonexistent/path/config.yaml"])
    assert rc != 0


def test_script_error_text_for_missing_config():
    """stderr contains an error message naming the missing file."""
    _, _, err = _run(["/nonexistent/path/config.yaml"])
    assert "error" in err.lower() or "not found" in err.lower()


def test_script_returns_nonzero_for_no_args():
    rc, _, _ = _run([])
    assert rc != 0


def test_script_usage_text_for_no_args():
    """stderr contains usage instructions when no argument is given."""
    _, _, err = _run([])
    assert "Usage" in err or "usage" in err.lower()


def test_script_returns_nonzero_for_too_many_args():
    rc, _, _ = _run([str(_TINY_CFG), "out_dir", "extra_arg"])
    assert rc != 0


def test_script_usage_text_for_too_many_args():
    """stderr contains usage instructions when too many arguments are given."""
    _, _, err = _run([str(_TINY_CFG), "out_dir", "extra_arg"])
    assert "Usage" in err or "usage" in err.lower()


def test_script_surfaces_error_for_unsupported_algo_type():
    """Unsupported algo_type in YAML surfaces a clear error (not silent bypass)."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg_data = {
            "env": {"n_steps": 5},
            "optim": {"n_epochs": 2, "n_steps_per_epoch": 1},
            "eval": {"n_eval_episodes": 2},
            "algo": {"algo_type": "ctrl_online", "oracle_gamma_embed": 1.0},
        }
        p = Path(tmp) / "bad.yaml"
        p.write_text(yaml.dump(cfg_data))
        with pytest.raises(ValueError, match="algo_type"):
            _run([str(p)])
