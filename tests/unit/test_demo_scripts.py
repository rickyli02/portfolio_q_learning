"""Unit tests for demo script entrypoints — Phase 16D.

Verifies both demo scripts (run_ctrl_demo.py and run_ctrl_oracle_demo.py)
and the smoke runner integration using importlib-based loading.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_CTRL_DEMO = _REPO_ROOT / "scripts" / "run_ctrl_demo.py"
_ORACLE_DEMO = _REPO_ROOT / "scripts" / "run_ctrl_oracle_demo.py"
_SMOKE_RUNNER = _REPO_ROOT / "scripts" / "run_smoke_test.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None, f"Cannot load {path}"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run_main_capture(path: Path, name: str) -> tuple[int, str]:
    """Load module, run main() capturing stdout, return (rc, output)."""
    mod = _load_module(path, name)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = mod.main()
    return rc, buf.getvalue()


# ===========================================================================
# run_ctrl_demo.py
# ===========================================================================


def test_ctrl_demo_returns_zero():
    rc, _ = _run_main_capture(_CTRL_DEMO, "run_ctrl_demo")
    assert rc == 0


def test_ctrl_demo_stdout_contains_summary_header():
    _, out = _run_main_capture(_CTRL_DEMO, "run_ctrl_demo")
    assert "--- CTRL demo summary ---" in out


def test_ctrl_demo_stdout_contains_w_final():
    _, out = _run_main_capture(_CTRL_DEMO, "run_ctrl_demo")
    assert "w_final" in out


def test_ctrl_demo_stdout_contains_last_terminal_wealth():
    _, out = _run_main_capture(_CTRL_DEMO, "run_ctrl_demo")
    assert "last_terminal_wealth" in out


def test_ctrl_demo_stdout_contains_history_len():
    _, out = _run_main_capture(_CTRL_DEMO, "run_ctrl_demo")
    assert "history_len" in out


# ===========================================================================
# run_ctrl_oracle_demo.py
# ===========================================================================


def test_oracle_demo_returns_zero():
    rc, _ = _run_main_capture(_ORACLE_DEMO, "run_ctrl_oracle_demo")
    assert rc == 0


def test_oracle_demo_stdout_contains_summary_header():
    _, out = _run_main_capture(_ORACLE_DEMO, "run_ctrl_oracle_demo")
    assert "--- CTRL-vs-oracle demo summary ---" in out


def test_oracle_demo_stdout_contains_post_training_w():
    _, out = _run_main_capture(_ORACLE_DEMO, "run_ctrl_oracle_demo")
    assert "post_training_w" in out


def test_oracle_demo_stdout_contains_ctrl_mean_tw():
    _, out = _run_main_capture(_ORACLE_DEMO, "run_ctrl_oracle_demo")
    assert "ctrl_mean_tw" in out


def test_oracle_demo_stdout_contains_oracle_mean_tw():
    _, out = _run_main_capture(_ORACLE_DEMO, "run_ctrl_oracle_demo")
    assert "oracle_mean_tw" in out


def test_oracle_demo_stdout_contains_mean_tw_delta():
    _, out = _run_main_capture(_ORACLE_DEMO, "run_ctrl_oracle_demo")
    assert "mean_tw_delta" in out


def test_oracle_demo_stdout_contains_ctrl_win_rate():
    _, out = _run_main_capture(_ORACLE_DEMO, "run_ctrl_oracle_demo")
    assert "ctrl_win_rate" in out


# ===========================================================================
# Smoke runner integration
# ===========================================================================


def test_smoke_runner_passes_all_nine():
    """Smoke runner exits 0 and reports 9/9 passed."""
    smoke = _load_module(_SMOKE_RUNNER, "run_smoke_test_integration")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = smoke.main(
            str(_REPO_ROOT / "configs" / "tests" / "smoke.yaml")
        )
    out = buf.getvalue()
    assert rc == 0, f"Smoke runner returned {rc}:\n{out}"
    assert "9/9 passed" in out, f"Expected '9/9 passed' in output:\n{out}"


def test_smoke_runner_output_contains_oracle_demo_check():
    """Smoke runner output mentions the oracle demo check."""
    smoke = _load_module(_SMOKE_RUNNER, "run_smoke_test_oracle_check")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        smoke.main(str(_REPO_ROOT / "configs" / "tests" / "smoke.yaml"))
    out = buf.getvalue()
    assert "run_ctrl_oracle_demo" in out, (
        f"Expected 'run_ctrl_oracle_demo' in smoke output:\n{out}"
    )
