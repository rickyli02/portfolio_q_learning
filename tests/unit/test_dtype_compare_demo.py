"""Unit tests for scripts/run_dtype_compare_demo.py — Phase 17D.

Verifies the dtype-comparison demo script entrypoint:
- returns success (0)
- stdout contains stable summary markers for both regimes
- both normal and extreme regimes are represented in output
- all four delta fields appear in the output
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DTYPE_DEMO = _REPO_ROOT / "scripts" / "run_dtype_compare_demo.py"


def _run_main_capture() -> tuple[int, str]:
    """Load the demo module, run main() capturing stdout, return (rc, output)."""
    spec = importlib.util.spec_from_file_location("run_dtype_compare_demo", _DTYPE_DEMO)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = mod.main()
    return rc, buf.getvalue()


# ===========================================================================
# Return code
# ===========================================================================


def test_dtype_compare_demo_returns_zero():
    rc, _ = _run_main_capture()
    assert rc == 0


# ===========================================================================
# Normal-regime markers
# ===========================================================================


def test_stdout_contains_normal_regime_header():
    _, out = _run_main_capture()
    assert "--- dtype comparison: normal regime ---" in out


def test_stdout_contains_actor_mean_action_delta():
    _, out = _run_main_capture()
    assert "actor_mean_action_delta" in out


def test_stdout_contains_actor_variance_delta():
    _, out = _run_main_capture()
    assert "actor_variance_delta" in out


def test_stdout_contains_critic_forward_delta():
    _, out = _run_main_capture()
    assert "critic_forward_delta" in out


def test_stdout_contains_oracle_action_delta():
    _, out = _run_main_capture()
    assert "oracle_action_delta" in out


# ===========================================================================
# Extreme-regime markers
# ===========================================================================


def test_stdout_contains_extreme_regime_header():
    _, out = _run_main_capture()
    assert "--- dtype comparison: extreme regime ---" in out


def test_both_regimes_have_end_markers():
    """Both regime blocks are closed with '--- end ---'."""
    _, out = _run_main_capture()
    assert out.count("--- end ---") >= 2


# ===========================================================================
# Both regimes are represented (each has its own delta lines)
# ===========================================================================


def test_actor_variance_delta_appears_in_both_regime_blocks():
    """actor_variance_delta label appears once per regime block (at least 2 times)."""
    _, out = _run_main_capture()
    assert out.count("actor_variance_delta") >= 2


def test_oracle_action_delta_appears_in_both_regime_blocks():
    _, out = _run_main_capture()
    assert out.count("oracle_action_delta") >= 2


# ===========================================================================
# Genuine distinction: extreme regime produces a larger actor_variance_delta
# ===========================================================================


def test_extreme_regime_actor_variance_delta_is_non_finite():
    """Prove the extreme block is a real boundary case, not just a re-labelled copy.

    phi3=90 at t=0.0 makes the variance exponent = phi3*(T-t) = 90*1.0 = 90.
    float32 overflows at exp(x) > exp(88.7), so exp(90) → inf in float32.
    float64 also overflows at t=0 with this phi3, so inf-inf → nan.
    Either way the delta is non-finite, which is clearly distinct from the
    small finite delta in the normal regime.
    """
    import math
    from src.utils.dtype_compare import compare_dtype_outputs

    normal = compare_dtype_outputs(
        mu=[0.08], sigma=[[0.20]], r=0.05, horizon=1.0, gamma_embed=1.5,
        t=0.3, wealth=1.2, w=1.0,
    )
    extreme = compare_dtype_outputs(
        mu=[0.08], sigma=[[0.20]], r=0.05, horizon=1.0, gamma_embed=1.5,
        t=0.0, wealth=1.2, w=1.0,
        init_phi3=90.0,
    )
    # normal regime: delta is small and finite
    assert math.isfinite(normal.actor_variance_delta)
    # extreme regime: delta is non-finite (nan or inf), proving genuine overflow
    assert not math.isfinite(extreme.actor_variance_delta)
