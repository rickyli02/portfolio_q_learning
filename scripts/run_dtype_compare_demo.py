#!/usr/bin/env python3
"""Float32-vs-float64 dtype-sensitivity diagnostic demo — Phase 17D.

Calls the approved ``compare_dtype_outputs`` helper in two regimes and prints
a compact human-readable report.  Intended for manual diagnostic use only;
not wired into the smoke harness or trainer pipeline.

Usage (run inside the project .venv):
    .venv/bin/python scripts/run_dtype_compare_demo.py
"""

import sys
from pathlib import Path

# Ensure repo root is on sys.path when run directly.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ---------------------------------------------------------------------------
# Fixed diagnostic inputs
# ---------------------------------------------------------------------------
_MU = [0.08]
_SIGMA = [[0.20]]
_R = 0.05
_HORIZON = 1.0
_GAMMA = 1.5
_T = 0.3
_WEALTH = 1.2
_W = 1.0

# Extreme-regime: phi3=90 at t=0.0 so exponent = phi3 * (T - t) = 90 * 1.0 = 90.
# float32 overflows at exp(x) where x > ~88.7, so exp(90) → inf in float32.
_EXTREME_PHI3 = 90.0
_EXTREME_T = 0.0


def _print_result(label: str, result) -> None:
    print(f"--- dtype comparison: {label} ---")
    print(f"  actor_mean_action_delta : {result.actor_mean_action_delta:.6e}")
    print(f"  actor_variance_delta    : {result.actor_variance_delta:.6e}")
    print(f"  critic_forward_delta    : {result.critic_forward_delta:.6e}")
    print(f"  oracle_action_delta     : {result.oracle_action_delta:.6e}")
    print("--- end ---")


def main() -> int:
    """Run dtype comparison in normal and extreme regimes.  Returns 0 on success."""
    from src.utils.dtype_compare import compare_dtype_outputs

    print("dtype compare demo: normal finite regime")
    normal = compare_dtype_outputs(
        mu=_MU, sigma=_SIGMA, r=_R, horizon=_HORIZON, gamma_embed=_GAMMA,
        t=_T, wealth=_WEALTH, w=_W,
    )
    print()
    _print_result("normal regime", normal)
    print()

    print("dtype compare demo: extreme regime (phi3=90, t=0.0, exp(90) overflows float32 and float64)")
    extreme = compare_dtype_outputs(
        mu=_MU, sigma=_SIGMA, r=_R, horizon=_HORIZON, gamma_embed=_GAMMA,
        t=_EXTREME_T, wealth=_WEALTH, w=_W,
        init_phi3=_EXTREME_PHI3,
    )
    print()
    _print_result("extreme regime", extreme)
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
