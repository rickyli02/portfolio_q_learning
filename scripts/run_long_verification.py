#!/usr/bin/env python3
"""Long verification runner for manual terminal use.

Runs slower verification stages (import timing, CTRL unit tests, pytest
timing profile, full unit suite) and saves timestamped output to
outputs/verification/.

Intended use: run this in your own terminal when you want full verification
results without blocking the fast smoke-test loop.

Usage (run from repo root inside .venv):
    .venv/bin/python scripts/run_long_verification.py
    .venv/bin/python scripts/run_long_verification.py --stages imports ctrl
    .venv/bin/python scripts/run_long_verification.py --stages imports durations
    .venv/bin/python scripts/run_long_verification.py --stages full

Available stages (run in order):
    imports    - Subprocess import-timing probe for numpy and torch
    ctrl       - CTRL unit tests (tests/unit/test_ctrl.py -v)
    durations  - pytest timing profile (tests/unit --durations=20 -q)
    full       - Full unit suite (tests/unit -q)
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_OUTPUT_DIR = _REPO_ROOT / "outputs" / "verification"

# Stage definitions for pytest-backed stages: key -> (display label, pytest args)
_PYTEST_STAGES = {
    "ctrl": (
        "CTRL unit tests (tests/unit/test_ctrl.py)",
        ["tests/unit/test_ctrl.py", "-v"],
    ),
    "durations": (
        "pytest timing profile (tests/unit --durations=20)",
        ["tests/unit", "--durations=20", "-q"],
    ),
    "full": (
        "Full unit suite (tests/unit -q)",
        ["tests/unit", "-q"],
    ),
}

_DEFAULT_STAGES = ["imports", "ctrl", "durations", "full"]
_VALID_STAGES = ["imports"] + list(_PYTEST_STAGES.keys())

# Packages to time in the imports stage.  Each entry: (label, import_statement)
_IMPORT_PROBES = [
    ("numpy", "import numpy"),
    ("torch", "import torch"),
]

# Python snippet run in a fresh subprocess for each probe.
# Prints elapsed seconds as a plain float so the parent can parse it.
_PROBE_SCRIPT = """\
import time as _t
_t0 = _t.monotonic()
{stmt}
_t1 = _t.monotonic()
print(f"{{_t1 - _t0:.3f}}")
"""


def _run_import_timing() -> tuple[bool, float, str]:
    """Measure import times for each probe in a fresh subprocess.

    Each probe is a separate process so results are not polluted by already-
    imported modules.  Returns (passed, total_elapsed, report_text).
    """
    lines: list[str] = ["Import timing (each probe is a fresh subprocess):"]
    total_start = time.monotonic()
    all_ok = True

    for label, stmt in _IMPORT_PROBES:
        script = _PROBE_SCRIPT.format(stmt=stmt)
        start = time.monotonic()
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            cwd=str(_REPO_ROOT),
        )
        wall = time.monotonic() - start

        if result.returncode != 0 or not result.stdout.strip():
            lines.append(f"  {label:<12} ERROR  (wall={wall:.1f}s)")
            lines.append(f"    stderr: {result.stderr.strip()}")
            all_ok = False
        else:
            reported = float(result.stdout.strip())
            lines.append(f"  {label:<12} {reported:.3f}s  (wall={wall:.1f}s)")

    total = time.monotonic() - total_start
    return all_ok, total, "\n".join(lines)


def _run_pytest_stage(label: str, pytest_args: list[str]) -> tuple[bool, float, str]:
    """Run one pytest stage. Returns (passed, elapsed_seconds, output_text)."""
    cmd = [sys.executable, "-m", "pytest"] + pytest_args
    start = time.monotonic()
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
    )
    elapsed = time.monotonic() - start
    output = result.stdout + result.stderr
    passed = result.returncode == 0
    return passed, elapsed, output


def main(stages: list[str]) -> int:
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_file = _OUTPUT_DIR / f"{timestamp}_verification.txt"

    lines: list[str] = []

    def _emit(text: str = "") -> None:
        """Print to terminal and buffer for file."""
        print(text)
        lines.append(text)

    _emit("Portfolio Q-Learning — Long Verification Run")
    _emit(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    _emit(f"Stages : {', '.join(stages)}")
    _emit(f"Output : {output_file.relative_to(_REPO_ROOT)}")
    _emit("=" * 60)

    n = len(stages)
    results: list[tuple[str, bool, float]] = []

    for idx, stage_key in enumerate(stages, start=1):
        if stage_key == "imports":
            _emit(f"\n[{idx}/{n}] Import timing probes (subprocess-isolated)")
            _emit("-" * 60)
            passed, elapsed, output = _run_import_timing()
            for line in output.splitlines():
                _emit(line)
            status = "PASS" if passed else "FAIL"
            _emit(f"\n→ {status} in {elapsed:.1f}s")
            results.append((stage_key, passed, elapsed))
        else:
            label, pytest_args = _PYTEST_STAGES[stage_key]
            _emit(f"\n[{idx}/{n}] {label}")
            _emit("-" * 60)
            passed, elapsed, output = _run_pytest_stage(label, pytest_args)
            for line in output.splitlines():
                _emit(line)
            status = "PASS" if passed else "FAIL"
            _emit(f"\n→ {status} in {elapsed:.1f}s")
            results.append((stage_key, passed, elapsed))

    # Summary table
    _emit("\n" + "=" * 60)
    _emit("=== Verification Summary ===")
    for idx, (stage_key, passed, elapsed) in enumerate(results, start=1):
        status = "PASS" if passed else "FAIL"
        _emit(f"  [{idx}/{n}] {stage_key:<12} {status}  ({elapsed:.1f}s)")

    overall = all(passed for _, passed, _ in results)
    _emit(f"\nOverall: {'PASS' if overall else 'FAIL'}")
    _emit(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Write to file
    output_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _emit(f"\nOutput saved to: {output_file.relative_to(_REPO_ROOT)}")

    return 0 if overall else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run slow verification stages and save timestamped output.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available stages: {', '.join(_VALID_STAGES)}",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=_VALID_STAGES,
        default=_DEFAULT_STAGES,
        metavar="STAGE",
        help=f"Stages to run (default: all). Choices: {', '.join(_VALID_STAGES)}",
    )
    args = parser.parse_args()
    sys.exit(main(args.stages))
