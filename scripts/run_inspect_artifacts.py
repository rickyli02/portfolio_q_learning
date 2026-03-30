#!/usr/bin/env python3
"""Saved-artifact inspection consumer — Phase 21A.

Loads a persisted ``report.json`` and ``resolved_config.yaml`` produced by
``scripts/run_config_experiment.py`` and prints a compact stable summary
combining key report scalars with identifying config fields.

Usage (run inside the project .venv):
    .venv/bin/python scripts/run_inspect_artifacts.py <report_json> <resolved_config_yaml>

Example:
    .venv/bin/python scripts/run_inspect_artifacts.py \\
        outputs/run_01/report.json \\
        outputs/run_01/resolved_config.yaml
"""

import sys
from pathlib import Path

# Ensure repo root is on sys.path when run directly.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def main(argv: list[str] | None = None) -> int:
    """Load and print saved experiment artifacts.  Returns 0 on success."""
    if argv is None:
        argv = sys.argv[1:]

    if len(argv) != 2:
        print(
            "Usage: run_inspect_artifacts.py <report_json> <resolved_config_yaml>",
            file=sys.stderr,
        )
        return 2

    report_path = Path(argv[0])
    config_path = Path(argv[1])

    from src.backtest.experiment_io import load_experiment_report
    from src.config.schema import load_config

    try:
        report = load_experiment_report(report_path)
    except FileNotFoundError as exc:
        print(f"run_inspect_artifacts: error — {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"run_inspect_artifacts: error loading report — {exc}", file=sys.stderr)
        return 1

    import yaml as _yaml

    try:
        cfg = load_config(config_path)
    except FileNotFoundError as exc:
        print(f"run_inspect_artifacts: error — {exc}", file=sys.stderr)
        return 1
    except (ValueError, _yaml.YAMLError) as exc:
        print(f"run_inspect_artifacts: error loading config — {exc}", file=sys.stderr)
        return 1

    print()
    print("--- artifact inspection summary ---")
    print(f"  report              : {report_path}")
    print(f"  config              : {config_path}")
    print(f"  seed                : {cfg.seed}")
    print(f"  env_type            : {cfg.env.env_type}")
    print(f"  algo_type           : {cfg.algo.algo_type}")
    print(f"  post_training_w     : {report.post_training_w:.6f}")
    print(f"  target_return_z     : {report.target_return_z:.6f}")
    print(f"  ctrl_mean_tw        : {report.ctrl_mean_terminal_wealth:.6f}")
    print(f"  oracle_mean_tw      : {report.oracle_mean_terminal_wealth:.6f}")
    print(f"  mean_tw_delta       : {report.mean_terminal_wealth_delta:.6f}")
    print(f"  ctrl_win_rate       : {report.ctrl_win_rate:.6f}")
    print("--- end ---")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
