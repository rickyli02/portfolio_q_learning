#!/usr/bin/env python3
"""YAML-driven config-backed experiment runner — Phase 20B / 20C.

Loads a YAML experiment config, calls the approved ``run_ctrl_experiment(...)``
helper for the supported GBM + CTRL baseline path, and prints a compact stable
scalar summary.  Optionally saves two run artifacts to a specified directory:

- ``report.json``          compact scalar report (all fields from CTRLTrainCompareReport)
- ``resolved_config.yaml`` effective config values used for the run (post-validation)

Unsupported config combinations (env_type, algo_type, policy_type, or
evaluation flags) surface the existing ``ValueError`` from the runner without
any additional wrapping — errors are intentionally explicit rather than silenced.

Usage (run inside the project .venv):
    .venv/bin/python scripts/run_config_experiment.py <config_path> [output_dir]

Examples:
    .venv/bin/python scripts/run_config_experiment.py \\
        configs/experiments/ctrl_baseline_tiny.yaml

    .venv/bin/python scripts/run_config_experiment.py \\
        configs/experiments/ctrl_baseline_tiny.yaml outputs/run_01
"""

import sys
from pathlib import Path

# Ensure repo root is on sys.path when run directly.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def main(argv: list[str] | None = None) -> int:
    """Load config, run experiment, optionally save artifacts.  Returns 0 on success."""
    if argv is None:
        argv = sys.argv[1:]

    if len(argv) not in (1, 2):
        print(
            "Usage: run_config_experiment.py <config_path> [output_dir]",
            file=sys.stderr,
        )
        return 2

    config_path = Path(argv[0])
    output_dir = Path(argv[1]) if len(argv) == 2 else None

    from src.backtest.experiment_io import save_experiment_config, save_experiment_report
    from src.backtest.experiment_runner import run_ctrl_experiment
    from src.config.schema import load_config

    print(f"run_config_experiment: loading config from {config_path}")
    try:
        cfg = load_config(config_path)
    except FileNotFoundError as exc:
        print(f"run_config_experiment: error — {exc}", file=sys.stderr)
        return 1

    print(
        f"run_config_experiment: running CTRL baseline experiment "
        f"(env={cfg.env.env_type}, algo={cfg.algo.algo_type}, "
        f"seed={cfg.seed}, n_outer_iters={cfg.optim.n_epochs}, "
        f"n_updates={cfg.optim.n_steps_per_epoch}, "
        f"n_eval_episodes={cfg.eval.n_eval_episodes})"
    )
    result = run_ctrl_experiment(cfg)

    r = result.report
    print()
    print("--- config experiment summary ---")
    print(f"  config              : {config_path}")
    print(f"  post_training_w     : {r.post_training_w:.6f}")
    print(f"  target_return_z     : {r.target_return_z:.6f}")
    print(f"  last_terminal_wealth: {r.last_terminal_wealth:.6f}")
    print(f"  ctrl_mean_tw        : {r.ctrl_mean_terminal_wealth:.6f}")
    print(f"  oracle_mean_tw      : {r.oracle_mean_terminal_wealth:.6f}")
    print(f"  mean_tw_delta       : {r.mean_terminal_wealth_delta:.6f}")
    print(f"  ctrl_win_rate       : {r.ctrl_win_rate:.6f}")
    print("--- end ---")
    print()

    if output_dir is not None:
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            print(
                f"run_config_experiment: error creating output directory — {exc}",
                file=sys.stderr,
            )
            return 1
        report_path = output_dir / "report.json"
        config_path_out = output_dir / "resolved_config.yaml"
        save_experiment_report(r, report_path)
        save_experiment_config(cfg, config_path_out)
        print(f"run_config_experiment: artifacts saved to {output_dir}")
        print(f"  report.json          → {report_path}")
        print(f"  resolved_config.yaml → {config_path_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
