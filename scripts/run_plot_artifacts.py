#!/usr/bin/env python3
"""Saved-artifact plotting consumer — Phase 21B.

Loads a persisted ``report.json`` and ``resolved_config.yaml`` produced by
``scripts/run_config_experiment.py`` and generates a simple bar chart comparing
CTRL vs oracle mean terminal wealth.  The chart title is annotated with
identifying fields from the config (seed, env_type, algo_type).

Usage (run inside the project .venv):
    .venv/bin/python scripts/run_plot_artifacts.py <report_json> <resolved_config_yaml> <output_image>

Example:
    .venv/bin/python scripts/run_plot_artifacts.py \\
        outputs/run_01/report.json \\
        outputs/run_01/resolved_config.yaml \\
        outputs/run_01/comparison.png
"""

import os
import sys
import tempfile
from pathlib import Path

# Ensure repo root is on sys.path when run directly.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Steer matplotlib to a writable config/cache directory before any matplotlib
# import so that systems where ~/.matplotlib is not writable do not emit
# cache-directory warnings on the success path.
if "MPLCONFIGDIR" not in os.environ:
    _mpl_cache = Path(tempfile.gettempdir()) / "portfolio_ql_mpl_cache"
    _mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(_mpl_cache)


def main(argv: list[str] | None = None) -> int:
    """Load saved artifacts and write a comparison bar chart.  Returns 0 on success."""
    if argv is None:
        argv = sys.argv[1:]

    if len(argv) != 3:
        print(
            "Usage: run_plot_artifacts.py <report_json> <resolved_config_yaml> <output_image>",
            file=sys.stderr,
        )
        return 2

    report_path = Path(argv[0])
    config_path = Path(argv[1])
    output_path = Path(argv[2])

    from src.backtest.experiment_io import load_experiment_report
    from src.config.schema import load_config

    import yaml as _yaml

    try:
        report = load_experiment_report(report_path)
    except FileNotFoundError as exc:
        print(f"run_plot_artifacts: error — {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"run_plot_artifacts: error loading report — {exc}", file=sys.stderr)
        return 1

    try:
        cfg = load_config(config_path)
    except FileNotFoundError as exc:
        print(f"run_plot_artifacts: error — {exc}", file=sys.stderr)
        return 1
    except (ValueError, _yaml.YAMLError) as exc:
        print(f"run_plot_artifacts: error loading config — {exc}", file=sys.stderr)
        return 1

    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend; must be set before pyplot import
    import matplotlib.pyplot as plt

    labels = ["CTRL", "Oracle"]
    values = [report.ctrl_mean_terminal_wealth, report.oracle_mean_terminal_wealth]

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(labels, values, color=["steelblue", "darkorange"])
    ax.set_ylabel("Mean Terminal Wealth")
    ax.set_title(
        f"CTRL vs Oracle — seed={cfg.seed} env={cfg.env.env_type} algo={cfg.algo.algo_type}"
    )
    ax.set_ylim(0, max(values) * 1.2 if max(values) > 0 else 1.2)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + max(values) * 0.01,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=100, bbox_inches="tight")
    except OSError as exc:
        print(f"run_plot_artifacts: error writing output — {exc}", file=sys.stderr)
        plt.close(fig)
        return 1
    finally:
        plt.close(fig)

    print(f"run_plot_artifacts: plot saved to {output_path}")
    print(
        f"  seed={cfg.seed}  env_type={cfg.env.env_type}  algo_type={cfg.algo.algo_type}"
    )
    print(f"  ctrl_mean_tw={report.ctrl_mean_terminal_wealth:.6f}")
    print(f"  oracle_mean_tw={report.oracle_mean_terminal_wealth:.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
