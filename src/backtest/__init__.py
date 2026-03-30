"""Backtesting: portfolio path simulation and summary statistics.

Current primitives:
- Phase 16A: deterministic CTRL-vs-oracle scalar comparison
  (``CTRLOracleComparisonSummary``, ``CTRLOracleBacktestComparison``,
   ``run_ctrl_oracle_comparison``).
- Phase 16C: training-to-backtest bridge
  (``CTRLTrainCompareResult``, ``train_and_compare``).
- Phase 18A: compact scalar report for the train-and-compare workflow
  (``CTRLTrainCompareReport``, ``summarize_train_compare``).
- Phase 20A: config-backed train-and-compare runner
  (``CTRLExperimentResult``, ``run_ctrl_experiment``).
- Phase 20C: run-artifact persistence helpers
  (``save_experiment_report``, ``load_experiment_report``, ``save_experiment_config``).
"""

from src.backtest.comparison import (
    CTRLOracleBacktestComparison,
    CTRLOracleComparisonSummary,
    run_ctrl_oracle_comparison,
)
from src.backtest.experiment_io import (
    load_experiment_report,
    save_experiment_config,
    save_experiment_report,
)
from src.backtest.experiment_runner import (
    CTRLExperimentResult,
    run_ctrl_experiment,
)
from src.backtest.train_compare import (
    CTRLTrainCompareResult,
    train_and_compare,
)
from src.backtest.train_compare_report import (
    CTRLTrainCompareReport,
    summarize_train_compare,
)

__all__ = [
    "CTRLOracleBacktestComparison",
    "CTRLOracleComparisonSummary",
    "run_ctrl_oracle_comparison",
    "CTRLExperimentResult",
    "run_ctrl_experiment",
    "load_experiment_report",
    "save_experiment_config",
    "save_experiment_report",
    "CTRLTrainCompareResult",
    "train_and_compare",
    "CTRLTrainCompareReport",
    "summarize_train_compare",
]
