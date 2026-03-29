"""Backtesting: portfolio path simulation and summary statistics.

Current primitives:
- Phase 16A: deterministic CTRL-vs-oracle scalar comparison
  (``CTRLOracleComparisonSummary``, ``CTRLOracleBacktestComparison``,
   ``run_ctrl_oracle_comparison``).
- Phase 16C: training-to-backtest bridge
  (``CTRLTrainCompareResult``, ``train_and_compare``).
- Phase 18A: compact scalar report for the train-and-compare workflow
  (``CTRLTrainCompareReport``, ``summarize_train_compare``).
"""

from src.backtest.comparison import (
    CTRLOracleBacktestComparison,
    CTRLOracleComparisonSummary,
    run_ctrl_oracle_comparison,
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
    "CTRLTrainCompareResult",
    "train_and_compare",
    "CTRLTrainCompareReport",
    "summarize_train_compare",
]
