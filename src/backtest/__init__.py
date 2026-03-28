"""Backtesting: portfolio path simulation and summary statistics.

Current primitives:
- Phase 16A: deterministic CTRL-vs-oracle scalar comparison
  (``CTRLOracleComparisonSummary``, ``CTRLOracleBacktestComparison``,
   ``run_ctrl_oracle_comparison``).
"""

from src.backtest.comparison import (
    CTRLOracleBacktestComparison,
    CTRLOracleComparisonSummary,
    run_ctrl_oracle_comparison,
)

__all__ = [
    "CTRLOracleBacktestComparison",
    "CTRLOracleComparisonSummary",
    "run_ctrl_oracle_comparison",
]
