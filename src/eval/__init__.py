"""Evaluation metrics and evaluator utilities.

Current primitives:
- Phase 15A: typed deterministic evaluation summary (``CTRLEvalSummary``, ``eval_summary``).
"""

from src.eval.summary import CTRLEvalSummary, eval_summary

__all__ = [
    "CTRLEvalSummary",
    "eval_summary",
]
