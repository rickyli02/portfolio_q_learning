"""Evaluation metrics and evaluator utilities.

Current primitives:
- Phase 15A: typed deterministic evaluation summary (``CTRLEvalSummary``, ``eval_summary``).
- Phase 15B: evaluation summary file IO helpers (``save_eval_summaries``, ``load_eval_summaries``).
- Phase 15C: multi-episode deterministic aggregate (``CTRLEvalAggregate``, ``eval_aggregate``).
"""

from src.eval.aggregate import CTRLEvalAggregate, eval_aggregate
from src.eval.io import load_eval_summaries, save_eval_summaries
from src.eval.summary import CTRLEvalSummary, eval_summary

__all__ = [
    "CTRLEvalAggregate",
    "CTRLEvalSummary",
    "eval_aggregate",
    "eval_summary",
    "load_eval_summaries",
    "save_eval_summaries",
]
