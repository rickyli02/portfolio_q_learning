"""Evaluation metrics and evaluator utilities.

Current primitives:
- Phase 15A: typed deterministic evaluation summary (``CTRLEvalSummary``, ``eval_summary``).
- Phase 15B: evaluation summary file IO helpers (``save_eval_summaries``, ``load_eval_summaries``).
- Phase 15C: multi-episode deterministic aggregate (``CTRLEvalAggregate``, ``eval_aggregate``).
- Phase 15D: evaluation aggregate file IO helpers (``save_eval_aggregates``, ``load_eval_aggregates``).
- Phase 15E: deterministic evaluation trajectory record (``CTRLEvalRecord``, ``eval_record``).
- Phase 15F: evaluation record file IO helpers (``save_eval_records``, ``load_eval_records``).
- Phase 15G: multi-seed deterministic record set (``CTRLEvalRecordSet``, ``eval_record_set``).
- Phase 15H: evaluation record-set file IO helpers (``save_eval_record_sets``, ``load_eval_record_sets``).
- Phase 15I: pure derivation helpers (``summary_from_record``, ``aggregate_from_record_set``).
"""

from src.eval.aggregate import CTRLEvalAggregate, eval_aggregate
from src.eval.aggregate_io import load_eval_aggregates, save_eval_aggregates
from src.eval.derive import aggregate_from_record_set, summary_from_record
from src.eval.io import load_eval_summaries, save_eval_summaries
from src.eval.record import CTRLEvalRecord, eval_record
from src.eval.record_io import load_eval_records, save_eval_records
from src.eval.record_set import CTRLEvalRecordSet, eval_record_set
from src.eval.record_set_io import load_eval_record_sets, save_eval_record_sets
from src.eval.summary import CTRLEvalSummary, eval_summary

__all__ = [
    "CTRLEvalAggregate",
    "CTRLEvalRecord",
    "CTRLEvalRecordSet",
    "CTRLEvalSummary",
    "aggregate_from_record_set",
    "eval_aggregate",
    "eval_record",
    "eval_record_set",
    "eval_summary",
    "load_eval_aggregates",
    "load_eval_record_sets",
    "load_eval_records",
    "load_eval_summaries",
    "save_eval_aggregates",
    "save_eval_record_sets",
    "save_eval_records",
    "save_eval_summaries",
    "summary_from_record",
]
