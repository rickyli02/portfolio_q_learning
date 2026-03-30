"""Compact scalar report for the train-and-compare workflow — Phase 18A.

Provides a typed scalar summary that flattens the key fields from a
``CTRLTrainCompareResult`` into a single inspection-friendly object.
The helper is a pure consumer of the existing result type; it does not
re-run training or evaluation.

SCOPE BOUNDARY
--------------
The following are NOT implemented here:
- file IO or report serialization
- config-driven experiment runners
- plotting or report visualization
- changes to train_and_compare semantics
"""

from __future__ import annotations

from dataclasses import dataclass

from src.backtest.train_compare import CTRLTrainCompareResult


@dataclass(frozen=True)
class CTRLTrainCompareReport:
    """Compact scalar summary of a train-and-compare run — Phase 18A.

    Flattens the key scalars from ``CTRLTrainCompareResult`` into a single
    flat dataclass for easy inspection.  All fields are plain Python scalars.

    Policy-role note: fields derived from the training run
    (``post_training_w``, ``last_n_updates``, ``last_terminal_wealth``)
    reflect the stochastic behavior policy used during data collection.
    Fields derived from the comparison
    (``ctrl_mean_terminal_wealth``, ``oracle_mean_terminal_wealth``,
    ``mean_terminal_wealth_delta``, ``ctrl_win_rate``) reflect the
    deterministic execution policy evaluated post-training.

    Attributes:
        post_training_w:              Outer-loop Lagrange multiplier after
                                      training (``post_training_snapshot.current_w``).
        target_return_z:              Target terminal wealth z
                                      (``post_training_snapshot.target_return_z``).
        last_n_updates:               Total inner actor/critic steps in the
                                      training run, or ``None`` if the snapshot
                                      has no completed run.
        last_terminal_wealth:         Terminal portfolio wealth from the final
                                      inner step of training, or ``None``.
        n_eval_seeds:                 Number of evaluation seeds used in the
                                      comparison.
        ctrl_mean_terminal_wealth:    Mean terminal wealth of the CTRL policy
                                      across eval seeds.
        oracle_mean_terminal_wealth:  Mean terminal wealth of the oracle policy
                                      across eval seeds.
        mean_terminal_wealth_delta:   Mean of (ctrl − oracle) terminal wealth
                                      across eval seeds.
        ctrl_win_rate:                Fraction of eval seeds where CTRL terminal
                                      wealth exceeded the oracle.
    """

    post_training_w: float
    target_return_z: float
    last_n_updates: int | None
    last_terminal_wealth: float | None
    n_eval_seeds: int
    ctrl_mean_terminal_wealth: float
    oracle_mean_terminal_wealth: float
    mean_terminal_wealth_delta: float
    ctrl_win_rate: float


def summarize_train_compare(result: CTRLTrainCompareResult) -> CTRLTrainCompareReport:
    """Derive a compact scalar report from a ``CTRLTrainCompareResult``.

    Reads the post-training snapshot and the CTRL-vs-oracle comparison
    scalar fields and returns them as a flat ``CTRLTrainCompareReport``.
    No training or evaluation is re-run.

    Args:
        result: Completed ``CTRLTrainCompareResult`` from ``train_and_compare``.

    Returns:
        ``CTRLTrainCompareReport`` with all scalar fields populated.
    """
    snap = result.post_training_snapshot
    comp = result.comparison

    return CTRLTrainCompareReport(
        post_training_w=snap.current_w,
        target_return_z=snap.target_return_z,
        last_n_updates=snap.last_n_updates,
        last_terminal_wealth=snap.last_terminal_wealth,
        n_eval_seeds=len(result.eval_seeds),
        ctrl_mean_terminal_wealth=comp.ctrl_bundle.aggregate.mean_terminal_wealth,
        oracle_mean_terminal_wealth=comp.oracle_bundle.aggregate.mean_terminal_wealth,
        mean_terminal_wealth_delta=comp.comparison.mean_terminal_wealth_delta,
        ctrl_win_rate=comp.comparison.ctrl_win_rate,
    )
