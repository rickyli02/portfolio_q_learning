"""Training-to-backtest bridge — Phase 16C.

Provides the first end-to-end typed seam that connects the approved stateful
trainer shell (``CTRLTrainerState``) to the approved deterministic
CTRL-vs-oracle comparison (``run_ctrl_oracle_comparison``).

The bridge runs a fixed outer-loop training schedule on a live trainer, then
immediately evaluates the resulting policy against the oracle using the
post-training Lagrange parameter as the evaluation ``w``.

SCOPE BOUNDARY
--------------
The following are NOT implemented here:
- file IO, logging, or checkpoint integration
- config-dispatch wiring
- experiment directory management
- plotting or report generation
- adaptive w schedules or early stopping
- multi-run sweeps or hyperparameter search
"""

from __future__ import annotations

from dataclasses import dataclass

from src.algos.oracle_mv import OracleMVPolicy
from src.backtest.comparison import CTRLOracleBacktestComparison, run_ctrl_oracle_comparison
from src.train.ctrl_state import CTRLTrainerSnapshot, CTRLTrainerState


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CTRLTrainCompareResult:
    """Typed result from a train-then-compare run — Phase 16C.

    Bundles the post-training trainer snapshot, the explicit evaluation seeds,
    and the full CTRL-vs-oracle comparison so callers can inspect all three
    without navigating nested structures.

    Attributes:
        eval_seeds:             Explicit evaluation seeds, in the order they
                                were passed to ``train_and_compare``.
        post_training_snapshot: Read-only trainer snapshot captured immediately
                                after ``run_outer_loop`` completes.  Reflects
                                the updated ``current_w``, total inner steps,
                                and final terminal wealth of the training run.
        comparison:             Full CTRL-vs-oracle deterministic comparison
                                built from the post-training actor and the
                                post-training ``current_w``.
    """

    eval_seeds: list[int]
    post_training_snapshot: CTRLTrainerSnapshot
    comparison: CTRLOracleBacktestComparison


# ---------------------------------------------------------------------------
# Bridge helper
# ---------------------------------------------------------------------------


def train_and_compare(
    trainer: CTRLTrainerState,
    oracle_policy: OracleMVPolicy,
    eval_seeds: list[int],
    n_outer_iters: int,
    n_updates: int,
    entropy_temp: float,
    base_seed: int | None = None,
    w_min: float | None = None,
    w_max: float | None = None,
) -> CTRLTrainCompareResult:
    """Run a fixed outer-loop training schedule then compare against oracle.

    Mutates ``trainer`` in place (as ``run_outer_loop`` always does), then
    evaluates the trained actor against ``oracle_policy`` using the
    post-training ``trainer.current_w`` and ``trainer.target_return_z``.

    Args:
        trainer:      Live ``CTRLTrainerState``; mutated by the training run.
        oracle_policy: Configured ``OracleMVPolicy`` for the same environment.
        eval_seeds:   Non-empty list of integer seeds for deterministic eval.
        n_outer_iters: Number of outer training iterations.
        n_updates:    Number of inner actor/critic steps per outer iteration.
        entropy_temp: Entropy regularisation temperature γ.
        base_seed:    Optional base seed forwarded to the training run for
                      reproducible trajectory scheduling.
        w_min:        Optional lower bound for w projection during training.
        w_max:        Optional upper bound for w projection during training.

    Returns:
        ``CTRLTrainCompareResult`` with post-training snapshot and
        CTRL-vs-oracle comparison.

    Raises:
        ValueError: If ``eval_seeds`` is empty (checked before training begins
                    so trainer state is never mutated on invalid input), or
                    propagated from ``run_outer_loop`` if training inputs are
                    invalid.
    """
    if not eval_seeds:
        raise ValueError("eval_seeds must be non-empty")

    trainer.run_outer_loop(
        n_outer_iters=n_outer_iters,
        n_updates=n_updates,
        entropy_temp=entropy_temp,
        base_seed=base_seed,
        w_min=w_min,
        w_max=w_max,
    )

    post_training_snapshot: CTRLTrainerSnapshot = trainer.snapshot()

    comparison = run_ctrl_oracle_comparison(
        actor=trainer.actor,
        env=trainer.env,
        w=trainer.current_w,
        oracle_policy=oracle_policy,
        seeds=eval_seeds,
        target_return_z=trainer.target_return_z,
    )

    return CTRLTrainCompareResult(
        eval_seeds=list(eval_seeds),
        post_training_snapshot=post_training_snapshot,
        comparison=comparison,
    )
