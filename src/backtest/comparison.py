"""Deterministic CTRL-vs-oracle scalar comparison — Phase 16A.

Provides the first real consumer of the eval stack: a typed scalar comparison
of the CTRL learned policy against the Zhou–Li (2000) analytic oracle over
an explicit seed set.

Both policies are evaluated deterministically on the same environment and
seed list.  Oracle results are wrapped into the approved ``CTRLEvalRecord`` /
``CTRLEvalRecordSet`` shapes so the same derivation and bundle helpers apply
to both sides without inventing a parallel type hierarchy.

SCOPE BOUNDARY
--------------
The following are NOT implemented here:
- plotting or report generation
- file IO
- training-loop integration
- config-dispatch wiring
- stochastic behavior-policy comparison
- broader run-directory / artifact-management policy

These belong in future bounded tasks.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from src.algos.oracle_mv import OracleMVPolicy, run_oracle_episode
from src.envs.gbm_env import GBMPortfolioEnv
from src.eval.derive import CTRLEvalScalarBundle, bundle_from_record_set
from src.eval.record import CTRLEvalRecord
from src.eval.record_set import CTRLEvalRecordSet, eval_record_set
from src.models.base import ActorBase


# ---------------------------------------------------------------------------
# Typed comparison structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CTRLOracleComparisonSummary:
    """Per-seed terminal-wealth delta summary for CTRL vs oracle — Phase 16A.

    All deltas are defined as ``ctrl − oracle`` so a positive value means
    CTRL outperformed the oracle on that metric.

    Attributes:
        mean_terminal_wealth_delta: Mean of (ctrl_x_T − oracle_x_T) over seeds.
        min_terminal_wealth_delta:  Minimum per-seed terminal wealth delta.
        max_terminal_wealth_delta:  Maximum per-seed terminal wealth delta.
        ctrl_win_rate:              Fraction of seeds where
                                    ctrl_x_T > oracle_x_T.
    """

    mean_terminal_wealth_delta: float
    min_terminal_wealth_delta: float
    max_terminal_wealth_delta: float
    ctrl_win_rate: float


@dataclass(frozen=True)
class CTRLOracleBacktestComparison:
    """Typed scalar comparison of CTRL vs oracle over an explicit seed set.

    Attributes:
        seeds:       Explicit seeds used for evaluation, in evaluation order.
        ctrl_bundle: Scalar bundle derived from CTRL deterministic evaluation.
        oracle_bundle: Scalar bundle derived from oracle deterministic evaluation.
        comparison:  Per-seed terminal-wealth delta summary.
    """

    seeds: list[int]
    ctrl_bundle: CTRLEvalScalarBundle
    oracle_bundle: CTRLEvalScalarBundle
    comparison: CTRLOracleComparisonSummary


# ---------------------------------------------------------------------------
# Oracle-episode helpers
# ---------------------------------------------------------------------------


def _oracle_episode_to_record(
    result: dict[str, torch.Tensor],
    target_return_z: float | None,
) -> CTRLEvalRecord:
    """Wrap a ``run_oracle_episode`` result dict into a ``CTRLEvalRecord``."""
    wealth_path = result["wealth_path"]
    times = result["times"]
    actions = result["actions"]

    terminal_wealth = float(wealth_path[-1])
    initial_wealth = float(wealth_path[0])
    n_steps = int(times.shape[0])
    min_wealth = float(wealth_path.min())
    max_wealth = float(wealth_path.max())

    terminal_gap: float | None
    if target_return_z is not None:
        terminal_gap = terminal_wealth - float(target_return_z)
    else:
        terminal_gap = None

    return CTRLEvalRecord(
        times=times.float(),
        wealth_path=wealth_path.float(),
        actions=actions.float(),
        terminal_wealth=terminal_wealth,
        initial_wealth=initial_wealth,
        n_steps=n_steps,
        min_wealth=min_wealth,
        max_wealth=max_wealth,
        target_return_z=target_return_z,
        terminal_gap=terminal_gap,
    )


def _eval_oracle_record_set(
    policy: OracleMVPolicy,
    env: GBMPortfolioEnv,
    seeds: list[int],
    target_return_z: float | None,
) -> CTRLEvalRecordSet:
    """Run oracle evaluation for each seed and return a ``CTRLEvalRecordSet``."""
    records = [
        _oracle_episode_to_record(
            run_oracle_episode(policy, env, seed=s),
            target_return_z=target_return_z,
        )
        for s in seeds
    ]
    return CTRLEvalRecordSet(seeds=list(seeds), records=records)


# ---------------------------------------------------------------------------
# Comparison summary computation
# ---------------------------------------------------------------------------


def _compute_comparison_summary(
    ctrl_bundle: CTRLEvalScalarBundle,
    oracle_bundle: CTRLEvalScalarBundle,
) -> CTRLOracleComparisonSummary:
    """Compute per-seed terminal-wealth deltas from two scalar bundles."""
    ctrl_tws = [s.terminal_wealth for s in ctrl_bundle.summaries]
    oracle_tws = [s.terminal_wealth for s in oracle_bundle.summaries]
    deltas = [c - o for c, o in zip(ctrl_tws, oracle_tws)]
    n = len(deltas)

    return CTRLOracleComparisonSummary(
        mean_terminal_wealth_delta=sum(deltas) / n,
        min_terminal_wealth_delta=min(deltas),
        max_terminal_wealth_delta=max(deltas),
        ctrl_win_rate=sum(1 for d in deltas if d > 0) / n,
    )


# ---------------------------------------------------------------------------
# Public comparison helper
# ---------------------------------------------------------------------------


def run_ctrl_oracle_comparison(
    actor: ActorBase,
    env: GBMPortfolioEnv,
    w: float,
    oracle_policy: OracleMVPolicy,
    seeds: list[int],
    target_return_z: float | None = None,
) -> CTRLOracleBacktestComparison:
    """Run deterministic CTRL and oracle evaluation and return a scalar comparison.

    Evaluates both policies on the same ``env`` and ``seeds``, derives scalar
    bundles for each, and computes per-seed terminal-wealth delta statistics.
    No rollout logic is duplicated — CTRL uses ``eval_record_set``; oracle
    results are wrapped into the same ``CTRLEvalRecord`` shape.

    CTRL evaluation chain (deterministic execution policy):
        ``eval_record_set`` → ``eval_record`` → ``evaluate_ctrl_deterministic``
        → ``execute_ctrl_action`` → ``actor.mean_action()``

    Args:
        actor:           Learned CTRL policy satisfying ``ActorBase``.
        env:             GBM portfolio environment shared by both policies.
        w:               Outer-loop Lagrange parameter for CTRL evaluation.
        oracle_policy:   Configured ``OracleMVPolicy`` (Zhou–Li 2000).
        seeds:           Non-empty list of integer RNG seeds.
        target_return_z: Optional target terminal wealth z, forwarded to
                         both CTRL and oracle record construction.

    Returns:
        ``CTRLOracleBacktestComparison`` with CTRL bundle, oracle bundle, and
        comparison summary.

    Raises:
        ValueError: If ``seeds`` is empty (propagated from ``eval_record_set``).
    """
    # CTRL side — reuse approved eval_record_set + bundle_from_record_set
    ctrl_rs = eval_record_set(actor, env, w=w, seeds=seeds, target_return_z=target_return_z)
    ctrl_bundle = bundle_from_record_set(ctrl_rs)

    # Oracle side — wrap run_oracle_episode results into approved record shapes
    oracle_rs = _eval_oracle_record_set(oracle_policy, env, seeds, target_return_z)
    oracle_bundle = bundle_from_record_set(oracle_rs)

    comparison = _compute_comparison_summary(ctrl_bundle, oracle_bundle)

    return CTRLOracleBacktestComparison(
        seeds=list(seeds),
        ctrl_bundle=ctrl_bundle,
        oracle_bundle=oracle_bundle,
        comparison=comparison,
    )
