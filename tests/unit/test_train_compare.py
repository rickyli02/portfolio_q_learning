"""Unit tests for src/backtest/train_compare.py — Phase 16C bridge.

All key invariants are verified against independently composed references
rather than just checking types or counts.
"""

from __future__ import annotations

import importlib.util
import math
import subprocess
import sys
from pathlib import Path

import torch

from src.algos.oracle_mv import OracleMVPolicy
from src.backtest import (
    CTRLOracleBacktestComparison,
    CTRLTrainCompareResult,
    run_ctrl_oracle_comparison,
    train_and_compare,
)
from src.config.schema import AssetConfig, EnvConfig
from src.envs.gbm_env import GBMPortfolioEnv
from src.models.gaussian_actor import GaussianActor
from src.models.quadratic_critic import QuadraticCritic
from src.train.ctrl_state import CTRLTrainerSnapshot, CTRLTrainerState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MU = [0.08]
SIGMA = [[0.20]]
R = 0.05
HORIZON = 1.0
N_STEPS = 4
Z = 1.0
GAMMA_EMBED = 1.5


def _make_env() -> GBMPortfolioEnv:
    cfg = EnvConfig(
        horizon=HORIZON,
        n_steps=N_STEPS,
        initial_wealth=1.0,
        mu=MU,
        sigma=SIGMA,
        assets=AssetConfig(n_risky=1, include_risk_free=True, risk_free_rate=R),
    )
    return GBMPortfolioEnv(cfg)


def _make_trainer(env: GBMPortfolioEnv | None = None, w_init: float = 1.0) -> CTRLTrainerState:
    if env is None:
        env = _make_env()
    actor = GaussianActor(n_risky=1, horizon=HORIZON)
    critic = QuadraticCritic(horizon=HORIZON, target_return_z=Z)
    actor_opt = torch.optim.SGD(actor.parameters(), lr=1e-3)
    critic_opt = torch.optim.SGD(critic.parameters(), lr=1e-3)
    return CTRLTrainerState(
        actor=actor,
        critic=critic,
        env=env,
        actor_optimizer=actor_opt,
        critic_optimizer=critic_opt,
        current_w=w_init,
        target_return_z=Z,
        w_step_size=0.01,
    )


def _make_oracle() -> OracleMVPolicy:
    return OracleMVPolicy.from_env_params(
        mu=MU, sigma=SIGMA, r=R, horizon=HORIZON, gamma_embed=GAMMA_EMBED
    )


# ===========================================================================
# Public API exports
# ===========================================================================


def test_public_exports_include_bridge_types():
    import src.backtest as bt

    assert hasattr(bt, "CTRLTrainCompareResult")
    assert hasattr(bt, "train_and_compare")
    assert "CTRLTrainCompareResult" in bt.__all__
    assert "train_and_compare" in bt.__all__


# ===========================================================================
# Return type
# ===========================================================================


def test_train_and_compare_returns_correct_type():
    trainer = _make_trainer()
    oracle = _make_oracle()
    seeds = [0, 1]
    result = train_and_compare(
        trainer, oracle, eval_seeds=seeds, n_outer_iters=2,
        n_updates=2, entropy_temp=0.1, base_seed=0,
    )
    assert isinstance(result, CTRLTrainCompareResult)
    assert isinstance(result.eval_seeds, list)
    assert result.eval_seeds == seeds
    assert isinstance(result.post_training_snapshot, CTRLTrainerSnapshot)
    assert isinstance(result.comparison, CTRLOracleBacktestComparison)


# ===========================================================================
# Post-training snapshot equality
# ===========================================================================


def test_post_training_snapshot_matches_trainer_state_exactly():
    """Snapshot in result is identical to trainer.snapshot() after training."""
    trainer = _make_trainer()
    oracle = _make_oracle()
    result = train_and_compare(
        trainer, oracle, eval_seeds=[0, 1], n_outer_iters=2,
        n_updates=2, entropy_temp=0.1, base_seed=0,
    )
    live_snap = trainer.snapshot()

    assert result.post_training_snapshot.current_w == live_snap.current_w
    assert result.post_training_snapshot.target_return_z == live_snap.target_return_z
    assert result.post_training_snapshot.w_step_size == live_snap.w_step_size
    assert result.post_training_snapshot.last_terminal_wealth == live_snap.last_terminal_wealth
    assert result.post_training_snapshot.last_w_prev == live_snap.last_w_prev
    assert result.post_training_snapshot.last_n_updates == live_snap.last_n_updates


def test_post_training_snapshot_current_w_is_finite():
    trainer = _make_trainer()
    oracle = _make_oracle()
    result = train_and_compare(
        trainer, oracle, eval_seeds=[0], n_outer_iters=2,
        n_updates=2, entropy_temp=0.1, base_seed=0,
    )
    assert math.isfinite(result.post_training_snapshot.current_w)


def test_post_training_snapshot_last_n_updates_is_total():
    """last_n_updates == n_outer_iters * n_updates after run_outer_loop."""
    n_outer, n_up = 3, 2
    trainer = _make_trainer()
    oracle = _make_oracle()
    result = train_and_compare(
        trainer, oracle, eval_seeds=[0], n_outer_iters=n_outer,
        n_updates=n_up, entropy_temp=0.1, base_seed=0,
    )
    assert result.post_training_snapshot.last_n_updates == n_outer * n_up


# ===========================================================================
# Post-training w used in comparison (not initial w)
# ===========================================================================


def test_bridge_uses_post_training_w_not_initial_w():
    """Comparison is built with post-training current_w, not initial w.

    After training completes, trainer.current_w has been updated.  The bridge
    must pass this post-training w to run_ctrl_oracle_comparison, not w_init.
    We verify by re-running the comparison with the known post_w and checking
    that results match.
    """
    trainer = _make_trainer()
    oracle = _make_oracle()
    seeds = [0, 1]
    result = train_and_compare(
        trainer, oracle, eval_seeds=seeds, n_outer_iters=2,
        n_updates=2, entropy_temp=0.1, base_seed=0,
    )
    post_w = result.post_training_snapshot.current_w

    # Re-run comparison with the same post-training w — results must match
    ref = run_ctrl_oracle_comparison(
        actor=trainer.actor,
        env=trainer.env,
        w=post_w,
        oracle_policy=oracle,
        seeds=seeds,
        target_return_z=Z,
    )

    for i, (got, exp) in enumerate(zip(
        result.comparison.ctrl_bundle.summaries, ref.ctrl_bundle.summaries
    )):
        assert got.terminal_wealth == exp.terminal_wealth, f"tw mismatch at seed {i}"


def test_bridge_post_training_w_differs_from_w_init_after_updates():
    """After training, current_w has been updated from w_init."""
    w_init = 5.0
    trainer = _make_trainer(w_init=w_init)
    oracle = _make_oracle()
    result = train_and_compare(
        trainer, oracle, eval_seeds=[0], n_outer_iters=2,
        n_updates=2, entropy_temp=0.1, base_seed=0,
    )
    # After outer-loop updates, w has been modified from w_init
    assert result.post_training_snapshot.current_w != w_init


# ===========================================================================
# Comparison equality against independent reference
# ===========================================================================


def test_comparison_equals_independent_reference_composition():
    """Bridge result comparison matches independent train-then-compare composition.

    Both paths start from identical model initialisation (same torch seed),
    use identical training schedules (same base_seed), and evaluate on the
    same oracle and eval_seeds.  Results must be bit-for-bit identical.
    """
    env = _make_env()
    oracle = _make_oracle()
    seeds = [1, 5, 9]

    # --- Bridge path ---
    torch.manual_seed(0)
    trainer_a = _make_trainer(env=env)
    result = train_and_compare(
        trainer_a, oracle, eval_seeds=seeds, n_outer_iters=2,
        n_updates=2, entropy_temp=0.1, base_seed=42,
    )

    # --- Reference path: reset to same initial conditions ---
    torch.manual_seed(0)
    trainer_ref = _make_trainer(env=env)
    trainer_ref.run_outer_loop(
        n_outer_iters=2, n_updates=2, entropy_temp=0.1, base_seed=42,
    )
    ref_comparison = run_ctrl_oracle_comparison(
        actor=trainer_ref.actor,
        env=trainer_ref.env,
        w=trainer_ref.current_w,
        oracle_policy=oracle,
        seeds=seeds,
        target_return_z=Z,
    )

    # The bridge result's comparison must match the independently composed one
    assert (
        result.comparison.comparison.mean_terminal_wealth_delta
        == ref_comparison.comparison.mean_terminal_wealth_delta
    )
    assert result.comparison.comparison.ctrl_win_rate == ref_comparison.comparison.ctrl_win_rate
    for i, (got, exp) in enumerate(zip(
        result.comparison.ctrl_bundle.summaries,
        ref_comparison.ctrl_bundle.summaries,
    )):
        assert got.terminal_wealth == exp.terminal_wealth, f"ctrl tw mismatch at seed {i}"
    for i, (got, exp) in enumerate(zip(
        result.comparison.oracle_bundle.summaries,
        ref_comparison.oracle_bundle.summaries,
    )):
        assert got.terminal_wealth == exp.terminal_wealth, f"oracle tw mismatch at seed {i}"


# ===========================================================================
# Eval seed order preservation
# ===========================================================================


def test_eval_seeds_preserved_exactly():
    """Seeds in top-level field and nested bundles match eval_seeds in order."""
    trainer = _make_trainer()
    oracle = _make_oracle()
    seeds = [7, 3, 42]
    result = train_and_compare(
        trainer, oracle, eval_seeds=seeds, n_outer_iters=2,
        n_updates=2, entropy_temp=0.1, base_seed=0,
    )
    assert result.eval_seeds == seeds
    assert result.comparison.seeds == seeds
    assert result.comparison.ctrl_bundle.seeds == seeds
    assert result.comparison.oracle_bundle.seeds == seeds


def test_empty_eval_seeds_rejected_before_training():
    """eval_seeds=[] raises ValueError without mutating trainer state."""
    trainer = _make_trainer()
    oracle = _make_oracle()
    w_before = trainer.current_w
    history_len_before = len(trainer.history)

    import pytest
    with pytest.raises(ValueError, match="eval_seeds"):
        train_and_compare(
            trainer, oracle, eval_seeds=[], n_outer_iters=2,
            n_updates=2, entropy_temp=0.1, base_seed=0,
        )

    # Trainer state must be unchanged
    assert trainer.current_w == w_before, "current_w was mutated before seeds validation"
    assert len(trainer.history) == history_len_before, "history was mutated before seeds validation"


def test_eval_seeds_count_matches_summaries():
    seeds = [0, 1, 2, 3]
    trainer = _make_trainer()
    oracle = _make_oracle()
    result = train_and_compare(
        trainer, oracle, eval_seeds=seeds, n_outer_iters=2,
        n_updates=2, entropy_temp=0.1, base_seed=0,
    )
    assert len(result.comparison.ctrl_bundle.summaries) == len(seeds)
    assert len(result.comparison.oracle_bundle.summaries) == len(seeds)


# ===========================================================================
# Bridge uses trainer.target_return_z for comparison
# ===========================================================================


def test_bridge_uses_trainer_target_return_z():
    """Comparison target_return_z matches trainer.target_return_z for each seed."""
    trainer = _make_trainer()
    oracle = _make_oracle()
    result = train_and_compare(
        trainer, oracle, eval_seeds=[0, 1], n_outer_iters=2,
        n_updates=2, entropy_temp=0.1, base_seed=0,
    )
    for s in result.comparison.ctrl_bundle.summaries:
        assert s.target_return_z == trainer.target_return_z
    for s in result.comparison.oracle_bundle.summaries:
        assert s.target_return_z == trainer.target_return_z


# ===========================================================================
# Trainer mutation: history updated by bridge
# ===========================================================================


def test_bridge_mutates_trainer_history():
    """After bridge call, trainer.history has exactly one entry."""
    trainer = _make_trainer()
    oracle = _make_oracle()
    assert len(trainer.history) == 0
    train_and_compare(
        trainer, oracle, eval_seeds=[0], n_outer_iters=2,
        n_updates=2, entropy_temp=0.1, base_seed=0,
    )
    assert len(trainer.history) == 1


# ===========================================================================
# Script entrypoint
# ===========================================================================

_SCRIPT_PATH = Path(__file__).resolve().parent.parent.parent / "scripts" / "run_ctrl_oracle_demo.py"


def test_demo_script_main_returns_zero():
    """main() returns 0 on success."""
    spec = importlib.util.spec_from_file_location("run_ctrl_oracle_demo", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.main() == 0


def test_demo_script_subprocess_exit_code_zero():
    """Script exits 0 when run as subprocess."""
    result = subprocess.run(
        [sys.executable, str(_SCRIPT_PATH)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"script failed:\n{result.stderr}"


def test_demo_script_stdout_contains_summary_markers():
    """Script stdout contains expected summary section markers."""
    result = subprocess.run(
        [sys.executable, str(_SCRIPT_PATH)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    out = result.stdout
    assert "post_training_w" in out
    assert "ctrl_mean_tw" in out
    assert "oracle_mean_tw" in out
    assert "mean_tw_delta" in out
    assert "ctrl_win_rate" in out


def test_demo_script_stdout_values_are_finite():
    """All numeric fields printed by the script are finite floats."""
    result = subprocess.run(
        [sys.executable, str(_SCRIPT_PATH)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    for line in result.stdout.splitlines():
        if ":" in line and any(
            key in line for key in (
                "post_training_w", "last_terminal_wealth",
                "ctrl_mean_tw", "oracle_mean_tw",
                "mean_tw_delta", "ctrl_win_rate",
            )
        ):
            value_str = line.split(":")[-1].strip()
            value = float(value_str)
            assert math.isfinite(value), f"non-finite value in output: {line}"
