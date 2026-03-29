"""Unit tests for src/backtest/train_compare_report.py — Phase 18A.

Verifies the compact scalar report seam:
- return type is CTRLTrainCompareReport
- all report fields match the source CTRLTrainCompareResult exactly
- helper works on a real small train_and_compare() run
- public exports are present
"""

from __future__ import annotations

import math

import torch

from src.algos.oracle_mv import OracleMVPolicy
from src.backtest import (
    CTRLTrainCompareReport,
    CTRLTrainCompareResult,
    summarize_train_compare,
    train_and_compare,
)
from src.config.schema import AssetConfig, EnvConfig
from src.envs.gbm_env import GBMPortfolioEnv
from src.models.gaussian_actor import GaussianActor
from src.models.quadratic_critic import QuadraticCritic
from src.train.ctrl_state import CTRLTrainerState

# ---------------------------------------------------------------------------
# Shared fixture values
# ---------------------------------------------------------------------------

MU = [0.08]
SIGMA = [[0.20]]
R = 0.05
HORIZON = 1.0
N_STEPS = 4
Z = 1.0
GAMMA_EMBED = 1.5
EVAL_SEEDS = [0, 1, 2]


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


def _make_trainer(w_init: float = 1.0) -> CTRLTrainerState:
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
        mu=MU, sigma=SIGMA, r=R, horizon=HORIZON, gamma_embed=GAMMA_EMBED,
    )


def _run_result() -> CTRLTrainCompareResult:
    return train_and_compare(
        _make_trainer(), _make_oracle(),
        eval_seeds=EVAL_SEEDS,
        n_outer_iters=2, n_updates=2, entropy_temp=0.1, base_seed=0,
    )


# ===========================================================================
# Public exports
# ===========================================================================


def test_public_exports_include_report_types():
    import src.backtest as bt

    assert hasattr(bt, "CTRLTrainCompareReport")
    assert hasattr(bt, "summarize_train_compare")
    assert "CTRLTrainCompareReport" in bt.__all__
    assert "summarize_train_compare" in bt.__all__


# ===========================================================================
# Return type
# ===========================================================================


def test_summarize_returns_report_type():
    assert isinstance(summarize_train_compare(_run_result()), CTRLTrainCompareReport)


# ===========================================================================
# Field mapping — each field must match the source CTRLTrainCompareResult
# ===========================================================================


def test_post_training_w_matches_snapshot():
    result = _run_result()
    report = summarize_train_compare(result)
    assert report.post_training_w == result.post_training_snapshot.current_w


def test_target_return_z_matches_snapshot():
    result = _run_result()
    report = summarize_train_compare(result)
    assert report.target_return_z == result.post_training_snapshot.target_return_z


def test_last_n_updates_matches_snapshot():
    result = _run_result()
    report = summarize_train_compare(result)
    assert report.last_n_updates == result.post_training_snapshot.last_n_updates


def test_last_terminal_wealth_matches_snapshot():
    result = _run_result()
    report = summarize_train_compare(result)
    assert report.last_terminal_wealth == result.post_training_snapshot.last_terminal_wealth


def test_n_eval_seeds_matches_eval_seeds_length():
    result = _run_result()
    report = summarize_train_compare(result)
    assert report.n_eval_seeds == len(result.eval_seeds)
    assert report.n_eval_seeds == len(EVAL_SEEDS)


def test_ctrl_mean_terminal_wealth_matches_bundle():
    result = _run_result()
    report = summarize_train_compare(result)
    assert report.ctrl_mean_terminal_wealth == result.comparison.ctrl_bundle.aggregate.mean_terminal_wealth


def test_oracle_mean_terminal_wealth_matches_bundle():
    result = _run_result()
    report = summarize_train_compare(result)
    assert report.oracle_mean_terminal_wealth == result.comparison.oracle_bundle.aggregate.mean_terminal_wealth


def test_mean_terminal_wealth_delta_matches_comparison():
    result = _run_result()
    report = summarize_train_compare(result)
    assert report.mean_terminal_wealth_delta == result.comparison.comparison.mean_terminal_wealth_delta


def test_ctrl_win_rate_matches_comparison():
    result = _run_result()
    report = summarize_train_compare(result)
    assert report.ctrl_win_rate == result.comparison.comparison.ctrl_win_rate


# ===========================================================================
# Scalar validity — all fields are finite plain Python scalars
# ===========================================================================


def test_all_non_optional_fields_are_finite():
    report = summarize_train_compare(_run_result())
    assert math.isfinite(report.post_training_w)
    assert math.isfinite(report.target_return_z)
    assert math.isfinite(report.ctrl_mean_terminal_wealth)
    assert math.isfinite(report.oracle_mean_terminal_wealth)
    assert math.isfinite(report.mean_terminal_wealth_delta)
    assert math.isfinite(report.ctrl_win_rate)


def test_last_n_updates_is_positive_after_run():
    report = summarize_train_compare(_run_result())
    assert report.last_n_updates is not None
    assert report.last_n_updates > 0


def test_n_eval_seeds_is_correct_count():
    report = summarize_train_compare(_run_result())
    assert report.n_eval_seeds == len(EVAL_SEEDS)


def test_ctrl_win_rate_in_unit_interval():
    report = summarize_train_compare(_run_result())
    assert 0.0 <= report.ctrl_win_rate <= 1.0


# ===========================================================================
# Live integration: helper works on a real train_and_compare() run
# (distinct from field-mapping tests which also use a real run)
# ===========================================================================


def test_report_is_consistent_with_raw_comparison_delta():
    """mean_terminal_wealth_delta = ctrl_mean_tw - oracle_mean_tw (up to per-seed avg)."""
    result = _run_result()
    report = summarize_train_compare(result)
    # The delta from the comparison summary is mean(ctrl_tw_i - oracle_tw_i),
    # which equals ctrl_mean_tw - oracle_mean_tw when computed per-seed.
    # Verify the sign is consistent with the mean terminal wealth values.
    expected_sign = report.ctrl_mean_terminal_wealth - report.oracle_mean_terminal_wealth
    assert (report.mean_terminal_wealth_delta >= 0) == (expected_sign >= 0)
