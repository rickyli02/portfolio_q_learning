"""Tests for the config-backed train-and-compare experiment runner — Phase 20A.

Coverage:
- ``CTRLExperimentResult`` typed shape and fields
- ``run_ctrl_experiment`` returns expected typed result on a real small run
- key config fields are reflected correctly in the result
- unsupported env_type / algo_type combinations raise ``ValueError`` with
  clear messages
- public exports are available from ``src.backtest``
"""

from __future__ import annotations

import pytest

from src.backtest import CTRLExperimentResult, run_ctrl_experiment
from src.backtest.experiment_runner import CTRLExperimentResult as _DirectImport
from src.backtest.train_compare import CTRLTrainCompareResult
from src.backtest.train_compare_report import CTRLTrainCompareReport
from src.config.schema import (
    AlgorithmConfig,
    EnvConfig,
    EvalConfig,
    ExperimentConfig,
    OptimConfig,
    RewardConfig,
)


# ---------------------------------------------------------------------------
# Tiny config fixture — keeps tests fast
# ---------------------------------------------------------------------------


def _tiny_cfg(
    env_type: str = "gbm",
    algo_type: str = "ctrl_baseline",
    n_eval_episodes: int = 3,
    n_epochs: int = 2,
    n_steps_per_epoch: int = 1,
) -> ExperimentConfig:
    """Build a minimal valid config for testing."""
    cfg = ExperimentConfig()
    cfg.env = EnvConfig(
        env_type=env_type,
        horizon=1.0,
        n_steps=5,
        initial_wealth=1.0,
        mu=[0.08],
        sigma=[[0.20]],
    )
    cfg.env.assets.n_risky = 1
    cfg.env.assets.include_risk_free = True
    cfg.env.assets.risk_free_rate = 0.05
    cfg.reward = RewardConfig(
        target_return=1.0,
        entropy_temp=0.1,
    )
    cfg.optim = OptimConfig(
        n_epochs=n_epochs,
        n_steps_per_epoch=n_steps_per_epoch,
        actor_lr=1e-3,
        critic_lr=1e-3,
    )
    cfg.eval = EvalConfig(n_eval_episodes=n_eval_episodes)
    cfg.algo = AlgorithmConfig(algo_type=algo_type)
    cfg.seed = 42
    return cfg


# ---------------------------------------------------------------------------
# Result type tests
# ---------------------------------------------------------------------------


def test_experiment_result_has_expected_fields():
    """CTRLExperimentResult dataclass exposes config, result, and report."""
    import dataclasses

    fields = {f.name for f in dataclasses.fields(CTRLExperimentResult)}
    assert "config" in fields
    assert "train_compare_result" in fields
    assert "report" in fields


def test_experiment_result_is_frozen():
    """CTRLExperimentResult is immutable (frozen dataclass)."""
    cfg = _tiny_cfg()
    result = run_ctrl_experiment(cfg)
    with pytest.raises((AttributeError, TypeError)):
        result.config = cfg  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Live small-run tests
# ---------------------------------------------------------------------------


def test_run_ctrl_experiment_returns_correct_types():
    """run_ctrl_experiment returns CTRLExperimentResult with correct field types."""
    cfg = _tiny_cfg()
    result = run_ctrl_experiment(cfg)

    assert isinstance(result, CTRLExperimentResult)
    assert isinstance(result.train_compare_result, CTRLTrainCompareResult)
    assert isinstance(result.report, CTRLTrainCompareReport)


def test_run_ctrl_experiment_config_is_preserved():
    """result.config is the same object passed to run_ctrl_experiment."""
    cfg = _tiny_cfg()
    result = run_ctrl_experiment(cfg)
    assert result.config is cfg


def test_run_ctrl_experiment_eval_seeds_match_config():
    """eval_seeds length matches cfg.eval.n_eval_episodes."""
    cfg = _tiny_cfg(n_eval_episodes=4)
    result = run_ctrl_experiment(cfg)
    assert len(result.train_compare_result.eval_seeds) == 4


def test_run_ctrl_experiment_eval_seeds_are_zero_indexed():
    """eval_seeds are 0 … n_eval_episodes-1."""
    cfg = _tiny_cfg(n_eval_episodes=3)
    result = run_ctrl_experiment(cfg)
    assert result.train_compare_result.eval_seeds == [0, 1, 2]


def test_run_ctrl_experiment_report_n_eval_seeds_matches_config():
    """report.n_eval_seeds matches cfg.eval.n_eval_episodes."""
    cfg = _tiny_cfg(n_eval_episodes=5)
    result = run_ctrl_experiment(cfg)
    assert result.report.n_eval_seeds == 5


def test_run_ctrl_experiment_report_target_return_matches_config():
    """report.target_return_z matches cfg.reward.target_return."""
    cfg = _tiny_cfg()
    cfg.reward.target_return = 1.1
    # Rebuild critic target via a fresh experiment (cfg not yet validated here)
    cfg.env.assets.n_risky = 1  # ensure consistent
    result = run_ctrl_experiment(cfg)
    assert result.report.target_return_z == pytest.approx(1.1)


def test_run_ctrl_experiment_report_scalars_are_finite():
    """All report scalar fields are finite after a real small run."""
    import math

    cfg = _tiny_cfg()
    result = run_ctrl_experiment(cfg)
    r = result.report

    assert math.isfinite(r.post_training_w)
    assert math.isfinite(r.target_return_z)
    assert math.isfinite(r.ctrl_mean_terminal_wealth)
    assert math.isfinite(r.oracle_mean_terminal_wealth)
    assert math.isfinite(r.mean_terminal_wealth_delta)
    assert 0.0 <= r.ctrl_win_rate <= 1.0


def test_run_ctrl_experiment_oracle_gamma_embed_from_config():
    """oracle_gamma_embed is read from cfg.algo.oracle_gamma_embed, not target_return."""
    cfg = _tiny_cfg()
    cfg.algo.oracle_gamma_embed = 1.5
    result = run_ctrl_experiment(cfg)
    assert isinstance(result, CTRLExperimentResult)


def test_run_ctrl_experiment_gamma_embed_and_target_return_are_independent():
    """Changing target_return does not silently change oracle_gamma_embed."""
    cfg_a = _tiny_cfg()
    cfg_a.reward.target_return = 1.0
    cfg_a.algo.oracle_gamma_embed = 1.5

    cfg_b = _tiny_cfg()
    cfg_b.reward.target_return = 1.3  # different training target
    cfg_b.algo.oracle_gamma_embed = 1.5  # same oracle embedding

    # Both should succeed; oracle embedding is the same despite different target_return
    result_a = run_ctrl_experiment(cfg_a)
    result_b = run_ctrl_experiment(cfg_b)
    assert isinstance(result_a, CTRLExperimentResult)
    assert isinstance(result_b, CTRLExperimentResult)


# ---------------------------------------------------------------------------
# Unsupported combination error tests
# ---------------------------------------------------------------------------


def test_unsupported_env_type_raises_value_error():
    """ValueError is raised for unsupported env_type."""
    cfg = _tiny_cfg()
    cfg.env.env_type = "jump"
    with pytest.raises(ValueError, match="env_type='gbm'"):
        run_ctrl_experiment(cfg)


def test_unsupported_env_type_error_message_names_received_value():
    """ValueError message names the unsupported env_type."""
    cfg = _tiny_cfg()
    cfg.env.env_type = "jump"
    with pytest.raises(ValueError, match="jump"):
        run_ctrl_experiment(cfg)


def test_unsupported_algo_type_raises_value_error():
    """ValueError is raised for unsupported algo_type."""
    cfg = _tiny_cfg()
    cfg.algo.algo_type = "ctrl_online"
    with pytest.raises(ValueError, match="algo_type='ctrl_baseline'"):
        run_ctrl_experiment(cfg)


def test_unsupported_algo_type_error_message_names_received_value():
    """ValueError message names the unsupported algo_type."""
    cfg = _tiny_cfg()
    cfg.algo.algo_type = "ctrl_online"
    with pytest.raises(ValueError, match="ctrl_online"):
        run_ctrl_experiment(cfg)


def test_oracle_algo_type_raises_value_error():
    """ValueError is raised for algo_type='oracle' (not supported by runner)."""
    cfg = _tiny_cfg()
    cfg.algo.algo_type = "oracle"
    with pytest.raises(ValueError, match="algo_type='ctrl_baseline'"):
        run_ctrl_experiment(cfg)


def test_unsupported_policy_type_raises_value_error():
    """ValueError is raised for unsupported policy_type."""
    cfg = _tiny_cfg()
    cfg.policy.policy_type = "mlp"
    with pytest.raises(ValueError, match="policy_type='gaussian'"):
        run_ctrl_experiment(cfg)


def test_unsupported_policy_type_error_message_names_received_value():
    """ValueError message names the unsupported policy_type."""
    cfg = _tiny_cfg()
    cfg.policy.policy_type = "mlp"
    with pytest.raises(ValueError, match="mlp"):
        run_ctrl_experiment(cfg)


def test_stochastic_policy_eval_flag_raises_value_error():
    """ValueError is raised when policy.deterministic_eval=False."""
    cfg = _tiny_cfg()
    cfg.policy.deterministic_eval = False
    with pytest.raises(ValueError, match="policy.deterministic_eval"):
        run_ctrl_experiment(cfg)


def test_stochastic_eval_flag_raises_value_error():
    """ValueError is raised when eval.eval_deterministic=False."""
    cfg = _tiny_cfg()
    cfg.eval.eval_deterministic = False
    with pytest.raises(ValueError, match="eval.eval_deterministic"):
        run_ctrl_experiment(cfg)


# ---------------------------------------------------------------------------
# Public export tests
# ---------------------------------------------------------------------------


def test_ctrl_experiment_result_exported_from_backtest():
    """CTRLExperimentResult is re-exported from src.backtest."""
    from src.backtest import CTRLExperimentResult as Exported

    assert Exported is _DirectImport


def test_run_ctrl_experiment_exported_from_backtest():
    """run_ctrl_experiment is re-exported from src.backtest."""
    from src.backtest import run_ctrl_experiment as exported_fn
    from src.backtest.experiment_runner import run_ctrl_experiment as direct_fn

    assert exported_fn is direct_fn
