"""Unit tests for the config system (src/config/)."""

import tempfile
from pathlib import Path

import pytest
import yaml

from src.config.schema import (
    AlgorithmConfig,
    ExperimentConfig,
    EnvConfig,
    PlottingConfig,
    RewardConfig,
    OptimConfig,
    load_config,
)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def test_experiment_config_defaults():
    cfg = ExperimentConfig()
    assert cfg.seed == 42
    assert cfg.env.env_type == "gbm"
    assert cfg.reward.entropy_temp == pytest.approx(0.1)
    assert cfg.optim.batch_size == 256
    assert cfg.logging.save_checkpoints is True


def test_env_config_defaults():
    env = EnvConfig()
    assert env.n_steps == 100
    assert env.horizon == pytest.approx(1.0)
    assert env.jump_intensity == pytest.approx(0.0)
    assert env.assets.n_risky == 1


def test_reward_config_names():
    """Verify unambiguous naming conventions are in place."""
    r = RewardConfig()
    assert hasattr(r, "entropy_temp"), "Should use 'entropy_temp' not 'lambda'"
    assert hasattr(r, "target_return"), "Should use 'target_return'"
    assert hasattr(r, "mv_penalty_coeff")


def test_optim_config_names():
    o = OptimConfig()
    assert hasattr(o, "trace_decay"), "Should use 'trace_decay'"
    assert hasattr(o, "online_update_interval")
    assert hasattr(o, "w_update_interval")


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------


def _write_yaml(tmp_dir: str, data: dict) -> Path:
    p = Path(tmp_dir) / "test_config.yaml"
    with p.open("w") as fh:
        yaml.dump(data, fh)
    return p


def test_load_config_empty_yaml():
    """An empty YAML should produce all-default config."""
    with tempfile.TemporaryDirectory() as tmp:
        p = _write_yaml(tmp, {})
        cfg = load_config(p)
    assert isinstance(cfg, ExperimentConfig)
    assert cfg.seed == 42


def test_load_config_overrides_seed():
    with tempfile.TemporaryDirectory() as tmp:
        p = _write_yaml(tmp, {"seed": 7})
        cfg = load_config(p)
    assert cfg.seed == 7


def test_load_config_nested_override():
    with tempfile.TemporaryDirectory() as tmp:
        p = _write_yaml(tmp, {"env": {"n_steps": 20}, "optim": {"batch_size": 64}})
        cfg = load_config(p)
    assert cfg.env.n_steps == 20
    assert cfg.optim.batch_size == 64
    # Unset fields retain defaults
    assert cfg.env.horizon == pytest.approx(1.0)


def test_load_config_smoke_yaml():
    """The checked-in smoke config must load cleanly."""
    smoke_path = Path(__file__).resolve().parents[2] / "configs" / "tests" / "smoke.yaml"
    assert smoke_path.exists(), f"smoke.yaml not found at {smoke_path}"
    cfg = load_config(smoke_path)
    assert cfg.seed == 0
    assert cfg.env.n_steps == 10
    assert cfg.optim.batch_size == 32
    assert cfg.logging.save_checkpoints is False


def test_load_config_base_yaml():
    """The checked-in base default config must load cleanly."""
    base_path = Path(__file__).resolve().parents[2] / "configs" / "base" / "default.yaml"
    assert base_path.exists(), f"default.yaml not found at {base_path}"
    cfg = load_config(base_path)
    assert cfg.seed == 42


def test_load_config_unknown_field_raises():
    with tempfile.TemporaryDirectory() as tmp:
        p = _write_yaml(tmp, {"nonexistent_key": 999})
        with pytest.raises(ValueError, match="Unknown config field"):
            load_config(p)


def test_load_config_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_config(Path("/nonexistent/path/config.yaml"))


# ---------------------------------------------------------------------------
# ExperimentConfig.validate() — via load_config
# ---------------------------------------------------------------------------


def _load_with_overrides(overrides: dict) -> "ExperimentConfig":
    with tempfile.TemporaryDirectory() as tmp:
        p = _write_yaml(tmp, overrides)
        return load_config(p)


def test_validate_non_positive_horizon_raises():
    with pytest.raises(ValueError, match="horizon"):
        _load_with_overrides({"env": {"horizon": 0.0}})


def test_validate_non_positive_n_steps_raises():
    with pytest.raises(ValueError, match="n_steps"):
        _load_with_overrides({"env": {"n_steps": 0}})


def test_validate_non_positive_initial_wealth_raises():
    with pytest.raises(ValueError, match="initial_wealth"):
        _load_with_overrides({"env": {"initial_wealth": -1.0}})


def test_validate_mu_dimension_mismatch_raises():
    with pytest.raises(ValueError, match="n_risky"):
        _load_with_overrides({
            "env": {
                "assets": {"n_risky": 2},
                "mu": [0.1],          # wrong length
                "sigma": [[0.2, 0.0], [0.0, 0.15]],
            }
        })


def test_validate_sigma_dimension_mismatch_raises():
    with pytest.raises(ValueError, match="sigma"):
        _load_with_overrides({
            "env": {
                "assets": {"n_risky": 2},
                "mu": [0.1, 0.08],
                "sigma": [[0.2]],     # wrong shape
            }
        })


def test_validate_negative_entropy_temp_raises():
    with pytest.raises(ValueError, match="entropy_temp"):
        _load_with_overrides({"reward": {"entropy_temp": -0.1}})


def test_validate_invalid_discount_raises():
    with pytest.raises(ValueError, match="discount"):
        _load_with_overrides({"reward": {"discount": 0.0}})


def test_validate_non_positive_batch_size_raises():
    with pytest.raises(ValueError, match="batch_size"):
        _load_with_overrides({"optim": {"batch_size": 0}})


def test_validate_defaults_pass():
    """Default ExperimentConfig must pass validation without errors."""
    cfg = ExperimentConfig()
    cfg.validate()  # should not raise


# ---------------------------------------------------------------------------
# AlgorithmConfig — defaults and validation
# ---------------------------------------------------------------------------


def test_algo_config_defaults():
    a = AlgorithmConfig()
    assert a.algo_type == "ctrl_baseline"
    assert a.n_oracle_episodes == 100


def test_algo_type_in_experiment_config():
    cfg = ExperimentConfig()
    assert cfg.algo.algo_type == "ctrl_baseline"


def test_validate_invalid_algo_type_raises():
    with pytest.raises(ValueError, match="algo.algo_type"):
        _load_with_overrides({"algo": {"algo_type": "unknown_algo"}})


def test_validate_oracle_algo_type_passes():
    cfg = _load_with_overrides({"algo": {"algo_type": "oracle"}})
    assert cfg.algo.algo_type == "oracle"


def test_validate_ctrl_online_algo_type_passes():
    cfg = _load_with_overrides({"algo": {"algo_type": "ctrl_online"}})
    assert cfg.algo.algo_type == "ctrl_online"


def test_validate_non_positive_n_oracle_episodes_raises():
    with pytest.raises(ValueError, match="n_oracle_episodes"):
        _load_with_overrides({"algo": {"n_oracle_episodes": 0}})


# ---------------------------------------------------------------------------
# PlottingConfig — defaults and validation
# ---------------------------------------------------------------------------


def test_plotting_config_defaults():
    p = PlottingConfig()
    assert p.enabled is True
    assert p.figure_format == "png"
    assert p.dpi == 150
    assert p.n_trajectory_samples == 10
    assert p.plot_losses is True
    assert p.plot_gradients is True
    assert p.plot_wealth_trajectories is True


def test_plotting_in_experiment_config():
    cfg = ExperimentConfig()
    assert cfg.plotting.enabled is True
    assert cfg.plotting.output_dir == "plots"


def test_validate_invalid_figure_format_raises():
    with pytest.raises(ValueError, match="figure_format"):
        _load_with_overrides({"plotting": {"figure_format": "bmp"}})


def test_validate_pdf_format_passes():
    cfg = _load_with_overrides({"plotting": {"figure_format": "pdf"}})
    assert cfg.plotting.figure_format == "pdf"


def test_validate_non_positive_dpi_raises():
    with pytest.raises(ValueError, match="dpi"):
        _load_with_overrides({"plotting": {"dpi": 0}})


def test_validate_non_positive_n_trajectory_samples_raises():
    with pytest.raises(ValueError, match="n_trajectory_samples"):
        _load_with_overrides({"plotting": {"n_trajectory_samples": 0}})


def test_validate_plotting_disabled_passes():
    """Disabled plotting with minimal settings should still pass validation."""
    cfg = _load_with_overrides({"plotting": {"enabled": False}})
    assert cfg.plotting.enabled is False
