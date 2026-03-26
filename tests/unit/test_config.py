"""Unit tests for the config system (src/config/)."""

import tempfile
from pathlib import Path

import pytest
import yaml

from src.config.schema import (
    ExperimentConfig,
    EnvConfig,
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
