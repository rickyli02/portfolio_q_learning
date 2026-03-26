"""Typed experiment configuration schema.

All config objects are plain Python dataclasses with explicit types so that
static analysis tools and tests can catch bad values early.  YAML loading
converts to these objects via ``load_config``.

Naming conventions follow the project style guide:
  - entropy_temp  (not lambda)
  - trace_decay   (TD(lambda) decay)
  - target_return (the w / wealth-target parameter in MV objective)
  - rebalance_interval, online_update_interval
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------


@dataclass
class AssetConfig:
    """Asset universe settings."""

    n_risky: int = 1
    """Number of risky assets."""

    include_risk_free: bool = True
    """Whether to include a risk-free asset in the portfolio."""

    risk_free_rate: float = 0.05
    """Annualised continuously-compounded risk-free rate."""


@dataclass
class EnvConfig:
    """Environment / simulation settings."""

    env_type: str = "gbm"
    """Environment identifier.  Supported: ``'gbm'``."""

    horizon: float = 1.0
    """Investment horizon T (in years)."""

    n_steps: int = 100
    """Number of discrete rebalance steps per episode."""

    initial_wealth: float = 1.0
    """Starting wealth x_0."""

    rebalance_interval: float = 0.0
    """Minimum time between rebalance events (0.0 = every step)."""

    # GBM parameters (used when env_type == 'gbm')
    mu: list[float] = field(default_factory=lambda: [0.1])
    """Expected log-return vector (annualised), one entry per risky asset."""

    sigma: list[list[float]] = field(
        default_factory=lambda: [[0.2]]
    )
    """Volatility / covariance matrix (annualised), shape [n_risky, n_risky]."""

    jump_intensity: float = 0.0
    """Poisson jump intensity (events per year).  0 disables jumps."""

    assets: AssetConfig = field(default_factory=AssetConfig)


@dataclass
class RewardConfig:
    """Reward / objective settings."""

    target_return: float = 1.0
    """Target terminal wealth w in the MV objective (x_T - w)^2."""

    mv_penalty_coeff: float = 1.0
    """Coefficient on the variance-penalty term in the MV objective."""

    entropy_temp: float = 0.1
    """Entropy regularisation temperature (lambda in Wang-Zhou notation).
    Higher values encourage more exploration."""

    discount: float = 1.0
    """Discount factor for future rewards (1.0 = undiscounted)."""


@dataclass
class PolicyConfig:
    """Behaviour and execution policy settings."""

    policy_type: str = "gaussian"
    """Policy class.  Supported: ``'gaussian'``."""

    hidden_dims: list[int] = field(default_factory=lambda: [64, 64])
    """Hidden layer sizes for MLP-based policies."""

    log_std_min: float = -5.0
    """Minimum log standard-deviation for clipping."""

    log_std_max: float = 2.0
    """Maximum log standard-deviation for clipping."""

    deterministic_eval: bool = True
    """If True, use mean action (no sampling) during evaluation."""


@dataclass
class OptimConfig:
    """Optimiser / training loop settings."""

    actor_lr: float = 3e-4
    """Actor learning rate."""

    critic_lr: float = 3e-4
    """Critic learning rate."""

    batch_size: int = 256
    """Mini-batch size for gradient updates."""

    n_epochs: int = 100
    """Number of training epochs (offline) or update rounds (online)."""

    n_steps_per_epoch: int = 1000
    """Environment steps collected per epoch (online) or gradient steps per
    epoch (offline)."""

    trace_decay: float = 0.95
    """TD(lambda) eligibility-trace decay factor."""

    online_update_interval: int = 1
    """Number of environment steps between parameter updates (online mode)."""

    w_update_interval: int = 10
    """Number of epochs between updates to the MV Lagrange multiplier w."""

    replay_buffer_size: int = 100_000
    """Maximum transitions stored in the replay buffer."""

    grad_clip: float = 1.0
    """Gradient norm clipping threshold (0 = disabled)."""


@dataclass
class EvalConfig:
    """Evaluation settings."""

    eval_interval: int = 10
    """Epochs between evaluation runs."""

    n_eval_episodes: int = 16
    """Number of episodes per evaluation."""

    eval_deterministic: bool = True
    """Evaluate with the deterministic execution policy."""


@dataclass
class LoggingConfig:
    """Logging and output settings."""

    experiment_name: str = "default"
    """Name used to organise output subdirectory."""

    log_level: str = "INFO"
    """Python logging level string."""

    save_checkpoints: bool = True
    """Whether to save parameter checkpoints."""

    checkpoint_interval: int = 10
    """Epochs between checkpoint saves."""


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration.

    All sub-configs have sensible defaults so that a minimal YAML only needs
    to specify the fields that differ from the defaults.
    """

    seed: int = 42
    """Global random seed."""

    env: EnvConfig = field(default_factory=EnvConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def validate(self) -> None:
        """Raise ``ValueError`` if any field holds an obviously invalid value.

        Called automatically by ``load_config`` after YAML overrides are applied.
        """
        # --- env ---
        if self.env.horizon <= 0:
            raise ValueError(f"env.horizon must be > 0, got {self.env.horizon}")
        if self.env.n_steps < 1:
            raise ValueError(f"env.n_steps must be >= 1, got {self.env.n_steps}")
        if self.env.initial_wealth <= 0:
            raise ValueError(
                f"env.initial_wealth must be > 0, got {self.env.initial_wealth}"
            )
        n = self.env.assets.n_risky
        if n < 1:
            raise ValueError(f"env.assets.n_risky must be >= 1, got {n}")
        if len(self.env.mu) != n:
            raise ValueError(
                f"len(env.mu)={len(self.env.mu)} must equal env.assets.n_risky={n}"
            )
        if len(self.env.sigma) != n or any(len(row) != n for row in self.env.sigma):
            raise ValueError(
                f"env.sigma must be {n}x{n} to match env.assets.n_risky={n}"
            )
        # --- reward ---
        if self.reward.entropy_temp < 0:
            raise ValueError(
                f"reward.entropy_temp must be >= 0, got {self.reward.entropy_temp}"
            )
        if self.reward.mv_penalty_coeff <= 0:
            raise ValueError(
                f"reward.mv_penalty_coeff must be > 0, got {self.reward.mv_penalty_coeff}"
            )
        if not (0 < self.reward.discount <= 1.0):
            raise ValueError(
                f"reward.discount must be in (0, 1], got {self.reward.discount}"
            )
        # --- optim ---
        if self.optim.batch_size < 1:
            raise ValueError(
                f"optim.batch_size must be >= 1, got {self.optim.batch_size}"
            )
        if self.optim.n_epochs < 1:
            raise ValueError(
                f"optim.n_epochs must be >= 1, got {self.optim.n_epochs}"
            )
        if self.optim.replay_buffer_size < 1:
            raise ValueError(
                f"optim.replay_buffer_size must be >= 1, "
                f"got {self.optim.replay_buffer_size}"
            )


# ---------------------------------------------------------------------------
# YAML loading helpers
# ---------------------------------------------------------------------------


def _apply_overrides(obj: Any, overrides: dict) -> None:
    """Recursively apply a nested dict of overrides onto a dataclass."""
    for key, value in overrides.items():
        if not hasattr(obj, key):
            raise ValueError(
                f"Unknown config field '{key}' for {type(obj).__name__}"
            )
        current = getattr(obj, key)
        if isinstance(current, dict):
            current.update(value)
        elif hasattr(current, "__dataclass_fields__"):
            _apply_overrides(current, value)
        else:
            setattr(obj, key, value)


def load_config(path: Path | str) -> ExperimentConfig:
    """Load an ExperimentConfig from a YAML file.

    The YAML may specify any subset of fields; unspecified fields retain their
    default values.  Raises ``ValueError`` for unknown field names.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open() as fh:
        raw: dict = yaml.safe_load(fh) or {}
    cfg = ExperimentConfig()
    _apply_overrides(cfg, raw)
    cfg.validate()
    return cfg
