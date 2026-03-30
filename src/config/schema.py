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
    """Price-SDE drift vector b (annualised), one entry per risky asset.

    This is not the expected log-return. The GBM environment computes
    per-step log-drift as ``(mu - 0.5 * diag(sigma sigma^T)) * dt``.
    """

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
class AlgorithmConfig:
    """Algorithm / strategy selection settings.

    Controls which learning or execution algorithm is instantiated at runtime.
    Only schema-level selection is handled here; implementation lives in the
    algorithm modules.

    Supported ``algo_type`` values:

    - ``'oracle'``: closed-form optimal policy from known synthetic parameters
      (Zhou–Li 2000).  Requires a synthetic environment with known coefficients.
    - ``'ctrl_baseline'``: theorem-aligned CTRL offline baseline
      (Huang–Jia–Zhou 2025).
    - ``'ctrl_online'``: practical online CTRL with 2022/2025 improvements
      (Huang–Jia–Zhou 2022 + 2025 E-companion).
    """

    algo_type: str = "ctrl_baseline"
    """Algorithm to run.  Supported: ``'oracle'``, ``'ctrl_baseline'``,
    ``'ctrl_online'``."""

    n_oracle_episodes: int = 100
    """Number of evaluation episodes for the oracle policy (``algo_type='oracle'``
    only).  Ignored for learned-policy algorithms."""

    oracle_gamma_embed: float = 1.0
    """Auxiliary embedding scalar γ for the Zhou–Li (2000) oracle policy.

    This is the virtual terminal wealth target in the oracle formula:

        ū(t, x) = [σσᵀ]⁻¹ B · (γ · exp(−r(T−t)) − x)

    **Not** the same as ``reward.target_return`` (z), which is the CTRL
    training target.  Changing ``reward.target_return`` does NOT implicitly
    change the oracle benchmark unless this field is also updated.
    Meaningful values are typically close to the expected terminal wealth
    (e.g. 1.0 for a unit-wealth experiment).
    """


@dataclass
class PlottingConfig:
    """YAML-configurable plotting settings.

    Controls which plots are generated and how they are saved.  Plotting
    implementation is handled by a separate module (``src/eval/plots.py``);
    this config block only records user preferences.

    Memory-pressure capture uses ``tracemalloc`` for CPU peak and
    ``torch.cuda.memory_allocated`` for GPU when ``plot_memory`` is enabled.
    """

    enabled: bool = True
    """Master switch.  If False, no plots are written."""

    output_dir: str = "plots"
    """Directory (relative to the run output root) to write plot files."""

    figure_format: str = "png"
    """Output format for saved figures.  Supported: ``'png'``, ``'pdf'``,
    ``'svg'``."""

    dpi: int = 150
    """Dots per inch for raster formats (``'png'``).  Ignored for vector
    formats."""

    # --- training diagnostics ---

    plot_losses: bool = True
    """Plot critic and actor loss curves over training episodes."""

    plot_gradients: bool = True
    """Plot gradient norms over training episodes."""

    plot_training_time: bool = False
    """Plot wall-clock training time per episode."""

    plot_memory: bool = False
    """Plot memory pressure (CPU peak via ``tracemalloc``; GPU via
    ``torch.cuda`` when available)."""

    # --- wealth and portfolio plots ---

    plot_wealth_trajectories: bool = True
    """Plot wealth trajectories comparing the learned policy against the oracle
    and/or baseline comparators."""

    plot_portfolio_weights: bool = True
    """Plot portfolio weight / dollar-allocation paths over time."""

    plot_eval_metrics: bool = True
    """Overlay evaluation metrics (e.g. Sharpe ratio, MV objective) on wealth
    trajectory plots."""

    n_trajectory_samples: int = 10
    """Number of sample trajectories to display on wealth and weight plots."""


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
    algo: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    plotting: PlottingConfig = field(default_factory=PlottingConfig)

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
        # --- algo ---
        _valid_algo_types = {"oracle", "ctrl_baseline", "ctrl_online"}
        if self.algo.algo_type not in _valid_algo_types:
            raise ValueError(
                f"algo.algo_type must be one of {sorted(_valid_algo_types)}, "
                f"got '{self.algo.algo_type}'"
            )
        if self.algo.n_oracle_episodes < 1:
            raise ValueError(
                f"algo.n_oracle_episodes must be >= 1, "
                f"got {self.algo.n_oracle_episodes}"
            )
        if self.algo.oracle_gamma_embed <= 0:
            raise ValueError(
                f"algo.oracle_gamma_embed must be > 0, "
                f"got {self.algo.oracle_gamma_embed}"
            )
        # --- plotting ---
        _valid_formats = {"png", "pdf", "svg"}
        if self.plotting.figure_format not in _valid_formats:
            raise ValueError(
                f"plotting.figure_format must be one of {sorted(_valid_formats)}, "
                f"got '{self.plotting.figure_format}'"
            )
        if self.plotting.dpi < 1:
            raise ValueError(
                f"plotting.dpi must be >= 1, got {self.plotting.dpi}"
            )
        if self.plotting.n_trajectory_samples < 1:
            raise ValueError(
                f"plotting.n_trajectory_samples must be >= 1, "
                f"got {self.plotting.n_trajectory_samples}"
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
