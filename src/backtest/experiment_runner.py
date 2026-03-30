"""Config-backed train-and-compare experiment runner — Phase 20A.

Provides the first typed end-to-end experiment runner helper that consumes
``ExperimentConfig`` for the supported baseline path (GBM env + CTRL baseline
algo) and runs the approved train-and-compare + scalar report workflow.

SCOPE BOUNDARY
--------------
The following are NOT implemented here:
- CLI parsing or YAML loading (handled by src/config/schema.py)
- output file IO, logging, or checkpoint management
- plotting or report visualization
- adaptive w schedules or early stopping
- multi-run sweeps or hyperparameter search
- unsupported env_type / algo_type combinations
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from src.algos.oracle_mv import OracleMVPolicy
from src.backtest.train_compare import CTRLTrainCompareResult, train_and_compare
from src.backtest.train_compare_report import CTRLTrainCompareReport, summarize_train_compare
from src.config.schema import ExperimentConfig
from src.envs.gbm_env import GBMPortfolioEnv
from src.models.gaussian_actor import GaussianActor
from src.models.quadratic_critic import QuadraticCritic
from src.train.ctrl_state import CTRLTrainerState


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CTRLExperimentResult:
    """Typed result from a config-backed train-and-compare run — Phase 20A.

    Bundles the resolved ``ExperimentConfig``, the full train-compare result,
    and the compact scalar report so callers can inspect all three without
    navigating nested structures.

    Attributes:
        config:               The ``ExperimentConfig`` used for this run.
        train_compare_result: Full ``CTRLTrainCompareResult`` from
                              ``train_and_compare``.
        report:               Compact scalar summary derived from the result
                              via ``summarize_train_compare``.
    """

    config: ExperimentConfig
    train_compare_result: CTRLTrainCompareResult
    report: CTRLTrainCompareReport


# ---------------------------------------------------------------------------
# Runner helper
# ---------------------------------------------------------------------------


def run_ctrl_experiment(
    cfg: ExperimentConfig,
    w_init: float = 1.0,
    w_step_size: float = 0.01,
) -> CTRLExperimentResult:
    """Run a config-backed train-then-compare experiment on the supported path.

    Constructs the GBM environment, CTRL actor/critic/trainer, and oracle
    comparator from ``cfg``, then runs ``train_and_compare`` followed by
    ``summarize_train_compare``.

    Supported path (raises ``ValueError`` for anything else):
        - ``cfg.env.env_type == "gbm"``
        - ``cfg.algo.algo_type == "ctrl_baseline"``
        - ``cfg.policy.policy_type == "gaussian"``
        - ``cfg.policy.deterministic_eval == True``
        - ``cfg.eval.eval_deterministic == True``

    Configuration mapping:
        - ``cfg.env``                     → ``GBMPortfolioEnv``
        - ``cfg.algo.oracle_gamma_embed`` → oracle embedding scalar γ
        - ``cfg.reward.target_return``    → critic ``target_return_z``
        - ``cfg.optim.n_epochs``          → ``n_outer_iters``
        - ``cfg.optim.n_steps_per_epoch`` → ``n_updates`` (inner gradient steps)
        - ``cfg.reward.entropy_temp``     → ``entropy_temp``
        - ``cfg.eval.n_eval_episodes``    → number of deterministic eval seeds
        - ``cfg.seed``                    → ``base_seed`` for the training run

    Args:
        cfg:          Resolved and validated ``ExperimentConfig``.
        w_init:       Initial Lagrange multiplier w for the trainer.
        w_step_size:  Step size for the outer-loop w update.

    Returns:
        ``CTRLExperimentResult`` with config, full train-compare result, and
        compact scalar report.

    Raises:
        ValueError: If any config selector names an unsupported combination.
    """
    if cfg.env.env_type != "gbm":
        raise ValueError(
            f"run_ctrl_experiment only supports env_type='gbm', "
            f"got '{cfg.env.env_type}'"
        )
    if cfg.algo.algo_type != "ctrl_baseline":
        raise ValueError(
            f"run_ctrl_experiment only supports algo_type='ctrl_baseline', "
            f"got '{cfg.algo.algo_type}'"
        )
    if cfg.policy.policy_type != "gaussian":
        raise ValueError(
            f"run_ctrl_experiment only supports policy_type='gaussian', "
            f"got '{cfg.policy.policy_type}'"
        )
    if not cfg.policy.deterministic_eval:
        raise ValueError(
            "run_ctrl_experiment requires policy.deterministic_eval=True; "
            "stochastic evaluation is not supported on this path"
        )
    if not cfg.eval.eval_deterministic:
        raise ValueError(
            "run_ctrl_experiment requires eval.eval_deterministic=True; "
            "stochastic evaluation is not supported on this path"
        )

    gamma_embed = cfg.algo.oracle_gamma_embed

    # Build GBM environment from config.
    env = GBMPortfolioEnv(cfg.env)

    n_risky = cfg.env.assets.n_risky
    horizon = cfg.env.horizon

    # Build actor and critic.
    actor = GaussianActor(n_risky=n_risky, horizon=horizon)
    critic = QuadraticCritic(
        horizon=horizon, target_return_z=cfg.reward.target_return
    )

    # Build optimizers from config learning rates.
    actor_opt = torch.optim.Adam(actor.parameters(), lr=cfg.optim.actor_lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=cfg.optim.critic_lr)

    # Build stateful trainer.
    trainer = CTRLTrainerState(
        actor=actor,
        critic=critic,
        env=env,
        actor_optimizer=actor_opt,
        critic_optimizer=critic_opt,
        current_w=w_init,
        target_return_z=cfg.reward.target_return,
        w_step_size=w_step_size,
    )

    # Build oracle policy from env parameters.
    oracle = OracleMVPolicy.from_env_params(
        mu=cfg.env.mu,
        sigma=cfg.env.sigma,
        r=cfg.env.assets.risk_free_rate,
        horizon=horizon,
        gamma_embed=gamma_embed,
    )

    # Deterministic evaluation seeds: 0 … n_eval_episodes-1.
    eval_seeds = list(range(cfg.eval.n_eval_episodes))

    # Run training then deterministic comparison.
    result = train_and_compare(
        trainer=trainer,
        oracle_policy=oracle,
        eval_seeds=eval_seeds,
        n_outer_iters=cfg.optim.n_epochs,
        n_updates=cfg.optim.n_steps_per_epoch,
        entropy_temp=cfg.reward.entropy_temp,
        base_seed=cfg.seed,
    )

    report = summarize_train_compare(result)

    return CTRLExperimentResult(
        config=cfg,
        train_compare_result=result,
        report=report,
    )
