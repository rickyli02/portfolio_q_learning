#!/usr/bin/env python3
"""Tiny CTRL-vs-oracle training and comparison demo — Phase 16C.

Runs a small fixed CTRL training schedule on a synthetic single-asset GBM
environment, then evaluates the trained policy against the Zhou–Li (2000)
analytic oracle using the first training-to-backtest bridge helper.

Hardcoded tiny defaults:

    N_RISKY        = 1       risky asset
    N_STEPS        = 5       time steps per episode
    HORIZON        = 1.0     year
    INITIAL_WEALTH = 1.0
    MU             = [0.08]  drift
    SIGMA          = [[0.20]] volatility (lower-triangular factor)
    R              = 0.05    risk-free rate
    W_INIT         = 1.0     initial Lagrange multiplier
    TARGET_RETURN  = 1.0     target terminal wealth z
    W_STEP_SIZE    = 0.01    outer-loop step size a_w
    ENTROPY_TEMP   = 0.1     entropy regularisation γ
    N_OUTER_ITERS  = 3       outer training iterations
    N_UPDATES      = 2       inner actor/critic steps per iteration
    BASE_SEED      = 42      deterministic training seed
    EVAL_SEEDS     = [0,1,2] evaluation seeds
    GAMMA_EMBED    = 1.5     oracle risk-aversion proxy

Usage (run inside the project .venv):
    .venv/bin/python scripts/run_ctrl_oracle_demo.py
"""

import sys
from pathlib import Path

# Ensure repo root is on sys.path when run directly.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ---------------------------------------------------------------------------
# Hardcoded tiny defaults
# ---------------------------------------------------------------------------
_N_RISKY = 1
_HORIZON = 1.0
_N_STEPS = 5
_INITIAL_WEALTH = 1.0
_MU = [0.08]
_SIGMA = [[0.20]]
_R = 0.05
_W_INIT = 1.0
_TARGET_RETURN_Z = 1.0
_W_STEP_SIZE = 0.01
_ENTROPY_TEMP = 0.1
_N_OUTER_ITERS = 3
_N_UPDATES = 2
_BASE_SEED = 42
_EVAL_SEEDS = [0, 1, 2]
_GAMMA_EMBED = 1.5


def main() -> int:
    """Run CTRL training then CTRL-vs-oracle comparison.  Returns 0 on success."""
    import torch

    from src.algos.oracle_mv import OracleMVPolicy
    from src.backtest.train_compare import train_and_compare
    from src.config.schema import AssetConfig, EnvConfig
    from src.envs.gbm_env import GBMPortfolioEnv
    from src.models.gaussian_actor import GaussianActor
    from src.models.quadratic_critic import QuadraticCritic
    from src.train.ctrl_state import CTRLTrainerState

    print("CTRL-vs-oracle demo: building tiny synthetic GBM environment")
    env_cfg = EnvConfig(
        horizon=_HORIZON,
        n_steps=_N_STEPS,
        initial_wealth=_INITIAL_WEALTH,
        mu=_MU,
        sigma=_SIGMA,
        assets=AssetConfig(
            n_risky=_N_RISKY,
            include_risk_free=True,
            risk_free_rate=_R,
        ),
    )
    env = GBMPortfolioEnv(env_cfg)

    print("CTRL-vs-oracle demo: instantiating models and trainer")
    actor = GaussianActor(n_risky=_N_RISKY, horizon=_HORIZON)
    critic = QuadraticCritic(horizon=_HORIZON, target_return_z=_TARGET_RETURN_Z)
    actor_opt = torch.optim.SGD(actor.parameters(), lr=1e-3)
    critic_opt = torch.optim.SGD(critic.parameters(), lr=1e-3)
    trainer = CTRLTrainerState(
        actor=actor,
        critic=critic,
        env=env,
        actor_optimizer=actor_opt,
        critic_optimizer=critic_opt,
        current_w=_W_INIT,
        target_return_z=_TARGET_RETURN_Z,
        w_step_size=_W_STEP_SIZE,
    )

    print("CTRL-vs-oracle demo: instantiating oracle policy")
    oracle = OracleMVPolicy.from_env_params(
        mu=_MU,
        sigma=_SIGMA,
        r=_R,
        horizon=_HORIZON,
        gamma_embed=_GAMMA_EMBED,
    )

    print(
        f"CTRL-vs-oracle demo: training "
        f"(n_outer_iters={_N_OUTER_ITERS}, n_updates={_N_UPDATES}, "
        f"seed={_BASE_SEED}) then evaluating on seeds {_EVAL_SEEDS}"
    )
    result = train_and_compare(
        trainer=trainer,
        oracle_policy=oracle,
        eval_seeds=_EVAL_SEEDS,
        n_outer_iters=_N_OUTER_ITERS,
        n_updates=_N_UPDATES,
        entropy_temp=_ENTROPY_TEMP,
        base_seed=_BASE_SEED,
    )

    snap = result.post_training_snapshot
    ctrl_agg = result.comparison.ctrl_bundle.aggregate
    oracle_agg = result.comparison.oracle_bundle.aggregate
    cmp = result.comparison.comparison

    print()
    print("--- CTRL-vs-oracle demo summary ---")
    print(f"  post_training_w         : {snap.current_w:.6f}")
    print(f"  last_terminal_wealth    : {snap.last_terminal_wealth:.6f}")
    print(f"  ctrl_mean_tw            : {ctrl_agg.mean_terminal_wealth:.6f}")
    print(f"  oracle_mean_tw          : {oracle_agg.mean_terminal_wealth:.6f}")
    print(f"  mean_tw_delta           : {cmp.mean_terminal_wealth_delta:.6f}")
    print(f"  ctrl_win_rate           : {cmp.ctrl_win_rate:.6f}")
    print("--- end ---")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
