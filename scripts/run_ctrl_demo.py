#!/usr/bin/env python3
"""Minimal end-to-end CTRL trainer demo script — Phase 13A.

Exercises the full trainer stack (actor → critic → CTRLTrainerState → outer
loop) on a tiny synthetic GBM environment and prints a concise scalar summary
to stdout.  Intended for manual smoke/demo use only; not a production
experiment runner.

Hardcoded defaults (kept small for fast demo runtime):

    N_RISKY        = 1        risky asset
    N_STEPS        = 5        time steps per episode
    HORIZON        = 1.0      year
    INITIAL_WEALTH = 1.0
    W_INIT         = 1.0      initial Lagrange multiplier
    TARGET_RETURN  = 1.0      target terminal wealth z
    W_STEP_SIZE    = 0.1      outer-loop step size a_w
    ENTROPY_TEMP   = 0.01     entropy regularisation γ
    N_OUTER_ITERS  = 3        outer iterations
    N_UPDATES      = 2        inner actor/critic steps per outer iteration
    BASE_SEED      = 42       deterministic seed

Usage (run inside the project .venv):
    .venv/bin/python scripts/run_ctrl_demo.py
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
_W_INIT = 1.0
_TARGET_RETURN_Z = 1.0
_W_STEP_SIZE = 0.1
_ENTROPY_TEMP = 0.01
_N_OUTER_ITERS = 3
_N_UPDATES = 2
_BASE_SEED = 42


def main() -> int:
    """Run the CTRL demo and print a scalar summary.  Returns 0 on success."""
    import torch

    from src.config.schema import AssetConfig, EnvConfig
    from src.envs.gbm_env import GBMPortfolioEnv
    from src.models.gaussian_actor import GaussianActor
    from src.models.quadratic_critic import QuadraticCritic
    from src.train import CTRLTrainerState

    print("CTRL demo: building tiny synthetic GBM environment")
    env_cfg = EnvConfig(
        horizon=_HORIZON,
        n_steps=_N_STEPS,
        initial_wealth=_INITIAL_WEALTH,
        mu=[0.08],
        sigma=[[0.20]],
        assets=AssetConfig(
            n_risky=_N_RISKY,
            include_risk_free=True,
            risk_free_rate=0.03,
        ),
    )
    env = GBMPortfolioEnv(env_cfg)

    print("CTRL demo: instantiating GaussianActor and QuadraticCritic")
    actor = GaussianActor(n_risky=_N_RISKY, horizon=_HORIZON)
    critic = QuadraticCritic(horizon=_HORIZON, target_return_z=_TARGET_RETURN_Z)
    actor_opt = torch.optim.SGD(actor.parameters(), lr=1e-3)
    critic_opt = torch.optim.SGD(critic.parameters(), lr=1e-3)

    print("CTRL demo: constructing CTRLTrainerState")
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

    print(
        f"CTRL demo: running outer loop "
        f"(n_outer_iters={_N_OUTER_ITERS}, n_updates={_N_UPDATES}, seed={_BASE_SEED})"
    )
    trainer.run_outer_loop(
        n_outer_iters=_N_OUTER_ITERS,
        n_updates=_N_UPDATES,
        entropy_temp=_ENTROPY_TEMP,
        base_seed=_BASE_SEED,
    )

    snap = trainer.snapshot()
    print()
    print("--- CTRL demo summary ---")
    print(f"  w_init              : {_W_INIT:.6f}")
    print(f"  w_final             : {snap.current_w:.6f}")
    print(f"  target_return_z     : {snap.target_return_z:.6f}")
    print(f"  last_terminal_wealth: {snap.last_terminal_wealth:.6f}")
    print(f"  last_w_prev         : {snap.last_w_prev:.6f}")
    print(f"  last_n_updates      : {snap.last_n_updates}")
    print(f"  history_len         : {len(trainer.history)}")
    print("--- end ---")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
