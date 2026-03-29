#!/usr/bin/env python3
"""Fast integration smoke test.

Loads the smoke config, exercises the config system, data layer, and GBM
synthetic generation, then exits with a non-zero status on any failure.

Usage (must be run inside the project .venv):
    source .venv/bin/activate
    python scripts/run_smoke_test.py
    python scripts/run_smoke_test.py --config configs/tests/smoke.yaml

Or without activating:
    .venv/bin/python scripts/run_smoke_test.py
"""

import argparse
import contextlib
import io
import sys
import traceback
from pathlib import Path

# Make sure repo root is on sys.path when run directly.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _check(label: str, fn):
    """Run *fn()* and print pass/fail.  Returns True on success."""
    print(f"  [RUN ] {label}", flush=True)
    try:
        fn()
        print(f"  [PASS] {label}")
        return True
    except Exception:
        print(f"  [FAIL] {label}")
        traceback.print_exc()
        return False


def main(config_path: str) -> int:
    results = []

    # ------------------------------------------------------------------
    # 1. Config loading
    # ------------------------------------------------------------------
    cfg = None

    def _load_config():
        nonlocal cfg
        from src.config import load_config
        cfg = load_config(config_path)
        assert cfg.seed == 0, f"Expected seed=0 for smoke config, got {cfg.seed}"
        assert cfg.env.n_steps == 10
        assert cfg.optim.batch_size == 32

    results.append(_check("Config loads and validates", _load_config))

    # ------------------------------------------------------------------
    # 2. Seeding
    # ------------------------------------------------------------------
    def _seeding():
        from src.utils.seed import set_seed
        set_seed(cfg.seed)

    results.append(_check("Deterministic seeding", _seeding))

    # ------------------------------------------------------------------
    # 3. Synthetic GBM paths
    # ------------------------------------------------------------------
    def _gbm():
        import torch
        from src.data.synthetic import generate_gbm_paths, generate_gbm_returns

        paths = generate_gbm_paths(
            n_paths=4,
            n_steps=cfg.env.n_steps,
            horizon=cfg.env.horizon,
            mu=cfg.env.mu,
            sigma=cfg.env.sigma,
            x0=cfg.env.initial_wealth,
            seed=cfg.seed,
        )
        assert paths.shape == (4, cfg.env.n_steps + 1, len(cfg.env.mu)), \
            f"Unexpected shape: {paths.shape}"
        assert torch.all(paths > 0), "GBM paths must be strictly positive"

        returns = generate_gbm_returns(
            n_paths=4,
            n_steps=cfg.env.n_steps,
            horizon=cfg.env.horizon,
            mu=cfg.env.mu,
            sigma=cfg.env.sigma,
            seed=cfg.seed,
        )
        assert returns.shape == (4, cfg.env.n_steps, len(cfg.env.mu))

        # Determinism check
        paths2 = generate_gbm_paths(
            n_paths=4,
            n_steps=cfg.env.n_steps,
            horizon=cfg.env.horizon,
            mu=cfg.env.mu,
            sigma=cfg.env.sigma,
            x0=cfg.env.initial_wealth,
            seed=cfg.seed,
        )
        assert torch.allclose(paths, paths2), "GBM paths must be deterministic under fixed seed"

    results.append(_check("GBM synthetic paths (shape + determinism)", _gbm))

    # ------------------------------------------------------------------
    # 4. Replay buffer
    # ------------------------------------------------------------------
    def _replay_buffer():
        import torch
        from src.data.replay_buffer import ReplayBuffer
        from src.data.types import Transition

        buf = ReplayBuffer(capacity=cfg.optim.replay_buffer_size)
        assert len(buf) == 0

        n_fill = cfg.optim.batch_size * 2  # ensure enough to sample
        for _ in range(n_fill):
            t = Transition(
                obs=torch.randn(3),
                action=torch.randn(1),
                reward=torch.tensor(0.1),
                next_obs=torch.randn(3),
                done=torch.tensor(0.0),
                time=torch.tensor(0.0),
                next_time=torch.tensor(0.1),
            )
            buf.add(t)

        assert len(buf) == min(n_fill, cfg.optim.replay_buffer_size)
        batch = buf.sample(cfg.optim.batch_size)
        assert batch.obs.shape == (cfg.optim.batch_size, 3)
        assert batch.reward.shape == (cfg.optim.batch_size,)

    results.append(_check("Replay buffer add + sample", _replay_buffer))

    # ------------------------------------------------------------------
    # 5. Episode dataset
    # ------------------------------------------------------------------
    def _dataset():
        import torch
        from src.data.datasets import EpisodeDataset
        from src.data.types import Transition

        n_ep, ep_len = 3, cfg.env.n_steps
        episodes = []
        for _ in range(n_ep):
            ep = [
                Transition(
                    obs=torch.randn(2),
                    action=torch.randn(1),
                    reward=torch.tensor(0.0),
                    next_obs=torch.randn(2),
                    done=torch.tensor(float(i == ep_len - 1)),
                    time=torch.tensor(float(i) / ep_len),
                    next_time=torch.tensor(float(i + 1) / ep_len),
                )
                for i in range(ep_len)
            ]
            episodes.append(ep)

        ds = EpisodeDataset(episodes)
        assert ds.n_episodes == n_ep
        assert ds.n_transitions == n_ep * ep_len

        all_batch = ds.get_all()
        assert all_batch.batch_size == n_ep * ep_len

    results.append(_check("EpisodeDataset construction + batch", _dataset))

    # ------------------------------------------------------------------
    # 6. GBM environment
    # ------------------------------------------------------------------
    def _gbm_env():
        import torch
        from src.envs.gbm_env import GBMPortfolioEnv

        env = GBMPortfolioEnv(cfg.env)
        obs, info = env.reset(seed=cfg.seed)
        assert obs.shape == (env.obs_dim,)
        assert info["time"].item() == 0.0

        all_done = False
        for _ in range(cfg.env.n_steps):
            action = torch.zeros(env.action_dim)  # risk-free only
            step = env.step(action)
            all_done = step.done

        assert all_done, "Episode should end after n_steps"
        # Risk-free-only: final wealth ≈ initial * exp(r * T)
        import math
        r, T = cfg.env.assets.risk_free_rate, cfg.env.horizon
        expected = cfg.env.initial_wealth * math.exp(r * T)
        assert abs(step.wealth.item() - expected) < 1e-4, (
            f"Risk-free wealth mismatch: got {step.wealth.item():.6f}, "
            f"expected {expected:.6f}"
        )

    results.append(_check("GBM environment step + risk-free sanity", _gbm_env))

    # ------------------------------------------------------------------
    # 7. CTRL trajectory and deterministic evaluation
    # ------------------------------------------------------------------
    def _ctrl():
        import torch
        from src.algos.ctrl import collect_ctrl_trajectory, evaluate_ctrl_deterministic
        from src.envs.gbm_env import GBMPortfolioEnv
        from src.models.gaussian_actor import GaussianActor

        env = GBMPortfolioEnv(cfg.env)
        n_risky = cfg.env.assets.n_risky
        actor = GaussianActor(n_risky=n_risky, horizon=cfg.env.horizon)
        n = cfg.env.n_steps
        w = cfg.env.initial_wealth

        # Stochastic behavior-policy rollout
        traj = collect_ctrl_trajectory(actor, env, w=w, seed=cfg.seed)
        assert traj.times.shape == (n,), f"times shape: {traj.times.shape}"
        assert traj.wealth_path.shape == (n + 1,), f"wealth_path shape: {traj.wealth_path.shape}"
        assert traj.actions.shape == (n, n_risky), f"actions shape: {traj.actions.shape}"
        assert traj.log_probs.shape == (n,), f"log_probs shape: {traj.log_probs.shape}"
        assert traj.entropy_terms.shape == (n,), f"entropy_terms shape: {traj.entropy_terms.shape}"
        assert torch.isfinite(traj.log_probs).all(), "log_probs contain non-finite values"
        assert torch.isfinite(traj.entropy_terms).all(), "entropy_terms contain non-finite values"
        assert abs(traj.terminal_wealth.item() - traj.wealth_path[-1].item()) < 1e-6, (
            f"terminal_wealth mismatch: {traj.terminal_wealth.item()} vs {traj.wealth_path[-1].item()}"
        )

        # Deterministic execution-policy evaluation
        result = evaluate_ctrl_deterministic(actor, env, w=w, seed=cfg.seed)
        assert result.times.shape == (n,), f"eval times shape: {result.times.shape}"
        assert result.wealth_path.shape == (n + 1,), f"eval wealth_path shape: {result.wealth_path.shape}"
        assert result.actions.shape == (n, n_risky), f"eval actions shape: {result.actions.shape}"
        assert torch.isfinite(result.wealth_path).all(), "eval wealth_path contains non-finite values"
        assert abs(result.terminal_wealth.item() - result.wealth_path[-1].item()) < 1e-6, (
            f"eval terminal_wealth mismatch: {result.terminal_wealth.item()} vs {result.wealth_path[-1].item()}"
        )

    results.append(_check("CTRL trajectory + deterministic eval", _ctrl))

    # ------------------------------------------------------------------
    # 8. CTRL trainer demo (run_ctrl_demo.py entrypoint)
    # ------------------------------------------------------------------
    def _ctrl_trainer_demo():
        import importlib.util
        demo_path = _REPO_ROOT / "scripts" / "run_ctrl_demo.py"
        spec = importlib.util.spec_from_file_location("run_ctrl_demo", demo_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = module.main()
        assert rc == 0, f"run_ctrl_demo.main() returned non-zero exit code: {rc}"
        out = buf.getvalue()
        for marker in ("--- CTRL demo summary ---", "w_final", "last_terminal_wealth", "history_len"):
            assert marker in out, f"run_ctrl_demo stdout missing marker: {marker!r}"

    results.append(_check("CTRL trainer demo (run_ctrl_demo.py)", _ctrl_trainer_demo))

    # ------------------------------------------------------------------
    # 9. CTRL-vs-oracle demo (run_ctrl_oracle_demo.py entrypoint)
    # ------------------------------------------------------------------
    def _ctrl_oracle_demo():
        import importlib.util
        demo_path = _REPO_ROOT / "scripts" / "run_ctrl_oracle_demo.py"
        spec = importlib.util.spec_from_file_location("run_ctrl_oracle_demo", demo_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = module.main()
        assert rc == 0, f"run_ctrl_oracle_demo.main() returned non-zero exit code: {rc}"
        out = buf.getvalue()
        for marker in (
            "--- CTRL-vs-oracle demo summary ---",
            "post_training_w",
            "ctrl_mean_tw",
            "oracle_mean_tw",
            "mean_tw_delta",
            "ctrl_win_rate",
        ):
            assert marker in out, f"run_ctrl_oracle_demo stdout missing marker: {marker!r}"

    results.append(_check("CTRL-vs-oracle demo (run_ctrl_oracle_demo.py)", _ctrl_oracle_demo))

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    n_pass = sum(results)
    n_total = len(results)
    print(f"\nSmoke test: {n_pass}/{n_total} passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Portfolio RL smoke test")
    parser.add_argument(
        "--config",
        default=str(_REPO_ROOT / "configs" / "tests" / "smoke.yaml"),
        help="Path to smoke config YAML",
    )
    args = parser.parse_args()
    sys.exit(main(args.config))
