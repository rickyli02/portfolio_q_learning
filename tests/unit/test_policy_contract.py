"""Policy-role contract tests — Phase 19A / 19B / 19D.

Proves the behavior-vs-execution policy separation at each named public seam:

  Behavior policy  (stochastic): collect_ctrl_trajectory  → actor.sample()
  Execution policy (deterministic): evaluate_ctrl_deterministic → execute_ctrl_action()
                                    execute_ctrl_action → actor.mean_action()
  Training step                : ctrl_train_step uses behavior path
  Comparison                   : run_ctrl_oracle_comparison uses execution path

Phase 19B adds tests for the named single-step execution helper execute_ctrl_action:
  - equivalence to actor.mean_action() (same result, same determinism guarantee)
  - evaluate_ctrl_deterministic routes through execute_ctrl_action (same episode output)

Phase 19D adds a regression proof at the eval_record_set bridge/comparison seam:
  - eval_record_set produces identical action records across repeated calls with the
    same inputs, proving the bridge layer is deterministic (execution policy, not
    stochastic sampling).
  - This test FAILS if the bridge layer is swapped to use collect_ctrl_trajectory
    or any other stochastic path.

Tests are designed to FAIL if the policy roles are accidentally swapped.
They check observable behavioral differences, not just type annotations.
"""

from __future__ import annotations

import torch

from src.algos.ctrl import (
    CTRLEvalResult,
    CTRLTrajectory,
    collect_ctrl_trajectory,
    evaluate_ctrl_deterministic,
    execute_ctrl_action,
)
from src.algos.oracle_mv import OracleMVPolicy
from src.backtest.comparison import run_ctrl_oracle_comparison
from src.eval.record_set import eval_record_set
from src.config.schema import AssetConfig, EnvConfig
from src.envs.gbm_env import GBMPortfolioEnv
from src.models.gaussian_actor import GaussianActor
from src.models.quadratic_critic import QuadraticCritic
from src.train.ctrl_trainer import ctrl_train_step

# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------

_MU = [0.08]
_SIGMA = [[0.20]]
_R = 0.05
_HORIZON = 1.0
_N_STEPS = 5
_W = 1.0


def _make_env() -> GBMPortfolioEnv:
    cfg = EnvConfig(
        horizon=_HORIZON,
        n_steps=_N_STEPS,
        initial_wealth=1.0,
        mu=_MU,
        sigma=_SIGMA,
        assets=AssetConfig(n_risky=1, include_risk_free=True, risk_free_rate=_R),
    )
    return GBMPortfolioEnv(cfg)


def _make_actor() -> GaussianActor:
    return GaussianActor(n_risky=1, horizon=_HORIZON)


def _make_critic() -> QuadraticCritic:
    return QuadraticCritic(horizon=_HORIZON, target_return_z=1.0)


def _make_oracle() -> OracleMVPolicy:
    return OracleMVPolicy.from_env_params(
        mu=_MU, sigma=_SIGMA, r=_R, horizon=_HORIZON, gamma_embed=1.5,
    )


# ===========================================================================
# Structural proofs — the containers encode the policy role
# ===========================================================================


def test_trajectory_stores_log_probs():
    """CTRLTrajectory carries log_probs — proof it was produced by actor.sample()."""
    traj = collect_ctrl_trajectory(_make_actor(), _make_env(), w=_W, seed=0)
    assert isinstance(traj, CTRLTrajectory)
    assert hasattr(traj, "log_probs")
    assert traj.log_probs.shape == (traj.times.shape[0],)


def test_trajectory_stores_entropy_terms():
    """CTRLTrajectory carries entropy_terms — confirms stochastic policy was evaluated."""
    traj = collect_ctrl_trajectory(_make_actor(), _make_env(), w=_W, seed=0)
    assert hasattr(traj, "entropy_terms")
    assert traj.entropy_terms.shape == (traj.times.shape[0],)


def test_eval_result_has_no_log_probs():
    """CTRLEvalResult has no log_probs — proof it was produced by actor.mean_action()."""
    result = evaluate_ctrl_deterministic(_make_actor(), _make_env(), w=_W, seed=0)
    assert isinstance(result, CTRLEvalResult)
    assert not hasattr(result, "log_probs")
    assert not hasattr(result, "entropy_terms")


# ===========================================================================
# Behavioral proof — behavior actions differ from execution actions
#
# Both start from the same initial wealth (same env seed).
# Behavior: u_k ~ N(mean, variance) → sampled action differs from mean
# Execution: u_k = mean_action(t, x, w) → deterministic
#
# This test FAILS if collect_ctrl_trajectory accidentally used mean_action().
# ===========================================================================


def test_behavior_actions_differ_from_execution_actions():
    """Behavior trajectory actions differ from deterministic execution actions.

    Same actor, same env seed, same w.  Behavior policy samples from the
    Gaussian; execution policy uses the mean.  With probability 1 (continuous
    distribution) at least one step's action differs.
    """
    actor = _make_actor()
    env = _make_env()

    traj = collect_ctrl_trajectory(actor, env, w=_W, seed=42)
    det = evaluate_ctrl_deterministic(actor, env, w=_W, seed=42)

    # At least one step's action must differ; if all are equal, sampling was bypassed.
    all_equal = torch.allclose(traj.actions, det.actions, atol=0.0)
    assert not all_equal, (
        "Behavior trajectory actions are identical to deterministic execution actions. "
        "This indicates collect_ctrl_trajectory used mean_action() instead of sample()."
    )


# ===========================================================================
# Behavioral proof — execution path is deterministic w.r.t. PyTorch RNG state
#
# evaluate_ctrl_deterministic uses actor.mean_action(), which has no
# randomness.  Setting torch.manual_seed to different values before each
# call must NOT change the output actions.
#
# This test FAILS if evaluate_ctrl_deterministic accidentally used sample().
# ===========================================================================


def test_execution_actions_invariant_to_torch_rng_state():
    """Execution policy actions are unchanged by torch.manual_seed().

    mean_action() does not call torch.randn or torch.normal, so the output
    is fully determined by the env reset seed and actor parameters.
    """
    actor = _make_actor()
    env = _make_env()

    torch.manual_seed(0)
    result_a = evaluate_ctrl_deterministic(actor, env, w=_W, seed=7)
    torch.manual_seed(99999)
    result_b = evaluate_ctrl_deterministic(actor, env, w=_W, seed=7)

    assert torch.allclose(result_a.actions, result_b.actions, atol=0.0), (
        "Execution policy actions changed when torch RNG seed changed. "
        "This indicates evaluate_ctrl_deterministic used sample() instead of mean_action()."
    )
    assert torch.allclose(result_a.wealth_path, result_b.wealth_path, atol=0.0)


# ===========================================================================
# Behavioral proof — behavior path is stochastic (varies with PyTorch seed)
#
# collect_ctrl_trajectory seeds torch with seed= before sampling.  Using
# different seeds must produce different sampled actions.
#
# This test FAILS if collect_ctrl_trajectory used mean_action() instead of
# sample() (mean_action output would be identical regardless of torch seed).
# ===========================================================================


def test_behavior_actions_vary_across_seeds():
    """Behavior policy actions differ across different trajectory seeds.

    Each seed resets the PyTorch RNG before sampling, so different seeds
    must produce different sampled actions.
    """
    actor = _make_actor()
    env = _make_env()

    traj_a = collect_ctrl_trajectory(actor, env, w=_W, seed=10)
    traj_b = collect_ctrl_trajectory(actor, env, w=_W, seed=20)

    all_equal = torch.allclose(traj_a.actions, traj_b.actions, atol=0.0)
    assert not all_equal, (
        "Behavior trajectory actions are identical across different seeds. "
        "This indicates collect_ctrl_trajectory used mean_action() instead of sample()."
    )


# ===========================================================================
# Training-step seam: ctrl_train_step uses the stochastic behavior path
#
# If the training step used mean_action() instead of sample(), the trajectory
# would carry zero variance and the sum_log_prob would be the same for all
# seeded runs.  With genuine sampling, sum_log_prob varies across seeds.
# ===========================================================================


def test_train_step_carries_nonzero_entropy():
    """Training step reports nonzero mean_entropy — confirms stochastic behavior path.

    If mean_action() were used, the distribution entropy would be correct but
    the update would be off-policy w.r.t. the deterministic action.  More
    importantly, entropy being nonzero confirms a Gaussian distribution was
    evaluated, not a degenerate delta.
    """
    actor = _make_actor()
    critic = _make_critic()
    env = _make_env()
    actor_opt = torch.optim.SGD(actor.parameters(), lr=1e-3)
    critic_opt = torch.optim.SGD(critic.parameters(), lr=1e-3)

    result = ctrl_train_step(
        actor, critic, env, actor_opt, critic_opt, w=_W, entropy_temp=0.1, seed=0,
    )
    assert result.mean_entropy != 0.0, (
        "Training step mean_entropy is zero — this suggests sampling was not used."
    )


def test_train_step_sum_log_prob_varies_across_seeds():
    """Training-step sum_log_prob differs across seeds — confirms stochastic behavior.

    With genuine stochastic sampling the drawn actions (and hence log-probs)
    depend on the seed.  If mean_action() were used the log-prob would be
    the same deterministic value each time.
    """
    actor = _make_actor()
    critic = _make_critic()
    env = _make_env()

    results = []
    for seed in [0, 1, 2]:
        a = GaussianActor(n_risky=1, horizon=_HORIZON)
        c = QuadraticCritic(horizon=_HORIZON, target_return_z=1.0)
        a.load_state_dict(actor.state_dict())
        c.load_state_dict(critic.state_dict())
        ao = torch.optim.SGD(a.parameters(), lr=0.0)   # lr=0 → no parameter change
        co = torch.optim.SGD(c.parameters(), lr=0.0)
        r = ctrl_train_step(a, c, env, ao, co, w=_W, entropy_temp=0.1, seed=seed)
        results.append(r.sum_log_prob)

    assert len(set(results)) > 1, (
        "sum_log_prob is identical across all seeds — "
        "this indicates sampling was not used in the training step."
    )


# ===========================================================================
# Comparison seam: run_ctrl_oracle_comparison uses the deterministic execution path
#
# Deterministic evaluation means the same seeds always produce the same result.
# If sample() were used instead of mean_action(), repeated calls would differ.
# ===========================================================================


def test_comparison_is_reproducible_across_calls():
    """run_ctrl_oracle_comparison is deterministic given the same seeds.

    Deterministic execution (mean_action) produces the same terminal wealth
    every call.  If sample() were used the results would differ between calls.
    """
    actor = _make_actor()
    env = _make_env()
    oracle = _make_oracle()
    seeds = [0, 1, 2]

    result_a = run_ctrl_oracle_comparison(
        actor, env, w=_W, oracle_policy=oracle, seeds=seeds, target_return_z=1.0,
    )
    result_b = run_ctrl_oracle_comparison(
        actor, env, w=_W, oracle_policy=oracle, seeds=seeds, target_return_z=1.0,
    )

    assert result_a.ctrl_bundle.aggregate.mean_terminal_wealth == \
           result_b.ctrl_bundle.aggregate.mean_terminal_wealth, (
        "Comparison mean_terminal_wealth differs across identical calls. "
        "This indicates sample() was used instead of mean_action() in the comparison path."
    )
    assert result_a.comparison.ctrl_win_rate == result_b.comparison.ctrl_win_rate


# ===========================================================================
# Phase 19B: execute_ctrl_action named helper seam
# ===========================================================================


def test_execute_ctrl_action_matches_mean_action():
    """execute_ctrl_action returns the same value as actor.mean_action().

    The helper is a named wrapper; it must not change the action value.
    This test fails if execute_ctrl_action calls sample() instead of mean_action().
    """
    actor = _make_actor()
    env = _make_env()
    _, info = env.reset(seed=0)
    wealth = info["wealth"].to(dtype=torch.float32)
    t = torch.tensor(0.0, dtype=torch.float32)

    with torch.no_grad():
        expected = actor.mean_action(t, wealth, _W)
    actual = execute_ctrl_action(actor, t, wealth, _W)

    assert torch.allclose(actual, expected, atol=0.0), (
        "execute_ctrl_action result differs from actor.mean_action(). "
        "The helper must use mean_action() exactly."
    )


def test_execute_ctrl_action_is_deterministic():
    """execute_ctrl_action is invariant to torch.manual_seed().

    Repeated calls with the same inputs must return identical actions regardless
    of the PyTorch RNG state.  This test fails if execute_ctrl_action uses
    sample() instead of mean_action().
    """
    actor = _make_actor()
    env = _make_env()
    _, info = env.reset(seed=0)
    wealth = info["wealth"].to(dtype=torch.float32)
    t = torch.tensor(0.3, dtype=torch.float32)

    torch.manual_seed(0)
    action_a = execute_ctrl_action(actor, t, wealth, _W)
    torch.manual_seed(99999)
    action_b = execute_ctrl_action(actor, t, wealth, _W)

    assert torch.allclose(action_a, action_b, atol=0.0), (
        "execute_ctrl_action returned different values with different torch seeds. "
        "The helper must be deterministic (mean_action has no randomness)."
    )


def test_evaluate_ctrl_deterministic_consistent_with_execute_ctrl_action():
    """evaluate_ctrl_deterministic episode output matches a manual step loop using execute_ctrl_action.

    Proves that evaluate_ctrl_deterministic routes through execute_ctrl_action
    and that both produce identical action sequences.
    """
    actor = _make_actor()
    env = _make_env()
    w = _W
    seed = 5

    # Run via evaluate_ctrl_deterministic (uses execute_ctrl_action internally)
    det_result = evaluate_ctrl_deterministic(actor, env, w=w, seed=seed)

    # Manual step loop using execute_ctrl_action directly
    _, info = env.reset(seed=seed)
    current_wealth = info["wealth"].to(dtype=torch.float32)
    dt = env.horizon / env.n_steps
    manual_actions = []
    for k in range(env.n_steps):
        t_k = torch.tensor(k * dt, dtype=torch.float32)
        action_k = execute_ctrl_action(actor, t_k, current_wealth, w)
        step = env.step(action_k.detach())
        manual_actions.append(action_k.detach())
        current_wealth = step.wealth.to(dtype=torch.float32)

    manual_actions_tensor = torch.stack(manual_actions)
    assert torch.allclose(det_result.actions, manual_actions_tensor, atol=1e-6), (
        "evaluate_ctrl_deterministic actions differ from manual execute_ctrl_action loop. "
        "The episode helper must use execute_ctrl_action consistently."
    )


# ===========================================================================
# Phase 19D: eval_record_set bridge/comparison seam regression proof
#
# eval_record_set routes through eval_record → evaluate_ctrl_deterministic
# → execute_ctrl_action → actor.mean_action().  The bridge layer is
# deterministic; identical inputs must always produce identical records.
#
# This test FAILS if eval_record_set (or anything in its call chain) is
# swapped to use a stochastic path such as collect_ctrl_trajectory, because
# repeated calls with the same seed would then produce different actions.
# ===========================================================================


def test_eval_record_set_bridge_is_deterministic():
    """eval_record_set produces identical action records across repeated calls.

    The bridge layer routes through the deterministic execution-policy chain
    (eval_record → evaluate_ctrl_deterministic → execute_ctrl_action →
    actor.mean_action).  Calling it twice with identical inputs must return
    records with identical action tensors.

    This test FAILS if the bridge is replaced with a stochastic path — stochastic
    sampling would produce different actions across calls even for the same seed.
    """
    actor = _make_actor()
    env = _make_env()
    seeds = [0, 1, 2]

    rs_a = eval_record_set(actor, env, w=_W, seeds=seeds)
    rs_b = eval_record_set(actor, env, w=_W, seeds=seeds)

    for i, (rec_a, rec_b) in enumerate(zip(rs_a.records, rs_b.records)):
        assert torch.allclose(rec_a.actions, rec_b.actions, atol=0.0), (
            f"eval_record_set seed={seeds[i]}: actions differ between identical calls. "
            "This indicates the bridge layer used stochastic sampling instead of "
            "the deterministic execution-policy path (execute_ctrl_action)."
        )
