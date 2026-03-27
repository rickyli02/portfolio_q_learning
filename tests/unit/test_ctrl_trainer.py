"""Unit tests for src/train/ — Phase 9A single-trajectory step, 9B fixed-length run, 10A w update."""

import math

import pytest
import torch

from src.config.schema import AssetConfig, EnvConfig
from src.envs.gbm_env import GBMPortfolioEnv
from src.models.gaussian_actor import GaussianActor
from src.models.quadratic_critic import QuadraticCritic
from src.train.ctrl_trainer import CTRLStepResult, ctrl_train_step


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_env(n_steps: int = 10) -> GBMPortfolioEnv:
    cfg = EnvConfig(
        horizon=1.0,
        n_steps=n_steps,
        initial_wealth=1.0,
        mu=[0.1],
        sigma=[[0.2]],
        assets=AssetConfig(n_risky=1, include_risk_free=True, risk_free_rate=0.05),
    )
    return GBMPortfolioEnv(cfg)


def _make_actor() -> GaussianActor:
    return GaussianActor(n_risky=1, horizon=1.0)


def _make_critic() -> QuadraticCritic:
    return QuadraticCritic(horizon=1.0, target_return_z=1.0)


def _make_optimizers(
    actor: GaussianActor,
    critic: QuadraticCritic,
    lr: float = 1e-2,
) -> tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
    actor_opt = torch.optim.SGD(actor.parameters(), lr=lr)
    critic_opt = torch.optim.SGD(critic.parameters(), lr=lr)
    return actor_opt, critic_opt


def _run_step(
    seed: int = 0,
    n_steps: int = 10,
    entropy_temp: float = 0.1,
    w: float = 1.0,
) -> tuple[CTRLStepResult, GaussianActor, QuadraticCritic]:
    actor = _make_actor()
    critic = _make_critic()
    env = _make_env(n_steps=n_steps)
    actor_opt, critic_opt = _make_optimizers(actor, critic)
    result = ctrl_train_step(
        actor, critic, env, actor_opt, critic_opt,
        w=w, entropy_temp=entropy_temp, seed=seed,
    )
    return result, actor, critic


# ---------------------------------------------------------------------------
# CTRLStepResult — dataclass structure
# ---------------------------------------------------------------------------

def test_step_returns_dataclass():
    result, _, _ = _run_step()
    assert isinstance(result, CTRLStepResult)


def test_step_critic_loss_is_float():
    result, _, _ = _run_step()
    assert isinstance(result.critic_loss, float)


def test_step_actor_loss_is_float():
    result, _, _ = _run_step()
    assert isinstance(result.actor_loss, float)


def test_step_terminal_wealth_is_float():
    result, _, _ = _run_step()
    assert isinstance(result.terminal_wealth, float)


def test_step_sum_log_prob_is_float():
    result, _, _ = _run_step()
    assert isinstance(result.sum_log_prob, float)


def test_step_mean_entropy_is_float():
    result, _, _ = _run_step()
    assert isinstance(result.mean_entropy, float)


def test_step_n_steps_is_int():
    result, _, _ = _run_step()
    assert isinstance(result.n_steps, int)


# ---------------------------------------------------------------------------
# CTRLStepResult — values
# ---------------------------------------------------------------------------

def test_step_critic_loss_finite():
    result, _, _ = _run_step()
    assert math.isfinite(result.critic_loss)


def test_step_actor_loss_finite():
    result, _, _ = _run_step()
    assert math.isfinite(result.actor_loss)


def test_step_terminal_wealth_finite():
    result, _, _ = _run_step()
    assert math.isfinite(result.terminal_wealth)


def test_step_sum_log_prob_finite():
    result, _, _ = _run_step()
    assert math.isfinite(result.sum_log_prob)


def test_step_mean_entropy_finite():
    result, _, _ = _run_step()
    assert math.isfinite(result.mean_entropy)


def test_step_n_steps_matches_env():
    n_steps = 8
    result, _, _ = _run_step(n_steps=n_steps)
    assert result.n_steps == n_steps


def test_step_terminal_wealth_positive():
    """GBM portfolio wealth should stay positive."""
    result, _, _ = _run_step()
    assert result.terminal_wealth > 0.0


# ---------------------------------------------------------------------------
# dt derived from env (not caller-supplied)
# ---------------------------------------------------------------------------

def test_step_runs_with_nonstandard_n_steps():
    """step must work for any env.n_steps, not just round dt values."""
    result, _, _ = _run_step(n_steps=7)
    assert result.n_steps == 7
    assert isinstance(result.critic_loss, float)


# ---------------------------------------------------------------------------
# Gradient and parameter update
# ---------------------------------------------------------------------------

def test_step_critic_params_have_grad():
    """Critic params must have gradients after one step."""
    actor = _make_actor()
    critic = _make_critic()
    env = _make_env()
    actor_opt, critic_opt = _make_optimizers(actor, critic)
    ctrl_train_step(actor, critic, env, actor_opt, critic_opt, w=1.0, entropy_temp=0.1, seed=0)
    assert critic.theta1.grad is not None


def test_step_actor_params_have_grad():
    """Actor params must have gradients after one step."""
    actor = _make_actor()
    critic = _make_critic()
    env = _make_env()
    actor_opt, critic_opt = _make_optimizers(actor, critic)
    ctrl_train_step(actor, critic, env, actor_opt, critic_opt, w=1.0, entropy_temp=0.1, seed=0)
    assert actor.log_phi1.grad is not None


def test_step_critic_params_updated():
    """Critic parameters must change after one gradient step."""
    actor = _make_actor()
    critic = _make_critic()
    env = _make_env()
    actor_opt, critic_opt = _make_optimizers(actor, critic)
    theta1_before = critic.theta1.item()
    ctrl_train_step(actor, critic, env, actor_opt, critic_opt, w=1.0, entropy_temp=0.1, seed=0)
    assert critic.theta1.item() != theta1_before


def test_step_actor_params_updated():
    """Actor parameters must change after one gradient step."""
    actor = _make_actor()
    critic = _make_critic()
    env = _make_env()
    actor_opt, critic_opt = _make_optimizers(actor, critic)
    phi1_before = actor.log_phi1.clone().detach()
    ctrl_train_step(actor, critic, env, actor_opt, critic_opt, w=1.0, entropy_temp=0.1, seed=0)
    phi1_after = actor.log_phi1.clone().detach()
    assert not torch.allclose(phi1_before, phi1_after)


def test_actor_loss_backward_no_grad_on_critic_params():
    """actor_loss.backward() must leave critic parameters with no gradient.

    This directly verifies the graph-independence claim underlying the
    two-stage backward ordering in ctrl_train_step: the actor and critic
    forward passes in reeval_ctrl_trajectory are independent, so
    actor_loss.backward() must not flow any gradient to critic parameters.
    """
    from src.algos.ctrl import (
        collect_ctrl_trajectory,
        compute_ctrl_actor_loss,
        compute_martingale_residuals,
        evaluate_critic_on_trajectory,
        reeval_ctrl_trajectory,
    )

    actor = _make_actor()
    critic = _make_critic()
    env = _make_env()

    traj = collect_ctrl_trajectory(actor, env, w=1.0, seed=0)
    dt = env.horizon / env.n_steps
    critic_eval = evaluate_critic_on_trajectory(critic, traj, dt=dt)
    residuals = compute_martingale_residuals(critic_eval, traj, entropy_temp=0.1)
    ge = reeval_ctrl_trajectory(actor, critic, traj, dt=dt)

    # Ensure all grads are clear before the actor-only backward pass.
    for p in critic.parameters():
        p.grad = None
    for p in actor.parameters():
        p.grad = None

    actor_loss = compute_ctrl_actor_loss(ge, residuals)
    actor_loss.backward()

    # Critic parameters must be untouched by the actor backward pass.
    assert critic.theta1.grad is None
    assert critic.theta2.grad is None
    assert critic.theta3.grad is None


def test_step_zero_entropy_temp_runs():
    """entropy_temp=0 is a valid edge case (no entropy regularisation)."""
    result, _, _ = _run_step(entropy_temp=0.0)
    assert isinstance(result.critic_loss, float)
    assert isinstance(result.actor_loss, float)


def test_step_two_sequential_steps_differ():
    """Two consecutive steps on the same models should produce different losses."""
    actor = _make_actor()
    critic = _make_critic()
    env = _make_env()
    actor_opt, critic_opt = _make_optimizers(actor, critic)
    r1 = ctrl_train_step(actor, critic, env, actor_opt, critic_opt, w=1.0, entropy_temp=0.1, seed=0)
    r2 = ctrl_train_step(actor, critic, env, actor_opt, critic_opt, w=1.0, entropy_temp=0.1, seed=1)
    # After one parameter update the loss values should generally differ
    assert r1.critic_loss != r2.critic_loss or r1.actor_loss != r2.actor_loss


def test_step_same_seed_same_terminal_wealth():
    """Same seed → same trajectory → same terminal_wealth (params unchanged between steps)."""
    actor = _make_actor()
    critic = _make_critic()
    env = _make_env()
    actor_opt1, critic_opt1 = _make_optimizers(actor, critic)

    actor2 = _make_actor()
    critic2 = _make_critic()
    env2 = _make_env()
    actor_opt2, critic_opt2 = _make_optimizers(actor2, critic2)

    r1 = ctrl_train_step(actor, critic, env, actor_opt1, critic_opt1, w=1.0, entropy_temp=0.1, seed=42)
    r2 = ctrl_train_step(actor2, critic2, env2, actor_opt2, critic_opt2, w=1.0, entropy_temp=0.1, seed=42)
    assert r1.terminal_wealth == pytest.approx(r2.terminal_wealth, abs=1e-5)


# ---------------------------------------------------------------------------
# Public API imports (Phase 9A)
# ---------------------------------------------------------------------------

def test_public_api_imports():
    from src.train import CTRLStepResult, ctrl_train_step
    assert CTRLStepResult is not None
    assert callable(ctrl_train_step)


# ---------------------------------------------------------------------------
# Phase 9B: ctrl_train_run / CTRLRunResult
# ---------------------------------------------------------------------------

from src.train.ctrl_runner import CTRLRunResult, ctrl_train_run


def _run_run(
    n_updates: int = 3,
    base_seed: int | None = 0,
    entropy_temp: float = 0.1,
    w: float = 1.0,
    n_steps: int = 10,
) -> tuple[CTRLRunResult, GaussianActor, QuadraticCritic]:
    actor = _make_actor()
    critic = _make_critic()
    env = _make_env(n_steps=n_steps)
    actor_opt, critic_opt = _make_optimizers(actor, critic)
    result = ctrl_train_run(
        actor, critic, env, actor_opt, critic_opt,
        w=w, entropy_temp=entropy_temp,
        n_updates=n_updates, base_seed=base_seed,
    )
    return result, actor, critic


# --- invalid n_updates ---

def test_run_invalid_n_updates_zero():
    actor = _make_actor()
    critic = _make_critic()
    env = _make_env()
    actor_opt, critic_opt = _make_optimizers(actor, critic)
    with pytest.raises(ValueError, match="n_updates"):
        ctrl_train_run(
            actor, critic, env, actor_opt, critic_opt,
            w=1.0, entropy_temp=0.1, n_updates=0,
        )


def test_run_invalid_n_updates_negative():
    actor = _make_actor()
    critic = _make_critic()
    env = _make_env()
    actor_opt, critic_opt = _make_optimizers(actor, critic)
    with pytest.raises(ValueError, match="n_updates"):
        ctrl_train_run(
            actor, critic, env, actor_opt, critic_opt,
            w=1.0, entropy_temp=0.1, n_updates=-1,
        )


# --- result structure ---

def test_run_returns_dataclass():
    result, _, _ = _run_run()
    assert isinstance(result, CTRLRunResult)


def test_run_history_length():
    n_updates = 5
    result, _, _ = _run_run(n_updates=n_updates)
    assert len(result.steps) == n_updates


def test_run_n_updates_stored():
    n_updates = 4
    result, _, _ = _run_run(n_updates=n_updates)
    assert result.n_updates == n_updates


def test_run_final_step_matches_last():
    result, _, _ = _run_run()
    assert result.final_step is result.steps[-1]


def test_run_steps_are_ctrl_step_results():
    result, _, _ = _run_run()
    for step in result.steps:
        assert isinstance(step, CTRLStepResult)


def test_run_single_update():
    result, _, _ = _run_run(n_updates=1)
    assert result.n_updates == 1
    assert len(result.steps) == 1


# --- finite values ---

def test_run_all_steps_finite():
    result, _, _ = _run_run(n_updates=3)
    for step in result.steps:
        assert math.isfinite(step.critic_loss)
        assert math.isfinite(step.actor_loss)
        assert math.isfinite(step.terminal_wealth)
        assert math.isfinite(step.sum_log_prob)
        assert math.isfinite(step.mean_entropy)


def test_run_n_steps_consistent():
    n_steps = 8
    result, _, _ = _run_run(n_updates=3, n_steps=n_steps)
    for step in result.steps:
        assert step.n_steps == n_steps


# --- reproducibility ---

def test_run_reproducible_with_base_seed():
    """Same base_seed and fresh identical model/env instances → identical scalar history."""
    # Instance A
    actor_a = _make_actor()
    critic_a = _make_critic()
    env_a = _make_env()
    actor_opt_a, critic_opt_a = _make_optimizers(actor_a, critic_a)
    result_a = ctrl_train_run(
        actor_a, critic_a, env_a, actor_opt_a, critic_opt_a,
        w=1.0, entropy_temp=0.1, n_updates=3, base_seed=7,
    )

    # Instance B — same default init, same seed schedule
    actor_b = _make_actor()
    critic_b = _make_critic()
    env_b = _make_env()
    actor_opt_b, critic_opt_b = _make_optimizers(actor_b, critic_b)
    result_b = ctrl_train_run(
        actor_b, critic_b, env_b, actor_opt_b, critic_opt_b,
        w=1.0, entropy_temp=0.1, n_updates=3, base_seed=7,
    )

    for k in range(3):
        assert result_a.steps[k].terminal_wealth == pytest.approx(
            result_b.steps[k].terminal_wealth, abs=1e-5
        )
        assert result_a.steps[k].critic_loss == pytest.approx(
            result_b.steps[k].critic_loss, abs=1e-5
        )


def test_run_no_base_seed_runs():
    """base_seed=None is valid; produces a stochastic but complete run."""
    result, _, _ = _run_run(base_seed=None, n_updates=2)
    assert len(result.steps) == 2
    assert math.isfinite(result.final_step.critic_loss)


# --- parameter updates ---

def test_run_critic_params_change_across_run():
    """Critic parameters must differ between start and end of a multi-step run."""
    actor = _make_actor()
    critic = _make_critic()
    env = _make_env()
    actor_opt, critic_opt = _make_optimizers(actor, critic)
    theta1_before = critic.theta1.item()
    ctrl_train_run(
        actor, critic, env, actor_opt, critic_opt,
        w=1.0, entropy_temp=0.1, n_updates=3, base_seed=0,
    )
    assert critic.theta1.item() != theta1_before


def test_run_actor_params_change_across_run():
    """Actor parameters must differ between start and end of a multi-step run."""
    actor = _make_actor()
    critic = _make_critic()
    env = _make_env()
    actor_opt, critic_opt = _make_optimizers(actor, critic)
    phi1_before = actor.log_phi1.clone().detach()
    ctrl_train_run(
        actor, critic, env, actor_opt, critic_opt,
        w=1.0, entropy_temp=0.1, n_updates=3, base_seed=0,
    )
    phi1_after = actor.log_phi1.clone().detach()
    assert not torch.allclose(phi1_before, phi1_after)


# --- public API (Phase 9B) ---

def test_phase9b_public_api_imports():
    from src.train import CTRLRunResult, ctrl_train_run
    assert CTRLRunResult is not None
    assert callable(ctrl_train_run)


# ---------------------------------------------------------------------------
# Phase 10A: ctrl_w_update / CTRLWUpdateResult
# ---------------------------------------------------------------------------

from src.train.w_update import CTRLWUpdateResult, ctrl_w_update


# --- validation ---

def test_w_update_zero_step_size_raises():
    with pytest.raises(ValueError, match="step_size"):
        ctrl_w_update(w=1.0, terminal_wealth=1.1, target_return_z=1.0, step_size=0.0)


def test_w_update_negative_step_size_raises():
    with pytest.raises(ValueError, match="step_size"):
        ctrl_w_update(w=1.0, terminal_wealth=1.1, target_return_z=1.0, step_size=-0.1)


def test_w_update_invalid_bounds_order_raises():
    """w_min > w_max is not a valid closed interval and must be rejected."""
    with pytest.raises(ValueError, match="w_min"):
        ctrl_w_update(
            w=1.0, terminal_wealth=1.1, target_return_z=1.0,
            step_size=0.1, w_min=1.2, w_max=0.8,
        )


# --- result structure ---

def test_w_update_returns_dataclass():
    result = ctrl_w_update(w=1.0, terminal_wealth=1.1, target_return_z=1.0, step_size=0.1)
    assert isinstance(result, CTRLWUpdateResult)


def test_w_update_w_prev_stored():
    result = ctrl_w_update(w=1.05, terminal_wealth=1.1, target_return_z=1.0, step_size=0.1)
    assert result.w_prev == pytest.approx(1.05)


# --- formula correctness (no bounds) ---

def test_w_update_signal_formula():
    """signal = x_T - z."""
    x_t, z = 1.12, 1.05
    result = ctrl_w_update(w=1.0, terminal_wealth=x_t, target_return_z=z, step_size=0.1)
    assert result.signal == pytest.approx(x_t - z)


def test_w_update_no_bounds_formula():
    """w_next = w - step_size * (x_T - z) when no projection bounds are supplied."""
    w, x_t, z, a = 1.0, 1.1, 1.0, 0.2
    result = ctrl_w_update(w=w, terminal_wealth=x_t, target_return_z=z, step_size=a)
    expected = w - a * (x_t - z)
    assert result.w_next_raw == pytest.approx(expected)
    assert result.w_next == pytest.approx(expected)


def test_w_update_no_bounds_w_next_equals_raw():
    """Without bounds, w_next must equal w_next_raw."""
    result = ctrl_w_update(w=1.0, terminal_wealth=1.2, target_return_z=1.0, step_size=0.5)
    assert result.w_next == pytest.approx(result.w_next_raw)


# --- projection ---

def test_w_update_projection_lower_clamp():
    """w_next must be clamped to w_min when unprojected value falls below it."""
    # w=1.0, x_T=0.5, z=1.0 → signal=-0.5 → w_next_raw = 1.0 - 0.1*(-0.5) = 1.05
    # set w_max=0.9 so raw=1.05 exceeds w_max... let's make it simpler:
    # w=1.0, x_T=1.5, z=1.0 → signal=0.5 → w_next_raw=1.0-1.0*0.5=0.5, clamp to w_min=0.8
    result = ctrl_w_update(
        w=1.0, terminal_wealth=1.5, target_return_z=1.0,
        step_size=1.0, w_min=0.8,
    )
    assert result.w_next_raw == pytest.approx(0.5)
    assert result.w_next == pytest.approx(0.8)


def test_w_update_projection_upper_clamp():
    """w_next must be clamped to w_max when unprojected value exceeds it."""
    # w=1.0, x_T=0.5, z=1.0 → signal=-0.5 → w_next_raw=1.0-1.0*(-0.5)=1.5, clamp to w_max=1.2
    result = ctrl_w_update(
        w=1.0, terminal_wealth=0.5, target_return_z=1.0,
        step_size=1.0, w_max=1.2,
    )
    assert result.w_next_raw == pytest.approx(1.5)
    assert result.w_next == pytest.approx(1.2)


def test_w_update_no_clamp_when_inside_bounds():
    """No projection should occur when w_next_raw is already inside [w_min, w_max]."""
    # w=1.0, x_T=1.05, z=1.0 → signal=0.05 → w_next_raw=1.0-0.1*0.05=0.995
    result = ctrl_w_update(
        w=1.0, terminal_wealth=1.05, target_return_z=1.0,
        step_size=0.1, w_min=0.9, w_max=1.1,
    )
    assert result.w_next == pytest.approx(result.w_next_raw)


def test_w_update_both_bounds_clamp_lower():
    result = ctrl_w_update(
        w=1.0, terminal_wealth=2.0, target_return_z=1.0,
        step_size=2.0, w_min=0.5, w_max=1.5,
    )
    # signal=1.0, w_next_raw=1.0-2.0=-1.0 → clamp to 0.5
    assert result.w_next == pytest.approx(0.5)


def test_w_update_both_bounds_clamp_upper():
    result = ctrl_w_update(
        w=1.0, terminal_wealth=0.0, target_return_z=1.0,
        step_size=2.0, w_min=0.5, w_max=1.5,
    )
    # signal=-1.0, w_next_raw=1.0+2.0=3.0 → clamp to 1.5
    assert result.w_next == pytest.approx(1.5)


# --- integration with trainer run result ---

def test_w_update_with_run_result():
    """ctrl_w_update should accept terminal_wealth from a CTRLRunResult."""
    actor = _make_actor()
    critic = _make_critic()
    env = _make_env()
    actor_opt, critic_opt = _make_optimizers(actor, critic)
    run_result = ctrl_train_run(
        actor, critic, env, actor_opt, critic_opt,
        w=1.0, entropy_temp=0.1, n_updates=2, base_seed=0,
    )
    x_t = run_result.final_step.terminal_wealth
    w_result = ctrl_w_update(
        w=1.0, terminal_wealth=x_t,
        target_return_z=1.0, step_size=0.1,
    )
    assert isinstance(w_result, CTRLWUpdateResult)
    assert math.isfinite(w_result.w_next)
    assert w_result.signal == pytest.approx(x_t - 1.0)


# --- public API (Phase 10A) ---

def test_phase10a_public_api_imports():
    from src.train import CTRLWUpdateResult, ctrl_w_update
    assert CTRLWUpdateResult is not None
    assert callable(ctrl_w_update)
