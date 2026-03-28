"""Unit tests for src/train/ — Phases 9A/9B (step/run), 10A (w update), 10B (outer iter), 10C (outer loop), 11A/11B (stateful shell + validation)."""

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


# ---------------------------------------------------------------------------
# Phase 10B: ctrl_outer_iter / CTRLOuterIterResult
# ---------------------------------------------------------------------------

from src.train.ctrl_outer_iter import CTRLOuterIterResult, ctrl_outer_iter


def _run_outer_iter(
    w: float = 1.0,
    target_return_z: float = 1.0,
    w_step_size: float = 0.1,
    n_updates: int = 3,
    entropy_temp: float = 0.1,
    base_seed: int | None = 0,
    w_min: float | None = None,
    w_max: float | None = None,
) -> CTRLOuterIterResult:
    actor = _make_actor()
    critic = _make_critic()
    env = _make_env()
    actor_opt, critic_opt = _make_optimizers(actor, critic)
    return ctrl_outer_iter(
        actor, critic, env, actor_opt, critic_opt,
        w=w, target_return_z=target_return_z,
        w_step_size=w_step_size, n_updates=n_updates,
        entropy_temp=entropy_temp, base_seed=base_seed,
        w_min=w_min, w_max=w_max,
    )


# --- result structure ---

def test_outer_iter_returns_dataclass():
    result = _run_outer_iter()
    assert isinstance(result, CTRLOuterIterResult)


def test_outer_iter_run_result_is_ctrl_run_result():
    result = _run_outer_iter()
    assert isinstance(result.run_result, CTRLRunResult)


def test_outer_iter_w_update_result_is_ctrl_w_update_result():
    result = _run_outer_iter()
    assert isinstance(result.w_update_result, CTRLWUpdateResult)


def test_outer_iter_w_prev_matches_input():
    w_in = 1.05
    result = _run_outer_iter(w=w_in)
    assert result.w_prev == pytest.approx(w_in)


def test_outer_iter_w_next_matches_w_update_result():
    """w_next must equal w_update_result.w_next."""
    result = _run_outer_iter()
    assert result.w_next == pytest.approx(result.w_update_result.w_next)


def test_outer_iter_n_updates_forwarded():
    n_updates = 4
    result = _run_outer_iter(n_updates=n_updates)
    assert result.run_result.n_updates == n_updates


# --- integration: w-update uses final_step.terminal_wealth ---

def test_outer_iter_w_update_uses_final_terminal_wealth():
    """w_update_result.signal must equal final_step.terminal_wealth - z."""
    z = 1.0
    result = _run_outer_iter(target_return_z=z)
    x_t = result.run_result.final_step.terminal_wealth
    assert result.w_update_result.signal == pytest.approx(x_t - z)


def test_outer_iter_w_update_formula():
    """w_next_raw must equal w - w_step_size * (x_T - z)."""
    w, z, a = 1.0, 1.0, 0.1
    result = _run_outer_iter(w=w, target_return_z=z, w_step_size=a)
    x_t = result.run_result.final_step.terminal_wealth
    expected_raw = w - a * (x_t - z)
    assert result.w_update_result.w_next_raw == pytest.approx(expected_raw)


# --- reproducibility ---

def test_outer_iter_reproducible_with_base_seed():
    """Fresh identical instances + same base_seed → same w_next."""
    def _fresh_run(seed: int) -> CTRLOuterIterResult:
        actor = _make_actor()
        critic = _make_critic()
        env = _make_env()
        actor_opt, critic_opt = _make_optimizers(actor, critic)
        return ctrl_outer_iter(
            actor, critic, env, actor_opt, critic_opt,
            w=1.0, target_return_z=1.0, w_step_size=0.1,
            n_updates=2, entropy_temp=0.1, base_seed=seed,
        )

    r1 = _fresh_run(seed=5)
    r2 = _fresh_run(seed=5)
    assert r1.w_next == pytest.approx(r2.w_next)
    assert r1.run_result.final_step.terminal_wealth == pytest.approx(
        r2.run_result.final_step.terminal_wealth, abs=1e-5
    )


# --- projection path ---

def test_outer_iter_projection_lower_clamp():
    """w_min projection must propagate through the combined helper."""
    # Use large w_step_size to force w_next_raw well below w_min.
    result = _run_outer_iter(
        w=1.0, target_return_z=1.0, w_step_size=50.0,
        w_min=0.5,
    )
    # If raw < 0.5 it gets clamped; if raw >= 0.5 the clamp does nothing.
    assert result.w_next >= 0.5


def test_outer_iter_projection_upper_clamp():
    """w_max projection must propagate through the combined helper."""
    result = _run_outer_iter(
        w=1.0, target_return_z=1.0, w_step_size=50.0,
        w_max=1.5,
    )
    assert result.w_next <= 1.5


def test_outer_iter_no_projection_when_unbounded():
    """Without bounds w_next must equal w_next_raw."""
    result = _run_outer_iter(w_min=None, w_max=None)
    assert result.w_next == pytest.approx(result.w_update_result.w_next_raw)


# --- public API (Phase 10B) ---

def test_phase10b_public_api_imports():
    from src.train import CTRLOuterIterResult, ctrl_outer_iter
    assert CTRLOuterIterResult is not None
    assert callable(ctrl_outer_iter)


# ---------------------------------------------------------------------------
# Phase 10C: ctrl_outer_loop / CTRLOuterLoopResult
# ---------------------------------------------------------------------------

from src.train.ctrl_outer_loop import CTRLOuterLoopResult, ctrl_outer_loop


def _make_outer_loop_args(
    n_outer_iters: int = 2,
    n_updates: int = 3,
    w_init: float = 1.0,
    target_return_z: float = 1.0,
    w_step_size: float = 0.1,
    entropy_temp: float = 0.1,
    base_seed: int | None = 0,
    w_min: float | None = None,
    w_max: float | None = None,
) -> dict:
    actor = _make_actor()
    critic = _make_critic()
    env = _make_env()
    actor_opt, critic_opt = _make_optimizers(actor, critic)
    return dict(
        actor=actor, critic=critic, env=env,
        actor_optimizer=actor_opt, critic_optimizer=critic_opt,
        w_init=w_init, target_return_z=target_return_z,
        w_step_size=w_step_size, n_outer_iters=n_outer_iters,
        n_updates=n_updates, entropy_temp=entropy_temp,
        base_seed=base_seed, w_min=w_min, w_max=w_max,
    )


def _run_outer_loop(**kwargs) -> CTRLOuterLoopResult:
    args = _make_outer_loop_args(**kwargs)
    return ctrl_outer_loop(**args)


# --- invalid n_outer_iters ---

def test_outer_loop_invalid_zero_raises():
    args = _make_outer_loop_args()
    args["n_outer_iters"] = 0
    with pytest.raises(ValueError, match="n_outer_iters"):
        ctrl_outer_loop(**args)


def test_outer_loop_invalid_negative_raises():
    args = _make_outer_loop_args()
    args["n_outer_iters"] = -1
    with pytest.raises(ValueError, match="n_outer_iters"):
        ctrl_outer_loop(**args)


# --- result structure ---

def test_outer_loop_returns_dataclass():
    result = _run_outer_loop()
    assert isinstance(result, CTRLOuterLoopResult)


def test_outer_loop_iters_length():
    n = 3
    result = _run_outer_loop(n_outer_iters=n)
    assert len(result.iters) == n


def test_outer_loop_n_outer_iters_stored():
    n = 3
    result = _run_outer_loop(n_outer_iters=n)
    assert result.n_outer_iters == n


def test_outer_loop_final_iter_is_last():
    result = _run_outer_loop(n_outer_iters=2)
    assert result.final_iter is result.iters[-1]


def test_outer_loop_w_init_stored():
    result = _run_outer_loop(w_init=1.1)
    assert result.w_init == pytest.approx(1.1)


def test_outer_loop_w_final_matches_last_w_next():
    result = _run_outer_loop(n_outer_iters=2)
    assert result.w_final == pytest.approx(result.iters[-1].w_next)


def test_outer_loop_single_iter():
    result = _run_outer_loop(n_outer_iters=1)
    assert result.n_outer_iters == 1
    assert len(result.iters) == 1


def test_outer_loop_iters_are_outer_iter_results():
    result = _run_outer_loop(n_outer_iters=2)
    for it in result.iters:
        assert isinstance(it, CTRLOuterIterResult)


# --- w threading ---

def test_outer_loop_w_threaded_across_iters():
    """Each iteration's w_prev must equal the previous iteration's w_next."""
    result = _run_outer_loop(n_outer_iters=3)
    assert result.iters[0].w_prev == pytest.approx(result.w_init)
    for j in range(1, 3):
        assert result.iters[j].w_prev == pytest.approx(result.iters[j - 1].w_next)


def test_outer_loop_w_first_iter_prev_is_w_init():
    w_init = 1.07
    result = _run_outer_loop(w_init=w_init, n_outer_iters=2)
    assert result.iters[0].w_prev == pytest.approx(w_init)


# --- reproducibility ---

def test_outer_loop_reproducible_with_base_seed():
    """Fresh identical instances + same base_seed → same w evolution."""
    def _fresh(seed: int) -> CTRLOuterLoopResult:
        return ctrl_outer_loop(**_make_outer_loop_args(
            n_outer_iters=2, n_updates=2, base_seed=seed,
        ))

    r1 = _fresh(seed=11)
    r2 = _fresh(seed=11)
    assert r1.w_final == pytest.approx(r2.w_final)
    for j in range(2):
        assert r1.iters[j].w_next == pytest.approx(r2.iters[j].w_next)


# --- projection propagation ---

def test_outer_loop_projection_upper_propagates():
    """w_max bound must be respected across all outer iterations."""
    w_max = 1.5
    result = _run_outer_loop(n_outer_iters=2, w_max=w_max)
    for it in result.iters:
        assert it.w_next <= w_max + 1e-9


def test_outer_loop_projection_lower_propagates():
    """w_min bound must be respected across all outer iterations."""
    w_min = 0.5
    result = _run_outer_loop(n_outer_iters=2, w_min=w_min)
    for it in result.iters:
        assert it.w_next >= w_min - 1e-9


# --- public API (Phase 10C) ---

def test_phase10c_public_api_imports():
    from src.train import CTRLOuterLoopResult, ctrl_outer_loop
    assert CTRLOuterLoopResult is not None
    assert callable(ctrl_outer_loop)


# ===========================================================================
# Phase 11A — CTRLTrainerState stateful shell
# ===========================================================================

from src.train.ctrl_state import CTRLTrainerState


def _make_trainer_state(
    w_init: float = 1.0,
    w_step_size: float = 0.1,
    target_return_z: float = 1.0,
) -> CTRLTrainerState:
    actor = _make_actor()
    critic = _make_critic()
    env = _make_env(n_steps=4)
    actor_opt, critic_opt = _make_optimizers(actor, critic)
    return CTRLTrainerState(
        actor=actor,
        critic=critic,
        env=env,
        actor_optimizer=actor_opt,
        critic_optimizer=critic_opt,
        current_w=w_init,
        target_return_z=target_return_z,
        w_step_size=w_step_size,
    )


# --- construction / stored fields ---

def test_trainer_state_stores_all_fields():
    """All 8 fields are accessible after construction."""
    actor = _make_actor()
    critic = _make_critic()
    env = _make_env(n_steps=4)
    actor_opt, critic_opt = _make_optimizers(actor, critic)
    state = CTRLTrainerState(
        actor=actor,
        critic=critic,
        env=env,
        actor_optimizer=actor_opt,
        critic_optimizer=critic_opt,
        current_w=2.5,
        target_return_z=1.1,
        w_step_size=0.05,
    )
    assert state.actor is actor
    assert state.critic is critic
    assert state.env is env
    assert state.actor_optimizer is actor_opt
    assert state.critic_optimizer is critic_opt
    assert state.current_w == pytest.approx(2.5)
    assert state.target_return_z == pytest.approx(1.1)
    assert state.w_step_size == pytest.approx(0.05)


# --- run_outer_iter mutates current_w ---

def test_trainer_state_run_outer_iter_mutates_w():
    """run_outer_iter updates current_w to result.w_next."""
    state = _make_trainer_state(w_init=1.0)
    result = state.run_outer_iter(n_updates=1, entropy_temp=0.01, base_seed=0)
    assert state.current_w == pytest.approx(result.w_next)
    assert isinstance(result.w_next, float)


def test_trainer_state_run_outer_iter_returns_correct_type():
    """run_outer_iter returns a CTRLOuterIterResult."""
    from src.train.ctrl_outer_iter import CTRLOuterIterResult
    state = _make_trainer_state()
    result = state.run_outer_iter(n_updates=1, entropy_temp=0.01, base_seed=0)
    assert isinstance(result, CTRLOuterIterResult)


def test_trainer_state_run_outer_iter_w_prev_matches_current_w_before_call():
    """result.w_prev matches the stored w at the time of the call."""
    w_init = 1.3
    state = _make_trainer_state(w_init=w_init)
    result = state.run_outer_iter(n_updates=1, entropy_temp=0.01, base_seed=0)
    assert result.w_prev == pytest.approx(w_init)


# --- run_outer_loop mutates current_w ---

def test_trainer_state_run_outer_loop_mutates_w():
    """run_outer_loop updates current_w to result.w_final."""
    state = _make_trainer_state(w_init=1.0)
    result = state.run_outer_loop(n_outer_iters=2, n_updates=1, entropy_temp=0.01, base_seed=0)
    assert state.current_w == pytest.approx(result.w_final)


def test_trainer_state_run_outer_loop_returns_correct_type():
    """run_outer_loop returns a CTRLOuterLoopResult."""
    from src.train.ctrl_outer_loop import CTRLOuterLoopResult
    state = _make_trainer_state()
    result = state.run_outer_loop(n_outer_iters=2, n_updates=1, entropy_temp=0.01, base_seed=0)
    assert isinstance(result, CTRLOuterLoopResult)


def test_trainer_state_run_outer_loop_w_init_matches_current_w_before_call():
    """result.w_init matches the stored w at the time of the call."""
    w_init = 0.8
    state = _make_trainer_state(w_init=w_init)
    result = state.run_outer_loop(n_outer_iters=1, n_updates=1, entropy_temp=0.01, base_seed=0)
    assert result.w_init == pytest.approx(w_init)


# --- consecutive calls use updated w ---

def test_trainer_state_consecutive_outer_iter_calls_thread_w():
    """Second run_outer_iter call starts from the w set by the first."""
    state = _make_trainer_state(w_init=1.0)
    state.run_outer_iter(n_updates=1, entropy_temp=0.01, base_seed=0)
    w_after_first = state.current_w
    r2 = state.run_outer_iter(n_updates=1, entropy_temp=0.01, base_seed=10)
    # second call's w_prev must equal w after first call
    assert r2.w_prev == pytest.approx(w_after_first)
    assert state.current_w == pytest.approx(r2.w_next)


def test_trainer_state_consecutive_outer_loop_calls_thread_w():
    """Second run_outer_loop call starts from the w set by the first."""
    state = _make_trainer_state(w_init=1.0)
    state.run_outer_loop(n_outer_iters=1, n_updates=1, entropy_temp=0.01, base_seed=0)
    w_after_first = state.current_w
    r2 = state.run_outer_loop(n_outer_iters=1, n_updates=1, entropy_temp=0.01, base_seed=10)
    assert r2.w_init == pytest.approx(w_after_first)
    assert state.current_w == pytest.approx(r2.w_final)


def test_trainer_state_mixed_consecutive_calls_thread_w():
    """run_outer_iter followed by run_outer_loop uses updated w."""
    state = _make_trainer_state(w_init=1.0)
    state.run_outer_iter(n_updates=1, entropy_temp=0.01, base_seed=0)
    w_mid = state.current_w
    r_loop = state.run_outer_loop(n_outer_iters=1, n_updates=1, entropy_temp=0.01, base_seed=5)
    assert r_loop.w_init == pytest.approx(w_mid)
    assert state.current_w == pytest.approx(r_loop.w_final)


# --- reproducibility ---

def test_trainer_state_outer_iter_reproducible():
    """Two fresh identical states + same seed → same w_next."""
    def _run(seed: int) -> float:
        state = _make_trainer_state(w_init=1.0)
        result = state.run_outer_iter(n_updates=2, entropy_temp=0.01, base_seed=seed)
        return result.w_next

    assert _run(7) == pytest.approx(_run(7))


def test_trainer_state_outer_loop_reproducible():
    """Two fresh identical states + same seed → same w_final."""
    def _run(seed: int) -> float:
        state = _make_trainer_state(w_init=1.0)
        result = state.run_outer_loop(n_outer_iters=2, n_updates=2, entropy_temp=0.01, base_seed=seed)
        return result.w_final

    assert _run(13) == pytest.approx(_run(13))


# --- public API (Phase 11A) ---

def test_phase11a_public_api_imports():
    from src.train import CTRLTrainerState
    assert CTRLTrainerState is not None


# ===========================================================================
# Phase 11B — CTRLTrainerState validation boundary
# ===========================================================================

# --- constructor validation ---

def test_trainer_state_invalid_w_step_size_zero_raises():
    """Constructor rejects w_step_size == 0."""
    with pytest.raises(ValueError, match="w_step_size"):
        _make_trainer_state(w_step_size=0.0)


def test_trainer_state_invalid_w_step_size_negative_raises():
    """Constructor rejects negative w_step_size."""
    with pytest.raises(ValueError, match="w_step_size"):
        _make_trainer_state(w_step_size=-1.0)


def test_trainer_state_invalid_w_step_size_inf_raises():
    """Constructor rejects infinite w_step_size."""
    with pytest.raises(ValueError, match="w_step_size"):
        _make_trainer_state(w_step_size=float("inf"))


def test_trainer_state_invalid_current_w_nan_raises():
    """Constructor rejects NaN current_w."""
    with pytest.raises(ValueError, match="current_w"):
        _make_trainer_state(w_init=float("nan"))


def test_trainer_state_invalid_current_w_inf_raises():
    """Constructor rejects infinite current_w."""
    with pytest.raises(ValueError, match="current_w"):
        _make_trainer_state(w_init=float("inf"))


def test_trainer_state_invalid_target_return_z_nan_raises():
    """Constructor rejects NaN target_return_z."""
    actor = _make_actor()
    critic = _make_critic()
    env = _make_env(n_steps=4)
    actor_opt, critic_opt = _make_optimizers(actor, critic)
    with pytest.raises(ValueError, match="target_return_z"):
        CTRLTrainerState(
            actor=actor, critic=critic, env=env,
            actor_optimizer=actor_opt, critic_optimizer=critic_opt,
            current_w=1.0, target_return_z=float("nan"), w_step_size=0.1,
        )


# --- run_outer_iter validation ---

def test_trainer_state_outer_iter_invalid_n_updates_zero_raises():
    """run_outer_iter rejects n_updates == 0."""
    state = _make_trainer_state()
    with pytest.raises(ValueError, match="n_updates"):
        state.run_outer_iter(n_updates=0, entropy_temp=0.01)


def test_trainer_state_outer_iter_invalid_n_updates_negative_raises():
    """run_outer_iter rejects negative n_updates."""
    state = _make_trainer_state()
    with pytest.raises(ValueError, match="n_updates"):
        state.run_outer_iter(n_updates=-1, entropy_temp=0.01)


def test_trainer_state_outer_iter_invalid_bound_order_raises():
    """run_outer_iter rejects w_min > w_max."""
    state = _make_trainer_state()
    with pytest.raises(ValueError, match="w_min"):
        state.run_outer_iter(n_updates=1, entropy_temp=0.01, w_min=2.0, w_max=1.0)


def test_trainer_state_outer_iter_equal_bounds_does_not_raise():
    """run_outer_iter accepts w_min == w_max (degenerate projection is valid)."""
    state = _make_trainer_state(w_init=1.5)
    result = state.run_outer_iter(n_updates=1, entropy_temp=0.01, base_seed=0, w_min=1.5, w_max=1.5)
    assert result.w_next == pytest.approx(1.5)


# --- run_outer_loop validation ---

def test_trainer_state_outer_loop_invalid_n_outer_iters_zero_raises():
    """run_outer_loop rejects n_outer_iters == 0."""
    state = _make_trainer_state()
    with pytest.raises(ValueError, match="n_outer_iters"):
        state.run_outer_loop(n_outer_iters=0, n_updates=1, entropy_temp=0.01)


def test_trainer_state_outer_loop_invalid_n_outer_iters_negative_raises():
    """run_outer_loop rejects negative n_outer_iters."""
    state = _make_trainer_state()
    with pytest.raises(ValueError, match="n_outer_iters"):
        state.run_outer_loop(n_outer_iters=-1, n_updates=1, entropy_temp=0.01)


def test_trainer_state_outer_loop_invalid_n_updates_zero_raises():
    """run_outer_loop rejects n_updates == 0."""
    state = _make_trainer_state()
    with pytest.raises(ValueError, match="n_updates"):
        state.run_outer_loop(n_outer_iters=1, n_updates=0, entropy_temp=0.01)


def test_trainer_state_outer_loop_invalid_bound_order_raises():
    """run_outer_loop rejects w_min > w_max."""
    state = _make_trainer_state()
    with pytest.raises(ValueError, match="w_min"):
        state.run_outer_loop(n_outer_iters=1, n_updates=1, entropy_temp=0.01, w_min=3.0, w_max=1.0)


# --- w_step_size caught through state shell ---

def test_trainer_state_w_step_size_zero_caught_at_construction():
    """w_step_size=0 is caught at the constructor, before any method call."""
    with pytest.raises(ValueError, match="w_step_size"):
        _make_trainer_state(w_step_size=0.0)


# --- happy path still works after Phase 11B ---

def test_trainer_state_valid_construction_and_outer_iter():
    """Valid state still runs run_outer_iter without error."""
    state = _make_trainer_state(w_init=1.0, w_step_size=0.05)
    result = state.run_outer_iter(n_updates=1, entropy_temp=0.01, base_seed=0)
    assert isinstance(result.w_next, float)


def test_trainer_state_valid_construction_and_outer_loop():
    """Valid state still runs run_outer_loop without error."""
    state = _make_trainer_state(w_init=1.0, w_step_size=0.05)
    result = state.run_outer_loop(n_outer_iters=1, n_updates=1, entropy_temp=0.01, base_seed=0)
    assert isinstance(result.w_final, float)


# --- public API unchanged (Phase 11B) ---

def test_phase11b_public_api_unchanged():
    from src.train import CTRLTrainerState
    state = _make_trainer_state()
    assert hasattr(state, "run_outer_iter")
    assert hasattr(state, "run_outer_loop")
    assert callable(state.run_outer_iter)
    assert callable(state.run_outer_loop)


# --- post-construction scalar mutation caught at method call time ---

def test_trainer_state_mutated_current_w_nan_caught_on_outer_iter():
    """Mutating current_w to NaN after construction is caught in run_outer_iter."""
    state = _make_trainer_state()
    state.current_w = float("nan")
    with pytest.raises(ValueError, match="current_w"):
        state.run_outer_iter(n_updates=1, entropy_temp=0.01)


def test_trainer_state_mutated_current_w_inf_caught_on_outer_iter():
    """Mutating current_w to inf after construction is caught in run_outer_iter."""
    state = _make_trainer_state()
    state.current_w = float("inf")
    with pytest.raises(ValueError, match="current_w"):
        state.run_outer_iter(n_updates=1, entropy_temp=0.01)


def test_trainer_state_mutated_target_return_z_nan_caught_on_outer_iter():
    """Mutating target_return_z to NaN after construction is caught in run_outer_iter."""
    state = _make_trainer_state()
    state.target_return_z = float("nan")
    with pytest.raises(ValueError, match="target_return_z"):
        state.run_outer_iter(n_updates=1, entropy_temp=0.01)


def test_trainer_state_mutated_w_step_size_zero_caught_on_outer_iter():
    """Mutating w_step_size to 0 after construction is caught in run_outer_iter."""
    state = _make_trainer_state()
    state.w_step_size = 0.0
    with pytest.raises(ValueError, match="w_step_size"):
        state.run_outer_iter(n_updates=1, entropy_temp=0.01)


def test_trainer_state_mutated_w_step_size_negative_caught_on_outer_iter():
    """Mutating w_step_size to negative after construction is caught in run_outer_iter."""
    state = _make_trainer_state()
    state.w_step_size = -0.5
    with pytest.raises(ValueError, match="w_step_size"):
        state.run_outer_iter(n_updates=1, entropy_temp=0.01)


def test_trainer_state_mutated_current_w_nan_caught_on_outer_loop():
    """Mutating current_w to NaN after construction is caught in run_outer_loop."""
    state = _make_trainer_state()
    state.current_w = float("nan")
    with pytest.raises(ValueError, match="current_w"):
        state.run_outer_loop(n_outer_iters=1, n_updates=1, entropy_temp=0.01)


def test_trainer_state_mutated_w_step_size_zero_caught_on_outer_loop():
    """Mutating w_step_size to 0 after construction is caught in run_outer_loop."""
    state = _make_trainer_state()
    state.w_step_size = 0.0
    with pytest.raises(ValueError, match="w_step_size"):
        state.run_outer_loop(n_outer_iters=1, n_updates=1, entropy_temp=0.01)


# ===========================================================================
# Phase 12A — CTRLTrainerSnapshot read-only diagnostics
# ===========================================================================

from src.train.ctrl_state import CTRLTrainerSnapshot


# --- snapshot dataclass shape and field values ---

def test_trainer_snapshot_is_dataclass():
    """CTRLTrainerSnapshot is a dataclass with the expected fields."""
    import dataclasses
    assert dataclasses.is_dataclass(CTRLTrainerSnapshot)
    field_names = {f.name for f in dataclasses.fields(CTRLTrainerSnapshot)}
    assert field_names == {
        "current_w", "target_return_z", "w_step_size",
        "last_terminal_wealth", "last_w_prev", "last_n_updates",
    }


def test_trainer_snapshot_direct_construction():
    """CTRLTrainerSnapshot can be constructed directly with expected field types."""
    snap = CTRLTrainerSnapshot(
        current_w=1.5,
        target_return_z=1.1,
        w_step_size=0.05,
        last_terminal_wealth=1.2,
        last_w_prev=1.0,
        last_n_updates=4,
    )
    assert snap.current_w == pytest.approx(1.5)
    assert snap.target_return_z == pytest.approx(1.1)
    assert snap.w_step_size == pytest.approx(0.05)
    assert snap.last_terminal_wealth == pytest.approx(1.2)
    assert snap.last_w_prev == pytest.approx(1.0)
    assert snap.last_n_updates == 4


# --- snapshot before any run ---

def test_trainer_snapshot_before_any_run_has_none_diagnostics():
    """Snapshot before any run has None for all diagnostic fields."""
    state = _make_trainer_state(w_init=1.0, target_return_z=1.0, w_step_size=0.1)
    snap = state.snapshot()
    assert snap.last_terminal_wealth is None
    assert snap.last_w_prev is None
    assert snap.last_n_updates is None


def test_trainer_snapshot_before_any_run_scalar_fields():
    """Snapshot before any run correctly reflects construction scalars."""
    state = _make_trainer_state(w_init=1.3, target_return_z=1.05, w_step_size=0.07)
    snap = state.snapshot()
    assert snap.current_w == pytest.approx(1.3)
    assert snap.target_return_z == pytest.approx(1.05)
    assert snap.w_step_size == pytest.approx(0.07)


# --- snapshot after run_outer_iter ---

def test_trainer_snapshot_after_outer_iter_has_diagnostics():
    """Snapshot after run_outer_iter populates all diagnostic fields."""
    state = _make_trainer_state()
    state.run_outer_iter(n_updates=2, entropy_temp=0.01, base_seed=0)
    snap = state.snapshot()
    assert snap.last_terminal_wealth is not None
    assert snap.last_w_prev is not None
    assert snap.last_n_updates is not None


def test_trainer_snapshot_after_outer_iter_current_w_updated():
    """Snapshot current_w reflects updated w after run_outer_iter."""
    state = _make_trainer_state(w_init=1.0)
    result = state.run_outer_iter(n_updates=1, entropy_temp=0.01, base_seed=0)
    snap = state.snapshot()
    assert snap.current_w == pytest.approx(result.w_next)


def test_trainer_snapshot_after_outer_iter_n_updates_stored():
    """Snapshot last_n_updates matches n_updates passed to run_outer_iter."""
    state = _make_trainer_state()
    state.run_outer_iter(n_updates=3, entropy_temp=0.01, base_seed=0)
    snap = state.snapshot()
    assert snap.last_n_updates == 3


def test_trainer_snapshot_after_outer_iter_w_prev_matches_result():
    """Snapshot last_w_prev matches result.w_prev from run_outer_iter."""
    state = _make_trainer_state(w_init=1.2)
    result = state.run_outer_iter(n_updates=1, entropy_temp=0.01, base_seed=0)
    snap = state.snapshot()
    assert snap.last_w_prev == pytest.approx(result.w_prev)


def test_trainer_snapshot_after_outer_iter_terminal_wealth_finite():
    """Snapshot last_terminal_wealth is a finite float after run_outer_iter."""
    state = _make_trainer_state()
    state.run_outer_iter(n_updates=1, entropy_temp=0.01, base_seed=0)
    snap = state.snapshot()
    assert isinstance(snap.last_terminal_wealth, float)
    assert math.isfinite(snap.last_terminal_wealth)


# --- snapshot after run_outer_loop ---

def test_trainer_snapshot_after_outer_loop_has_diagnostics():
    """Snapshot after run_outer_loop populates all diagnostic fields."""
    state = _make_trainer_state()
    state.run_outer_loop(n_outer_iters=2, n_updates=2, entropy_temp=0.01, base_seed=0)
    snap = state.snapshot()
    assert snap.last_terminal_wealth is not None
    assert snap.last_w_prev is not None
    assert snap.last_n_updates is not None


def test_trainer_snapshot_after_outer_loop_current_w_updated():
    """Snapshot current_w reflects updated w after run_outer_loop."""
    state = _make_trainer_state(w_init=1.0)
    result = state.run_outer_loop(n_outer_iters=2, n_updates=1, entropy_temp=0.01, base_seed=0)
    snap = state.snapshot()
    assert snap.current_w == pytest.approx(result.w_final)


def test_trainer_snapshot_after_outer_loop_n_updates_is_total():
    """Snapshot last_n_updates equals n_outer_iters × n_updates for outer-loop calls."""
    state = _make_trainer_state()
    state.run_outer_loop(n_outer_iters=3, n_updates=2, entropy_temp=0.01, base_seed=0)
    snap = state.snapshot()
    assert snap.last_n_updates == 6  # 3 * 2


# --- snapshot does not mutate trainer state ---

def test_trainer_snapshot_does_not_mutate_current_w():
    """Calling snapshot() does not change current_w."""
    state = _make_trainer_state(w_init=1.7)
    state.run_outer_iter(n_updates=1, entropy_temp=0.01, base_seed=0)
    w_before_snap = state.current_w
    state.snapshot()
    assert state.current_w == pytest.approx(w_before_snap)


def test_trainer_snapshot_repeated_calls_are_consistent():
    """Two consecutive snapshot() calls without intervening runs return equal values."""
    state = _make_trainer_state()
    state.run_outer_iter(n_updates=1, entropy_temp=0.01, base_seed=0)
    snap1 = state.snapshot()
    snap2 = state.snapshot()
    assert snap1.current_w == pytest.approx(snap2.current_w)
    assert snap1.last_n_updates == snap2.last_n_updates
    assert snap1.last_w_prev == pytest.approx(snap2.last_w_prev)


# --- public API export ---

def test_phase12a_public_api_imports():
    from src.train import CTRLTrainerSnapshot, CTRLTrainerState
    assert CTRLTrainerSnapshot is not None
    state = _make_trainer_state()
    assert callable(state.snapshot)
    snap = state.snapshot()
    assert isinstance(snap, CTRLTrainerSnapshot)


# ===========================================================================
# Phase 12B — CTRLTrainerState in-memory history
# ===========================================================================

# --- history empty on fresh construction ---

def test_trainer_history_empty_on_construction():
    """history is empty on a freshly constructed state."""
    state = _make_trainer_state()
    assert len(state.history) == 0


def test_trainer_history_is_tuple():
    """history property returns a tuple (immutable)."""
    state = _make_trainer_state()
    assert isinstance(state.history, tuple)


# --- run_outer_iter appends one entry ---

def test_trainer_history_one_outer_iter_appends_one_entry():
    """One run_outer_iter call appends exactly one history entry."""
    state = _make_trainer_state()
    state.run_outer_iter(n_updates=1, entropy_temp=0.01, base_seed=0)
    assert len(state.history) == 1


def test_trainer_history_outer_iter_entry_is_snapshot():
    """History entry after run_outer_iter is a CTRLTrainerSnapshot."""
    state = _make_trainer_state()
    state.run_outer_iter(n_updates=1, entropy_temp=0.01, base_seed=0)
    assert isinstance(state.history[0], CTRLTrainerSnapshot)


def test_trainer_history_outer_iter_entry_current_w_matches():
    """History entry current_w matches state.current_w after run_outer_iter."""
    state = _make_trainer_state()
    result = state.run_outer_iter(n_updates=1, entropy_temp=0.01, base_seed=0)
    assert state.history[0].current_w == pytest.approx(result.w_next)


# --- run_outer_loop appends one entry ---

def test_trainer_history_one_outer_loop_appends_one_entry():
    """One run_outer_loop call appends exactly one history entry."""
    state = _make_trainer_state()
    state.run_outer_loop(n_outer_iters=2, n_updates=1, entropy_temp=0.01, base_seed=0)
    assert len(state.history) == 1


def test_trainer_history_outer_loop_entry_is_snapshot():
    """History entry after run_outer_loop is a CTRLTrainerSnapshot."""
    state = _make_trainer_state()
    state.run_outer_loop(n_outer_iters=1, n_updates=1, entropy_temp=0.01, base_seed=0)
    assert isinstance(state.history[0], CTRLTrainerSnapshot)


def test_trainer_history_outer_loop_entry_current_w_matches():
    """History entry current_w matches state.current_w after run_outer_loop."""
    state = _make_trainer_state()
    result = state.run_outer_loop(n_outer_iters=1, n_updates=1, entropy_temp=0.01, base_seed=0)
    assert state.history[0].current_w == pytest.approx(result.w_final)


# --- repeated calls append in order ---

def test_trainer_history_two_outer_iter_calls_append_in_order():
    """Two run_outer_iter calls produce two ordered history entries."""
    state = _make_trainer_state()
    r1 = state.run_outer_iter(n_updates=1, entropy_temp=0.01, base_seed=0)
    r2 = state.run_outer_iter(n_updates=1, entropy_temp=0.01, base_seed=10)
    assert len(state.history) == 2
    assert state.history[0].current_w == pytest.approx(r1.w_next)
    assert state.history[1].current_w == pytest.approx(r2.w_next)


def test_trainer_history_mixed_calls_append_in_order():
    """run_outer_iter then run_outer_loop produces two ordered history entries."""
    state = _make_trainer_state()
    r_iter = state.run_outer_iter(n_updates=1, entropy_temp=0.01, base_seed=0)
    r_loop = state.run_outer_loop(n_outer_iters=1, n_updates=1, entropy_temp=0.01, base_seed=5)
    assert len(state.history) == 2
    assert state.history[0].current_w == pytest.approx(r_iter.w_next)
    assert state.history[1].current_w == pytest.approx(r_loop.w_final)


# --- history is read-only from caller perspective ---

def test_trainer_history_tuple_is_immutable():
    """Returned history tuple cannot be mutated by the caller."""
    state = _make_trainer_state()
    state.run_outer_iter(n_updates=1, entropy_temp=0.01, base_seed=0)
    hist = state.history
    with pytest.raises((AttributeError, TypeError)):
        hist.append(state.snapshot())  # type: ignore[attr-defined]


# --- clear_history removes entries without mutating live state ---

def test_trainer_clear_history_empties_list():
    """clear_history() removes all history entries."""
    state = _make_trainer_state()
    state.run_outer_iter(n_updates=1, entropy_temp=0.01, base_seed=0)
    state.run_outer_iter(n_updates=1, entropy_temp=0.01, base_seed=10)
    state.clear_history()
    assert len(state.history) == 0


def test_trainer_clear_history_does_not_change_current_w():
    """clear_history() does not change current_w."""
    state = _make_trainer_state()
    state.run_outer_iter(n_updates=1, entropy_temp=0.01, base_seed=0)
    w_after_run = state.current_w
    state.clear_history()
    assert state.current_w == pytest.approx(w_after_run)


def test_trainer_clear_history_does_not_change_diagnostics():
    """clear_history() does not change the last_* diagnostic fields in snapshot."""
    state = _make_trainer_state()
    state.run_outer_iter(n_updates=1, entropy_temp=0.01, base_seed=0)
    snap_before = state.snapshot()
    state.clear_history()
    snap_after = state.snapshot()
    assert snap_after.last_n_updates == snap_before.last_n_updates
    assert snap_after.last_w_prev == pytest.approx(snap_before.last_w_prev)


def test_trainer_history_resumes_after_clear():
    """Runs after clear_history() append new entries starting from index 0."""
    state = _make_trainer_state()
    state.run_outer_iter(n_updates=1, entropy_temp=0.01, base_seed=0)
    state.clear_history()
    result = state.run_outer_iter(n_updates=1, entropy_temp=0.01, base_seed=10)
    assert len(state.history) == 1
    assert state.history[0].current_w == pytest.approx(result.w_next)


def test_trainer_history_entry_field_mutation_raises():
    """Mutating a field on a history entry is rejected (frozen dataclass)."""
    state = _make_trainer_state()
    state.run_outer_iter(n_updates=1, entropy_temp=0.01, base_seed=0)
    entry = state.history[0]
    with pytest.raises(AttributeError):
        entry.current_w = 999.0  # type: ignore[misc]


# ===========================================================================
# Phase 12C — CTRLTrainerState reset boundary
# ===========================================================================

# --- reset to original w ---

def test_trainer_reset_restores_initial_w():
    """reset() with no argument restores current_w to construction-time value."""
    state = _make_trainer_state(w_init=1.5)
    state.run_outer_iter(n_updates=1, entropy_temp=0.01, base_seed=0)
    assert state.current_w != pytest.approx(1.5)  # w was updated by run
    state.reset()
    assert state.current_w == pytest.approx(1.5)


def test_trainer_reset_after_multiple_runs_restores_initial_w():
    """reset() restores initial w even after multiple consecutive runs."""
    state = _make_trainer_state(w_init=2.0)
    state.run_outer_iter(n_updates=1, entropy_temp=0.01, base_seed=0)
    state.run_outer_iter(n_updates=1, entropy_temp=0.01, base_seed=5)
    state.reset()
    assert state.current_w == pytest.approx(2.0)


# --- reset to explicit w ---

def test_trainer_reset_to_explicit_w():
    """reset(w=x) sets current_w to the supplied value."""
    state = _make_trainer_state(w_init=1.0)
    state.run_outer_iter(n_updates=1, entropy_temp=0.01, base_seed=0)
    state.reset(w=3.0)
    assert state.current_w == pytest.approx(3.0)


def test_trainer_reset_explicit_w_negative_is_valid():
    """reset accepts a negative finite w (no sign restriction)."""
    state = _make_trainer_state()
    state.reset(w=-0.5)
    assert state.current_w == pytest.approx(-0.5)


# --- reset clears diagnostics ---

def test_trainer_reset_clears_snapshot_diagnostics():
    """reset() sets last_* diagnostic fields back to None."""
    state = _make_trainer_state()
    state.run_outer_iter(n_updates=1, entropy_temp=0.01, base_seed=0)
    state.reset()
    snap = state.snapshot()
    assert snap.last_terminal_wealth is None
    assert snap.last_w_prev is None
    assert snap.last_n_updates is None


# --- reset clears history ---

def test_trainer_reset_clears_history():
    """reset() empties in-memory history."""
    state = _make_trainer_state()
    state.run_outer_iter(n_updates=1, entropy_temp=0.01, base_seed=0)
    state.reset()
    assert len(state.history) == 0


# --- reset does not replace object references ---

def test_trainer_reset_preserves_actor_reference():
    """reset() does not replace the stored actor object."""
    state = _make_trainer_state()
    original_actor = state.actor
    state.run_outer_iter(n_updates=1, entropy_temp=0.01, base_seed=0)
    state.reset()
    assert state.actor is original_actor


def test_trainer_reset_preserves_critic_reference():
    """reset() does not replace the stored critic object."""
    state = _make_trainer_state()
    original_critic = state.critic
    state.reset()
    assert state.critic is original_critic


def test_trainer_reset_preserves_env_reference():
    """reset() does not replace the stored env object."""
    state = _make_trainer_state()
    original_env = state.env
    state.reset()
    assert state.env is original_env


def test_trainer_reset_preserves_optimizer_references():
    """reset() does not replace the stored optimizer objects."""
    state = _make_trainer_state()
    original_actor_opt = state.actor_optimizer
    original_critic_opt = state.critic_optimizer
    state.reset()
    assert state.actor_optimizer is original_actor_opt
    assert state.critic_optimizer is original_critic_opt


# --- reset rejects invalid explicit w ---

def test_trainer_reset_nan_w_raises():
    """reset(w=nan) raises ValueError."""
    state = _make_trainer_state()
    with pytest.raises(ValueError, match="reset w"):
        state.reset(w=float("nan"))


def test_trainer_reset_inf_w_raises():
    """reset(w=inf) raises ValueError."""
    state = _make_trainer_state()
    with pytest.raises(ValueError, match="reset w"):
        state.reset(w=float("inf"))


# --- runs after reset use updated w ---

def test_trainer_reset_next_run_starts_from_reset_w():
    """After reset(), the next run_outer_iter uses the reset current_w."""
    state = _make_trainer_state(w_init=1.0)
    state.run_outer_iter(n_updates=1, entropy_temp=0.01, base_seed=0)
    state.reset()
    result = state.run_outer_iter(n_updates=1, entropy_temp=0.01, base_seed=0)
    assert result.w_prev == pytest.approx(1.0)


# ===========================================================================
# Phase 13B — CTRLCheckpointPayload export / restore
# ===========================================================================

import dataclasses

from src.train.ctrl_state import CTRLCheckpointPayload


# --- export returns expected payload shape ---

def test_export_checkpoint_returns_payload_type():
    """export_checkpoint() returns a CTRLCheckpointPayload."""
    state = _make_trainer_state()
    payload = state.export_checkpoint()
    assert isinstance(payload, CTRLCheckpointPayload)


def test_export_checkpoint_has_all_fields():
    """CTRLCheckpointPayload has the required seven fields."""
    expected = {
        "actor_state_dict", "critic_state_dict",
        "actor_optimizer_state_dict", "critic_optimizer_state_dict",
        "current_w", "target_return_z", "w_step_size",
    }
    assert {f.name for f in dataclasses.fields(CTRLCheckpointPayload)} == expected


def test_export_checkpoint_scalar_fields_match_state():
    """Exported scalars match the trainer's current values."""
    state = _make_trainer_state(w_init=1.3, target_return_z=1.05, w_step_size=0.07)
    payload = state.export_checkpoint()
    assert payload.current_w == pytest.approx(1.3)
    assert payload.target_return_z == pytest.approx(1.05)
    assert payload.w_step_size == pytest.approx(0.07)


def test_export_checkpoint_state_dicts_are_dicts():
    """All four state_dict fields are plain dicts."""
    state = _make_trainer_state()
    payload = state.export_checkpoint()
    assert isinstance(payload.actor_state_dict, dict)
    assert isinstance(payload.critic_state_dict, dict)
    assert isinstance(payload.actor_optimizer_state_dict, dict)
    assert isinstance(payload.critic_optimizer_state_dict, dict)


def test_export_checkpoint_after_run_captures_updated_w():
    """Exported payload reflects current_w after a training run."""
    state = _make_trainer_state(w_init=1.0)
    result = state.run_outer_iter(n_updates=1, entropy_temp=0.01, base_seed=0)
    payload = state.export_checkpoint()
    assert payload.current_w == pytest.approx(result.w_next)


# --- payload is stable after live mutations ---

def test_export_checkpoint_actor_payload_stable_after_live_mutation():
    """Mutating live actor parameters after export does not mutate the payload."""
    import torch
    state = _make_trainer_state()
    payload = state.export_checkpoint()
    # Snapshot payload values before mutation
    saved = {k: v.clone() for k, v in payload.actor_state_dict.items()}
    # Mutate live actor parameters via a training run
    state.run_outer_iter(n_updates=2, entropy_temp=0.01, base_seed=0)
    # Payload must be unchanged
    for k, v_saved in saved.items():
        assert torch.allclose(payload.actor_state_dict[k], v_saved), (
            f"actor payload key '{k}' was mutated by a live training run"
        )


def test_export_checkpoint_critic_payload_stable_after_live_mutation():
    """Mutating live critic parameters after export does not mutate the payload."""
    import torch
    state = _make_trainer_state()
    payload = state.export_checkpoint()
    saved = {k: v.clone() for k, v in payload.critic_state_dict.items()}
    state.run_outer_iter(n_updates=2, entropy_temp=0.01, base_seed=0)
    for k, v_saved in saved.items():
        assert torch.allclose(payload.critic_state_dict[k], v_saved), (
            f"critic payload key '{k}' was mutated by a live training run"
        )


# --- restore reproduces scalar trainer state ---

def test_restore_checkpoint_scalar_state():
    """restore_checkpoint() sets current_w, target_return_z, w_step_size from payload."""
    state = _make_trainer_state(w_init=1.0, target_return_z=1.0, w_step_size=0.1)
    payload = state.export_checkpoint()
    # Mutate scalars to something different
    state.current_w = 99.0
    state.target_return_z = 99.0
    state.w_step_size = 99.0
    state.restore_checkpoint(payload)
    assert state.current_w == pytest.approx(1.0)
    assert state.target_return_z == pytest.approx(1.0)
    assert state.w_step_size == pytest.approx(0.1)


# --- restore updates model/optimizer state in place ---

def test_restore_checkpoint_restores_actor_params():
    """restore_checkpoint() loads actor parameters back to their captured values."""
    import torch
    state = _make_trainer_state()
    payload_before = state.export_checkpoint()
    # Run to change actor parameters
    state.run_outer_iter(n_updates=2, entropy_temp=0.01, base_seed=0)
    # Verify at least one actor param changed after the run
    any_changed = any(
        not torch.allclose(payload_before.actor_state_dict[k], v)
        for k, v in state.actor.state_dict().items()
    )
    assert any_changed, "Expected actor parameters to change after run_outer_iter"
    # Restore and verify actor params match the captured payload
    state.restore_checkpoint(payload_before)
    for k, v in state.actor.state_dict().items():
        assert torch.allclose(v, payload_before.actor_state_dict[k]), (
            f"actor param '{k}' mismatch after restore"
        )


def test_restore_checkpoint_restores_critic_params():
    """restore_checkpoint() loads critic parameters back to their captured values."""
    import torch
    state = _make_trainer_state()
    payload_before = state.export_checkpoint()
    state.run_outer_iter(n_updates=2, entropy_temp=0.01, base_seed=0)
    state.restore_checkpoint(payload_before)
    for k, v in state.critic.state_dict().items():
        assert torch.allclose(v, payload_before.critic_state_dict[k]), (
            f"critic param '{k}' mismatch after restore"
        )


# --- restore does not replace stored object references ---

def test_restore_checkpoint_preserves_actor_reference():
    """restore_checkpoint() does not replace the actor object reference."""
    state = _make_trainer_state()
    original_actor = state.actor
    payload = state.export_checkpoint()
    state.restore_checkpoint(payload)
    assert state.actor is original_actor


def test_restore_checkpoint_preserves_critic_reference():
    """restore_checkpoint() does not replace the critic object reference."""
    state = _make_trainer_state()
    original_critic = state.critic
    payload = state.export_checkpoint()
    state.restore_checkpoint(payload)
    assert state.critic is original_critic


def test_restore_checkpoint_preserves_optimizer_references():
    """restore_checkpoint() does not replace optimizer object references."""
    state = _make_trainer_state()
    original_actor_opt = state.actor_optimizer
    original_critic_opt = state.critic_optimizer
    payload = state.export_checkpoint()
    state.restore_checkpoint(payload)
    assert state.actor_optimizer is original_actor_opt
    assert state.critic_optimizer is original_critic_opt


# --- restore clears history and diagnostics ---

def test_restore_checkpoint_clears_history():
    """restore_checkpoint() resets history to empty."""
    state = _make_trainer_state()
    state.run_outer_iter(n_updates=1, entropy_temp=0.01, base_seed=0)
    payload = state.export_checkpoint()
    state.run_outer_iter(n_updates=1, entropy_temp=0.01, base_seed=5)
    state.restore_checkpoint(payload)
    assert len(state.history) == 0


def test_restore_checkpoint_clears_diagnostics():
    """restore_checkpoint() resets latest snapshot diagnostics to None."""
    state = _make_trainer_state()
    state.run_outer_iter(n_updates=1, entropy_temp=0.01, base_seed=0)
    payload = state.export_checkpoint()
    state.restore_checkpoint(payload)
    snap = state.snapshot()
    assert snap.last_terminal_wealth is None
    assert snap.last_w_prev is None
    assert snap.last_n_updates is None


# --- invalid payloads are rejected ---

def test_restore_checkpoint_invalid_current_w_nan_raises():
    """restore_checkpoint() rejects payload with NaN current_w."""
    state = _make_trainer_state()
    payload = state.export_checkpoint()
    bad = CTRLCheckpointPayload(
        actor_state_dict=payload.actor_state_dict,
        critic_state_dict=payload.critic_state_dict,
        actor_optimizer_state_dict=payload.actor_optimizer_state_dict,
        critic_optimizer_state_dict=payload.critic_optimizer_state_dict,
        current_w=float("nan"),
        target_return_z=payload.target_return_z,
        w_step_size=payload.w_step_size,
    )
    with pytest.raises(ValueError, match="current_w"):
        state.restore_checkpoint(bad)


def test_restore_checkpoint_invalid_w_step_size_zero_raises():
    """restore_checkpoint() rejects payload with w_step_size == 0."""
    state = _make_trainer_state()
    payload = state.export_checkpoint()
    bad = CTRLCheckpointPayload(
        actor_state_dict=payload.actor_state_dict,
        critic_state_dict=payload.critic_state_dict,
        actor_optimizer_state_dict=payload.actor_optimizer_state_dict,
        critic_optimizer_state_dict=payload.critic_optimizer_state_dict,
        current_w=payload.current_w,
        target_return_z=payload.target_return_z,
        w_step_size=0.0,
    )
    with pytest.raises(ValueError, match="w_step_size"):
        state.restore_checkpoint(bad)


# --- public API export ---

def test_phase13b_public_api_imports():
    from src.train import CTRLCheckpointPayload, CTRLTrainerState
    assert CTRLCheckpointPayload is not None
    state = _make_trainer_state()
    assert callable(state.export_checkpoint)
    assert callable(state.restore_checkpoint)
    payload = state.export_checkpoint()
    assert isinstance(payload, CTRLCheckpointPayload)


# ===========================================================================
# Phase 13C — checkpoint file IO (save_checkpoint / load_checkpoint)
# ===========================================================================

from src.train.checkpoints import load_checkpoint, save_checkpoint


# --- save / load roundtrip ---

def test_checkpoint_roundtrip_returns_payload_type(tmp_path):
    """save then load roundtrip returns a CTRLCheckpointPayload."""
    state = _make_trainer_state()
    payload = state.export_checkpoint()
    ckpt_file = tmp_path / "ckpt.pt"
    save_checkpoint(payload, ckpt_file)
    loaded = load_checkpoint(ckpt_file)
    assert isinstance(loaded, CTRLCheckpointPayload)


def test_checkpoint_roundtrip_scalar_fields(tmp_path):
    """Roundtrip preserves all scalar fields."""
    state = _make_trainer_state(w_init=1.7, target_return_z=1.05, w_step_size=0.08)
    payload = state.export_checkpoint()
    ckpt_file = tmp_path / "ckpt.pt"
    save_checkpoint(payload, ckpt_file)
    loaded = load_checkpoint(ckpt_file)
    assert loaded.current_w == pytest.approx(payload.current_w)
    assert loaded.target_return_z == pytest.approx(payload.target_return_z)
    assert loaded.w_step_size == pytest.approx(payload.w_step_size)


def test_checkpoint_roundtrip_actor_state_dict(tmp_path):
    """Roundtrip preserves actor state_dict tensor values."""
    import torch
    state = _make_trainer_state()
    payload = state.export_checkpoint()
    ckpt_file = tmp_path / "ckpt.pt"
    save_checkpoint(payload, ckpt_file)
    loaded = load_checkpoint(ckpt_file)
    for k in payload.actor_state_dict:
        assert torch.allclose(loaded.actor_state_dict[k], payload.actor_state_dict[k]), (
            f"actor_state_dict key '{k}' mismatch after roundtrip"
        )


def test_checkpoint_roundtrip_critic_state_dict(tmp_path):
    """Roundtrip preserves critic state_dict tensor values."""
    import torch
    state = _make_trainer_state()
    payload = state.export_checkpoint()
    ckpt_file = tmp_path / "ckpt.pt"
    save_checkpoint(payload, ckpt_file)
    loaded = load_checkpoint(ckpt_file)
    for k in payload.critic_state_dict:
        assert torch.allclose(loaded.critic_state_dict[k], payload.critic_state_dict[k]), (
            f"critic_state_dict key '{k}' mismatch after roundtrip"
        )


# --- save / load / restore into fresh trainer ---

def test_checkpoint_save_load_restore_scalar_state(tmp_path):
    """save → load → restore_checkpoint reproduces scalar trainer state."""
    state_a = _make_trainer_state(w_init=1.0)
    result = state_a.run_outer_iter(n_updates=1, entropy_temp=0.01, base_seed=0)
    ckpt_file = tmp_path / "ckpt.pt"
    save_checkpoint(state_a.export_checkpoint(), ckpt_file)

    state_b = _make_trainer_state(w_init=99.0)
    state_b.restore_checkpoint(load_checkpoint(ckpt_file))
    assert state_b.current_w == pytest.approx(result.w_next)


def test_checkpoint_save_load_restore_actor_params(tmp_path):
    """save → load → restore_checkpoint reproduces actor model parameters."""
    import torch
    state_a = _make_trainer_state()
    state_a.run_outer_iter(n_updates=2, entropy_temp=0.01, base_seed=0)
    payload_a = state_a.export_checkpoint()
    ckpt_file = tmp_path / "ckpt.pt"
    save_checkpoint(payload_a, ckpt_file)

    state_b = _make_trainer_state()
    state_b.restore_checkpoint(load_checkpoint(ckpt_file))
    for k, v in state_b.actor.state_dict().items():
        assert torch.allclose(v, payload_a.actor_state_dict[k]), (
            f"actor param '{k}' mismatch after save/load/restore"
        )


# --- invalid cases are rejected ---

def test_checkpoint_load_nonexistent_path_raises(tmp_path):
    """load_checkpoint raises FileNotFoundError for a nonexistent path."""
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "does_not_exist.pt")


def test_checkpoint_load_invalid_shape_raises(tmp_path):
    """load_checkpoint raises ValueError if the file contains the wrong type."""
    import torch
    bad_file = tmp_path / "bad.pt"
    torch.save({"not": "a payload"}, bad_file)
    with pytest.raises(ValueError, match="CTRLCheckpointPayload"):
        load_checkpoint(bad_file)


def test_checkpoint_save_invalid_payload_type_raises(tmp_path):
    """save_checkpoint raises TypeError if passed a non-payload object."""
    with pytest.raises(TypeError):
        save_checkpoint({"not": "a payload"}, tmp_path / "ckpt.pt")


# --- public API ---

def test_phase13c_public_api_imports():
    from src.train import load_checkpoint, save_checkpoint
    assert callable(save_checkpoint)
    assert callable(load_checkpoint)
