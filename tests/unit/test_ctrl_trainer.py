"""Unit tests for src/train/ctrl_trainer.py — Phase 9A single-trajectory step."""

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
# Public API imports
# ---------------------------------------------------------------------------

def test_public_api_imports():
    from src.train import CTRLStepResult, ctrl_train_step
    assert CTRLStepResult is not None
    assert callable(ctrl_train_step)
