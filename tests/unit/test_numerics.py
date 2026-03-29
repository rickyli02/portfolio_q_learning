"""Unit tests for src/utils/numerics.py — Phase 17A.

Verifies the warn_if_unstable helper boundary exactly:
- no warning for ordinary finite values
- warning for large-finite values above threshold
- warning for inf / nan
- model validate_parameters() raises on non-finite actor/critic parameters
- GaussianActor warns on unstable computed quantities
- QuadraticCritic warns on unstable computed quantities
- ctrl_train_step fails fast on non-finite parameters
"""

from __future__ import annotations

import math
import warnings

import pytest
import torch
import torch.nn as nn

from src.models.base import ActorBase, CriticBase
from src.utils.numerics import warn_if_unstable


# ===========================================================================
# warn_if_unstable — core behaviour
# ===========================================================================


def test_no_warning_for_ordinary_finite_tensor():
    t = torch.tensor([1.0, -2.5, 0.3])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warn_if_unstable(t, "test_tensor")  # must not raise


def test_no_warning_below_threshold():
    t = torch.tensor([999_999.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warn_if_unstable(t, "test_tensor", threshold=1e6)  # exactly at limit, not over


def test_warning_for_large_finite_value():
    t = torch.tensor([2e6])
    with pytest.warns(UserWarning, match="large absolute value"):
        warn_if_unstable(t, "my_tensor", threshold=1e6)


def test_warning_message_contains_name():
    t = torch.tensor([2e6])
    with pytest.warns(UserWarning, match="my_special_name"):
        warn_if_unstable(t, "my_special_name", threshold=1e6)


def test_warning_message_contains_max_abs():
    t = torch.tensor([3e7])
    with pytest.warns(UserWarning, match="3.000e\\+07"):
        warn_if_unstable(t, "t", threshold=1e6)


def test_warning_for_positive_inf():
    t = torch.tensor([float("inf")])
    with pytest.warns(UserWarning, match="non-finite"):
        warn_if_unstable(t, "inf_tensor")


def test_warning_for_negative_inf():
    t = torch.tensor([float("-inf")])
    with pytest.warns(UserWarning, match="non-finite"):
        warn_if_unstable(t, "neg_inf_tensor")


def test_warning_for_nan():
    t = torch.tensor([float("nan")])
    with pytest.warns(UserWarning, match="non-finite"):
        warn_if_unstable(t, "nan_tensor")


def test_warning_for_mixed_nan_and_finite():
    t = torch.tensor([1.0, float("nan"), 2.0])
    with pytest.warns(UserWarning, match="non-finite"):
        warn_if_unstable(t, "mixed_tensor")


def test_inf_triggers_non_finite_not_large_finite_warning():
    """inf falls under non-finite branch, not the large-finite branch."""
    t = torch.tensor([float("inf")])
    with pytest.warns(UserWarning, match="non-finite"):
        warn_if_unstable(t, "t", threshold=1e6)


def test_no_mutation_of_tensor():
    t = torch.tensor([2e6, 1.0])
    original = t.clone()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        warn_if_unstable(t, "t", threshold=1e6)
    assert torch.equal(t, original)


def test_custom_threshold_respected():
    t = torch.tensor([500.0])
    # threshold=100 → 500 > 100 → should warn
    with pytest.warns(UserWarning, match="large absolute value"):
        warn_if_unstable(t, "t", threshold=100.0)


def test_custom_threshold_no_warning_below():
    t = torch.tensor([50.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warn_if_unstable(t, "t", threshold=100.0)


# ===========================================================================
# ActorBase / CriticBase validate_parameters()
# ===========================================================================


class _ConcreteActor(ActorBase):
    """Minimal concrete actor for parameter validation tests."""

    def __init__(self, value: float):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(value))

    def sample(self, t, wealth, w):
        return torch.zeros(1)

    def mean_action(self, t, wealth, w):
        return torch.zeros(1)

    def log_prob(self, action, t, wealth, w):
        return torch.tensor(0.0)

    def entropy(self, t):
        return torch.tensor(0.0)


class _ConcreteCritic(CriticBase):
    """Minimal concrete critic for parameter validation tests."""

    def __init__(self, value: float):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(value))

    def forward(self, t, wealth, w):
        return torch.tensor(0.0)


def test_actor_validate_parameters_passes_for_finite():
    actor = _ConcreteActor(1.0)
    actor.validate_parameters()  # must not raise


def test_actor_validate_parameters_raises_for_nan():
    actor = _ConcreteActor(float("nan"))
    with pytest.raises(ValueError, match="non-finite"):
        actor.validate_parameters()


def test_actor_validate_parameters_raises_for_inf():
    actor = _ConcreteActor(float("inf"))
    with pytest.raises(ValueError, match="non-finite"):
        actor.validate_parameters()


def test_actor_validate_parameters_error_message_contains_param_name():
    actor = _ConcreteActor(float("nan"))
    with pytest.raises(ValueError, match="'p'"):
        actor.validate_parameters()


def test_critic_validate_parameters_passes_for_finite():
    critic = _ConcreteCritic(2.0)
    critic.validate_parameters()  # must not raise


def test_critic_validate_parameters_raises_for_nan():
    critic = _ConcreteCritic(float("nan"))
    with pytest.raises(ValueError, match="non-finite"):
        critic.validate_parameters()


def test_critic_validate_parameters_raises_for_inf():
    critic = _ConcreteCritic(float("inf"))
    with pytest.raises(ValueError, match="non-finite"):
        critic.validate_parameters()


# ===========================================================================
# GaussianActor instability warnings
# ===========================================================================


def test_gaussian_actor_phi1_warns_when_log_phi1_very_large():
    """Extremely large log_phi1 → exp blows up → warn on phi1."""
    from src.models.gaussian_actor import GaussianActor

    actor = GaussianActor(n_risky=1, horizon=1.0)
    with torch.no_grad():
        actor.log_phi1.fill_(100.0)  # exp(100) >> 1e6

    with pytest.warns(UserWarning, match="GaussianActor.phi1"):
        _ = actor.phi1


def test_gaussian_actor_phi2_warns_when_log_phi2_inv_very_negative():
    """Very negative log_phi2_inv → exp(-log_phi2_inv) = exp(large) >> 1e6."""
    from src.models.gaussian_actor import GaussianActor

    actor = GaussianActor(n_risky=1, horizon=1.0)
    with torch.no_grad():
        actor.log_phi2_inv.fill_(-100.0)  # phi2 = exp(100) >> 1e6

    with pytest.warns(UserWarning, match="GaussianActor.phi2"):
        _ = actor.phi2


def test_gaussian_actor_variance_warns_when_phi3_very_large():
    """Large phi3 → exp(phi3 * T) >> 1e6 at t=0."""
    from src.models.gaussian_actor import GaussianActor

    actor = GaussianActor(n_risky=1, horizon=1.0)
    with torch.no_grad():
        actor.phi3.fill_(100.0)  # exp(100 * 1.0) >> 1e6

    with pytest.warns(UserWarning, match="GaussianActor.variance"):
        _ = actor.variance(0.0)


def test_gaussian_actor_no_warning_for_normal_params():
    """Default-initialised actor must not emit any warnings."""
    from src.models.gaussian_actor import GaussianActor

    actor = GaussianActor(n_risky=1, horizon=1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _ = actor.phi1
        _ = actor.phi2
        _ = actor.variance(0.0)


# ===========================================================================
# QuadraticCritic instability warnings
# ===========================================================================


def test_quadratic_critic_decay_warns_when_theta3_very_negative():
    """Very negative theta3 → exp(-theta3 * time_to_go) = exp(large) >> 1e6."""
    from src.models.quadratic_critic import QuadraticCritic

    critic = QuadraticCritic(horizon=1.0, target_return_z=1.0)
    with torch.no_grad():
        critic.theta3.fill_(-100.0)  # decay = exp(100) >> 1e6

    with pytest.warns(UserWarning, match="QuadraticCritic.decay"):
        critic.forward(t=0.0, wealth=torch.tensor(1.0), w=1.0)


def test_quadratic_critic_quad_warns_for_extreme_wealth():
    """Very large wealth gap → quad >> 1e6."""
    from src.models.quadratic_critic import QuadraticCritic

    critic = QuadraticCritic(horizon=1.0, target_return_z=1.0)
    # (x - w)^2 = (1e4)^2 = 1e8 >> 1e6
    with pytest.warns(UserWarning, match="QuadraticCritic.quad"):
        critic.forward(t=0.0, wealth=torch.tensor(1e4 + 1.0), w=1.0)


def test_quadratic_critic_no_warning_for_normal_inputs():
    """Normal inputs must not emit any warnings."""
    from src.models.quadratic_critic import QuadraticCritic

    critic = QuadraticCritic(horizon=1.0, target_return_z=1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        critic.forward(t=0.0, wealth=torch.tensor(1.0), w=1.0)


# ===========================================================================
# ctrl_train_step fails fast on non-finite parameters
# ===========================================================================


def test_ctrl_train_step_raises_on_non_finite_actor_parameter():
    from src.config.schema import AssetConfig, EnvConfig
    from src.envs.gbm_env import GBMPortfolioEnv
    from src.models.gaussian_actor import GaussianActor
    from src.models.quadratic_critic import QuadraticCritic
    from src.train.ctrl_trainer import ctrl_train_step

    env_cfg = EnvConfig(
        horizon=1.0, n_steps=4, initial_wealth=1.0,
        mu=[0.08], sigma=[[0.20]],
        assets=AssetConfig(n_risky=1, include_risk_free=True, risk_free_rate=0.05),
    )
    env = GBMPortfolioEnv(env_cfg)
    actor = GaussianActor(n_risky=1, horizon=1.0)
    critic = QuadraticCritic(horizon=1.0, target_return_z=1.0)
    actor_opt = torch.optim.SGD(actor.parameters(), lr=1e-3)
    critic_opt = torch.optim.SGD(critic.parameters(), lr=1e-3)

    # Corrupt a parameter
    with torch.no_grad():
        actor.log_phi1.fill_(float("nan"))

    with pytest.raises(ValueError, match="non-finite"):
        ctrl_train_step(
            actor=actor, critic=critic, env=env,
            actor_optimizer=actor_opt, critic_optimizer=critic_opt,
            w=1.0, entropy_temp=0.1,
        )


def test_ctrl_train_step_raises_on_non_finite_critic_parameter():
    from src.config.schema import AssetConfig, EnvConfig
    from src.envs.gbm_env import GBMPortfolioEnv
    from src.models.gaussian_actor import GaussianActor
    from src.models.quadratic_critic import QuadraticCritic
    from src.train.ctrl_trainer import ctrl_train_step

    env_cfg = EnvConfig(
        horizon=1.0, n_steps=4, initial_wealth=1.0,
        mu=[0.08], sigma=[[0.20]],
        assets=AssetConfig(n_risky=1, include_risk_free=True, risk_free_rate=0.05),
    )
    env = GBMPortfolioEnv(env_cfg)
    actor = GaussianActor(n_risky=1, horizon=1.0)
    critic = QuadraticCritic(horizon=1.0, target_return_z=1.0)
    actor_opt = torch.optim.SGD(actor.parameters(), lr=1e-3)
    critic_opt = torch.optim.SGD(critic.parameters(), lr=1e-3)

    with torch.no_grad():
        critic.theta3.fill_(float("inf"))

    with pytest.raises(ValueError, match="non-finite"):
        ctrl_train_step(
            actor=actor, critic=critic, env=env,
            actor_optimizer=actor_opt, critic_optimizer=critic_opt,
            w=1.0, entropy_temp=0.1,
        )
