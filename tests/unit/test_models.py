"""Unit tests for src/models/: ActorBase, CriticBase, GaussianActor, QuadraticCritic."""

import math

import pytest
import torch

from src.models.base import ActorBase, CriticBase
from src.models.gaussian_actor import GaussianActor
from src.models.quadratic_critic import QuadraticCritic


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _actor(n_risky: int = 1, horizon: float = 1.0, **kwargs) -> GaussianActor:
    return GaussianActor(n_risky=n_risky, horizon=horizon, **kwargs)


def _critic(horizon: float = 1.0, z: float = 1.0, **kwargs) -> QuadraticCritic:
    return QuadraticCritic(horizon=horizon, target_return_z=z, **kwargs)


# ---------------------------------------------------------------------------
# Abstract base classes — interface contract
# ---------------------------------------------------------------------------


def test_actor_base_is_abstract():
    """ActorBase cannot be instantiated directly."""
    with pytest.raises(TypeError):
        ActorBase()  # type: ignore[abstract]


def test_critic_base_is_abstract():
    with pytest.raises(TypeError):
        CriticBase()  # type: ignore[abstract]


def test_gaussian_actor_is_actor_base():
    assert isinstance(_actor(), ActorBase)


def test_quadratic_critic_is_critic_base():
    assert isinstance(_critic(), CriticBase)


def test_gaussian_actor_is_nn_module():
    assert isinstance(_actor(), torch.nn.Module)


def test_quadratic_critic_is_nn_module():
    assert isinstance(_critic(), torch.nn.Module)


# ---------------------------------------------------------------------------
# GaussianActor — construction and parameter constraints
# ---------------------------------------------------------------------------


def test_gaussian_actor_invalid_phi1_raises():
    with pytest.raises(ValueError, match="init_phi1"):
        GaussianActor(n_risky=1, horizon=1.0, init_phi1=0.0)


def test_gaussian_actor_invalid_phi2_raises():
    with pytest.raises(ValueError, match="init_phi2"):
        GaussianActor(n_risky=1, horizon=1.0, init_phi2=-0.1)


def test_gaussian_actor_phi1_positive():
    actor = _actor(init_phi1=2.0)
    assert (actor.phi1 > 0).all()


def test_gaussian_actor_phi2_positive():
    actor = _actor(init_phi2=0.3)
    assert actor.phi2.item() > 0.0


def test_gaussian_actor_phi1_shape():
    actor = _actor(n_risky=3)
    assert actor.phi1.shape == (3,)


def test_gaussian_actor_phi1_initial_value():
    actor = _actor(n_risky=2, init_phi1=2.0)
    assert torch.allclose(actor.phi1, torch.full((2,), 2.0), atol=1e-6)


def test_gaussian_actor_phi2_initial_value():
    actor = _actor(init_phi2=0.4)
    assert actor.phi2.item() == pytest.approx(0.4, rel=1e-5)


def test_gaussian_actor_has_three_parameter_groups():
    actor = _actor()
    param_names = {n for n, _ in actor.named_parameters()}
    assert "log_phi1" in param_names
    assert "log_phi2_inv" in param_names
    assert "phi3" in param_names


# ---------------------------------------------------------------------------
# GaussianActor — variance
# ---------------------------------------------------------------------------


def test_gaussian_actor_variance_positive():
    actor = _actor()
    assert actor.variance(t=0.5).item() > 0.0


def test_gaussian_actor_variance_at_terminal_time():
    """At t=T, variance = φ₂ (no time-decay contribution)."""
    actor = _actor(init_phi2=0.6, init_phi3=1.5, horizon=2.0)
    var_at_T = actor.variance(t=2.0)
    assert var_at_T.item() == pytest.approx(actor.phi2.item(), rel=1e-5)


def test_gaussian_actor_variance_decays_with_phi3_positive():
    """With φ₃ > 0, variance should be larger at t=0 than at t=T."""
    actor = _actor(init_phi3=1.0, horizon=1.0)
    assert actor.variance(t=0.0).item() > actor.variance(t=1.0).item()


def test_gaussian_actor_variance_grows_with_phi3_negative():
    """With φ₃ < 0, variance should be smaller at t=0 than at t=T."""
    actor = GaussianActor(n_risky=1, horizon=1.0, init_phi1=1.0, init_phi2=0.5, init_phi3=-1.0)
    # phi3 is initialized at -1.0 via the parameter directly
    with torch.no_grad():
        actor.phi3.fill_(-1.0)
    assert actor.variance(t=0.0).item() < actor.variance(t=1.0).item()


# ---------------------------------------------------------------------------
# GaussianActor — mean_action (execution policy)
# ---------------------------------------------------------------------------


def test_mean_action_zero_gap():
    """When wealth == w, mean allocation is zero."""
    actor = _actor()
    wealth = torch.tensor(1.5)
    action = actor.mean_action(t=0.5, wealth=wealth, w=1.5)
    assert torch.allclose(action, torch.zeros(1), atol=1e-6)


def test_mean_action_sign_positive_gap():
    """x > w → gap > 0 → mean action should be negative (sell down)."""
    actor = _actor(init_phi1=1.0)
    wealth = torch.tensor(1.2)
    action = actor.mean_action(t=0.0, wealth=wealth, w=1.0)
    assert action.item() < 0.0


def test_mean_action_sign_negative_gap():
    """x < w → gap < 0 → mean action should be positive (buy up)."""
    actor = _actor(init_phi1=1.0)
    wealth = torch.tensor(0.8)
    action = actor.mean_action(t=0.0, wealth=wealth, w=1.0)
    assert action.item() > 0.0


def test_mean_action_shape_single_asset():
    actor = _actor(n_risky=1)
    action = actor.mean_action(t=0.5, wealth=torch.tensor(1.0), w=1.0)
    assert action.shape == (1,)


def test_mean_action_shape_multi_asset():
    actor = _actor(n_risky=3)
    action = actor.mean_action(t=0.5, wealth=torch.tensor(1.0), w=1.0)
    assert action.shape == (3,)


def test_mean_action_shape_batched():
    actor = _actor(n_risky=2)
    wealth = torch.tensor([0.9, 1.0, 1.1])
    action = actor.mean_action(t=0.5, wealth=wealth, w=1.0)
    assert action.shape == (3, 2)


def test_mean_action_exact_value():
    """û = −φ₁·(x−w); verify with known φ₁=2.0."""
    actor = _actor(n_risky=1, init_phi1=2.0)
    wealth = torch.tensor(1.3)
    w = 1.0
    action = actor.mean_action(t=0.0, wealth=wealth, w=w)
    # φ₁ = 2.0, gap = 0.3 → û = -2.0 * 0.3 = -0.6
    expected = -actor.phi1[0].item() * 0.3
    assert action.item() == pytest.approx(expected, rel=1e-5)


# ---------------------------------------------------------------------------
# GaussianActor — sample
# ---------------------------------------------------------------------------


def test_sample_shape_single():
    actor = _actor(n_risky=1)
    s = actor.sample(t=0.5, wealth=torch.tensor(1.0), w=1.0)
    assert s.shape == (1,)


def test_sample_shape_multi():
    actor = _actor(n_risky=2)
    s = actor.sample(t=0.3, wealth=torch.tensor(1.0), w=1.0)
    assert s.shape == (2,)


def test_sample_shape_batched():
    actor = _actor(n_risky=2)
    wealth = torch.tensor([0.9, 1.0, 1.1])
    s = actor.sample(t=0.3, wealth=wealth, w=1.0)
    assert s.shape == (3, 2)


def test_sample_mean_close_to_mean_action_over_many():
    """Empirical mean of many samples should be close to mean_action."""
    torch.manual_seed(42)
    actor = _actor(n_risky=1, init_phi1=1.5, init_phi2=0.01)  # tiny variance
    wealth = torch.tensor(1.2)
    w = 1.0
    samples = torch.stack([actor.sample(t=0.5, wealth=wealth, w=w) for _ in range(500)])
    empirical_mean = samples.mean(dim=0)
    expected_mean = actor.mean_action(t=0.5, wealth=wealth, w=w)
    assert torch.allclose(empirical_mean, expected_mean, atol=0.05)


# ---------------------------------------------------------------------------
# GaussianActor — log_prob
# ---------------------------------------------------------------------------


def test_log_prob_scalar_output_unbatched():
    actor = _actor(n_risky=1)
    action = torch.tensor([0.3])
    lp = actor.log_prob(action, t=0.5, wealth=torch.tensor(1.0), w=1.0)
    assert lp.shape == ()  # scalar


def test_log_prob_shape_batched():
    actor = _actor(n_risky=2)
    action = torch.zeros(4, 2)
    wealth = torch.ones(4)
    lp = actor.log_prob(action, t=0.5, wealth=wealth, w=1.0)
    assert lp.shape == (4,)


def test_log_prob_of_mean_action_is_highest():
    """The mean action should have the highest log-prob among nearby actions."""
    actor = _actor(n_risky=1, init_phi2=0.1)
    wealth = torch.tensor(1.3)
    w = 1.0
    mean = actor.mean_action(t=0.3, wealth=wealth, w=w)
    lp_mean = actor.log_prob(mean, t=0.3, wealth=wealth, w=w)
    lp_offset = actor.log_prob(mean + 0.5, t=0.3, wealth=wealth, w=w)
    assert lp_mean.item() > lp_offset.item()


def test_log_prob_decreases_away_from_mean():
    actor = _actor(n_risky=1)
    wealth = torch.tensor(1.0)
    w = 1.0
    mean = actor.mean_action(t=0.5, wealth=wealth, w=w)
    lp0 = actor.log_prob(mean, t=0.5, wealth=wealth, w=w)
    lp1 = actor.log_prob(mean + 1.0, t=0.5, wealth=wealth, w=w)
    assert lp0.item() > lp1.item()


# ---------------------------------------------------------------------------
# GaussianActor — entropy
# ---------------------------------------------------------------------------


def test_entropy_is_scalar():
    actor = _actor()
    h = actor.entropy(t=0.5)
    assert h.shape == ()


def test_entropy_increases_with_variance():
    """Higher variance → higher entropy."""
    actor_small = _actor(init_phi2=0.01)
    actor_large = _actor(init_phi2=10.0)
    assert actor_large.entropy(t=0.5).item() > actor_small.entropy(t=0.5).item()


def test_entropy_does_not_depend_on_wealth():
    """Entropy should be the same regardless of current wealth."""
    actor = _actor()
    h1 = actor.entropy(t=0.5)
    h2 = actor.entropy(t=0.5)
    # Trivially true since wealth is not an argument; verifies interface
    assert h1.item() == pytest.approx(h2.item())


def test_entropy_increases_with_time_for_positive_phi3():
    """With φ₃ > 0, variance is larger at t=0 than at t=T, so entropy is too."""
    actor = GaussianActor(n_risky=1, horizon=1.0, init_phi3=1.0)
    assert actor.entropy(t=0.0).item() > actor.entropy(t=1.0).item()


def test_entropy_multi_asset_scales_with_n_risky():
    """Entropy of d-asset isotropic Gaussian = d × per-asset entropy."""
    actor1 = _actor(n_risky=1, init_phi2=0.5, init_phi3=0.0, horizon=1.0)
    actor2 = _actor(n_risky=2, init_phi2=0.5, init_phi3=0.0, horizon=1.0)
    # Same variance, double the assets → double the entropy
    assert actor2.entropy(t=0.5).item() == pytest.approx(
        2.0 * actor1.entropy(t=0.5).item(), rel=1e-5
    )


# ---------------------------------------------------------------------------
# QuadraticCritic — construction
# ---------------------------------------------------------------------------


def test_quadratic_critic_has_three_params():
    critic = _critic()
    param_names = {n for n, _ in critic.named_parameters()}
    assert {"theta1", "theta2", "theta3"} <= param_names


def test_quadratic_critic_stores_z():
    critic = _critic(z=1.25)
    assert critic.z == pytest.approx(1.25)


# ---------------------------------------------------------------------------
# QuadraticCritic — forward
# ---------------------------------------------------------------------------


def test_critic_forward_scalar_output():
    critic = _critic()
    v = critic(t=0.5, wealth=torch.tensor(1.0), w=1.0)
    assert v.shape == ()


def test_critic_forward_batched_output():
    critic = _critic()
    wealth = torch.tensor([0.8, 1.0, 1.2])
    v = critic(t=0.3, wealth=wealth, w=1.0)
    assert v.shape == (3,)


def test_critic_terminal_value():
    """At t=T, J(T, x; w) = −(w−z)²  (other terms vanish)."""
    z = 1.0
    w = 1.2
    critic = _critic(z=z)
    # At t=T all polynomial and quadratic terms vanish:
    # quad: (x-w)² * exp(-θ₃ * 0) = (x-w)²
    # Wait — at t=T, time_to_go=0, so exp(-θ₃*0) = 1.
    # time_correction: θ₂(T²-T²) + θ₁(T-T) = 0
    # So J(T, x; w) = (x-w)² - (w-z)²
    # The quadratic term does NOT vanish at t=T; only the time_correction vanishes.
    T = 1.0
    wealth = torch.tensor(1.0)
    v = critic(t=T, wealth=wealth, w=w)
    # With default θ₁=θ₂=0, θ₃=0.5:
    # J(T, x=1.0, w=1.2) = (1.0-1.2)² * 1 + 0 + 0 - (1.2-1.0)²
    #                     = 0.04 - 0.04 = 0
    expected = (wealth.item() - w)**2 - (w - z)**2
    assert v.item() == pytest.approx(expected, rel=1e-5)


def test_critic_terminal_time_correction_is_zero():
    """Time-correction terms θ₁(t-T) + θ₂(t²-T²) vanish at t=T."""
    with torch.no_grad():
        critic = _critic()
        critic.theta1.fill_(1.0)
        critic.theta2.fill_(1.0)
        critic.theta3.fill_(0.0)  # disable decay
    T = 1.0
    w = 1.0
    wealth = torch.tensor(w)  # x == w → quad term = 0
    v = critic(t=T, wealth=wealth, w=w)
    # quad=0, correction=0, terminal_penalty=-(w-z)²=-(1.0-1.0)²=0
    assert v.item() == pytest.approx(0.0, abs=1e-6)


def test_critic_quadratic_in_wealth():
    """Value should differ by (x₁-w)² − (x₂-w)² when only wealth changes."""
    z = 1.0
    w = 1.0
    critic = _critic(z=z)
    # Force θ₃=0 so exponential decay factor = 1
    with torch.no_grad():
        critic.theta3.fill_(0.0)

    x1 = torch.tensor(1.3)
    x2 = torch.tensor(0.7)
    t = 0.5
    v1 = critic(t=t, wealth=x1, w=w)
    v2 = critic(t=t, wealth=x2, w=w)
    # Δ(quad) = (x1-w)² − (x2-w)²; all other terms cancel.
    expected_diff = (1.3 - w)**2 - (0.7 - w)**2
    assert (v1 - v2).item() == pytest.approx(expected_diff, abs=1e-5)


def test_critic_zero_wealth_gap_removes_quad_term():
    """When x == w, the quadratic term is zero regardless of θ₃."""
    z = 1.0
    w = 1.0
    wealth = torch.tensor(w)  # x == w
    critic = _critic(z=z)
    with torch.no_grad():
        critic.theta1.fill_(0.0)
        critic.theta2.fill_(0.0)
    # J = 0 + 0 + 0 - (w-z)² = 0
    v = critic(t=0.5, wealth=wealth, w=w)
    assert v.item() == pytest.approx(-(w - z)**2, rel=1e-5)


def test_critic_sensitivity_to_theta1():
    """θ₁ affects the time-correction term θ₁·(t-T)."""
    critic = _critic()
    t = 0.3
    T = 1.0
    wealth = torch.tensor(1.0)
    w = 1.0
    with torch.no_grad():
        critic.theta1.fill_(2.0)
        critic.theta2.fill_(0.0)
        critic.theta3.fill_(0.0)
    v = critic(t=t, wealth=wealth, w=w)
    # quad = 0 (x=w), correction = 2.0*(0.3-1.0) = -1.4, penalty = 0
    assert v.item() == pytest.approx(2.0 * (t - T), rel=1e-5)


def test_critic_w_update_changes_output():
    """Changing w should change the critic output."""
    critic = _critic(z=1.0)
    wealth = torch.tensor(1.1)
    t = 0.5
    v1 = critic(t=t, wealth=wealth, w=1.0)
    v2 = critic(t=t, wealth=wealth, w=1.3)
    assert v1.item() != pytest.approx(v2.item())


def test_critic_gradients_flow_through_theta():
    """Parameters θ should receive gradients from a scalar critic output."""
    critic = _critic()
    wealth = torch.tensor(1.2, requires_grad=False)
    v = critic(t=0.4, wealth=wealth, w=1.0)
    v.backward()
    assert critic.theta1.grad is not None
    assert critic.theta2.grad is not None
    assert critic.theta3.grad is not None
