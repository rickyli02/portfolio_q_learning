"""Unit tests for src/algos/oracle_mv.py."""

import math

import pytest
import torch

from src.algos.oracle_mv import (
    OracleMVPolicy,
    compute_oracle_coefficients,
    oracle_action,
    run_oracle_episode,
)
from src.config.schema import AssetConfig, EnvConfig
from src.envs.gbm_env import GBMPortfolioEnv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_single_asset_policy(
    mu: float = 0.1,
    sigma_val: float = 0.2,
    r: float = 0.05,
    horizon: float = 1.0,
    gamma_embed: float = 1.1,
) -> OracleMVPolicy:
    return OracleMVPolicy.from_env_params(
        mu=[mu],
        sigma=[[sigma_val]],
        r=r,
        horizon=horizon,
        gamma_embed=gamma_embed,
    )


def _make_env(n_risky: int = 1, n_steps: int = 10) -> GBMPortfolioEnv:
    if n_risky == 1:
        mu, sigma = [0.1], [[0.2]]
    else:
        mu = [0.1, 0.08]
        sigma = [[0.2, 0.0], [0.0, 0.15]]
    cfg = EnvConfig(
        horizon=1.0,
        n_steps=n_steps,
        initial_wealth=1.0,
        mu=mu,
        sigma=sigma,
        assets=AssetConfig(n_risky=n_risky, include_risk_free=True, risk_free_rate=0.05),
    )
    return GBMPortfolioEnv(cfg)


# ---------------------------------------------------------------------------
# compute_oracle_coefficients — shapes and types
# ---------------------------------------------------------------------------


def test_coefficients_single_asset_shapes():
    coeffs = compute_oracle_coefficients(
        mu=[0.1], sigma=[[0.2]], r=0.05, horizon=1.0, gamma_embed=1.1
    )
    assert coeffs.B.shape == (1,)
    assert coeffs.cov.shape == (1, 1)
    assert coeffs.sensitivity.shape == (1,)


def test_coefficients_multi_asset_shapes():
    coeffs = compute_oracle_coefficients(
        mu=[0.1, 0.08],
        sigma=[[0.2, 0.0], [0.0, 0.15]],
        r=0.05,
        horizon=1.0,
        gamma_embed=1.1,
    )
    assert coeffs.B.shape == (2,)
    assert coeffs.cov.shape == (2, 2)
    assert coeffs.sensitivity.shape == (2,)


def test_coefficients_B_equals_mu_minus_r():
    """B = mu - r (no Ito correction: env.mu is the price-SDE drift b)."""
    mu = [0.1]
    r = 0.05
    coeffs = compute_oracle_coefficients(mu=mu, sigma=[[0.2]], r=r, horizon=1.0, gamma_embed=1.0)
    expected_B = torch.tensor([mu[0] - r], dtype=torch.float64)
    assert torch.allclose(coeffs.B, expected_B)


def test_coefficients_stored_scalars():
    coeffs = compute_oracle_coefficients(
        mu=[0.1], sigma=[[0.2]], r=0.05, horizon=2.0, gamma_embed=1.3
    )
    assert coeffs.r == pytest.approx(0.05)
    assert coeffs.horizon == pytest.approx(2.0)
    assert coeffs.gamma_embed == pytest.approx(1.3)


def test_coefficients_sensitivity_satisfies_normal_equation():
    """cov @ sensitivity should equal B (verifies the linear solve is correct)."""
    coeffs = compute_oracle_coefficients(
        mu=[0.1, 0.08],
        sigma=[[0.2, 0.0], [0.0, 0.15]],
        r=0.05,
        horizon=1.0,
        gamma_embed=1.0,
    )
    # [σσᵀ] sensitivity = B  ⟺  sensitivity = [σσᵀ]⁻¹ B
    reconstructed_B = coeffs.cov @ coeffs.sensitivity
    assert torch.allclose(reconstructed_B, coeffs.B, atol=1e-10)


def test_coefficients_singular_sigma_raises():
    with pytest.raises(ValueError, match="singular"):
        compute_oracle_coefficients(
            mu=[0.1, 0.08],
            sigma=[[0.2, 0.2], [0.2, 0.2]],  # rank-1 matrix
            r=0.05,
            horizon=1.0,
            gamma_embed=1.0,
        )


# ---------------------------------------------------------------------------
# oracle_action — sanity and shape checks
# ---------------------------------------------------------------------------


def test_zero_risk_premium_gives_zero_action():
    """When B = 0 (mu == r), oracle allocation is zero for any (t, x)."""
    r = 0.05
    # Setting mu = r means B = mu - r = 0
    coeffs = compute_oracle_coefficients(
        mu=[r], sigma=[[0.2]], r=r, horizon=1.0, gamma_embed=1.0
    )
    wealth = torch.tensor(1.0)
    action = oracle_action(coeffs, t=0.3, wealth=wealth)
    assert torch.allclose(action, torch.zeros(1, dtype=action.dtype), atol=1e-10)


def test_zero_risk_premium_multi_asset():
    """Multi-asset: all excess returns zero → zero allocation vector."""
    r = 0.05
    coeffs = compute_oracle_coefficients(
        mu=[r, r], sigma=[[0.2, 0.0], [0.0, 0.15]], r=r, horizon=1.0, gamma_embed=1.0
    )
    wealth = torch.tensor(1.0)
    action = oracle_action(coeffs, t=0.0, wealth=wealth)
    assert torch.allclose(action, torch.zeros(2, dtype=action.dtype), atol=1e-10)


def test_action_output_shape_unbatched():
    policy = _make_single_asset_policy()
    wealth = torch.tensor(1.0)
    action = oracle_action(policy.coeffs, t=0.5, wealth=wealth)
    assert action.shape == (1,)


def test_action_output_shape_multi_asset_unbatched():
    coeffs = compute_oracle_coefficients(
        mu=[0.1, 0.08], sigma=[[0.2, 0.0], [0.0, 0.15]],
        r=0.05, horizon=1.0, gamma_embed=1.1
    )
    wealth = torch.tensor(1.0)
    action = oracle_action(coeffs, t=0.0, wealth=wealth)
    assert action.shape == (2,)


def test_action_output_shape_batched():
    policy = _make_single_asset_policy()
    wealth = torch.tensor([0.9, 1.0, 1.1])
    action = oracle_action(policy.coeffs, t=0.5, wealth=wealth)
    assert action.shape == (3, 1)


def test_action_sign_positive_excess_return():
    """With mu > r and wealth below virtual target, oracle should go long."""
    gamma_embed = 1.5
    r = 0.05
    horizon = 1.0
    t = 0.0
    # Virtual target at t=0: gamma_embed * exp(-r*T) = 1.5 * exp(-0.05)
    virtual_target_t0 = gamma_embed * math.exp(-r * horizon)
    wealth = torch.tensor(virtual_target_t0 - 0.2)  # below target -> gap > 0
    coeffs = compute_oracle_coefficients(
        mu=[0.1], sigma=[[0.2]], r=r, horizon=horizon, gamma_embed=gamma_embed
    )
    action = oracle_action(coeffs, t=t, wealth=wealth)
    # B > 0, gap > 0 -> action > 0
    assert action.item() > 0.0


def test_action_sign_flips_above_virtual_target():
    """Wealth above virtual target should produce negative (short) allocation."""
    gamma_embed = 1.0
    r = 0.05
    horizon = 1.0
    t = 0.0
    virtual_target_t0 = gamma_embed * math.exp(-r * horizon)
    wealth = torch.tensor(virtual_target_t0 + 0.2)  # above target -> gap < 0
    coeffs = compute_oracle_coefficients(
        mu=[0.1], sigma=[[0.2]], r=r, horizon=horizon, gamma_embed=gamma_embed
    )
    action = oracle_action(coeffs, t=t, wealth=wealth)
    assert action.item() < 0.0


def test_action_exact_single_asset():
    """Verify the analytic formula value for a known simple case."""
    mu_val = 0.1
    sigma_val = 0.2
    r = 0.05
    T = 1.0
    gamma = 1.2
    t = 0.0

    # Expected: u = (mu - r)/sigma^2 * (gamma * exp(-r*T) - x)
    x = 1.0
    B_val = mu_val - r
    expected = B_val / sigma_val**2 * (gamma * math.exp(-r * T) - x)

    coeffs = compute_oracle_coefficients(
        mu=[mu_val], sigma=[[sigma_val]], r=r, horizon=T, gamma_embed=gamma
    )
    wealth = torch.tensor(x, dtype=torch.float32)
    action = oracle_action(coeffs, t=t, wealth=wealth)
    assert action.item() == pytest.approx(expected, rel=1e-5)


def test_action_at_terminal_time():
    """At t = T, the discount factor is 1; action depends on (gamma - x)."""
    gamma = 1.1
    r = 0.05
    T = 1.0
    sigma_val = 0.2
    mu_val = 0.1

    coeffs = compute_oracle_coefficients(
        mu=[mu_val], sigma=[[sigma_val]], r=r, horizon=T, gamma_embed=gamma
    )
    wealth = torch.tensor(1.0, dtype=torch.float32)
    action = oracle_action(coeffs, t=T, wealth=wealth)

    # At t=T: virtual_target = gamma * exp(0) = gamma; gap = gamma - x = 0.1
    b_val = mu_val - r
    expected = b_val / sigma_val**2 * (gamma - 1.0)
    assert action.item() == pytest.approx(expected, rel=1e-5)


def test_action_exact_correlated_multi_asset():
    """Exact formula check for a correlated 2-asset case.

    Parameters chosen so sensitivity can be verified by hand:
      sigma = [[0.20, 0.05], [0.00, 0.15]]
      cov   = [[0.0425, 0.0075], [0.0075, 0.0225]]   (det = 0.0009)
      B     = [0.05, 0.03]
      sensitivity = cov^{-1} B = [1.0, 1.0]  (exact, verified by hand)

    So u(t=0, x=1.0) = [1.0, 1.0] * (gamma * exp(-r*T) - 1.0).
    """
    mu = [0.10, 0.08]
    sigma = [[0.20, 0.05], [0.00, 0.15]]
    r = 0.05
    T = 1.0
    gamma = 1.2
    x = 1.0
    t = 0.0

    # Hand-derived: sensitivity = [1.0, 1.0] for these parameters
    gap = gamma * math.exp(-r * T) - x
    expected = torch.tensor([gap, gap], dtype=torch.float32)

    coeffs = compute_oracle_coefficients(mu=mu, sigma=sigma, r=r, horizon=T, gamma_embed=gamma)
    wealth = torch.tensor(x, dtype=torch.float32)
    action = oracle_action(coeffs, t=t, wealth=wealth)

    assert action.shape == (2,)
    assert torch.allclose(action, expected, atol=1e-5)


def test_action_correlated_sensitivity_satisfies_normal_equation():
    """For the correlated case, verify cov @ sensitivity == B holds."""
    coeffs = compute_oracle_coefficients(
        mu=[0.10, 0.08],
        sigma=[[0.20, 0.05], [0.00, 0.15]],
        r=0.05, horizon=1.0, gamma_embed=1.2,
    )
    assert torch.allclose(coeffs.cov @ coeffs.sensitivity, coeffs.B, atol=1e-10)


# ---------------------------------------------------------------------------
# dtype behavior
# ---------------------------------------------------------------------------


def test_coefficients_are_float64():
    """compute_oracle_coefficients should always return float64 tensors."""
    coeffs = compute_oracle_coefficients(
        mu=[0.1], sigma=[[0.2]], r=0.05, horizon=1.0, gamma_embed=1.1
    )
    assert coeffs.B.dtype == torch.float64
    assert coeffs.cov.dtype == torch.float64
    assert coeffs.sensitivity.dtype == torch.float64


def test_oracle_action_inherits_wealth_dtype_float32():
    """oracle_action output dtype should match the wealth tensor dtype."""
    coeffs = compute_oracle_coefficients(
        mu=[0.1], sigma=[[0.2]], r=0.05, horizon=1.0, gamma_embed=1.1
    )
    wealth = torch.tensor(1.0, dtype=torch.float32)
    action = oracle_action(coeffs, t=0.5, wealth=wealth)
    assert action.dtype == torch.float32


def test_oracle_action_inherits_wealth_dtype_float64():
    """oracle_action output dtype should match the wealth tensor dtype."""
    coeffs = compute_oracle_coefficients(
        mu=[0.1], sigma=[[0.2]], r=0.05, horizon=1.0, gamma_embed=1.1
    )
    wealth = torch.tensor(1.0, dtype=torch.float64)
    action = oracle_action(coeffs, t=0.5, wealth=wealth)
    assert action.dtype == torch.float64


def test_oracle_action_batched_dtype_matches_wealth():
    """Batched oracle_action dtype should match the batched wealth tensor dtype."""
    coeffs = compute_oracle_coefficients(
        mu=[0.1], sigma=[[0.2]], r=0.05, horizon=1.0, gamma_embed=1.1
    )
    wealth = torch.tensor([0.9, 1.0, 1.1], dtype=torch.float32)
    action = oracle_action(coeffs, t=0.5, wealth=wealth)
    assert action.dtype == torch.float32
    assert action.shape == (3, 1)


# ---------------------------------------------------------------------------
# OracleMVPolicy — class interface
# ---------------------------------------------------------------------------


def test_policy_callable():
    policy = _make_single_asset_policy()
    action = policy(t=0.0, wealth=torch.tensor(1.0))
    assert action.shape == (1,)


def test_policy_from_env_params_matches_direct():
    """from_env_params should produce the same result as compute_oracle_coefficients."""
    mu, sigma, r, T, gamma = [0.1], [[0.2]], 0.05, 1.0, 1.1
    policy = OracleMVPolicy.from_env_params(mu=mu, sigma=sigma, r=r, horizon=T, gamma_embed=gamma)
    coeffs_direct = compute_oracle_coefficients(mu=mu, sigma=sigma, r=r, horizon=T, gamma_embed=gamma)
    wealth = torch.tensor(1.0, dtype=torch.float32)
    assert torch.allclose(
        oracle_action(coeffs_direct, t=0.5, wealth=wealth),
        policy(t=0.5, wealth=wealth),
        atol=1e-6,
    )


# ---------------------------------------------------------------------------
# run_oracle_episode — rollout
# ---------------------------------------------------------------------------


def test_run_oracle_episode_output_keys():
    env = _make_env(n_risky=1, n_steps=5)
    policy = _make_single_asset_policy()
    result = run_oracle_episode(policy, env, seed=0)
    assert set(result.keys()) == {"wealth_path", "actions", "terminal_wealth", "times"}


def test_run_oracle_episode_wealth_path_shape():
    n_steps = 8
    env = _make_env(n_risky=1, n_steps=n_steps)
    policy = _make_single_asset_policy()
    result = run_oracle_episode(policy, env, seed=0)
    assert result["wealth_path"].shape == (n_steps + 1,)


def test_run_oracle_episode_actions_shape():
    n_steps = 6
    env = _make_env(n_risky=1, n_steps=n_steps)
    policy = _make_single_asset_policy()
    result = run_oracle_episode(policy, env, seed=0)
    assert result["actions"].shape == (n_steps, 1)


def test_run_oracle_episode_multi_asset_actions_shape():
    n_steps = 4
    env = _make_env(n_risky=2, n_steps=n_steps)
    policy = OracleMVPolicy.from_env_params(
        mu=[0.1, 0.08], sigma=[[0.2, 0.0], [0.0, 0.15]],
        r=0.05, horizon=1.0, gamma_embed=1.1
    )
    result = run_oracle_episode(policy, env, seed=0)
    assert result["actions"].shape == (n_steps, 2)


def test_run_oracle_episode_wealth_path_starts_at_x0():
    env = _make_env(n_steps=5)
    policy = _make_single_asset_policy()
    result = run_oracle_episode(policy, env, seed=0)
    assert result["wealth_path"][0].item() == pytest.approx(1.0, rel=1e-5)


def test_run_oracle_episode_terminal_wealth_matches_path_end():
    env = _make_env(n_steps=5)
    policy = _make_single_asset_policy()
    result = run_oracle_episode(policy, env, seed=0)
    assert result["terminal_wealth"].item() == pytest.approx(
        result["wealth_path"][-1].item(), rel=1e-6
    )


def test_run_oracle_episode_deterministic_with_same_seed():
    """Same seed should produce identical trajectories."""
    env = _make_env(n_steps=10)
    policy = _make_single_asset_policy()
    result1 = run_oracle_episode(policy, env, seed=42)
    result2 = run_oracle_episode(policy, env, seed=42)
    assert torch.allclose(result1["wealth_path"], result2["wealth_path"])
    assert torch.allclose(result1["actions"], result2["actions"])


def test_run_oracle_episode_different_seeds_differ():
    """Different seeds should almost surely produce different trajectories."""
    env = _make_env(n_steps=10)
    policy = _make_single_asset_policy()
    result1 = run_oracle_episode(policy, env, seed=0)
    result2 = run_oracle_episode(policy, env, seed=99)
    assert not torch.allclose(result1["wealth_path"], result2["wealth_path"])


def test_run_oracle_episode_times_shape():
    n_steps = 5
    env = _make_env(n_steps=n_steps)
    policy = _make_single_asset_policy()
    result = run_oracle_episode(policy, env, seed=0)
    assert result["times"].shape == (n_steps,)


def test_zero_premium_oracle_gives_zero_actions_in_rollout():
    """With zero excess return (mu==r), oracle should produce zero actions."""
    r = 0.05
    env = _make_env(n_risky=1, n_steps=5)
    policy = OracleMVPolicy.from_env_params(
        mu=[r],  # B = mu - r = 0
        sigma=[[0.2]],
        r=r,
        horizon=1.0,
        gamma_embed=1.1,
    )
    result = run_oracle_episode(policy, env, seed=0)
    assert torch.allclose(result["actions"], torch.zeros_like(result["actions"]), atol=1e-6)
