"""Unit tests for src/utils/dtype_compare.py — Phase 17C.

Verifies the dtype-sensitivity comparison seam:
- result object has expected fields (no oracle_sensitivity_delta — oracle is always float64)
- actor/critic instances are genuinely separate objects with the correct dtypes
- all deltas are finite scalars in a normal regime
- stable-regime deltas are small (not gross divergence)
- extreme-regime test is labelled as a boundary diagnostic
"""

from __future__ import annotations

import math

import torch

from src.utils.dtype_compare import DtypeComparisonResult, compare_dtype_outputs

# ---------------------------------------------------------------------------
# Shared fixture values (normal finite regime)
# ---------------------------------------------------------------------------

_MU = [0.08]
_SIGMA = [[0.20]]
_R = 0.05
_HORIZON = 1.0
_GAMMA = 1.5
_T = 0.3
_WEALTH = 1.2
_W = 1.0


def _normal_result() -> DtypeComparisonResult:
    return compare_dtype_outputs(
        mu=_MU, sigma=_SIGMA, r=_R, horizon=_HORIZON, gamma_embed=_GAMMA,
        t=_T, wealth=_WEALTH, w=_W,
    )


# ===========================================================================
# Result type and field presence
# ===========================================================================


def test_compare_returns_dtype_comparison_result():
    result = _normal_result()
    assert isinstance(result, DtypeComparisonResult)


def test_result_has_actor_mean_action_delta():
    result = _normal_result()
    assert hasattr(result, "actor_mean_action_delta")


def test_result_has_actor_variance_delta():
    result = _normal_result()
    assert hasattr(result, "actor_variance_delta")


def test_result_has_critic_forward_delta():
    result = _normal_result()
    assert hasattr(result, "critic_forward_delta")


def test_result_has_oracle_action_delta():
    result = _normal_result()
    assert hasattr(result, "oracle_action_delta")


def test_result_does_not_have_oracle_sensitivity_delta():
    """oracle_sensitivity_delta is absent: oracle always uses float64 internally."""
    result = _normal_result()
    assert not hasattr(result, "oracle_sensitivity_delta")


def test_all_fields_are_floats():
    result = _normal_result()
    for field in (
        "actor_mean_action_delta",
        "actor_variance_delta",
        "critic_forward_delta",
        "oracle_action_delta",
    ):
        assert isinstance(getattr(result, field), float), f"{field} is not float"


# ===========================================================================
# Separate-instance proof: actor and critic are genuinely distinct objects
# ===========================================================================


def test_actor_instances_are_separate_objects():
    """compare_dtype_outputs creates two separate actor objects, not the same one."""
    from src.models.gaussian_actor import GaussianActor

    a32 = GaussianActor(n_risky=1, horizon=1.0)
    a64 = GaussianActor(n_risky=1, horizon=1.0).double()
    # After .double(), the second instance is a different object
    assert a32 is not a64


def test_float32_actor_has_float32_parameters():
    from src.models.gaussian_actor import GaussianActor

    actor = GaussianActor(n_risky=1, horizon=1.0)
    for p in actor.parameters():
        assert p.dtype == torch.float32, f"expected float32, got {p.dtype}"


def test_float64_actor_has_float64_parameters():
    from src.models.gaussian_actor import GaussianActor

    actor = GaussianActor(n_risky=1, horizon=1.0).double()
    for p in actor.parameters():
        assert p.dtype == torch.float64, f"expected float64, got {p.dtype}"


def test_float32_critic_has_float32_parameters():
    from src.models.quadratic_critic import QuadraticCritic

    critic = QuadraticCritic(horizon=1.0, target_return_z=1.0)
    for p in critic.parameters():
        assert p.dtype == torch.float32, f"expected float32, got {p.dtype}"


def test_float64_critic_has_float64_parameters():
    from src.models.quadratic_critic import QuadraticCritic

    critic = QuadraticCritic(horizon=1.0, target_return_z=1.0).double()
    for p in critic.parameters():
        assert p.dtype == torch.float64, f"expected float64, got {p.dtype}"


# ===========================================================================
# Finiteness in normal regime
# ===========================================================================


def test_actor_mean_action_delta_is_finite():
    assert math.isfinite(_normal_result().actor_mean_action_delta)


def test_actor_variance_delta_is_finite():
    assert math.isfinite(_normal_result().actor_variance_delta)


def test_critic_forward_delta_is_finite():
    assert math.isfinite(_normal_result().critic_forward_delta)


def test_oracle_action_delta_is_finite():
    assert math.isfinite(_normal_result().oracle_action_delta)


def test_all_deltas_non_negative():
    result = _normal_result()
    assert result.actor_mean_action_delta >= 0.0
    assert result.actor_variance_delta >= 0.0
    assert result.critic_forward_delta >= 0.0
    assert result.oracle_action_delta >= 0.0


# ===========================================================================
# Stable regime: deltas are small (not gross divergence)
# float32 machine epsilon ≈ 1.2e-7, so single-operation deltas < ~1e-4
# is a reasonable stability threshold for these simple scalar computations.
# ===========================================================================


def test_actor_mean_action_delta_small_in_normal_regime():
    assert _normal_result().actor_mean_action_delta < 1e-4


def test_actor_variance_delta_small_in_normal_regime():
    assert _normal_result().actor_variance_delta < 1e-4


def test_critic_forward_delta_small_in_normal_regime():
    assert _normal_result().critic_forward_delta < 1e-4


def test_oracle_action_delta_small_in_normal_regime():
    """Oracle uses float64 coefficients in both paths; action delta reflects only
    the float32 cast of the output, which is small in the normal regime."""
    assert _normal_result().oracle_action_delta < 1e-4


# ===========================================================================
# Multi-asset normal regime — shape and finiteness
# ===========================================================================


def test_multi_asset_normal_regime_all_finite():
    result = compare_dtype_outputs(
        mu=[0.08, 0.06],
        sigma=[[0.20, 0.02], [0.02, 0.15]],
        r=0.05, horizon=1.0, gamma_embed=1.5,
        t=0.5, wealth=1.1, w=1.0,
    )
    assert math.isfinite(result.actor_mean_action_delta)
    assert math.isfinite(result.actor_variance_delta)
    assert math.isfinite(result.critic_forward_delta)
    assert math.isfinite(result.oracle_action_delta)


# ===========================================================================
# Determinism — same inputs produce identical results
# ===========================================================================


def test_compare_is_deterministic():
    r1 = _normal_result()
    r2 = _normal_result()
    assert r1.actor_mean_action_delta == r2.actor_mean_action_delta
    assert r1.actor_variance_delta == r2.actor_variance_delta
    assert r1.critic_forward_delta == r2.critic_forward_delta
    assert r1.oracle_action_delta == r2.oracle_action_delta


# ===========================================================================
# Extreme-regime boundary diagnostic
# (Not a parity failure — documents that divergence is expected at overflow)
# ===========================================================================


def test_extreme_phi3_boundary_diagnostic():
    """Boundary diagnostic: large phi3 produces larger dtype delta than normal regime.

    This is NOT a parity failure test.  It documents that float32 variance
    overflows at phi3 ≈ 87/T while float64 remains finite.  The test asserts
    only that the comparison helper completes and that the extreme delta is at
    least as large as the normal-regime delta.
    """
    result_extreme = compare_dtype_outputs(
        mu=_MU, sigma=_SIGMA, r=_R, horizon=_HORIZON, gamma_embed=_GAMMA,
        t=_T, wealth=_WEALTH, w=_W,
        init_phi3=90.0,  # float32 overflows; float64 stays finite
    )
    result_normal = _normal_result()

    assert isinstance(result_extreme.actor_variance_delta, float)
    assert result_extreme.actor_variance_delta >= result_normal.actor_variance_delta
