"""Float32 vs float64 dtype-sensitivity comparison — Phase 17C.

Provides a deterministic diagnostic seam for measuring how much the core
actor, critic, and oracle computations differ between float32 and float64
precision.  The results quantify precision sensitivity without modifying any
model math or warning thresholds.

This module is diagnostic only; it has no effect on training or evaluation.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import torch

from src.algos.oracle_mv import compute_oracle_coefficients, oracle_action
from src.models.gaussian_actor import GaussianActor
from src.models.quadratic_critic import QuadraticCritic


@dataclass(frozen=True)
class DtypeComparisonResult:
    """Scalar absolute-difference diagnostics: float32 vs float64 outputs.

    All fields are non-negative scalar floats.  Larger values indicate greater
    dtype sensitivity for that component.

    Note on oracle fields: ``compute_oracle_coefficients`` always promotes
    inputs to float64 internally, so there is no separate float32/float64
    sensitivity code path.  The oracle is represented here only by
    ``oracle_action_delta``, which measures the difference in ``oracle_action``
    output when called with a float32 wealth tensor vs a float64 wealth tensor
    (the output dtype follows the wealth dtype in that function).

    Attributes:
        actor_mean_action_delta:  Max absolute difference in ``mean_action``
            output between a float32 actor instance and a separate float64
            actor instance with identical initial parameter values.
        actor_variance_delta:     Absolute difference in ``variance`` output
            between the same two actor instances.
        critic_forward_delta:     Absolute difference in ``QuadraticCritic.forward``
            output between a float32 critic instance and a separate float64
            critic instance with identical initial parameter values.
        oracle_action_delta:      Max absolute difference in ``oracle_action``
            output when evaluated with float32 wealth vs float64 wealth.
            The oracle uses float64 coefficients in both cases; the difference
            reflects the float32 cast of those coefficients in the output path.
    """

    actor_mean_action_delta: float
    actor_variance_delta: float
    critic_forward_delta: float
    oracle_action_delta: float


def compare_dtype_outputs(
    mu: list[float],
    sigma: list[list[float]],
    r: float,
    horizon: float,
    gamma_embed: float,
    t: float,
    wealth: float,
    w: float,
    init_phi1: float = 1.0,
    init_phi2: float = 0.5,
    init_phi3: float = 0.0,
    init_theta1: float = 0.0,
    init_theta2: float = 0.0,
    init_theta3: float = 0.5,
    target_return_z: float = 1.0,
) -> DtypeComparisonResult:
    """Run core component outputs in float32 and float64, return absolute deltas.

    Creates two independent GaussianActor instances (float32 and float64) and
    two independent QuadraticCritic instances with identical initial parameter
    values, then evaluates each on inputs cast to the respective dtype.  Both
    instances are genuinely separate Python objects; no in-place dtype mutation
    is used.

    Oracle coefficients always use float64 internally.  The oracle comparison
    measures only ``oracle_action`` output with float32 vs float64 wealth inputs.

    All computations run under ``torch.no_grad()``.  Numerical-stability
    warnings from the comparison are suppressed so the caller sees only the
    result.

    Args:
        mu:              Price-process drift vector, length n_risky.
        sigma:           Volatility factor matrix, shape (n_risky, n_risky).
        r:               Risk-free rate.
        horizon:         Investment horizon T.
        gamma_embed:     Oracle embedding scalar γ.
        t:               Time point for evaluation (scalar).
        wealth:          Wealth for evaluation (scalar).
        w:               Lagrange / target-wealth parameter (scalar).
        init_phi1:       Initial φ₁ value for both actor instances (> 0).
        init_phi2:       Initial φ₂ value for both actor instances (> 0).
        init_phi3:       Initial φ₃ value for both actor instances.
        init_theta1:     Initial θ₁ for both critic instances.
        init_theta2:     Initial θ₂ for both critic instances.
        init_theta3:     Initial θ₃ for both critic instances.
        target_return_z: Target return z for both critic instances.

    Returns:
        ``DtypeComparisonResult`` with scalar absolute deltas for each component.
    """
    n_risky = len(mu)

    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # ------------------------------------------------------------------
        # Actor: two separate instances — float32 and float64
        # nn.Module.double() is in-place so we create each independently.
        # ------------------------------------------------------------------
        actor32 = GaussianActor(
            n_risky=n_risky, horizon=horizon,
            init_phi1=init_phi1, init_phi2=init_phi2, init_phi3=init_phi3,
        )  # default dtype = float32
        actor64 = GaussianActor(
            n_risky=n_risky, horizon=horizon,
            init_phi1=init_phi1, init_phi2=init_phi2, init_phi3=init_phi3,
        ).double()  # separate instance promoted to float64

        w32 = torch.tensor(w, dtype=torch.float32)
        w64 = torch.tensor(w, dtype=torch.float64)
        x32 = torch.tensor(wealth, dtype=torch.float32)
        x64 = torch.tensor(wealth, dtype=torch.float64)

        mean32 = actor32.mean_action(t=t, wealth=x32, w=w32)
        mean64 = actor64.mean_action(t=t, wealth=x64, w=w64).float()
        actor_mean_action_delta = float((mean32 - mean64).abs().max().item())

        var32 = actor32.variance(t=t)
        var64 = actor64.variance(t=t).float()
        actor_variance_delta = float((var32 - var64).abs().item())

        # ------------------------------------------------------------------
        # Critic: two separate instances — float32 and float64
        # ------------------------------------------------------------------
        critic32 = QuadraticCritic(
            horizon=horizon, target_return_z=target_return_z,
            init_theta1=init_theta1, init_theta2=init_theta2, init_theta3=init_theta3,
        )
        critic64 = QuadraticCritic(
            horizon=horizon, target_return_z=target_return_z,
            init_theta1=init_theta1, init_theta2=init_theta2, init_theta3=init_theta3,
        ).double()

        val32 = critic32.forward(t=t, wealth=x32, w=w32)
        val64 = critic64.forward(t=t, wealth=x64, w=w64).float()
        critic_forward_delta = float((val32 - val64).abs().item())

        # ------------------------------------------------------------------
        # Oracle: one set of float64 coefficients (the only code path), but
        # oracle_action output dtype follows the wealth tensor dtype.
        # Compare output for float32 wealth vs float64 wealth.
        # ------------------------------------------------------------------
        coeffs = compute_oracle_coefficients(
            mu=[float(v) for v in mu],
            sigma=[[float(v) for v in row] for row in sigma],
            r=r, horizon=horizon, gamma_embed=gamma_embed,
        )

        act32 = oracle_action(coeffs, t=t, wealth=x32)   # output cast to float32
        act64 = oracle_action(coeffs, t=t, wealth=x64)   # output stays float64
        oracle_action_delta = float((act32 - act64.float()).abs().max().item())

    return DtypeComparisonResult(
        actor_mean_action_delta=actor_mean_action_delta,
        actor_variance_delta=actor_variance_delta,
        critic_forward_delta=critic_forward_delta,
        oracle_action_delta=oracle_action_delta,
    )
