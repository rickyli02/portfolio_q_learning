"""Trainer/demo pipeline stress tests across broader GBM parameter regimes — Phase 16B.

Exercises the full ctrl_outer_loop pipeline across cases that differ in
drift magnitude, volatility, initial wealth, asset count, and correlation
structure.  All tests assert concrete pipeline invariants rather than
just checking for no-crash behavior.

Cases A–I follow the task specification exactly.  Sigma matrices are
defined below as module-level constants with inline covariance commentary.
"""

from __future__ import annotations

import math

import pytest
import torch

from src.config.schema import AssetConfig, EnvConfig
from src.envs.gbm_env import GBMPortfolioEnv
from src.models.gaussian_actor import GaussianActor
from src.models.quadratic_critic import QuadraticCritic
from src.train.ctrl_outer_loop import CTRLOuterLoopResult
from src.train.ctrl_state import CTRLTrainerState, CTRLTrainerSnapshot


# ---------------------------------------------------------------------------
# Sigma matrices for multi-asset cases
# ---------------------------------------------------------------------------

# Case E — correlated 2-asset (from task spec)
# cov ≈ [[0.09, 0.054], [0.054, 0.0900]]; positive off-diagonal
_SIGMA_E = [[0.30, 0.00], [0.18, 0.24]]

# Case F — strongly positively correlated, n_risky=5
# Structure: one dominant common factor (col 0, loading 0.30) + small
# idiosyncratic terms on cols 1–4 for assets 0–3; asset 4 has no idio.
# cov[i,j] = 0.09 for i≠j; cov[i,i] ≈ 0.0925 (0-3) or 0.09 (4).
# All eigenvalues of cov positive (PD); pairwise corr ≈ 0.97.
_SIGMA_F = [
    [0.30, 0.05, 0.00, 0.00, 0.00],
    [0.30, 0.00, 0.05, 0.00, 0.00],
    [0.30, 0.00, 0.00, 0.05, 0.00],
    [0.30, 0.00, 0.00, 0.00, 0.05],
    [0.30, 0.00, 0.00, 0.00, 0.00],
]

# Case G — negatively correlated across two groups, n_risky=5
# Assets 0,1 load +0.25 on factor 0; assets 2,3,4 load −0.25.
# cov[i,j] = (sign_i)*(sign_j)*0.0625 + idio; negative for cross-group pairs.
_SIGMA_G = [
    [+0.25, 0.10, 0.00, 0.00, 0.00],
    [+0.25, 0.00, 0.10, 0.00, 0.00],
    [-0.25, 0.00, 0.00, 0.10, 0.00],
    [-0.25, 0.00, 0.00, 0.00, 0.10],
    [-0.25, 0.00, 0.00, 0.00, 0.00],
]

# Case I — near-deterministic low-volatility, n_risky=2
# Per-asset implied vol ≈ 1e-3; returns nearly deterministic.
_SIGMA_I = [[1e-3, 0.0], [0.0, 1e-3]]


def _make_sigma_h() -> list[list[float]]:
    """Build the 10×10 sigma matrix for Case H (two-cluster structure).

    Cluster 1 (assets 0–4): load +0.25 on col 0, −0.05 on col 5,
                             idiosyncratic 0.08 on cols 1–4 (assets 0–3).
    Cluster 2 (assets 5–9): load +0.25 on col 5, −0.05 on col 0,
                             idiosyncratic 0.08 on cols 6–9 (assets 5–8).
    Assets 4 and 9 carry only common factor loadings.

    Implied covariance: strong positive within-cluster off-diagonals
    (≈ 0.0625), negative between-cluster off-diagonals (≈ −0.0125).
    Matrix is full-rank (10), cov = sigma @ sigma.T is PD.
    """
    a = 0.25   # within-cluster factor loading
    x = -0.05  # cross-cluster loading
    e = 0.08   # idiosyncratic loading

    sigma = [[0.0] * 10 for _ in range(10)]
    for i in range(5):          # cluster 1: assets 0–4
        sigma[i][0] = a
        sigma[i][5] = x
        if i < 4:               # unique idiosyncratic column for assets 0–3
            sigma[i][i + 1] = e
    for i in range(5):          # cluster 2: assets 5–9
        sigma[5 + i][5] = a
        sigma[5 + i][0] = x
        if i < 4:               # unique idiosyncratic column for assets 5–8
            sigma[5 + i][i + 6] = e
    return sigma


_SIGMA_H = _make_sigma_h()


# ---------------------------------------------------------------------------
# Pipeline runner and invariant checker
# ---------------------------------------------------------------------------


def _run_pipeline(
    n_risky: int,
    mu: list[float],
    sigma: list[list[float]],
    initial_wealth: float,
    r: float = 0.05,
    n_steps: int = 4,
    n_updates: int = 2,
    n_outer_iters: int = 2,
    base_seed: int | None = 0,
    w_init: float = 1.0,
    z: float = 1.0,
    entropy_temp: float = 0.1,
    w_step_size: float = 0.01,
) -> tuple[CTRLOuterLoopResult, GaussianActor, QuadraticCritic, CTRLTrainerState]:
    cfg = EnvConfig(
        horizon=1.0,
        n_steps=n_steps,
        initial_wealth=initial_wealth,
        mu=mu,
        sigma=sigma,
        assets=AssetConfig(n_risky=n_risky, include_risk_free=True, risk_free_rate=r),
    )
    env = GBMPortfolioEnv(cfg)
    actor = GaussianActor(n_risky=n_risky, horizon=1.0)
    critic = QuadraticCritic(horizon=1.0, target_return_z=z)
    actor_opt = torch.optim.SGD(actor.parameters(), lr=1e-3)
    critic_opt = torch.optim.SGD(critic.parameters(), lr=1e-3)

    trainer = CTRLTrainerState(
        actor=actor,
        critic=critic,
        env=env,
        actor_optimizer=actor_opt,
        critic_optimizer=critic_opt,
        current_w=w_init,
        target_return_z=z,
        w_step_size=w_step_size,
    )
    result = trainer.run_outer_loop(
        n_outer_iters=n_outer_iters,
        n_updates=n_updates,
        entropy_temp=entropy_temp,
        base_seed=base_seed,
    )
    return result, actor, critic, trainer


def _assert_pipeline_invariants(
    result: CTRLOuterLoopResult,
    actor: GaussianActor,
    critic: QuadraticCritic,
    trainer: CTRLTrainerState,
    n_outer_iters: int,
    n_updates: int,
    label: str = "",
) -> None:
    pfx = f"[{label}] " if label else ""

    # History length
    assert result.n_outer_iters == n_outer_iters, f"{pfx}n_outer_iters mismatch"
    assert len(result.iters) == n_outer_iters, f"{pfx}len(iters) mismatch"
    assert result.final_iter.run_result.n_updates == n_updates, f"{pfx}n_updates mismatch"

    # Scalar outputs finite
    final_step = result.final_iter.run_result.final_step
    assert math.isfinite(final_step.critic_loss), f"{pfx}critic_loss not finite"
    assert math.isfinite(final_step.actor_loss), f"{pfx}actor_loss not finite"
    assert math.isfinite(final_step.terminal_wealth), f"{pfx}terminal_wealth not finite"
    assert math.isfinite(result.w_final), f"{pfx}w_final not finite"

    # All per-step scalars across all iters and inner steps
    for ji, it in enumerate(result.iters):
        for ki, step in enumerate(it.run_result.steps):
            assert math.isfinite(step.critic_loss), f"{pfx}critic_loss NaN at iter {ji} step {ki}"
            assert math.isfinite(step.actor_loss), f"{pfx}actor_loss NaN at iter {ji} step {ki}"

    # No NaN/inf in model parameters
    for i, p in enumerate(actor.parameters()):
        assert torch.isfinite(p).all(), f"{pfx}actor param[{i}] has NaN/inf"
    for i, p in enumerate(critic.parameters()):
        assert torch.isfinite(p).all(), f"{pfx}critic param[{i}] has NaN/inf"

    # Trainer stateful shell — snapshot and history checks
    snap = trainer.snapshot()
    assert isinstance(snap, CTRLTrainerSnapshot), f"{pfx}snapshot wrong type"
    assert math.isfinite(snap.current_w), f"{pfx}snapshot.current_w not finite"
    assert snap.current_w == result.w_final, f"{pfx}snapshot.current_w != result.w_final"
    assert snap.last_terminal_wealth is not None, f"{pfx}snapshot.last_terminal_wealth is None"
    assert math.isfinite(snap.last_terminal_wealth), f"{pfx}snapshot.last_terminal_wealth not finite"
    assert snap.last_n_updates == n_outer_iters * n_updates, (
        f"{pfx}snapshot.last_n_updates {snap.last_n_updates} != {n_outer_iters * n_updates}"
    )
    # run_outer_loop appends exactly one snapshot to history per call
    assert len(trainer.history) == 1, f"{pfx}trainer.history length != 1"
    hist_snap = trainer.history[0]
    assert hist_snap.current_w == snap.current_w, f"{pfx}history[0].current_w mismatch"
    assert hist_snap.last_n_updates == snap.last_n_updates, f"{pfx}history[0].last_n_updates mismatch"


# ===========================================================================
# Case A — baseline single-asset
# ===========================================================================


def test_stress_case_a_baseline_single_asset():
    """Case A: mu=0.08, sigma=0.20, initial_wealth=1.0 — standard baseline."""
    result, actor, critic, trainer = _run_pipeline(
        n_risky=1, mu=[0.08], sigma=[[0.20]], initial_wealth=1.0
    )
    _assert_pipeline_invariants(result, actor, critic, trainer, n_outer_iters=2, n_updates=2, label="A")


# ===========================================================================
# Case B — very small initial wealth
# ===========================================================================


def test_stress_case_b_small_initial_wealth():
    """Case B: initial_wealth=0.05 — near-zero wealth regime produces finite outputs."""
    result, actor, critic, trainer = _run_pipeline(
        n_risky=1, mu=[0.08], sigma=[[0.20]], initial_wealth=0.05
    )
    _assert_pipeline_invariants(result, actor, critic, trainer, n_outer_iters=2, n_updates=2, label="B")
    # Terminal wealth should remain positive and finite
    final_step = result.final_iter.run_result.final_step
    assert final_step.terminal_wealth > 0, "Case B: terminal_wealth should be positive"


# ===========================================================================
# Case C — very high volatility
# ===========================================================================


def test_stress_case_c_high_volatility():
    """Case C: mu=0.005, sigma=0.80 — very high vol relative to drift."""
    result, actor, critic, trainer = _run_pipeline(
        n_risky=1, mu=[0.005], sigma=[[0.80]], initial_wealth=1.0
    )
    _assert_pipeline_invariants(result, actor, critic, trainer, n_outer_iters=2, n_updates=2, label="C")


# ===========================================================================
# Case D — adverse drift with large volatility
# ===========================================================================


def test_stress_case_d_adverse_drift():
    """Case D: mu=-0.20, sigma=0.60 — negative drift with high volatility."""
    result, actor, critic, trainer = _run_pipeline(
        n_risky=1, mu=[-0.20], sigma=[[0.60]], initial_wealth=1.0
    )
    _assert_pipeline_invariants(result, actor, critic, trainer, n_outer_iters=2, n_updates=2, label="D")


# ===========================================================================
# Case E — correlated 2-asset
# ===========================================================================


def test_stress_case_e_correlated_2_asset():
    """Case E: n_risky=2, correlated sigma — positive off-diagonal covariance."""
    result, actor, critic, trainer = _run_pipeline(
        n_risky=2, mu=[0.03, 0.04], sigma=_SIGMA_E, initial_wealth=1.0
    )
    _assert_pipeline_invariants(result, actor, critic, trainer, n_outer_iters=2, n_updates=2, label="E")


# ===========================================================================
# Case F — strongly positively correlated, n_risky=5
# ===========================================================================


def test_stress_case_f_strongly_positive_corr_5_assets():
    """Case F: n_risky=5, dominant common factor — pairwise corr ≈ 0.97."""
    result, actor, critic, trainer = _run_pipeline(
        n_risky=5, mu=[0.03] * 5, sigma=_SIGMA_F, initial_wealth=1.0
    )
    _assert_pipeline_invariants(result, actor, critic, trainer, n_outer_iters=2, n_updates=2, label="F")


# ===========================================================================
# Case G — negatively correlated, n_risky=5
# ===========================================================================


def test_stress_case_g_negative_corr_5_assets():
    """Case G: n_risky=5, two-group negative cross-correlation structure."""
    result, actor, critic, trainer = _run_pipeline(
        n_risky=5, mu=[0.03, 0.03, 0.04, 0.04, 0.035], sigma=_SIGMA_G, initial_wealth=1.0
    )
    _assert_pipeline_invariants(result, actor, critic, trainer, n_outer_iters=2, n_updates=2, label="G")


# ===========================================================================
# Case H — two-cluster correlated, n_risky=10
# ===========================================================================


def test_stress_case_h_two_cluster_10_assets():
    """Case H: n_risky=10, two 5-asset clusters with weak negative between-cluster cov."""
    result, actor, critic, trainer = _run_pipeline(
        n_risky=10, mu=[0.03] * 10, sigma=_SIGMA_H, initial_wealth=1.0
    )
    _assert_pipeline_invariants(result, actor, critic, trainer, n_outer_iters=2, n_updates=2, label="H")


# ===========================================================================
# Case I — near-deterministic low-volatility
# ===========================================================================


def test_stress_case_i_near_deterministic():
    """Case I: n_risky=2, sigma ≈ 1e-3 — near-deterministic returns."""
    result, actor, critic, trainer = _run_pipeline(
        n_risky=2, mu=[0.05, 0.04], sigma=_SIGMA_I, initial_wealth=1.0
    )
    _assert_pipeline_invariants(result, actor, critic, trainer, n_outer_iters=2, n_updates=2, label="I")


# ===========================================================================
# Very small wealth: terminal_wealth remains positive and finite
# ===========================================================================


def test_stress_small_wealth_terminal_wealth_is_positive_and_finite():
    """Very small initial wealth stays positive and finite throughout pipeline."""
    result, actor, critic, trainer = _run_pipeline(
        n_risky=1, mu=[0.08], sigma=[[0.20]], initial_wealth=0.02,
        n_outer_iters=3, n_updates=2,
    )
    _assert_pipeline_invariants(result, actor, critic, trainer, n_outer_iters=3, n_updates=2, label="small_wealth")
    for it in result.iters:
        final_step = it.run_result.final_step
        assert final_step.terminal_wealth > 0, "terminal_wealth must remain positive"
        assert math.isfinite(final_step.terminal_wealth), "terminal_wealth must be finite"


# ===========================================================================
# Determinism — repeated run on nontrivial case produces identical outputs
# ===========================================================================


def test_stress_determinism_repeated_run_identical_scalars():
    """Same base_seed on case E produces identical scalar outputs across two runs."""
    kwargs = dict(
        n_risky=2, mu=[0.03, 0.04], sigma=_SIGMA_E, initial_wealth=1.0,
        n_outer_iters=3, n_updates=3, base_seed=99,
    )
    result_a, _, _, _ = _run_pipeline(**kwargs)
    result_b, _, _, _ = _run_pipeline(**kwargs)

    # w_final must be identical
    assert result_a.w_final == result_b.w_final, "w_final differs between runs"

    # Final step losses must be identical
    step_a = result_a.final_iter.run_result.final_step
    step_b = result_b.final_iter.run_result.final_step
    assert step_a.critic_loss == step_b.critic_loss, "critic_loss differs between runs"
    assert step_a.actor_loss == step_b.actor_loss, "actor_loss differs between runs"
    assert step_a.terminal_wealth == step_b.terminal_wealth, "terminal_wealth differs between runs"

    # Every per-step loss across all iterations must match
    for ji, (it_a, it_b) in enumerate(zip(result_a.iters, result_b.iters)):
        for ki, (s_a, s_b) in enumerate(zip(it_a.run_result.steps, it_b.run_result.steps)):
            assert s_a.critic_loss == s_b.critic_loss, f"critic_loss differs at iter {ji} step {ki}"
            assert s_a.actor_loss == s_b.actor_loss, f"actor_loss differs at iter {ji} step {ki}"


# ===========================================================================
# History length is exact across longer runs
# ===========================================================================


def test_stress_history_length_exact_for_longer_run():
    """n_outer_iters=5, n_updates=3 produces exactly the right history lengths."""
    n_outer, n_up = 5, 3
    result, _, _, trainer = _run_pipeline(
        n_risky=1, mu=[0.08], sigma=[[0.20]], initial_wealth=1.0,
        n_outer_iters=n_outer, n_updates=n_up,
    )
    assert result.n_outer_iters == n_outer
    assert len(result.iters) == n_outer
    for ji, it in enumerate(result.iters):
        assert it.run_result.n_updates == n_up, f"n_updates mismatch at iter {ji}"
        assert len(it.run_result.steps) == n_up, f"steps length mismatch at iter {ji}"
    # Trainer history: one entry per run_outer_loop call
    assert len(trainer.history) == 1
    assert trainer.history[0].last_n_updates == n_outer * n_up
