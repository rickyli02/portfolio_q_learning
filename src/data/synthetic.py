"""Synthetic path generation for portfolio RL environments.

Currently supports multi-asset Geometric Brownian Motion (GBM).  All
functions are deterministic under a fixed seed and return plain torch tensors
so they can be used by both environment stepping and offline dataset creation.

GBM model (log-price / wealth dynamics):
    dX_t / X_t = mu dt + sigma dW_t

Discretised with Euler-Maruyama (exact for log-normal):
    X_{t+dt} = X_t * exp((mu - 0.5 * diag(sigma @ sigma.T)) * dt
                          + sigma @ sqrt(dt) * z),  z ~ N(0, I)
"""

from __future__ import annotations

import torch


def generate_gbm_paths(
    n_paths: int,
    n_steps: int,
    horizon: float,
    mu: list[float] | torch.Tensor,
    sigma: list[list[float]] | torch.Tensor,
    x0: float = 1.0,
    seed: int | None = None,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Generate multi-asset GBM sample paths.

    Args:
        n_paths: Number of independent sample paths.
        n_steps: Discrete time steps per path.
        horizon: Total time horizon T (years).
        mu: Expected log-return vector, shape (n_assets,).
        sigma: Volatility / covariance matrix, shape (n_assets, n_assets).
            This is the *volatility* matrix (not covariance); the full
            covariance is sigma @ sigma.T.
        x0: Common initial wealth / price level for all paths and assets.
        seed: Optional RNG seed for reproducibility.
        device: Target device.

    Returns:
        Tensor of shape ``(n_paths, n_steps + 1, n_assets)`` containing
        simulated wealth / price levels, including the initial value at t=0.
    """
    mu_t = torch.as_tensor(mu, dtype=torch.float32, device=device)        # (n_assets,)
    sigma_t = torch.as_tensor(sigma, dtype=torch.float32, device=device)  # (n_assets, n_assets)
    n_assets = mu_t.shape[0]
    dt = horizon / n_steps

    gen = torch.Generator(device=device)
    if seed is not None:
        gen.manual_seed(seed)

    # z ~ N(0, I), shape (n_paths, n_steps, n_assets)
    z = torch.randn(n_paths, n_steps, n_assets, generator=gen, device=device)

    # Drift: (mu - 0.5 * diag(sigma @ sigma.T)) * dt, shape (n_assets,)
    cov_diag = (sigma_t @ sigma_t.T).diagonal()              # (n_assets,)
    drift = (mu_t - 0.5 * cov_diag) * dt                    # (n_assets,)

    # Diffusion: sigma.T @ z[i,j]  — increments, shape (n_paths, n_steps, n_assets)
    # sigma_t has shape (n_assets, n_assets); we want z @ sigma_t.T
    diffusion = (z @ sigma_t.T) * (dt ** 0.5)               # (n_paths, n_steps, n_assets)

    log_increments = drift + diffusion                       # (n_paths, n_steps, n_assets)
    log_paths = torch.cumsum(log_increments, dim=1)          # (n_paths, n_steps, n_assets)

    # Prepend log(x0) = 0 at t=0, then exponentiate
    log_init = torch.zeros(n_paths, 1, n_assets, device=device)
    log_all = torch.cat([log_init, log_paths], dim=1)        # (n_paths, n_steps+1, n_assets)
    paths = x0 * torch.exp(log_all)                          # (n_paths, n_steps+1, n_assets)

    return paths


def generate_gbm_returns(
    n_paths: int,
    n_steps: int,
    horizon: float,
    mu: list[float] | torch.Tensor,
    sigma: list[list[float]] | torch.Tensor,
    seed: int | None = None,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Generate per-step gross returns for a multi-asset GBM.

    Returns:
        Tensor of shape ``(n_paths, n_steps, n_assets)`` where each entry is
        the gross return R_{t,t+dt} = X_{t+dt} / X_t.
    """
    paths = generate_gbm_paths(
        n_paths=n_paths,
        n_steps=n_steps,
        horizon=horizon,
        mu=mu,
        sigma=sigma,
        seed=seed,
        device=device,
    )
    # Ratio of consecutive time steps: paths[:, 1:] / paths[:, :-1]
    returns = paths[:, 1:, :] / paths[:, :-1, :]
    return returns
