"""Stateful CTRL trainer shell — Phase 11A/11B.

REPO ENGINEERING NOTES
----------------------
This module provides the narrowest useful stateful shell on top of the
already-approved functional helpers from Phases 10B and 10C:

- ``ctrl_outer_iter``  (Phase 10B): one inner run + one w update
- ``ctrl_outer_loop``  (Phase 10C): fixed-length sequence of outer iterations

``CTRLTrainerState`` holds live objects (actor, critic, env, optimizers) and a
persistent ``current_w`` that is updated in place after each call.  Callers
can chain multiple calls without manually threading ``w`` between them.

Phase 11B adds a centralized validation boundary at the stateful shell layer:
- constructor validates finite scalar fields and w_step_size > 0
- ``run_outer_iter`` validates ``n_updates >= 1`` and bound order
- ``run_outer_loop`` validates ``n_outer_iters >= 1``, ``n_updates >= 1``, and bound order

SCOPE BOUNDARY
--------------
The following are NOT implemented here:
- checkpoint or logging infrastructure
- config-dispatch wiring
- offline / online trainer classes
- adaptive or learned w schedules
- convergence checks / early stopping
- callback / progress infrastructure

These belong in future bounded tasks.
"""

from __future__ import annotations

import math

import torch

from src.envs.base_env import PortfolioEnv
from src.models.base import ActorBase, CriticBase
from src.train.ctrl_outer_iter import CTRLOuterIterResult, ctrl_outer_iter
from src.train.ctrl_outer_loop import CTRLOuterLoopResult, ctrl_outer_loop


class CTRLTrainerState:
    """Stateful shell holding live trainer objects and persistent current_w.

    REPO ENGINEERING: thin wrapper over the approved Phase 10B/10C functional
    helpers.  The only mutable state is ``current_w``; all other fields are
    stored references to the caller-owned objects.

    Attributes:
        actor:             Stochastic behavior policy (``ActorBase``).
        critic:            Value function (``CriticBase``).
        env:               Portfolio environment.
        actor_optimizer:   PyTorch optimiser bound to actor parameters.
        critic_optimizer:  PyTorch optimiser bound to critic parameters.
        current_w:         Current outer-loop Lagrange multiplier.  Updated
                           in place after each ``run_outer_iter`` or
                           ``run_outer_loop`` call.
        target_return_z:   Target terminal wealth z for w-update signals.
        w_step_size:       Positive outer-loop step size a_w.
    """

    def __init__(
        self,
        actor: ActorBase,
        critic: CriticBase,
        env: PortfolioEnv,
        actor_optimizer: torch.optim.Optimizer,
        critic_optimizer: torch.optim.Optimizer,
        current_w: float,
        target_return_z: float,
        w_step_size: float,
    ) -> None:
        if not math.isfinite(current_w):
            raise ValueError(f"current_w must be finite, got {current_w}")
        if not math.isfinite(target_return_z):
            raise ValueError(f"target_return_z must be finite, got {target_return_z}")
        if w_step_size <= 0.0 or not math.isfinite(w_step_size):
            raise ValueError(f"w_step_size must be finite and > 0, got {w_step_size}")
        self.actor = actor
        self.critic = critic
        self.env = env
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.current_w = current_w
        self.target_return_z = target_return_z
        self.w_step_size = w_step_size

    def _validate_stored_scalars(self) -> None:
        """Raise ValueError if any stored scalar field is in an invalid state.

        Called at the start of each method that delegates to sub-helpers so
        that post-construction mutations to invalid values are caught at the
        state-shell boundary rather than propagating silently.
        """
        if not math.isfinite(self.current_w):
            raise ValueError(f"current_w must be finite, got {self.current_w}")
        if not math.isfinite(self.target_return_z):
            raise ValueError(f"target_return_z must be finite, got {self.target_return_z}")
        if self.w_step_size <= 0.0 or not math.isfinite(self.w_step_size):
            raise ValueError(f"w_step_size must be finite and > 0, got {self.w_step_size}")

    def run_outer_iter(
        self,
        n_updates: int,
        entropy_temp: float,
        base_seed: int | None = None,
        w_min: float | None = None,
        w_max: float | None = None,
    ) -> CTRLOuterIterResult:
        """Run one outer iteration and update ``current_w`` to ``result.w_next``.

        Delegates to the approved Phase 10B helper ``ctrl_outer_iter``, using
        the stored objects and ``current_w``.  On return ``self.current_w`` is
        set to ``result.w_next`` so the next call starts from the updated value.

        Args:
            n_updates:    Number of inner actor/critic steps.
            entropy_temp: Entropy regularisation temperature γ.
            base_seed:    Optional base seed for deterministic inner steps.
            w_min:        Optional lower bound for w projection.
            w_max:        Optional upper bound for w projection.

        Returns:
            ``CTRLOuterIterResult`` from ``ctrl_outer_iter``.

        Raises:
            ValueError: if stored scalar state is invalid, ``n_updates < 1``,
                or bound order is invalid.
        """
        self._validate_stored_scalars()
        if n_updates < 1:
            raise ValueError(f"n_updates must be >= 1, got {n_updates}")
        if w_min is not None and w_max is not None and w_min > w_max:
            raise ValueError(
                f"w_min must be <= w_max when both are provided, got w_min={w_min}, w_max={w_max}"
            )
        result = ctrl_outer_iter(
            actor=self.actor,
            critic=self.critic,
            env=self.env,
            actor_optimizer=self.actor_optimizer,
            critic_optimizer=self.critic_optimizer,
            w=self.current_w,
            target_return_z=self.target_return_z,
            w_step_size=self.w_step_size,
            n_updates=n_updates,
            entropy_temp=entropy_temp,
            base_seed=base_seed,
            w_min=w_min,
            w_max=w_max,
        )
        self.current_w = result.w_next
        return result

    def run_outer_loop(
        self,
        n_outer_iters: int,
        n_updates: int,
        entropy_temp: float,
        base_seed: int | None = None,
        w_min: float | None = None,
        w_max: float | None = None,
    ) -> CTRLOuterLoopResult:
        """Run a fixed-length outer loop and update ``current_w`` to ``result.w_final``.

        Delegates to the approved Phase 10C helper ``ctrl_outer_loop``, using
        the stored objects and ``current_w`` as ``w_init``.  On return
        ``self.current_w`` is set to ``result.w_final`` so the next call starts
        from the updated value.

        Args:
            n_outer_iters: Number of outer iterations to execute.
            n_updates:     Number of inner actor/critic steps per outer iteration.
            entropy_temp:  Entropy regularisation temperature γ.
            base_seed:     Optional base seed for deterministic scheduling.
            w_min:         Optional lower bound for w projection.
            w_max:         Optional upper bound for w projection.

        Returns:
            ``CTRLOuterLoopResult`` from ``ctrl_outer_loop``.

        Raises:
            ValueError: if stored scalar state is invalid, ``n_outer_iters < 1``,
                ``n_updates < 1``, or bound order is invalid.
        """
        self._validate_stored_scalars()
        if n_outer_iters < 1:
            raise ValueError(f"n_outer_iters must be >= 1, got {n_outer_iters}")
        if n_updates < 1:
            raise ValueError(f"n_updates must be >= 1, got {n_updates}")
        if w_min is not None and w_max is not None and w_min > w_max:
            raise ValueError(
                f"w_min must be <= w_max when both are provided, got w_min={w_min}, w_max={w_max}"
            )
        result = ctrl_outer_loop(
            actor=self.actor,
            critic=self.critic,
            env=self.env,
            actor_optimizer=self.actor_optimizer,
            critic_optimizer=self.critic_optimizer,
            w_init=self.current_w,
            target_return_z=self.target_return_z,
            w_step_size=self.w_step_size,
            n_outer_iters=n_outer_iters,
            n_updates=n_updates,
            entropy_temp=entropy_temp,
            base_seed=base_seed,
            w_min=w_min,
            w_max=w_max,
        )
        self.current_w = result.w_final
        return result
