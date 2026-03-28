"""Stateful CTRL trainer shell — Phases 11A/11B/12A–12C/13B.

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

Phase 12A adds a read-only snapshot / scalar-summary layer:
- ``CTRLTrainerSnapshot`` dataclass captures current scalar state + recent diagnostics
- ``CTRLTrainerState.snapshot()`` returns a snapshot without mutating state

Phase 12B adds an in-memory history layer:
- ``CTRLTrainerState`` records a ``CTRLTrainerSnapshot`` after each successful run
- ``CTRLTrainerState.history`` exposes accumulated snapshots as an immutable tuple
- ``CTRLTrainerState.clear_history()`` resets history without touching live trainer state

Phase 12C adds a scalar-state reset boundary:
- ``CTRLTrainerState.reset(w=None)`` resets ``current_w``, clears diagnostics and history
- Supports reset to construction-time ``w`` (default) or an explicitly supplied finite ``w``
- Actor / critic / env / optimizer references are never replaced

Phase 13B adds an in-memory checkpoint payload layer:
- ``CTRLCheckpointPayload`` captures actor/critic/optimizer state_dicts + scalar trainer state
- ``CTRLTrainerState.export_checkpoint()`` captures the current in-memory payload
- ``CTRLTrainerState.restore_checkpoint(payload)`` restores state in place; history and
  latest snapshot diagnostics are intentionally omitted from the payload boundary

SCOPE BOUNDARY
--------------
The following are NOT implemented here:
- writing checkpoint files to disk or loading from disk paths
- config-dispatch wiring
- offline / online trainer classes
- adaptive or learned w schedules
- convergence checks / early stopping
- callback / progress infrastructure

These belong in future bounded tasks.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import torch

from src.envs.base_env import PortfolioEnv
from src.models.base import ActorBase, CriticBase
from src.train.ctrl_outer_iter import CTRLOuterIterResult, ctrl_outer_iter
from src.train.ctrl_outer_loop import CTRLOuterLoopResult, ctrl_outer_loop


@dataclass(frozen=True)
class CTRLTrainerSnapshot:
    """Read-only snapshot of ``CTRLTrainerState`` at a point in time.

    REPO ENGINEERING (Phase 12A): structured in-memory diagnostics boundary.
    All fields are plain Python scalars; no tensor graphs or live object
    references are stored.  Intended for future logging/checkpoint consumers.

    Attributes:
        current_w:            Current outer-loop Lagrange multiplier.
        target_return_z:      Target terminal wealth z.
        w_step_size:          Outer-loop step size a_w.
        last_terminal_wealth: Terminal portfolio wealth from the final inner
                              step of the most recent run.  ``None`` if no run
                              has been executed yet.
        last_w_prev:          Lagrange multiplier before the most recent w
                              update.  ``None`` if no run has been executed yet.
        last_n_updates:       Total inner actor/critic steps in the most recent
                              call (n_outer_iters × n_updates for outer-loop
                              calls).  ``None`` if no run has been executed yet.
    """

    current_w: float
    target_return_z: float
    w_step_size: float
    last_terminal_wealth: float | None
    last_w_prev: float | None
    last_n_updates: int | None


@dataclass
class CTRLCheckpointPayload:
    """In-memory checkpoint payload for ``CTRLTrainerState`` — Phase 13B.

    REPO ENGINEERING: captures everything needed to resume training from a
    known state.  Intentionally omits in-memory history and latest snapshot
    diagnostics; those are ephemeral and not required for a resume boundary.

    Attributes:
        actor_state_dict:            ``actor.state_dict()`` at capture time.
        critic_state_dict:           ``critic.state_dict()`` at capture time.
        actor_optimizer_state_dict:  ``actor_optimizer.state_dict()`` at capture time.
        critic_optimizer_state_dict: ``critic_optimizer.state_dict()`` at capture time.
        current_w:                   Lagrange multiplier at capture time.
        target_return_z:             Target terminal wealth z at capture time.
        w_step_size:                 Outer-loop step size a_w at capture time.
    """

    actor_state_dict: dict
    critic_state_dict: dict
    actor_optimizer_state_dict: dict
    critic_optimizer_state_dict: dict
    current_w: float
    target_return_z: float
    w_step_size: float


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

    Private diagnostic fields (populated after each run, used by snapshot()):
        _initial_w:            Construction-time current_w; used by reset().
        _last_terminal_wealth: Terminal wealth from last run's final step.
        _last_w_prev:          w before the last w-update.
        _last_n_updates:       Total inner steps in the last call.
        _history:              Ordered list of snapshots, one per completed run.
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
        self._initial_w: float = current_w
        self._last_terminal_wealth: float | None = None
        self._last_w_prev: float | None = None
        self._last_n_updates: int | None = None
        self._history: list[CTRLTrainerSnapshot] = []

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
        self._last_terminal_wealth = result.run_result.final_step.terminal_wealth
        self._last_w_prev = result.w_prev
        self._last_n_updates = result.run_result.n_updates
        self._history.append(self.snapshot())
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
        self._last_terminal_wealth = result.final_iter.run_result.final_step.terminal_wealth
        self._last_w_prev = result.final_iter.w_prev
        self._last_n_updates = result.n_outer_iters * result.final_iter.run_result.n_updates
        self._history.append(self.snapshot())
        return result

    def snapshot(self) -> CTRLTrainerSnapshot:
        """Return a read-only snapshot of current trainer state and diagnostics.

        REPO ENGINEERING (Phase 12A): does not mutate any trainer state.
        Scalar diagnostics are ``None`` until the first ``run_outer_iter`` or
        ``run_outer_loop`` call has completed.

        Returns:
            ``CTRLTrainerSnapshot`` with current scalar fields and optional
            diagnostics from the most recent run.
        """
        return CTRLTrainerSnapshot(
            current_w=self.current_w,
            target_return_z=self.target_return_z,
            w_step_size=self.w_step_size,
            last_terminal_wealth=self._last_terminal_wealth,
            last_w_prev=self._last_w_prev,
            last_n_updates=self._last_n_updates,
        )

    @property
    def history(self) -> tuple[CTRLTrainerSnapshot, ...]:
        """Ordered, immutable sequence of snapshots recorded after each run.

        REPO ENGINEERING (Phase 12B): one snapshot is appended after each
        successful ``run_outer_iter`` or ``run_outer_loop`` call.  Returns a
        tuple so callers cannot accidentally mutate the internal list.

        Returns:
            Tuple of ``CTRLTrainerSnapshot`` entries in call order.  Empty
            before any run.
        """
        return tuple(self._history)

    def clear_history(self) -> None:
        """Clear all accumulated history entries.

        REPO ENGINEERING (Phase 12B): removes all recorded snapshots without
        changing ``current_w``, ``target_return_z``, ``w_step_size``, or any
        other live trainer state.
        """
        self._history.clear()

    def reset(self, w: float | None = None) -> None:
        """Reset scalar trainer state to a clean starting point.

        REPO ENGINEERING (Phase 12C): resets ``current_w``, clears all scalar
        diagnostics (``_last_*`` fields), and clears in-memory history.  Does
        NOT reinitialize model parameters, optimizers, or environment internals.

        Args:
            w: New starting value for ``current_w``.  Must be finite.
               If ``None``, resets to the construction-time ``current_w``
               (``_initial_w``).

        Raises:
            ValueError: if ``w`` is provided but not finite.
        """
        if w is not None:
            if not math.isfinite(w):
                raise ValueError(f"reset w must be finite, got {w}")
            self.current_w = w
        else:
            self.current_w = self._initial_w
        self._last_terminal_wealth = None
        self._last_w_prev = None
        self._last_n_updates = None
        self._history.clear()

    def export_checkpoint(self) -> CTRLCheckpointPayload:
        """Capture the current in-memory trainer state as a checkpoint payload.

        REPO ENGINEERING (Phase 13B): calls ``state_dict()`` on the stored
        actor, critic, and optimizers, then packages them with the current
        scalar fields.  History and latest snapshot diagnostics are
        intentionally omitted (see ``CTRLCheckpointPayload`` docstring).

        Returns:
            ``CTRLCheckpointPayload`` with copies of all state_dicts and
            current scalar trainer state.
        """
        return CTRLCheckpointPayload(
            actor_state_dict=copy.deepcopy(self.actor.state_dict()),
            critic_state_dict=copy.deepcopy(self.critic.state_dict()),
            actor_optimizer_state_dict=copy.deepcopy(self.actor_optimizer.state_dict()),
            critic_optimizer_state_dict=copy.deepcopy(self.critic_optimizer.state_dict()),
            current_w=self.current_w,
            target_return_z=self.target_return_z,
            w_step_size=self.w_step_size,
        )

    def restore_checkpoint(self, payload: CTRLCheckpointPayload) -> None:
        """Restore trainer state in place from a checkpoint payload.

        REPO ENGINEERING (Phase 13B): calls ``load_state_dict()`` on the
        stored actor, critic, and optimizers, then updates the scalar fields.
        Object references (actor, critic, env, optimizers) are never replaced.
        History and latest snapshot diagnostics are reset to empty / None
        because they are not captured in the payload boundary.

        Args:
            payload: ``CTRLCheckpointPayload`` produced by ``export_checkpoint``.

        Raises:
            ValueError: if any scalar field in the payload is invalid (non-finite
                ``current_w`` or ``target_return_z``, or ``w_step_size <= 0``).
        """
        if not math.isfinite(payload.current_w):
            raise ValueError(f"payload current_w must be finite, got {payload.current_w}")
        if not math.isfinite(payload.target_return_z):
            raise ValueError(
                f"payload target_return_z must be finite, got {payload.target_return_z}"
            )
        if payload.w_step_size <= 0.0 or not math.isfinite(payload.w_step_size):
            raise ValueError(
                f"payload w_step_size must be finite and > 0, got {payload.w_step_size}"
            )
        self.actor.load_state_dict(payload.actor_state_dict)
        self.critic.load_state_dict(payload.critic_state_dict)
        self.actor_optimizer.load_state_dict(payload.actor_optimizer_state_dict)
        self.critic_optimizer.load_state_dict(payload.critic_optimizer_state_dict)
        self.current_w = payload.current_w
        self.target_return_z = payload.target_return_z
        self.w_step_size = payload.w_step_size
        self._last_terminal_wealth = None
        self._last_w_prev = None
        self._last_n_updates = None
        self._history.clear()
