"""Shared data types and batch schema for offline and online training.

Both trainers consume ``Batch`` objects built from ``Transition`` instances so
that the training loop is agnostic to whether data came from a replay buffer
or a live environment interaction.

Tensor shapes (B = batch size, A = n_actions, F = n_features):
  obs          : (B, F)   or  (B,)  for scalar wealth state
  action       : (B, A)   or  (B,)  for single-asset case
  reward       : (B,)
  next_obs     : (B, F)   or  (B,)
  done         : (B,)     bool / float, 1.0 at terminal step
  time         : (B,)     current time t in [0, T]
  next_time    : (B,)     next time t + dt
  log_prob     : (B,)     optional log pi(a|s) at collection time
  context      : (B, C)   optional auxiliary conditioning features
  context_mask : (B, C)   bool mask, True where context is present
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Transition:
    """A single environment transition stored as CPU tensors."""

    obs: torch.Tensor
    """Current observation / state."""

    action: torch.Tensor
    """Action taken."""

    reward: torch.Tensor
    """Scalar reward received."""

    next_obs: torch.Tensor
    """Next observation / state."""

    done: torch.Tensor
    """Terminal flag (1.0 if episode ended)."""

    time: torch.Tensor
    """Time at current step."""

    next_time: torch.Tensor
    """Time at next step."""

    log_prob: torch.Tensor | None = None
    """Log-probability of action under behaviour policy (optional)."""

    context: torch.Tensor | None = None
    """Optional auxiliary conditioning features."""

    context_mask: torch.Tensor | None = None
    """Boolean mask for context (True = feature present)."""


@dataclass
class Batch:
    """A mini-batch of transitions as stacked tensors.

    All tensors are on the same device and have a leading batch dimension B.
    """

    obs: torch.Tensor           # (B, *obs_shape)
    action: torch.Tensor        # (B, *action_shape)
    reward: torch.Tensor        # (B,)
    next_obs: torch.Tensor      # (B, *obs_shape)
    done: torch.Tensor          # (B,)
    time: torch.Tensor          # (B,)
    next_time: torch.Tensor     # (B,)

    log_prob: torch.Tensor | None = None        # (B,)
    context: torch.Tensor | None = None         # (B, C)
    context_mask: torch.Tensor | None = None    # (B, C)

    def to(self, device: torch.device | str) -> "Batch":
        """Return a copy of this batch moved to ``device``."""
        def _move(t: torch.Tensor | None) -> torch.Tensor | None:
            return t.to(device) if t is not None else None

        return Batch(
            obs=self.obs.to(device),
            action=self.action.to(device),
            reward=self.reward.to(device),
            next_obs=self.next_obs.to(device),
            done=self.done.to(device),
            time=self.time.to(device),
            next_time=self.next_time.to(device),
            log_prob=_move(self.log_prob),
            context=_move(self.context),
            context_mask=_move(self.context_mask),
        )

    @property
    def batch_size(self) -> int:
        return self.obs.shape[0]


def collate_transitions(transitions: list[Transition]) -> Batch:
    """Stack a list of ``Transition`` objects into a single ``Batch``."""

    def _stack(attr: str) -> torch.Tensor:
        tensors = [getattr(t, attr) for t in transitions]
        return torch.stack(tensors, dim=0)

    def _stack_optional(attr: str) -> torch.Tensor | None:
        values = [getattr(t, attr) for t in transitions]
        if all(v is None for v in values):
            return None
        if any(v is None for v in values):
            raise ValueError(
                f"Cannot collate '{attr}': some transitions have it, some do not."
            )
        return torch.stack(values, dim=0)

    return Batch(
        obs=_stack("obs"),
        action=_stack("action"),
        reward=_stack("reward"),
        next_obs=_stack("next_obs"),
        done=_stack("done"),
        time=_stack("time"),
        next_time=_stack("next_time"),
        log_prob=_stack_optional("log_prob"),
        context=_stack_optional("context"),
        context_mask=_stack_optional("context_mask"),
    )
