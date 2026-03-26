"""Fixed-size replay buffer for off-policy RL training.

The buffer stores transitions as pre-allocated tensors and wraps around when
full (ring-buffer semantics).  Sampling is uniform random without replacement.
"""

from __future__ import annotations

import torch

from src.data.types import Batch, Transition, collate_transitions


class ReplayBuffer:
    """Circular replay buffer backed by a list of ``Transition`` objects.

    Args:
        capacity: Maximum number of transitions to store.
    """

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        self._capacity = capacity
        self._buffer: list[Transition] = []
        self._ptr: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def capacity(self) -> int:
        return self._capacity

    def __len__(self) -> int:
        return len(self._buffer)

    def is_full(self) -> bool:
        return len(self._buffer) == self._capacity

    def add(self, transition: Transition) -> None:
        """Insert one transition, overwriting oldest entry when full."""
        if len(self._buffer) < self._capacity:
            self._buffer.append(transition)
        else:
            self._buffer[self._ptr] = transition
        self._ptr = (self._ptr + 1) % self._capacity

    def add_batch(self, transitions: list[Transition]) -> None:
        """Insert multiple transitions."""
        for t in transitions:
            self.add(t)

    def sample(self, batch_size: int) -> Batch:
        """Sample ``batch_size`` transitions uniformly at random.

        Raises:
            ValueError: If the buffer contains fewer transitions than
                ``batch_size``.
        """
        n = len(self._buffer)
        if batch_size > n:
            raise ValueError(
                f"Cannot sample {batch_size} transitions from buffer of size {n}"
            )
        indices = torch.randint(0, n, (batch_size,)).tolist()
        selected = [self._buffer[i] for i in indices]
        return collate_transitions(selected)

    def clear(self) -> None:
        """Remove all stored transitions."""
        self._buffer.clear()
        self._ptr = 0
