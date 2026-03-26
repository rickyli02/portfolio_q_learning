"""Offline dataset containers for portfolio RL.

An ``EpisodeDataset`` holds complete episodes (trajectories) as tensors and
provides a ``Batch``-compatible iterator for offline training.  It wraps a
list of ``Transition`` sequences and does not grow after construction.
"""

from __future__ import annotations

import torch

from src.data.types import Batch, Transition, collate_transitions


class EpisodeDataset:
    """Immutable dataset of complete episodes stored as ``Transition`` lists.

    Args:
        episodes: A list of episodes, where each episode is a list of
            ``Transition`` objects ordered from t=0 to terminal.
    """

    def __init__(self, episodes: list[list[Transition]]) -> None:
        if not episodes:
            raise ValueError("episodes must be non-empty")
        self._episodes = episodes
        self._flat: list[Transition] = [t for ep in episodes for t in ep]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_episodes(self) -> int:
        return len(self._episodes)

    @property
    def n_transitions(self) -> int:
        return len(self._flat)

    def episode(self, idx: int) -> list[Transition]:
        """Return all transitions for episode ``idx``."""
        return self._episodes[idx]

    # ------------------------------------------------------------------
    # Batch access
    # ------------------------------------------------------------------

    def get_all(self) -> Batch:
        """Return all transitions as a single ``Batch``."""
        return collate_transitions(self._flat)

    def sample_batch(self, batch_size: int) -> Batch:
        """Sample ``batch_size`` transitions uniformly at random."""
        n = self.n_transitions
        if batch_size > n:
            raise ValueError(
                f"Cannot sample {batch_size} from dataset of size {n}"
            )
        indices = torch.randint(0, n, (batch_size,)).tolist()
        selected = [self._flat[i] for i in indices]
        return collate_transitions(selected)

    def iter_batches(
        self, batch_size: int, shuffle: bool = True
    ):
        """Yield non-overlapping ``Batch`` objects over all transitions.

        The last incomplete batch is dropped.
        """
        n = self.n_transitions
        order = torch.randperm(n).tolist() if shuffle else list(range(n))
        for start in range(0, n - batch_size + 1, batch_size):
            indices = order[start : start + batch_size]
            selected = [self._flat[i] for i in indices]
            yield collate_transitions(selected)
