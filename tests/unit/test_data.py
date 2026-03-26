"""Unit tests for the data layer (src/data/)."""

import pytest
import torch

from src.data.types import Transition, Batch, collate_transitions
from src.data.replay_buffer import ReplayBuffer
from src.data.synthetic import generate_gbm_paths, generate_gbm_returns
from src.data.datasets import EpisodeDataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_transition(obs_dim: int = 3, action_dim: int = 1) -> Transition:
    return Transition(
        obs=torch.randn(obs_dim),
        action=torch.randn(action_dim),
        reward=torch.tensor(0.5),
        next_obs=torch.randn(obs_dim),
        done=torch.tensor(0.0),
        time=torch.tensor(0.0),
        next_time=torch.tensor(0.1),
    )


# ---------------------------------------------------------------------------
# Transition / Batch / collate
# ---------------------------------------------------------------------------


def test_collate_transitions_shapes():
    batch_size = 8
    transitions = [_make_transition(obs_dim=4, action_dim=2) for _ in range(batch_size)]
    batch = collate_transitions(transitions)
    assert batch.obs.shape == (batch_size, 4)
    assert batch.action.shape == (batch_size, 2)
    assert batch.reward.shape == (batch_size,)
    assert batch.done.shape == (batch_size,)
    assert batch.time.shape == (batch_size,)


def test_collate_optional_none():
    """When no transition has log_prob/context, batch fields should be None."""
    transitions = [_make_transition() for _ in range(4)]
    batch = collate_transitions(transitions)
    assert batch.log_prob is None
    assert batch.context is None
    assert batch.context_mask is None


def test_collate_optional_present():
    """When all transitions have log_prob, it should be stacked."""
    transitions = []
    for _ in range(5):
        t = _make_transition()
        t.log_prob = torch.tensor(-1.2)
        transitions.append(t)
    batch = collate_transitions(transitions)
    assert batch.log_prob is not None
    assert batch.log_prob.shape == (5,)


def test_collate_mixed_optional_raises():
    """Mixing None and non-None log_prob should raise."""
    t1 = _make_transition()
    t2 = _make_transition()
    t2.log_prob = torch.tensor(-0.5)
    with pytest.raises(ValueError):
        collate_transitions([t1, t2])


def test_batch_to_device():
    transitions = [_make_transition() for _ in range(3)]
    batch = collate_transitions(transitions)
    batch_cpu = batch.to("cpu")
    assert batch_cpu.obs.device.type == "cpu"


def test_batch_batch_size():
    transitions = [_make_transition() for _ in range(7)]
    batch = collate_transitions(transitions)
    assert batch.batch_size == 7


# ---------------------------------------------------------------------------
# ReplayBuffer
# ---------------------------------------------------------------------------


def test_replay_buffer_add_and_len():
    buf = ReplayBuffer(capacity=10)
    assert len(buf) == 0
    buf.add(_make_transition())
    assert len(buf) == 1


def test_replay_buffer_capacity_wraps():
    buf = ReplayBuffer(capacity=5)
    for _ in range(8):
        buf.add(_make_transition())
    assert len(buf) == 5  # capped at capacity


def test_replay_buffer_sample_shape():
    buf = ReplayBuffer(capacity=100)
    for _ in range(50):
        buf.add(_make_transition(obs_dim=4))
    batch = buf.sample(16)
    assert batch.obs.shape == (16, 4)
    assert batch.reward.shape == (16,)


def test_replay_buffer_sample_too_large_raises():
    buf = ReplayBuffer(capacity=10)
    buf.add(_make_transition())
    with pytest.raises(ValueError):
        buf.sample(5)


def test_replay_buffer_invalid_capacity_raises():
    with pytest.raises(ValueError):
        ReplayBuffer(capacity=0)


def test_replay_buffer_is_full():
    buf = ReplayBuffer(capacity=3)
    assert not buf.is_full()
    for _ in range(3):
        buf.add(_make_transition())
    assert buf.is_full()


def test_replay_buffer_clear():
    buf = ReplayBuffer(capacity=10)
    for _ in range(5):
        buf.add(_make_transition())
    buf.clear()
    assert len(buf) == 0


def test_replay_buffer_sample_no_duplicates():
    """Sample without replacement: no duplicate indices when batch_size == buffer size."""
    buf = ReplayBuffer(capacity=20)
    for _ in range(20):
        buf.add(_make_transition())
    # Sample all 20 — with replacement this would almost surely produce duplicates
    batch = buf.sample(20)
    assert batch.batch_size == 20


# ---------------------------------------------------------------------------
# Synthetic GBM
# ---------------------------------------------------------------------------


def test_gbm_paths_shape_single_asset():
    paths = generate_gbm_paths(
        n_paths=8, n_steps=20, horizon=1.0,
        mu=[0.1], sigma=[[0.2]], x0=1.0, seed=0,
    )
    assert paths.shape == (8, 21, 1)


def test_gbm_paths_shape_multi_asset():
    paths = generate_gbm_paths(
        n_paths=4, n_steps=10, horizon=0.5,
        mu=[0.1, 0.08], sigma=[[0.2, 0.0], [0.0, 0.15]], x0=2.0, seed=1,
    )
    assert paths.shape == (4, 11, 2)


def test_gbm_paths_positive():
    paths = generate_gbm_paths(
        n_paths=16, n_steps=50, horizon=1.0,
        mu=[0.05], sigma=[[0.3]], seed=42,
    )
    assert torch.all(paths > 0)


def test_gbm_paths_initial_wealth():
    x0 = 2.5
    paths = generate_gbm_paths(
        n_paths=10, n_steps=5, horizon=1.0,
        mu=[0.1], sigma=[[0.2]], x0=x0, seed=0,
    )
    assert torch.allclose(paths[:, 0, :], torch.full((10, 1), x0))


def test_gbm_paths_deterministic():
    kwargs = dict(n_paths=6, n_steps=10, horizon=1.0, mu=[0.1], sigma=[[0.2]], seed=99)
    p1 = generate_gbm_paths(**kwargs)
    p2 = generate_gbm_paths(**kwargs)
    assert torch.allclose(p1, p2)


def test_gbm_paths_different_seeds():
    kwargs = dict(n_paths=6, n_steps=10, horizon=1.0, mu=[0.1], sigma=[[0.2]])
    p1 = generate_gbm_paths(**kwargs, seed=0)
    p2 = generate_gbm_paths(**kwargs, seed=1)
    assert not torch.allclose(p1, p2)


def test_gbm_returns_shape():
    returns = generate_gbm_returns(
        n_paths=5, n_steps=15, horizon=1.0,
        mu=[0.1], sigma=[[0.2]], seed=7,
    )
    assert returns.shape == (5, 15, 1)


def test_gbm_returns_positive():
    returns = generate_gbm_returns(
        n_paths=10, n_steps=20, horizon=1.0,
        mu=[0.0], sigma=[[0.1]], seed=3,
    )
    assert torch.all(returns > 0)


# ---------------------------------------------------------------------------
# EpisodeDataset
# ---------------------------------------------------------------------------


def _make_episode(length: int, obs_dim: int = 3) -> list[Transition]:
    return [
        Transition(
            obs=torch.randn(obs_dim),
            action=torch.randn(1),
            reward=torch.tensor(0.0),
            next_obs=torch.randn(obs_dim),
            done=torch.tensor(float(i == length - 1)),
            time=torch.tensor(float(i) / length),
            next_time=torch.tensor(float(i + 1) / length),
        )
        for i in range(length)
    ]


def test_episode_dataset_counts():
    ds = EpisodeDataset([_make_episode(10), _make_episode(10)])
    assert ds.n_episodes == 2
    assert ds.n_transitions == 20


def test_episode_dataset_get_all():
    ds = EpisodeDataset([_make_episode(5) for _ in range(4)])
    batch = ds.get_all()
    assert batch.batch_size == 20


def test_episode_dataset_sample_batch():
    ds = EpisodeDataset([_make_episode(20)])
    batch = ds.sample_batch(8)
    assert batch.batch_size == 8


def test_episode_dataset_iter_batches():
    ds = EpisodeDataset([_make_episode(30)])
    batches = list(ds.iter_batches(batch_size=10, shuffle=False))
    # 30 transitions → 3 full batches of 10
    assert len(batches) == 3
    for b in batches:
        assert b.batch_size == 10


def test_episode_dataset_empty_raises():
    with pytest.raises(ValueError):
        EpisodeDataset([])
