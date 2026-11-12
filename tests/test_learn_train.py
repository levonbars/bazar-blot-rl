"""Training-loop helpers that are cheap to test without running a training loop."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from bazarblot.learn.train import TrainConfig, rollout_seed_for


def test_consecutive_iterations_play_disjoint_shuffles() -> None:
    """Regression test: the rollout base seed used to advance by 1 per iteration while each
    iteration used `base + i` for i < n_deals, so consecutive iterations replayed almost the same
    shuffles (~655 distinct deals in a 400 x 256 run)."""
    for n_deals in (1, 64, 256, 1024):
        config = TrainConfig(n_deals_per_iter=n_deals, seed=3)
        seen: set[int] = set()
        for it in range(50):
            base = rollout_seed_for(config, it)
            shuffles = {base + i for i in range(n_deals)}
            assert not (shuffles & seen)
            seen |= shuffles
    a = {rollout_seed_for(TrainConfig(seed=0), it) + i for it in range(400) for i in range(256)}
    b = {rollout_seed_for(TrainConfig(seed=1), it) + i for it in range(400) for i in range(256)}
    assert not (a & b)  # different run seeds do not overlap either
