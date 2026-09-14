"""M7: end-to-end smoke test for the training loop -- tiny sizes (a handful of deals/iterations),
checking the whole pipeline (rollout -> PPO update -> opponent-pool snapshot -> paired eval against
`RandomAgent`/`HeuristicAgent` -> checkpoint) runs without crashing and produces a loadable
checkpoint. Not a claim about training QUALITY at this scale -- see
`docs/03-implementation-roadmap.md` M7 for the actual reported run."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from bazarblot.core.rules import load_default  # noqa: E402
from bazarblot.learn.checkpoint import load_checkpoint  # noqa: E402
from bazarblot.learn.ppo import PPOConfig  # noqa: E402
from bazarblot.learn.train import (  # noqa: E402
    TrainConfig,
    _entropy_coef_at,
    _forced_bid_prob_at,
    train,
)

RULES = load_default()


def test_entropy_coef_anneals_linearly_from_start_to_end() -> None:
    config = TrainConfig(n_iterations=101, entropy_coef_start=0.05, entropy_coef_end=0.02)
    assert _entropy_coef_at(0, config) == pytest.approx(0.05)
    assert _entropy_coef_at(100, config) == pytest.approx(0.02)
    assert _entropy_coef_at(50, config) == pytest.approx(0.035)


def test_entropy_coef_at_handles_single_iteration_run() -> None:
    config = TrainConfig(n_iterations=1, entropy_coef_start=0.05, entropy_coef_end=0.02)
    assert _entropy_coef_at(0, config) == pytest.approx(0.05)


def test_forced_bid_prob_anneals_then_holds_at_end_value() -> None:
    config = TrainConfig(
        n_iterations=100,
        forced_bid_prob_start=0.4,
        forced_bid_prob_end=0.0,
        forced_bid_anneal_frac=0.5,
    )
    assert _forced_bid_prob_at(0, config) == pytest.approx(0.4)
    assert _forced_bid_prob_at(25, config) == pytest.approx(0.2)
    # past the anneal fraction (iteration >= 50), held at the end value
    assert _forced_bid_prob_at(50, config) == pytest.approx(0.0)
    assert _forced_bid_prob_at(99, config) == pytest.approx(0.0)


def test_train_smoke_runs_end_to_end(tmp_path) -> None:  # type: ignore[no-untyped-def]
    config = TrainConfig(
        n_iterations=3,
        n_deals_per_iter=6,
        seed=0,
        ppo=PPOConfig(epochs=1, minibatch_size=32),
        opponent_prob=0.5,
        snapshot_every=1,
        eval_every=2,
        eval_n_pairs=3,
        checkpoint_dir=tmp_path / "checkpoints",
        log_dir=tmp_path / "runs",
    )
    train(config, rules=RULES)

    final = tmp_path / "checkpoints" / "final.pt"
    assert final.exists()
    net, payload = load_checkpoint(final, RULES)
    assert payload["iteration"] == 3
    assert net is not None

    mid_checkpoints = list((tmp_path / "checkpoints").glob("iter_*.pt"))
    assert len(mid_checkpoints) >= 1
