"""Parallel rollouts (`learn/parallel.py`): packing must be lossless, and a parallel collection
must be reproducible and consistent with the serial contract."""

from __future__ import annotations

import random

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from bazarblot.agents.nn.network import apply_action_prior, build_network  # noqa: E402
from bazarblot.core.rules import load_default  # noqa: E402
from bazarblot.env.actions import build_action_space  # noqa: E402
from bazarblot.learn.parallel import (  # noqa: E402
    ParallelRollout,
    pack_trajectories,
    unpack_trajectories,
)
from bazarblot.learn.rollout import play_rollout_deals  # noqa: E402

RULES = load_default()
SPACE = build_action_space(RULES)


def _net() -> torch.nn.Module:
    torch.manual_seed(0)
    net = build_network(RULES)
    apply_action_prior(net, raise_decay=0.5, capot_bias=-4.0, double_bias=-3.0)
    return net


def test_pack_unpack_round_trip() -> None:
    net = _net()
    trajectories, _ = play_rollout_deals(RULES, SPACE, net, range(12), 5, random.Random(0))
    back = unpack_trajectories(pack_trajectories(trajectories))
    assert [len(t) for t in back] == [len(t) for t in trajectories]
    for ta, tb in zip(
        [t for tr in trajectories for t in tr], [t for tr in back for t in tr], strict=True
    ):
        da, db = ta.decision, tb.decision
        assert (da.action_idx, da.log_prob, da.value, da.phase, da.base_level, da.forced) == (
            db.action_idx,
            db.log_prob,
            db.value,
            db.phase,
            db.base_level,
            db.forced,
        )
        assert np.array_equal(da.obs, db.obs) and np.array_equal(da.mask, db.mask)
        assert ta.belief_target is not None and tb.belief_target is not None
        assert np.array_equal(ta.belief_target, tb.belief_target)
        assert (ta.reward, ta.outcome_target, ta.value) == (tb.reward, tb.outcome_target, tb.value)


def test_parallel_collection_is_reproducible_and_complete() -> None:
    net = _net()
    snap = _net()
    with ParallelRollout(RULES, 2, hidden=(256, 256), fixed_opponents=("random",)) as pr:
        a, sa = pr.collect(net, [(0, snap)], n_deals=16, seed=7, opponent_prob=0.5)
        b, sb = pr.collect(net, [(0, snap)], n_deals=16, seed=7, opponent_prob=0.5)
    assert [t.decision.action_idx for t in a] == [t.decision.action_idx for t in b]
    assert sa.n_deals == sb.n_deals == 16 + sa.n_opponent_deals  # paired opponent shuffles x2
    assert sa.n_transitions == len(a) > 0
    assert sum(t.reward != 0.0 for t in a) > 0
