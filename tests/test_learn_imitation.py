"""M7: behavior cloning of the heuristic's card play. The checks that matter: the recorded
dataset is internally consistent (each label is a legal card at its position), noise injection
changes the executed play but never the label, and cloning actually learns the teacher on a
small dataset."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import numpy as np  # noqa: E402

from bazarblot.agents.heuristic import HeuristicAgent  # noqa: E402
from bazarblot.agents.nn.network import build_network  # noqa: E402
from bazarblot.core.rules import load_default  # noqa: E402
from bazarblot.env.actions import build_action_space  # noqa: E402
from bazarblot.learn.belief import IGNORE  # noqa: E402
from bazarblot.learn.imitation import behavior_clone, collect_imitation_data  # noqa: E402

RULES = load_default()
SPACE = build_action_space(RULES)


def test_collected_labels_are_legal_and_aligned() -> None:
    data = collect_imitation_data(RULES, SPACE, HeuristicAgent(RULES), n_deals=40, seed=0)
    n = len(data)
    assert n > 100
    assert data.obs.shape[0] == data.card_mask.shape[0] == data.belief.shape[0] == n
    rows = np.arange(n)
    assert data.card_mask[rows, data.action].all(), "teacher label must be a legal card"
    # 24 unseen cards at a deal's first play, shrinking as others play; none left on its last.
    labelled = (data.belief != IGNORE).sum(axis=1)
    assert labelled.max() == 24 and labelled.min() == 0


def test_noise_changes_play_but_not_labels_being_legal() -> None:
    clean = collect_imitation_data(RULES, SPACE, HeuristicAgent(RULES), 30, seed=1, noise_prob=0.0)
    noisy = collect_imitation_data(RULES, SPACE, HeuristicAgent(RULES), 30, seed=1, noise_prob=0.5)
    rows = np.arange(len(noisy))
    assert noisy.card_mask[rows, noisy.action].all()
    # Executed random plays send later positions down different paths, so the datasets differ.
    assert len(clean) != len(noisy) or not np.array_equal(clean.action, noisy.action)


def test_collection_is_deterministic_for_a_seed() -> None:
    a = collect_imitation_data(RULES, SPACE, HeuristicAgent(RULES), 20, seed=3)
    b = collect_imitation_data(RULES, SPACE, HeuristicAgent(RULES), 20, seed=3)
    assert np.array_equal(a.action, b.action)
    assert np.array_equal(a.obs, b.obs)


def test_behavior_clone_learns_the_teacher() -> None:
    torch.manual_seed(0)
    data = collect_imitation_data(RULES, SPACE, HeuristicAgent(RULES), n_deals=250, seed=0)
    net = build_network(RULES)
    stats = behavior_clone(net, data, epochs=12, batch_size=256, verbose=False)
    # Untrained, a legal-masked guess among ~4-8 cards is well under 0.5.
    assert stats.holdout_card_acc > 0.6
