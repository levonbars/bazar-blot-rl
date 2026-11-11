"""M7: behavior cloning of the heuristic's card play. The checks that matter: the recorded
dataset is internally consistent (each label is a legal card at its position), noise injection
changes the executed play but never the label, and cloning actually learns the teacher on a
small dataset."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import numpy as np  # noqa: E402

from bazarblot.agents.heuristic import HeuristicAgent  # noqa: E402
from bazarblot.agents.nn.network import (  # noqa: E402
    META_BID,
    META_CONTRA,
    META_PASS,
    build_network,
)
from bazarblot.core.rules import load_default  # noqa: E402
from bazarblot.env.actions import build_action_space  # noqa: E402
from bazarblot.learn.belief import IGNORE  # noqa: E402
from bazarblot.learn.imitation import (  # noqa: E402
    NO_LABEL,
    behavior_clone,
    collect_imitation_data,
)

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


def test_auction_labels_are_legal_under_the_sampler_masks() -> None:
    """Every auction sample's label must be a legal choice under the exact masks the sampler and
    PPO use, factor by factor (meta, then delta, type given delta, capot given delta and type)."""
    for noise in (0.0, 0.5):
        data = collect_imitation_data(
            RULES, SPACE, HeuristicAgent(RULES), n_deals=60, seed=2, auction_noise_prob=noise
        )
        m = data.n_auction
        assert m > 100
        rows = np.arange(m)
        assert data.a_meta_mask[rows, data.a_meta].all()
        is_bid = data.a_meta == META_BID
        assert is_bid.any() and (~is_bid).any()
        b = rows[is_bid]
        assert data.a_delta_mask[b, data.a_delta[b]].all()
        assert data.a_type_mask[b, data.a_type[b]].all()
        assert data.a_capot_mask[b, data.a_capot[b]].all()
        nb = rows[~is_bid]
        assert (data.a_delta[nb] == NO_LABEL).all()
        assert (data.a_type[nb] == NO_LABEL).all()
        assert (data.a_capot[nb] == NO_LABEL).all()


def test_the_teacher_only_passes_or_bids() -> None:
    """The fixed heuristic never doubles, so noise-free labels have no CONTRA/RECONTRA."""
    data = collect_imitation_data(
        RULES, SPACE, HeuristicAgent(RULES), n_deals=60, seed=2, auction_noise_prob=0.0
    )
    assert set(np.unique(data.a_meta)) <= {META_PASS, META_BID}


def test_cloned_network_actually_bids_and_completes_auctions() -> None:
    """Regression test for the original M7 failure: agents trained against the broken heuristic
    learned to concede every auction (0 declarations, 0 completed self-play auctions). A network
    cloned from the fixed heuristic must bid, declare and reach contracts."""
    import random

    from bazarblot.agents.nn.ppo_agent import NNAgent
    from bazarblot.eval.behavior import evaluate_behavior

    torch.manual_seed(0)
    teacher = HeuristicAgent(RULES)
    data = collect_imitation_data(RULES, SPACE, teacher, n_deals=500, seed=0)
    net = build_network(RULES)
    behavior_clone(net, data, epochs=8, batch_size=256, verbose=False)
    net.eval()

    rng = random.Random(1)
    seeds = [rng.randrange(1 << 30) for _ in range(20)]
    dealers = [rng.randrange(4) for _ in range(20)]
    report = evaluate_behavior(RULES, SPACE, NNAgent(net, sample=False), teacher, seeds, dealers)
    assert report.declare_rate is not None and report.declare_rate > 0.1
    assert report.selfplay_contract_rate > 0.3


def test_opponents_restrict_recording_to_the_teachers_seats() -> None:
    """With an opponent table, only the teacher's decisions are recorded (about half as many as
    when the teacher plays all four seats), and `opponent is teacher` is the all-teacher case."""
    teacher = HeuristicAgent(RULES)
    everyone = collect_imitation_data(RULES, SPACE, teacher, n_deals=80, seed=5)
    vs_aggressive = collect_imitation_data(
        RULES, SPACE, teacher, n_deals=80, seed=5, opponents=[HeuristicAgent(RULES, bid_offset=4)]
    )
    ratio = vs_aggressive.n_auction / everyone.n_auction
    assert 0.3 < ratio < 0.8
    assert vs_aggressive.a_obs.shape[0] == vs_aggressive.n_auction > 0

    same = collect_imitation_data(RULES, SPACE, teacher, n_deals=80, seed=5, opponents=[teacher])
    assert np.array_equal(same.action, everyone.action)
    assert np.array_equal(same.a_meta, everyone.a_meta)


def test_a_rule_doubler_teacher_produces_doubling_labels_against_overbidders() -> None:
    """Doubling only comes up against bids worth doubling: the data must contain doubling labels
    when the rule-wrapped teacher faces reckless bidders, and (essentially) none when it plays a
    plain heuristic table."""
    from bazarblot.agents.doubling import RuleDoubler

    base = HeuristicAgent(RULES)
    teacher = RuleDoubler(base, RULES, slack=0)
    reckless = collect_imitation_data(
        RULES, SPACE, teacher, n_deals=120, seed=1, opponents=[HeuristicAgent(RULES, bid_offset=10)]
    )
    assert int((reckless.a_meta == META_CONTRA).sum()) > 5

    plain = collect_imitation_data(RULES, SPACE, base, n_deals=120, seed=1)
    assert int((plain.a_meta == META_CONTRA).sum()) == 0
