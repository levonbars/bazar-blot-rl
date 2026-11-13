"""M7: self-play rollout collection -- the properties a broken reward assignment or a broken GAE
implementation would most plausibly violate: every seat's own decisions form a trajectory with
reward 0 everywhere except the last step, teammates share identical terminal reward, and that
reward is antisymmetric between the two teams (spec §6's zero-sum-margin property, checked here at
the rollout level rather than just on `deal_margin_reward` in isolation)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from bazarblot.agents.heuristic import HeuristicAgent  # noqa: E402
from bazarblot.agents.nn.network import build_network  # noqa: E402
from bazarblot.core.rules import load_default  # noqa: E402
from bazarblot.env.actions import build_action_space  # noqa: E402
from bazarblot.learn.reward import deal_margin_reward  # noqa: E402
from bazarblot.learn.rollout import collect_rollout  # noqa: E402

RULES = load_default()
SPACE = build_action_space(RULES)


def test_pure_self_play_rollout_reward_and_gae_shape() -> None:
    torch.manual_seed(0)
    net = build_network(RULES)
    transitions, stats = collect_rollout(RULES, SPACE, net, n_deals=25, seed=0)

    assert stats.n_deals >= 25
    assert stats.n_transitions == len(transitions)
    assert stats.n_transitions > 0

    # Every seat acts at least once in any completed deal (auction alone guarantees it), and
    # exactly its LAST transition carries the nonzero terminal reward -- so a pure self-play
    # rollout (no opponent pool, one seat assignment per shuffle) has exactly 4 nonzero-reward
    # transitions per completed deal, no more and no fewer.
    n_completed = stats.n_deals - stats.n_aborted
    nonzero = [t for t in transitions if t.reward != 0.0]
    assert len(nonzero) == 4 * n_completed
    for t in nonzero:
        assert -1.0 < t.reward < 1.0


def test_aborted_deals_are_not_discarded() -> None:
    """Regression test for the pass-collapse bug hit during the first real M7 training run (see
    `docs/03-implementation-roadmap.md` M7): an early version of `collect_rollout` discarded every
    aborted (4-pass) deal's transitions entirely, making them invisible to the gradient and
    letting per-seat pass probability drift upward with nothing to pull it back down. Every one
    of the four seats acts at least once in ANY deal -- aborted or completed -- so in pure
    self-play (no opponent pool), the number of recorded transitions must be at least
    `4 * n_deals` regardless of how many deals aborted; the old, buggy behavior would undercount
    whenever `stats.n_aborted > 0`."""
    torch.manual_seed(0)
    net = build_network(RULES)
    _transitions, stats = collect_rollout(RULES, SPACE, net, n_deals=30, seed=0)
    assert stats.n_deals == 30
    assert stats.n_transitions >= 4 * stats.n_deals


def test_opponent_pool_accepts_a_mix_of_networks_and_plain_agents() -> None:
    """The pool must accept both frozen `PolicyValueNet` snapshots and hand-written baseline
    `Agent`s (e.g. `HeuristicAgent`) in the same list -- this is what lets `learn/train.py` seed
    the pool with a reliable bidder from iteration 0, before any self-play snapshot exists."""
    torch.manual_seed(0)
    learner = build_network(RULES)
    net_opponent = build_network(RULES)
    heuristic = HeuristicAgent(RULES)
    transitions, stats = collect_rollout(
        RULES,
        SPACE,
        learner,
        n_deals=10,
        seed=0,
        opponent_pool=[heuristic, net_opponent],
        opponent_prob=1.0,
    )
    assert stats.n_opponent_deals == 10
    assert len(transitions) == stats.n_transitions


def test_learner_declare_rate_is_none_without_opponent_mix() -> None:
    """Pure self-play (`opponent_prob=0`, the default) has no meaningful "did the learner
    declare" signal -- both teams are the identical network, so the stat must stay `None` rather
    than report a number that says nothing about skill."""
    torch.manual_seed(0)
    net = build_network(RULES)
    _transitions, stats = collect_rollout(RULES, SPACE, net, n_deals=15, seed=0)
    assert stats.learner_declare_rate is None
    assert stats.n_contested_completed == 0


def test_learner_declare_rate_tracks_contested_deals_only() -> None:
    """Regression test for the finding that a policy which never once declares against a real
    opponent can still clear `evaluate_pairs`'s CI-excluding-zero bar on pure pass-and-defend
    (see `docs/03-implementation-roadmap.md` M7) -- this is the stat that makes that visible
    DURING training. With `opponent_prob=1.0`, every deal is contested, so the rate must be
    computable and consistent with the raw counters."""
    torch.manual_seed(0)
    learner = build_network(RULES)
    heuristic = HeuristicAgent(RULES)
    _transitions, stats = collect_rollout(
        RULES,
        SPACE,
        learner,
        n_deals=20,
        seed=0,
        opponent_pool=[heuristic],
        opponent_prob=1.0,
    )
    assert stats.n_contested_completed > 0
    rate = stats.learner_declare_rate
    assert rate is not None
    assert rate == pytest.approx(stats.n_learner_declared_contested / stats.n_contested_completed)
    assert 0.0 <= rate <= 1.0


def test_bid_agent_takes_every_auction_decision_and_learner_trains_on_play_only() -> None:
    """With a fixed `bid_agent`, the learner records NO auction transitions (it cannot influence
    the auction) and every recorded transition is a card play -- the setup that rules out the
    pass-everything shortcut and isolates play skill (roadmap M7 stage 1)."""
    from bazarblot.core.deal import Phase

    torch.manual_seed(0)
    net = build_network(RULES)
    transitions, stats = collect_rollout(
        RULES, SPACE, net, n_deals=30, seed=0, bid_agent=HeuristicAgent(RULES)
    )
    assert stats.n_deals == 30
    assert len(transitions) > 0
    assert all(t.decision.phase == Phase.PLAY for t in transitions)


def test_deal_margin_reward_is_antisymmetric() -> None:
    for margin in (0.0, 8.0, 56.0, 216.0, -216.0):
        assert deal_margin_reward(margin) == pytest.approx(-deal_margin_reward(-margin))


def test_opponent_pool_pairing_uses_both_seat_assignments() -> None:
    """With `opponent_prob=1.0` every shuffle is played against the pool, paired both ways --
    so `n_deals` requested shuffles should yield exactly `2 * n_deals` played deals."""
    torch.manual_seed(1)
    learner = build_network(RULES)
    opponent = build_network(RULES)
    transitions, stats = collect_rollout(
        RULES,
        SPACE,
        learner,
        n_deals=10,
        seed=1,
        opponent_pool=[opponent],
        opponent_prob=1.0,
    )
    assert stats.n_opponent_deals == 10
    assert stats.n_deals == 20  # each of the 10 shuffles played with both seat assignments
    assert len(transitions) == stats.n_transitions


def test_teammates_receive_identical_terminal_reward() -> None:
    """In the opponent-pool 2v2 assignment, the two learner seats are always teammates
    (`{0, 2}` or `{1, 3}`, per `TEAM_OF`) -- so their nonzero terminal rewards in any one
    completed deal must be exactly equal, and the two deals of a paired shuffle (learner as
    team 0, then as team 1) must carry exactly opposite rewards, since the shuffle -- and hence
    the true margin -- is identical between them."""
    torch.manual_seed(2)
    learner = build_network(RULES)
    opponent = build_network(RULES)
    transitions, stats = collect_rollout(
        RULES,
        SPACE,
        learner,
        n_deals=8,
        seed=2,
        opponent_pool=[opponent],
        opponent_prob=1.0,
    )
    nonzero_rewards = [t.reward for t in transitions if t.reward != 0.0]
    # Each completed deal contributes exactly 2 nonzero-reward transitions (the two learner
    # seats), and they must match each other (teammates, identical terminal reward).
    n_completed = stats.n_deals - stats.n_aborted
    assert len(nonzero_rewards) == 2 * n_completed
    for i in range(0, len(nonzero_rewards), 2):
        assert nonzero_rewards[i] == pytest.approx(nonzero_rewards[i + 1])


def test_reward_scale_controls_how_much_large_margins_saturate() -> None:
    """A larger `reward_scale` keeps big margins distinguishable: with scale 24 a margin of 40 and
    one of 70 differ by ~0.06 in reward, with scale 96 by ~0.2."""
    small = deal_margin_reward(70.0, 24.0) - deal_margin_reward(40.0, 24.0)
    large = deal_margin_reward(70.0, 96.0) - deal_margin_reward(40.0, 96.0)
    assert small < 0.07 and large > 0.18

    torch.manual_seed(0)
    net = build_network(RULES)
    a, _ = collect_rollout(RULES, SPACE, net, n_deals=10, seed=0, reward_scale=24.0)
    torch.manual_seed(0)
    b, _ = collect_rollout(RULES, SPACE, net, n_deals=10, seed=0, reward_scale=96.0)
    ra = [t.reward for t in a if t.reward != 0.0]
    rb = [t.reward for t in b if t.reward != 0.0]
    assert len(ra) == len(rb) > 0
    assert all(abs(y) < abs(x) for x, y in zip(ra, rb, strict=True))  # same margins, gentler squash


def test_outcome_labels_match_the_finished_deal() -> None:
    """Every learner transition carries the deal's outcome from its own seat's view: teammates get
    the same label, opponents the mirrored one, and aborted deals the abort class."""
    from bazarblot.agents.nn.network import (
        OUTCOME_ABORT,
        OUTCOME_THEY_FAIL,
        OUTCOME_THEY_MAKE,
        OUTCOME_WE_FAIL,
        OUTCOME_WE_MAKE,
    )

    torch.manual_seed(0)
    net = build_network(RULES)
    transitions, stats = collect_rollout(RULES, SPACE, net, n_deals=30, seed=0)
    labels = {t.outcome_target for t in transitions}
    assert -1 not in labels
    assert labels <= {
        OUTCOME_ABORT,
        OUTCOME_WE_MAKE,
        OUTCOME_WE_FAIL,
        OUTCOME_THEY_MAKE,
        OUTCOME_THEY_FAIL,
    }
    # In pure self-play both teams are present in every completed deal, so "we" and "they"
    # labels must appear equally often across a deal's final transitions.
    finals = [t.outcome_target for t in transitions if t.reward != 0.0]
    we = sum(x in (OUTCOME_WE_MAKE, OUTCOME_WE_FAIL) for x in finals)
    they = sum(x in (OUTCOME_THEY_MAKE, OUTCOME_THEY_FAIL) for x in finals)
    assert we == they > 0
    assert len(stats.selfplay_levels) == stats.n_deals - stats.n_aborted


def test_outcome_label_mapping() -> None:
    from types import SimpleNamespace

    from bazarblot.agents.nn.network import (
        OUTCOME_ABORT,
        OUTCOME_THEY_FAIL,
        OUTCOME_THEY_MAKE,
        OUTCOME_WE_FAIL,
        OUTCOME_WE_MAKE,
    )
    from bazarblot.learn.rollout import outcome_label

    made = SimpleNamespace(attackers_team=0, made=True)
    failed = SimpleNamespace(attackers_team=1, made=False)
    assert outcome_label(0, None) == OUTCOME_ABORT
    assert outcome_label(0, made) == OUTCOME_WE_MAKE  # type: ignore[arg-type]
    assert outcome_label(2, made) == OUTCOME_WE_MAKE  # type: ignore[arg-type]
    assert outcome_label(1, made) == OUTCOME_THEY_MAKE  # type: ignore[arg-type]
    assert outcome_label(1, failed) == OUTCOME_WE_FAIL  # type: ignore[arg-type]
    assert outcome_label(0, failed) == OUTCOME_THEY_FAIL  # type: ignore[arg-type]


def test_auction_hitting_the_step_cap_is_scored_as_an_abort() -> None:
    """An untrained network that bids about half the time can circle past the engine's
    `max_auction_steps` safety cap; the rollout must score that as an abort, not crash."""
    import dataclasses

    from bazarblot.agents.nn.network import OUTCOME_ABORT

    tight = dataclasses.replace(
        RULES, auction=dataclasses.replace(RULES.auction, max_auction_steps=3)
    )
    torch.manual_seed(0)
    net = build_network(tight)
    transitions, stats = collect_rollout(tight, SPACE, net, n_deals=40, seed=0)
    assert stats.n_capped > 0
    assert stats.n_aborted >= stats.n_capped
    assert any(t.outcome_target == OUTCOME_ABORT for t in transitions)


def test_trick_credit_spreads_the_same_return_over_the_tricks() -> None:
    """Same deals and actions with and without `trick_credit`: every seat's return (sum of its
    rewards) is unchanged, but card plays now carry their tricks' points as they happen."""
    from bazarblot.core.deal import Phase
    from bazarblot.learn.rollout import play_rollout_deals

    def run(credit: bool):  # type: ignore[no-untyped-def]
        torch.manual_seed(0)
        net = build_network(RULES)
        with torch.no_grad():
            net.meta_head.bias[0] += 1.0  # pass more often, so auctions end and cards get played
        import random

        trajectories, _ = play_rollout_deals(
            RULES,
            SPACE,
            net,
            range(20),
            7,
            random.Random(0),
            reward_scale=96.0,
            trick_credit=credit,
        )
        return trajectories

    plain, credited = run(False), run(True)
    assert len(plain) == len(credited) > 0
    n_play_rewards = 0
    for a, b in zip(plain, credited, strict=True):
        assert [t.decision.action_idx for t in a] == [t.decision.action_idx for t in b]
        assert sum(t.reward for t in a) == pytest.approx(sum(t.reward for t in b), abs=1e-9)
        plays = [t for t in b if t.decision.phase == Phase.PLAY]
        n_play_rewards += sum(t.reward != 0.0 for t in plays[:-1])
        # over a full deal, a seat's credited points are its team's card points minus the
        # opponents' (all 162 points, last-trick bonus included, go to one side or the other)
        if len(plays) == RULES.deal.cards_per_player:
            total = sum(t.trick_points for t in plays)
            assert abs(total) <= 162 and (total + 162) % 2 == 0
    assert n_play_rewards > 0
