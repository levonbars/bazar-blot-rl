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
