"""`RuleDoubler` adds exactly one behaviour -- doubling a standing opponent bid when the rule says
the bidder can't reach it -- and otherwise delegates untouched."""

from __future__ import annotations

import random

from bazarblot.agents.doubling import RuleDoubler
from bazarblot.agents.heuristic import HeuristicAgent
from bazarblot.core.auction import BidAction
from bazarblot.core.deal import Deal, Phase
from bazarblot.core.dealing import deal_hands
from bazarblot.core.rules import load_default
from bazarblot.env.actions import CONTRA_IDX, build_action_space, legal_mask
from bazarblot.env.infoset import info_set
from bazarblot.env.tracked_deal import TrackedDeal

RULES = load_default()
SPACE = build_action_space(RULES)


def _facing_a_bid(level: int, seed: int = 0):  # type: ignore[no-untyped-def]
    """The info set of the opposing seat right after a bid of `level` in hearts."""
    deal = Deal(RULES, dealer=0, hands=deal_hands(random.Random(seed), RULES), deal_id=0)
    tracked = TrackedDeal(deal)
    tracked.step(BidAction(level=level, contract_type="H", capot=False))
    assert deal.phase == Phase.AUCTION
    info = info_set(tracked, deal.to_act, match_score=(0, 0), deal_number=0)
    return info, legal_mask(info, SPACE)


def test_doubles_a_bid_it_judges_unreachable_and_not_one_it_cannot_judge_so() -> None:
    heuristic = HeuristicAgent(RULES)
    info, mask = _facing_a_bid(80)  # the top of the ladder: nobody reaches this
    assert mask[CONTRA_IDX]
    assert RuleDoubler(heuristic, RULES, slack=0).act(info, SPACE, mask) == CONTRA_IDX
    # With enough doubt the same bid is left alone.
    assert RuleDoubler(heuristic, RULES, slack=200).act(info, SPACE, mask) != CONTRA_IDX


def test_delegates_when_doubling_is_not_available_and_in_play() -> None:
    heuristic = HeuristicAgent(RULES)
    rule = RuleDoubler(heuristic, RULES, slack=0)

    deal = Deal(RULES, dealer=0, hands=deal_hands(random.Random(1), RULES), deal_id=0)
    tracked = TrackedDeal(deal)
    info = info_set(tracked, deal.to_act, match_score=(0, 0), deal_number=0)
    mask = legal_mask(info, SPACE)
    assert not mask[CONTRA_IDX]  # nothing to double yet
    assert rule.act(info, SPACE, mask) == heuristic.act(info, SPACE, mask)

    for seed in range(40):
        deal = Deal(RULES, dealer=0, hands=deal_hands(random.Random(seed), RULES), deal_id=0)
        tracked = TrackedDeal(deal)
        while deal.phase == Phase.AUCTION:
            i = info_set(tracked, deal.to_act, match_score=(0, 0), deal_number=0)
            tracked.step(SPACE.decode(heuristic.act(i, SPACE, legal_mask(i, SPACE))))
        if deal.phase == Phase.PLAY:
            i = info_set(tracked, deal.to_act, match_score=(0, 0), deal_number=0)
            m = legal_mask(i, SPACE)
            assert rule.act(i, SPACE, m) == heuristic.act(i, SPACE, m)
            return
    raise AssertionError("no seed reached the play phase")
