"""Regression tests for the combination-units bug (roadmap M7): combination values are RAW points
(a tierce is 20) and count toward a contract exactly like card points, so they must be summed with
card points BEFORE scaling to bid units. Three places had added them after scaling, overstating any
hand with a combination by up to 18 bid levels per tierce."""

from __future__ import annotations

import random

from bazarblot.agents.heuristic import HeuristicAgent, _estimate_scaled_value
from bazarblot.agents.pass_bidder import PassBidder
from bazarblot.core.cards import card_id
from bazarblot.core.declarations import detect_hand_melds
from bazarblot.core.rules import load_default
from bazarblot.env.actions import build_action_space
from bazarblot.env.obs import _fulfilment_margin
from bazarblot.eval.duplicate import play_deal_once
from bazarblot.eval.metrics import scaled_value

RULES = load_default()
SPACE = build_action_space(RULES)


def test_heuristic_estimate_adds_a_tierce_as_two_bid_levels_not_twenty() -> None:
    # Same suit-S tierce (7-8-9) plus fillers; remove the tierce's middle card to break it.
    with_tierce = frozenset(
        card_id(s, r) for s, r in [("S", "7"), ("S", "8"), ("S", "9"), ("C", "A"), ("D", "A")]
    )
    without = frozenset(
        card_id(s, r) for s, r in [("S", "7"), ("H", "8"), ("S", "9"), ("C", "A"), ("D", "A")]
    )
    melds = detect_hand_melds(with_tierce, 0, "S", RULES)
    assert [m.value for m in melds] == [20]  # a tierce is worth 20 RAW points
    gain = _estimate_scaled_value(with_tierce, "S", RULES) - _estimate_scaled_value(
        without, "S", RULES
    )
    # 20 raw points are ~2 bid levels; the old arithmetic gave ~+20. Card values also differ
    # slightly between the two hands (8 of spades vs 8 of hearts), hence the tolerance.
    assert 0 <= gain <= 4


def test_heuristic_mostly_makes_the_contracts_it_bids() -> None:
    """With the bug, 11% of the heuristic's contracts were made; it should now make most."""
    heuristic = HeuristicAgent(RULES)
    results = [
        r
        for r in (
            play_deal_once(RULES, SPACE, i, 0, i % 4, dict.fromkeys(range(4), heuristic))
            for i in range(300)
        )
        if r is not None
    ]
    assert len(results) > 150
    made = sum(r.made for r in results) / len(results)
    assert made > 0.6, f"heuristic made only {made:.2f} of its contracts"


def test_fulfilment_margin_counts_combinations_as_raw_points() -> None:
    # A team with no card points, one 20-point tierce, bidding 10 (needs 100 raw): 80 short.
    # The old scaled-then-added arithmetic reported it as already over its bid.
    assert _fulfilment_margin(0, 20, 10, RULES) == -80
    assert _fulfilment_margin(90, 20, 10, RULES) == 10  # 110 raw vs the 100 needed


def test_oracle_scaled_value_sums_raw_points_before_scaling() -> None:
    assert scaled_value(84, 20, RULES) == 10  # 104 raw -> 10 bid levels (old code: 8 + 20 = 28)
    assert scaled_value(84, 0, RULES) == 8


def test_pass_bidder_always_passes_and_refuses_to_play() -> None:
    import numpy as np
    import pytest

    from bazarblot.core.deal import Deal, Phase
    from bazarblot.core.dealing import deal_hands
    from bazarblot.env.actions import PASS_IDX, legal_mask
    from bazarblot.env.infoset import info_set
    from bazarblot.env.tracked_deal import TrackedDeal

    deal = Deal(RULES, dealer=0, hands=deal_hands(random.Random(0), RULES), deal_id=0)
    tracked = TrackedDeal(deal)
    info = info_set(tracked, deal.to_act, match_score=(0, 0), deal_number=0)
    mask = legal_mask(info, SPACE)
    assert PassBidder().act(info, SPACE, mask) == PASS_IDX

    # drive to the play phase with a bidder that opens, then check the refusal
    heuristic = HeuristicAgent(RULES)
    for seed in range(50):
        deal = Deal(RULES, dealer=0, hands=deal_hands(random.Random(seed), RULES), deal_id=0)
        tracked = TrackedDeal(deal)
        while deal.phase == Phase.AUCTION:
            info = info_set(tracked, deal.to_act, match_score=(0, 0), deal_number=0)
            m = legal_mask(info, SPACE)
            tracked.step(SPACE.decode(heuristic.act(info, SPACE, m)))
        if deal.phase == Phase.PLAY:
            info = info_set(tracked, deal.to_act, match_score=(0, 0), deal_number=0)
            with pytest.raises(ValueError, match="auction"):
                PassBidder().act(info, SPACE, np.ones_like(legal_mask(info, SPACE)))
            return
    raise AssertionError("no seed reached the play phase")


def test_heuristic_bid_offset_shifts_how_often_it_bids() -> None:
    """`bid_offset` builds held-out opponents of different bidding styles: more aggressive means
    more auctions end in a contract. The default (0) must be the unchanged reference heuristic."""

    def contract_rate(agent: HeuristicAgent) -> float:
        results = [
            play_deal_once(RULES, SPACE, i, 0, i % 4, dict.fromkeys(range(4), agent))
            for i in range(200)
        ]
        return sum(r is not None for r in results) / len(results)

    timid = contract_rate(HeuristicAgent(RULES, bid_offset=-6))
    reference = contract_rate(HeuristicAgent(RULES))
    aggressive = contract_rate(HeuristicAgent(RULES, bid_offset=6))
    assert timid < reference < aggressive
    assert contract_rate(HeuristicAgent(RULES, bid_offset=0)) == reference
