"""M7: the `(Delta, type, capot)` factoring of the flat bid action segment, cross-checked against
`core.auction.is_legal` directly rather than trusting the reshape/re-base arithmetic by
inspection."""

from __future__ import annotations

import random

from bazarblot.agents.nn.bid_factor import base_level, bid_legality, flat_bid_index, n_delta_slots
from bazarblot.core.auction import BidAction, is_legal
from bazarblot.core.deal import Deal, Phase
from bazarblot.core.dealing import deal_hands
from bazarblot.core.rules import load_default
from bazarblot.env.actions import build_action_space, legal_mask
from bazarblot.env.infoset import info_set
from bazarblot.env.tracked_deal import TrackedDeal

RULES = load_default()
SPACE = build_action_space(RULES)


def test_n_delta_slots_covers_an_opening_at_max_bid() -> None:
    # Opening (virtual base = min_bid - 1 = 7) at the ladder's very top (80) is a real, legal
    # bid -- see the module docstring on why this is 73, not the spec's stated 72.
    virtual_base = RULES.auction.min_bid - 1
    delta_for_max_opening = RULES.auction.max_bid - virtual_base
    assert delta_for_max_opening == n_delta_slots(SPACE)


def test_base_level_is_virtual_min_bid_minus_one_before_any_bid() -> None:
    deal = Deal(RULES, dealer=0, hands=deal_hands(random.Random(0), RULES), deal_id=0)
    tracked = TrackedDeal(deal)
    info = info_set(tracked, deal.to_act, match_score=(0, 0), deal_number=0)
    assert base_level(info) == RULES.auction.min_bid - 1


def test_bid_legality_matches_core_legality_on_random_auctions() -> None:
    """Drive many random auctions; at every AUCTION decision, cross-check the factored cube
    against `is_legal` called directly for every `(level, type, capot)` triple -- the actual
    ground truth, not `legal_mask`'s own reshape (which this module's correctness must not
    circularly depend on)."""
    rng = random.Random(1)
    n_checked = 0
    for deal_id in range(60):
        deal = Deal(RULES, dealer=deal_id % 4, hands=deal_hands(rng, RULES), deal_id=deal_id)
        tracked = TrackedDeal(deal)
        while deal.phase == Phase.AUCTION:
            info = info_set(tracked, deal.to_act, match_score=(0, 0), deal_number=deal_id)
            mask = legal_mask(info, SPACE)
            base = base_level(info)
            legality = bid_legality(mask, SPACE, base)

            for level in range(SPACE.min_bid, SPACE.max_bid + 1):
                delta = level - base
                for type_idx, contract_type in enumerate(SPACE.contract_types):
                    for capot_idx in range(SPACE.capot_states):
                        capot = bool(capot_idx) if SPACE.capot_states == 2 else False
                        actual = is_legal(
                            info.auction_state,
                            BidAction(level=level, contract_type=contract_type, capot=capot),
                            info.rules,
                        )
                        if 1 <= delta <= legality.delta_mask.shape[0]:
                            factored = bool(
                                legality.capot_mask_given_delta_type[delta - 1, type_idx, capot_idx]
                            )
                        else:
                            factored = False
                        assert factored == actual, (level, contract_type, capot, delta)
                        n_checked += 1

            # Advance with any legal action so later decisions get exercised too.
            legal_idx = mask.nonzero()[0]
            action_idx = int(rng.choice(legal_idx))
            tracked.step(SPACE.decode(action_idx))
    assert n_checked > 10_000  # sanity: the loop above actually ran


def test_flat_bid_index_round_trips_through_decode() -> None:
    rng = random.Random(2)
    deal = Deal(RULES, dealer=0, hands=deal_hands(rng, RULES), deal_id=0)
    tracked = TrackedDeal(deal)
    info = info_set(tracked, deal.to_act, match_score=(0, 0), deal_number=0)
    base = base_level(info)
    for level in range(SPACE.min_bid, SPACE.max_bid + 1):
        for type_idx, contract_type in enumerate(SPACE.contract_types):
            for capot_idx in range(SPACE.capot_states):
                idx = flat_bid_index(SPACE, base, level - base, type_idx, capot_idx)
                decoded = SPACE.decode(idx)
                assert isinstance(decoded, BidAction)
                assert decoded.level == level
                assert decoded.contract_type == contract_type
                expected_capot = bool(capot_idx) if SPACE.capot_states == 2 else False
                assert decoded.capot == expected_capot


def test_bid_legality_shapes() -> None:
    deal = Deal(RULES, dealer=0, hands=deal_hands(random.Random(3), RULES), deal_id=0)
    tracked = TrackedDeal(deal)
    info = info_set(tracked, deal.to_act, match_score=(0, 0), deal_number=0)
    mask = legal_mask(info, SPACE)
    legality = bid_legality(mask, SPACE, base_level(info))
    n_delta = n_delta_slots(SPACE)
    n_types = len(SPACE.contract_types)
    assert legality.delta_mask.shape == (n_delta,)
    assert legality.type_mask_given_delta.shape == (n_delta, n_types)
    assert legality.capot_mask_given_delta_type.shape == (n_delta, n_types, SPACE.capot_states)
    assert legality.any_bid_legal() == bool(mask[SPACE.bid_start : SPACE.play_start].any())
