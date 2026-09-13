"""M3: the flat action space. `test_action_dim_is_pinned` is the guard against the space
silently desyncing from the preset (see `env/actions.py`'s module docstring for why `ACTION_DIM`
is 765, not the spec's worked 796 — that number assumes the combination protocol is live)."""

from __future__ import annotations

import random

from bazarblot.core.auction import BidAction, ContraAction, PassAction, RecontraAction, is_legal
from bazarblot.core.deal import Deal, Phase
from bazarblot.core.dealing import deal_hands
from bazarblot.core.play import PlayCardAction, legal_moves
from bazarblot.core.rules import load_default
from bazarblot.env.actions import build_action_space, legal_mask
from bazarblot.env.infoset import info_set
from bazarblot.env.tracked_deal import TrackedDeal

RULES = load_default()
SPACE = build_action_space(RULES)


def test_action_dim_is_pinned() -> None:
    assert SPACE.min_bid == 8
    assert SPACE.max_bid == 80
    assert SPACE.contract_types == ("C", "D", "H", "S", "NT")
    assert SPACE.capot_states == 2
    assert SPACE.n_bid_actions == (80 - 8 + 1) * 5 * 2  # 730
    assert SPACE.action_dim == 3 + 730 + 32  # 765 — see module docstring for why not 796


def test_encode_decode_roundtrip_every_index() -> None:
    for idx in range(SPACE.action_dim):
        action = SPACE.decode(idx)
        assert SPACE.encode(action) == idx


def test_decode_out_of_range_raises() -> None:
    import pytest

    with pytest.raises(ValueError):
        SPACE.decode(-1)
    with pytest.raises(ValueError):
        SPACE.decode(SPACE.action_dim)


def test_pass_contra_recontra_fixed_indices() -> None:
    assert SPACE.encode(PassAction()) == 0
    assert SPACE.encode(ContraAction()) == 1
    assert SPACE.encode(RecontraAction()) == 2


def test_bid_index_formula_matches_spec_shape() -> None:
    # index = 3 + combos_per_level*(level-MIN) + type_idx*capot_states + capot_idx
    combos_per_level = len(SPACE.contract_types) * SPACE.capot_states
    for level in (SPACE.min_bid, SPACE.min_bid + 1, SPACE.max_bid):
        for ti, t in enumerate(SPACE.contract_types):
            for capot in (False, True):
                level_offset = combos_per_level * (level - SPACE.min_bid)
                expected = 3 + level_offset + ti * SPACE.capot_states + int(capot)
                assert SPACE.bid_index(level, t, capot) == expected


def test_play_index_offset() -> None:
    assert SPACE.play_index(0) == SPACE.play_start
    assert SPACE.play_index(31) == SPACE.play_start + 31


def test_legal_mask_matches_core_legality_on_random_deals() -> None:
    """`legal_mask` must agree exactly with `core.auction.is_legal` / `core.play.legal_moves` —
    the whole point of computing it from the `InfoSet` via those functions rather than
    re-deriving the rules is that it CAN'T disagree with them."""
    for seed in range(150):
        rng = random.Random(seed)
        hands = deal_hands(rng, RULES)
        deal = Deal(RULES, dealer=seed % 4, hands=hands, deal_id=seed)
        tracked = TrackedDeal(deal)
        steps = 0
        while deal.phase in (Phase.AUCTION, Phase.PLAY) and steps < 60:
            seat = deal.to_act
            info = info_set(tracked, seat, match_score=(0, 0), deal_number=0)
            mask = legal_mask(info, SPACE)

            if deal.phase == Phase.AUCTION:
                expected_legal = set()
                for auction_action, idx in (
                    (PassAction(), 0),
                    (ContraAction(), 1),
                    (RecontraAction(), 2),
                ):
                    if is_legal(deal.auction_state, auction_action, RULES):
                        expected_legal.add(idx)
                for level in range(SPACE.min_bid, SPACE.max_bid + 1):
                    for t in SPACE.contract_types:
                        for capot in (False, True):
                            bid = BidAction(level=level, contract_type=t, capot=capot)
                            if is_legal(deal.auction_state, bid, RULES):
                                expected_legal.add(SPACE.bid_index(level, t, capot))
            else:
                assert deal.contract is not None and deal.tables is not None
                legal_cards = legal_moves(
                    deal.hands[seat],
                    deal.current_trick,
                    seat,
                    deal.contract.contract_type,
                    RULES,
                    deal.tables,
                )
                expected_legal = {SPACE.play_index(c) for c in legal_cards}

            actual_legal = set(mask.nonzero()[0].tolist())
            assert actual_legal == expected_legal, f"seed={seed} step={steps} phase={deal.phase}"

            idx = rng.choice(sorted(actual_legal))
            action = SPACE.decode(idx)
            if isinstance(action, PlayCardAction):
                tracked.step(action)
            else:
                tracked.step(action)
            steps += 1


def test_legal_mask_empty_when_terminal_or_aborted() -> None:
    rng = random.Random(0)
    hands = deal_hands(rng, RULES)
    deal = Deal(RULES, dealer=0, hands=hands, deal_id=0)
    tracked = TrackedDeal(deal)
    steps = 0
    while deal.phase in (Phase.AUCTION, Phase.PLAY) and steps < 500:
        seat = deal.to_act
        info = info_set(tracked, seat, match_score=(0, 0), deal_number=0)
        mask = legal_mask(info, SPACE)
        legal = mask.nonzero()[0]
        action = SPACE.decode(int(rng.choice(legal)))
        tracked.step(action)
        steps += 1
    assert deal.phase in (Phase.TERMINAL, Phase.ABORTED)
    info = info_set(tracked, 0, match_score=(0, 0), deal_number=0)
    mask = legal_mask(info, SPACE)
    assert mask.sum() == 0
