"""The standard suit order (`env/canonical.py`) and the v4 trick / combination features."""

from __future__ import annotations

import random

import numpy as np
import torch

from bazarblot.agents.nn.network import build_network
from bazarblot.agents.nn.ppo_agent import decide
from bazarblot.core.auction import BidAction, PassAction
from bazarblot.core.cards import card_id
from bazarblot.core.deal import Deal, Phase
from bazarblot.core.dealing import deal_hands
from bazarblot.core.play import PlayCardAction, current_winner
from bazarblot.core.rules import load_default
from bazarblot.env.actions import build_action_space, legal_mask
from bazarblot.env.augment import permute_card, permute_contract_type, permute_hands
from bazarblot.env.canonical import (
    canonical_permutation,
    permute_infoset,
    to_canonical_mask,
    to_env_action,
)
from bazarblot.env.infoset import info_set
from bazarblot.env.obs import encode, flatten
from bazarblot.env.tracked_deal import TrackedDeal

RULES = load_default()
SPACE = build_action_space(RULES)


def _relabel_action(action, perm):  # type: ignore[no-untyped-def]
    if isinstance(action, BidAction):
        return BidAction(
            action.level, permute_contract_type(action.contract_type, perm), action.capot
        )
    if isinstance(action, PlayCardAction):
        return PlayCardAction(permute_card(action.card, perm))
    return action


def _no_ties(info) -> bool:  # type: ignore[no-untyped-def]
    keys = []
    for si in range(4):
        held = [c % 8 for c in info.original_hand if c // 8 == si]
        trump = info.contract is not None and info.contract.contract_type == "CDHS"[si]
        keys.append((trump, len(held), sum(held)))
    return len(set(keys)) == 4


def test_relabeled_deal_gives_the_same_observation() -> None:
    """A deal and its suit-relabeled twin, played the same way, look identical to the network
    whenever the deciding seat's suits are not tied (ties fall back to the real suit order)."""
    rng = random.Random(0)
    compared = 0
    for d in range(40):
        hands = deal_hands(rng, RULES)
        perm = tuple(rng.sample(range(4), 4))
        deal_a = Deal(RULES, dealer=d % 4, hands=hands, deal_id=d)
        deal_b = Deal(RULES, dealer=d % 4, hands=permute_hands(hands, perm), deal_id=d)  # type: ignore[arg-type]
        ta, tb = TrackedDeal(deal_a), TrackedDeal(deal_b)
        while deal_a.phase in (Phase.AUCTION, Phase.PLAY):
            seat = deal_a.to_act
            ia = info_set(ta, seat, (0, 0), 0)
            ib = info_set(tb, seat, (0, 0), 0)
            if _no_ties(ia):
                oa = flatten(encode(permute_infoset(ia, canonical_permutation(ia))))
                ob = flatten(encode(permute_infoset(ib, canonical_permutation(ib))))
                assert np.array_equal(oa, ob), f"deal {d} seat {seat} phase {deal_a.phase}"
                compared += 1
            if deal_a.phase == Phase.AUCTION:
                legal_idx = [int(i) for i in legal_mask(ia, SPACE).nonzero()[0]]
                bids = [SPACE.decode(i) for i in legal_idx if i != 0]
                if len(ta.auction_log) > 6 or not bids or rng.random() < 0.4:
                    action = PassAction()
                else:
                    action = rng.choice(
                        [b for b in bids if isinstance(b, BidAction)] or [PassAction()]
                    )
            else:
                action = rng.choice(deal_a.legal_actions())
            ta.step(action)
            tb.step(_relabel_action(action, perm))
    assert compared > 200


def test_masks_and_actions_round_trip() -> None:
    torch.manual_seed(0)
    net = build_network(RULES)
    rng = random.Random(3)
    for d in range(10):
        deal = Deal(RULES, dealer=d % 4, hands=deal_hands(rng, RULES), deal_id=d)
        tracked = TrackedDeal(deal)
        while deal.phase in (Phase.AUCTION, Phase.PLAY):
            if deal.phase == Phase.AUCTION and len(tracked.auction_log) > 8:
                tracked.step(PassAction())
                continue
            info = info_set(tracked, deal.to_act, (0, 0), 0)
            mask = legal_mask(info, SPACE)
            perm = canonical_permutation(info)
            cmask = to_canonical_mask(mask, SPACE, perm)
            assert np.array_equal(cmask, legal_mask(permute_infoset(info, perm), SPACE))
            decision = decide(net, info, SPACE, mask, sample=True)
            assert mask[decision.env_action_idx]
            assert decision.env_action_idx == to_env_action(decision.action_idx, SPACE, perm)
            tracked.step(SPACE.decode(decision.env_action_idx))


def test_card_power_matches_the_engine() -> None:
    rng = random.Random(5)
    checked = 0
    for d in range(30):
        deal = Deal(RULES, dealer=d % 4, hands=deal_hands(rng, RULES), deal_id=d)
        tracked = TrackedDeal(deal)
        tracked.step(BidAction(8, rng.choice(["C", "D", "H", "S", "NT"])))  # type: ignore[arg-type]
        while deal.phase == Phase.AUCTION:
            tracked.step(PassAction())
        while deal.phase == Phase.PLAY:
            seat = deal.to_act
            info = info_set(tracked, seat, (0, 0), 0)
            blocks = encode(info)
            power, state = blocks["card_power"], blocks["trick_state"]
            t = deal.tables
            assert t is not None
            if deal.current_trick:
                w = current_winner(deal.current_trick, t)
                assert state[(w - seat) % 4] == 1.0
                for c in deal.hands[seat]:
                    wins = current_winner([*deal.current_trick, (seat, c)], t) == seat
                    assert power[c, 0] == float(wins)
                    checked += 1
            else:
                assert state[5] == 1.0
            # exactly one master card per suit that still has unplayed cards
            for si in range(4):
                assert power[si * 8 : si * 8 + 8, 4].sum() <= 1.0
            tracked.step(rng.choice(deal.legal_actions()))
    assert checked > 100


def test_blot_is_not_visible_before_it_is_played() -> None:
    """Seat 1 holds the trump K and Q: nobody's observation shows it until seat 1 plays one."""
    h1 = frozenset(
        {
            card_id("H", "K"),
            card_id("H", "Q"),
            *(card_id("C", r) for r in "789"),
            card_id("D", "7"),
            card_id("D", "8"),
            card_id("S", "7"),
        }
    )
    rest = sorted(set(range(32)) - h1)
    hands = (frozenset(rest[:8]), h1, frozenset(rest[8:16]), frozenset(rest[16:]))
    deal = Deal(RULES, dealer=3, hands=hands, deal_id=0)
    tracked = TrackedDeal(deal)
    tracked.step(BidAction(8, "H"))
    while deal.phase == Phase.AUCTION:
        tracked.step(PassAction())
    rng = random.Random(0)
    said = False
    while deal.phase == Phase.PLAY:
        for seat in range(4):
            info = info_set(tracked, seat, (0, 0), 0)
            row = encode(info)["melds_public"][(1 - seat) % 4]
            assert row[26] == float(said)  # blot flag
        if deal.to_act == 1:
            legal = deal.legal_actions()
            kq = [
                a
                for a in legal
                if isinstance(a, PlayCardAction)
                and a.card in (card_id("H", "K"), card_id("H", "Q"))
            ]
            action = kq[0] if kq and not said else rng.choice(legal)
            said = said or bool(kq and action in kq)
        else:
            action = rng.choice(deal.legal_actions())
        tracked.step(action)
