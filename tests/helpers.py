"""Shared test utilities: a random-legal policy for exercising the engine end to end.

Not part of the public `bazarblot` API — this is a deliberately unsophisticated policy whose
only job is to terminate quickly while visiting a wide variety of legal states. `agents/` (M4)
will have a real heuristic bidder; this is not that.
"""

from __future__ import annotations

import random

from bazarblot.core.auction import (
    AuctionAction,
    AuctionState,
    BidAction,
    ContraAction,
    PassAction,
    RecontraAction,
    is_legal,
)
from bazarblot.core.cards import TEAM_OF
from bazarblot.core.deal import Deal, Phase
from bazarblot.core.dealing import deal_hands
from bazarblot.core.rules import RuleConfig


def random_auction_action(
    state: AuctionState, rules: RuleConfig, rng: random.Random
) -> AuctionAction:
    standing = state.standing_bid

    if (
        standing is not None
        and state.doubling == "none"
        and TEAM_OF[state.to_act] != TEAM_OF[standing.seat]
        and rng.random() < 0.08
    ):
        return ContraAction()
    if (
        standing is not None
        and state.doubling == "contra"
        and TEAM_OF[state.to_act] == TEAM_OF[standing.seat]
        and rng.random() < 0.3
    ):
        return RecontraAction()

    if standing is not None and standing.level >= rules.auction.max_bid:
        return PassAction()
    if rng.random() < 0.55:
        return PassAction()

    base = standing.level if standing is not None else rules.auction.min_bid - 1
    level = min(base + rng.randint(1, 5), rules.auction.max_bid)
    level = max(level, rules.auction.min_bid)
    contract_type = rng.choice(rules.contracts.types)
    capot = bool(standing and standing.capot) or rng.random() < 0.02
    action = BidAction(level=level, contract_type=contract_type, capot=capot)
    if is_legal(state, action, rules):
        return action
    return PassAction()


def play_random_legal_deal(
    rules: RuleConfig,
    dealer: int,
    rng: random.Random,
    deal_id: int = 0,
    check_invariants: bool = True,
) -> Deal:
    """Deal, auction and play one full deal with a random-legal policy.

    Returns the `Deal` once it reaches `TERMINAL` or `ABORTED`. `check_invariants` runs
    `Deal.check_invariants()` after every single step — cheap, and it's the whole point of a
    random-legal-playout test.
    """
    hands = deal_hands(rng, rules)
    d = Deal(rules, dealer=dealer, hands=hands, deal_id=deal_id)
    if check_invariants:
        d.check_invariants()

    while d.phase == Phase.AUCTION:
        d.step(random_auction_action(d.auction_state, rules, rng))
        if check_invariants:
            d.check_invariants()

    while d.phase == Phase.PLAY:
        legal = d.legal_actions()
        assert legal, "legal_actions() returned empty in PLAY phase"
        d.step(rng.choice(legal))
        if check_invariants:
            d.check_invariants()

    return d


def play_until_terminal(
    rules: RuleConfig, dealer: int, rng: random.Random, deal_id: int = 0
) -> Deal:
    """Like `play_random_legal_deal`, but retries on ABORTED (4-pass redeal) with the same
    dealer, matching how `Match` expects the caller to drive redeals."""
    while True:
        d = play_random_legal_deal(rules, dealer, rng, deal_id=deal_id)
        if d.phase == Phase.TERMINAL:
            return d
