"""A minimal bot for the UI to play against. NOT the M4 "club player" heuristic agent.

This exists only so `watch`/`play` mode has something more interesting than uniform-random to
look at while M4 doesn't exist yet. It should be deleted or replaced once `agents/heuristic.py`
lands — do not extend this file into M4's scope.

Bidding: rough hand-strength estimate per candidate contract, bid modestly if strong, else pass.
Never contras, never recontras, never bids capot — all judged too risky for a bot this simple.
Play: leads arbitrarily; when following, wins as cheaply as possible if it can (checked by
asking the engine's own `current_winner`, not reimplemented), else discards the weakest legal
card. No signalling, no trump-drawing strategy, no combination awareness.
"""

from __future__ import annotations

import random

from bazarblot.core.auction import AuctionState, BidAction, PassAction
from bazarblot.core.cards import ContractType, build_tables
from bazarblot.core.deal import Action, Deal, Phase
from bazarblot.core.play import PlayCardAction, current_winner
from bazarblot.core.rules import RuleConfig


def _hand_strength(hand: frozenset[int], contract_type: ContractType, rules: RuleConfig) -> int:
    """Sum of card point values under `contract_type` — a crude but serviceable proxy for
    "how many raw points can this hand plausibly take"."""
    tables = build_tables(contract_type, rules)
    return sum(tables.points[c] for c in hand)


def _best_contract(hand: frozenset[int], rules: RuleConfig) -> tuple[ContractType, int]:
    best_type: ContractType = "NT"
    best_value = -1
    for contract_type in rules.contracts.types:
        value = _hand_strength(hand, contract_type, rules)
        if value > best_value:
            best_type, best_value = contract_type, value
    return best_type, best_value


def bot_auction_action(state: AuctionState, hand: frozenset[int], rules: RuleConfig) -> Action:
    if state.doubling == "contra":
        return PassAction()  # never recontras — always accepts

    contract_type, strength = _best_contract(hand, rules)
    # `strength` is raw points (0..~90 for one 8-card hand); bid levels live on the scaled
    # (raw/10) axis. Scale up by ~1.4x on the assumption partner contributes comparably, so
    # only genuinely strong single hands clear the level-8 minimum on their own — roughly
    # what a real opening-bid judgment should do.
    target_level = round(strength * 1.4 / 10)

    standing = state.standing_bid
    if standing is None:
        if target_level >= rules.auction.min_bid:
            return BidAction(level=rules.auction.min_bid, contract_type=contract_type)
        return PassAction()

    if standing.capot:
        return PassAction()  # never chases a capot bid
    if target_level > standing.level and target_level <= rules.auction.max_bid:
        return BidAction(level=standing.level + 1, contract_type=contract_type)
    return PassAction()


def bot_play_action(d: Deal, seat: int, rng: random.Random) -> PlayCardAction:
    assert d.contract is not None and d.tables is not None
    legal = d.legal_actions()
    assert legal

    if not d.current_trick:
        return rng.choice(legal)  # type: ignore[return-value]

    tables = d.tables

    def wins_if_played(card: int) -> bool:
        # Delegate to the engine's own trick-resolution logic rather than reimplementing
        # it — strength values are only comparable within the trump/plain-suit group they
        # came from, and a bespoke comparison here would get that wrong.
        hypothetical = [*d.current_trick, (seat, card)]
        return current_winner(hypothetical, tables) == seat

    winning = [a for a in legal if isinstance(a, PlayCardAction) and wins_if_played(a.card)]
    pool = winning if winning else legal
    return min(pool, key=lambda a: tables.strength[a.card])  # type: ignore[union-attr,return-value]


def bot_action(d: Deal, seat: int, rng: random.Random) -> Action:
    if d.phase == Phase.AUCTION:
        return bot_auction_action(d.auction_state, d.hands[seat], d.rules)
    if d.phase == Phase.PLAY:
        return bot_play_action(d, seat, rng)
    raise ValueError(f"bot asked to act outside AUCTION/PLAY (phase={d.phase})")
