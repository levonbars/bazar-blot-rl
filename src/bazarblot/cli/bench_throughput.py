"""Throughput benchmark (M6): measures deals/second for two configurations — the pure engine
(`core/` alone) and the engine plus observation encoding (`InfoSet` construction, `legal_mask`,
and full `encode()`/`flatten()` at every decision point — what a real self-play loop actually
pays per step). Uses independent, uniformly-random-legal play throughout: no agent objects, no
DD solving, nothing but the engine and (in the second configuration) the environment layer — so
the measurement reflects the environment's own raw throughput ceiling, not any particular
policy's cost.

Run directly: `python -m bazarblot.cli.bench_throughput [n_deals]`
"""

from __future__ import annotations

import random
import sys
import time

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
from bazarblot.core.rules import RuleConfig, load_default
from bazarblot.env.actions import build_action_space, legal_mask
from bazarblot.env.infoset import info_set
from bazarblot.env.obs import encode, flatten
from bazarblot.env.tracked_deal import TrackedDeal

_DEFAULT_N_DEALS = 2000


def _cheap_random_auction_action(
    state: AuctionState, rules: RuleConfig, rng: random.Random
) -> AuctionAction:
    """A deliberately cheap random-legal auction policy, mirroring `tests/helpers.py`'s
    `random_auction_action` (kept as an independent copy rather than imported — `tests/` isn't a
    dependency of `src/`): constructs one plausible action and validates it once, rather than
    enumerating the full ~730-action space the way `env/actions.py::legal_mask` does for a real
    agent's mask. Using that exhaustive enumeration here would benchmark ITS cost, not the
    engine's own — the "engine-only" configuration is meant to isolate `core/` alone."""
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


def _play_one_deal_engine_only(
    rules: RuleConfig, dealer: int, rng: random.Random, deal_id: int
) -> Deal:
    """One full deal to `TERMINAL`, redealing transparently on a 4-pass abort (matching
    `Match`'s documented "same dealer, new shuffle" behavior) — engine-only: no `InfoSet`, no
    action-space machinery, no encoding, just `core/`."""
    while True:
        hands = deal_hands(rng, rules)
        deal = Deal(rules, dealer=dealer, hands=hands, deal_id=deal_id)
        while deal.phase == Phase.AUCTION:
            deal.step(_cheap_random_auction_action(deal.auction_state, rules, rng))
        while deal.phase == Phase.PLAY:
            legal = deal.legal_actions()
            deal.step(rng.choice(legal))
        if deal.phase == Phase.TERMINAL:
            return deal


def bench_engine_only(rules: RuleConfig, n_deals: int, seed: int = 0) -> float:
    """Deals/second for `core/` alone."""
    rng = random.Random(seed)
    start = time.perf_counter()
    for i in range(n_deals):
        _play_one_deal_engine_only(rules, dealer=i % 4, rng=rng, deal_id=i)
    elapsed = time.perf_counter() - start
    return n_deals / elapsed


def bench_engine_plus_encode(rules: RuleConfig, n_deals: int, seed: int = 0) -> float:
    """Deals/second including `InfoSet` construction, `legal_mask`, and full observation
    encoding at EVERY decision point — what a real self-play loop actually pays per step, not
    just the engine's own cost."""
    space = build_action_space(rules)
    rng = random.Random(seed)
    start = time.perf_counter()
    for i in range(n_deals):
        dealer = i % 4
        while True:
            hands = deal_hands(rng, rules)
            deal = Deal(rules, dealer=dealer, hands=hands, deal_id=i)
            tracked = TrackedDeal(deal)
            while deal.phase in (Phase.AUCTION, Phase.PLAY):
                seat = deal.to_act
                info = info_set(tracked, seat, match_score=(0, 0), deal_number=i)
                mask = legal_mask(info, space)
                flatten(encode(info))
                legal_idx = mask.nonzero()[0]
                action_idx = int(rng.choice(legal_idx))
                tracked.step(space.decode(action_idx))
            if deal.phase == Phase.TERMINAL:
                break
    elapsed = time.perf_counter() - start
    return n_deals / elapsed


def main(argv: list[str]) -> None:
    n_deals = int(argv[1]) if len(argv) > 1 else _DEFAULT_N_DEALS
    rules = load_default()

    engine_rate = bench_engine_only(rules, n_deals)
    print(f"engine-only:      {engine_rate:,.0f} deals/s  (n={n_deals})")

    # Encoding is far more expensive per decision than the engine alone (numpy array
    # construction per feature block, ~20 blocks) — a smaller n keeps this runnable in seconds
    # while still stabilizing the rate.
    encode_n = max(200, n_deals // 5)
    encode_rate = bench_engine_plus_encode(rules, encode_n)
    print(f"engine + encode:  {encode_rate:,.0f} deals/s  (n={encode_n})")


if __name__ == "__main__":
    main(sys.argv)
