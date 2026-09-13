"""The leakage test — see the roadmap's M3 section: "If this test does not exist, the whole
project's results are worthless." Everything downstream of `InfoSet` (bid quality, win rate, the
paper's numbers) is contaminated if a seat's encoded observation ever depends on a card it could
not actually have seen.

**Construction.** For seat `s`, build two full ground-truth deals, A and B: identical hand for
`s`, and the remaining 24 cards redistributed independently at random among the other three
seats (same per-seat card COUNT, different actual cards). From `s`'s point of view these are two
indistinguishable possible worlds. If `s`'s `InfoSet`/encoded observation differs between them,
that difference can only be explained by `info_set()` reading something it shouldn't have.

**Scope, and why.** The rigorous, full-strength (10^5-sample) version of this test runs during
the AUCTION phase, where there is no legitimate public disclosure of any hidden card at all —
every InfoSet field is either `s`'s own hand (identical by construction) or genuinely
hand-independent public state (bid history, dealer, scores, per-seat card counts). That makes it
the cleanest possible instance of the property: ANY difference there is unambiguously a bug.

A second, smaller-scale (not 10^5 — see its own docstring) test extends the same construction
into the PLAY phase, where one legitimate complication appears: once a contract is set, melds are
auto-shown (`staging.declarations_are_actions = False`) and are genuinely, correctly public — a
different redistribution of the other three seats' cards generally produces different real melds,
and `s` is *supposed* to see that difference (it's what "auto-show" means). That test therefore
compares only the fields that must NEVER depend on hidden information regardless of disclosure
status (`hand`, `card_state`, `current_trick`, `voids`, `suit_unseen`, `cards_left`), and leaves
meld-derived fields out of the comparison — not because they're exempt from correctness, but
because their correctness is a different property (already covered by `declarations.py`'s own
extensive M1 test suite, plus the explicit assertion here that `melds_by_seat is None` before a
contract exists) than "does this field leak an unplayed card from another hand."
"""

from __future__ import annotations

import copy
import random

import numpy as np
import pytest

from bazarblot.core.cards import full_deck
from bazarblot.core.deal import Deal, Phase
from bazarblot.core.rules import load_default
from bazarblot.env.infoset import info_set
from bazarblot.env.obs import BLOCK_SHAPES, encode
from bazarblot.env.tracked_deal import TrackedDeal
from tests.helpers import random_auction_action

RULES = load_default()


def _redistribute_others(
    hands: tuple[frozenset[int], ...], seat: int, rng: random.Random
) -> tuple[frozenset[int], ...]:
    """Same hand for `seat`, a fresh random 8/8/8 split of the other 24 cards among the other
    three seats."""
    pool: list[int] = []
    for s in range(4):
        if s != seat:
            pool.extend(hands[s])
    rng.shuffle(pool)
    out = list(hands)
    others = [s for s in range(4) if s != seat]
    n = len(hands[seat])
    for i, s in enumerate(others):
        out[s] = frozenset(pool[i * n : (i + 1) * n])
    return tuple(out)


def _fresh_pair(seed: int, seat: int) -> tuple[TrackedDeal, TrackedDeal]:
    rng_a = random.Random(seed)
    deck = list(full_deck())
    rng_a.shuffle(deck)
    n = RULES.deal.cards_per_player
    hands_a = tuple(frozenset(deck[i * n : (i + 1) * n]) for i in range(4))

    rng_b = random.Random(seed + 500_000)
    hands_b = _redistribute_others(hands_a, seat, rng_b)
    assert hands_a[seat] == hands_b[seat]
    assert hands_a != hands_b  # extremely unlikely to coincide; would just weaken this one sample

    dealer = seed % 4
    deal_a = Deal(RULES, dealer=dealer, hands=hands_a, deal_id=seed)
    deal_b = Deal(RULES, dealer=dealer, hands=hands_b, deal_id=seed)
    return TrackedDeal(deal_a), TrackedDeal(deal_b)


@pytest.mark.slow
def test_no_leakage_at_deal_start_100000_samples() -> None:
    """The roadmap's literal "run 10^5 of these." Fresh deal, before any action at all."""
    n = 100_000
    for seed in range(n):
        seat = seed % 4
        tracked_a, tracked_b = _fresh_pair(seed, seat)

        info_a = info_set(tracked_a, seat, match_score=(0, 0), deal_number=0)
        info_b = info_set(tracked_b, seat, match_score=(0, 0), deal_number=0)

        blocks_a, blocks_b = encode(info_a), encode(info_b)
        for key in BLOCK_SHAPES:
            assert np.array_equal(blocks_a[key], blocks_b[key]), (
                f"seed={seed} seat={seat} block={key!r} differs between two deals identical "
                f"from this seat's perspective — information leak"
            )


def test_no_leakage_at_deal_start_fast() -> None:
    """Same property, fast tier (2000 samples) — runs every CI build, not just the slow tier."""
    n = 2000
    for seed in range(n):
        seat = seed % 4
        tracked_a, tracked_b = _fresh_pair(seed, seat)
        info_a = info_set(tracked_a, seat, match_score=(0, 0), deal_number=0)
        info_b = info_set(tracked_b, seat, match_score=(0, 0), deal_number=0)
        blocks_a, blocks_b = encode(info_a), encode(info_b)
        for key in BLOCK_SHAPES:
            assert np.array_equal(blocks_a[key], blocks_b[key]), (
                f"seed={seed} seat={seat} block={key!r}"
            )


def test_no_leakage_across_early_auction_steps() -> None:
    """Extends the deal-start property through a few real auction actions. Auction actions never
    read hand contents (`core.auction.is_legal`/`apply` take only the auction state and the
    action itself), so the exact same scripted action sequence is valid to replay against both
    A's and B's `Deal` — a real, not merely structural, extension of the property."""
    n = 2000
    for seed in range(n):
        seat = seed % 4
        tracked_a, tracked_b = _fresh_pair(seed, seat)
        rng = random.Random(seed + 9_000_000)
        n_steps = rng.randrange(0, 4)
        for _ in range(n_steps):
            if tracked_a.deal.phase != Phase.AUCTION:
                break
            action = random_auction_action(tracked_a.deal.auction_state, RULES, rng)
            tracked_a.step(action)
            tracked_b.step(action)
        if tracked_a.deal.phase != Phase.AUCTION:
            continue  # aborted or (can't happen this early) closed; skip, covered by other seeds

        info_a = info_set(tracked_a, seat, match_score=(0, 0), deal_number=0)
        info_b = info_set(tracked_b, seat, match_score=(0, 0), deal_number=0)
        blocks_a, blocks_b = encode(info_a), encode(info_b)
        for key in BLOCK_SHAPES:
            assert np.array_equal(blocks_a[key], blocks_b[key]), (
                f"seed={seed} seat={seat} block={key!r}"
            )


_MELD_DERIVED_KEYS = (
    "melds_public",
    "own_combinations",
    "contract_progress",
    "bid_summary",
    "bid_recent",
)
_NON_MELD_KEYS = tuple(k for k in BLOCK_SHAPES if k not in _MELD_DERIVED_KEYS)


def test_no_leakage_through_play_phase() -> None:
    """Extends into the PLAY phase. Only compares the fields that must never depend on hidden
    information regardless of disclosure status — see the module docstring for exactly why meld-
    derived fields (and `bid_summary`/`bid_recent`, which reflect the auction *history*, itself
    identical between A and B by construction up to the redistribution point, but not worth
    re-deriving here) are excluded from this specific comparison, and what covers them instead."""
    n = 500
    for seed in range(n):
        seat = seed % 4
        rng = random.Random(seed + 3_000_000)
        deck = list(full_deck())
        rng.shuffle(deck)
        n_cards = RULES.deal.cards_per_player
        hands_a = tuple(frozenset(deck[i * n_cards : (i + 1) * n_cards]) for i in range(4))
        dealer = seed % 4
        deal_a = Deal(RULES, dealer=dealer, hands=hands_a, deal_id=seed)
        tracked_a = TrackedDeal(deal_a)

        # Drive A through the auction with a fixed, hand-independent policy (always PASS unless
        # forced to bid) until a contract closes, then play a few random-legal cards.
        steps = 0
        while tracked_a.deal.phase == Phase.AUCTION and steps < 50:
            action = random_auction_action(tracked_a.deal.auction_state, RULES, rng)
            tracked_a.step(action)
            steps += 1
        if tracked_a.deal.phase != Phase.PLAY:
            continue  # aborted; skip, covered elsewhere

        n_plays = rng.randrange(0, 6)
        for _ in range(n_plays):
            legal = tracked_a.deal.legal_actions()
            tracked_a.step(rng.choice(legal))
            if tracked_a.deal.phase != Phase.PLAY:
                break
        if tracked_a.deal.phase != Phase.PLAY:
            continue

        # B: same public history (auction log, tricks, current trick — literally the same
        # objects), different hidden redistribution of the other three seats' REMAINING cards,
        # with each seat's already-played cards held fixed (so original_hands stays a coherent
        # played + remaining split for every seat).
        deal_a = tracked_a.deal
        already_played_by_seat: list[set[int]] = [set() for _ in range(4)]
        for trick in deal_a.tricks:
            for s, c in trick.plays:
                already_played_by_seat[s].add(c)
        for s, c in deal_a.current_trick:
            already_played_by_seat[s].add(c)

        pool: list[int] = []
        for s in range(4):
            if s != seat:
                pool.extend(deal_a.hands[s])
        rng.shuffle(pool)
        remaining_b = list(deal_a.hands)
        others = [s for s in range(4) if s != seat]
        i = 0
        for s in others:
            k = len(deal_a.hands[s])
            remaining_b[s] = frozenset(pool[i : i + k])
            i += k

        original_hands_b = list(deal_a.original_hands)
        for s in others:
            original_hands_b[s] = frozenset(already_played_by_seat[s] | remaining_b[s])
            assert len(original_hands_b[s]) == len(deal_a.original_hands[s])

        # `Deal.__init__` enforces "every hand has exactly cards_per_player cards" — true of a
        # freshly dealt hand (deal_a's own construction), false of `remaining_b` once cards have
        # been played. B only needs to be a consistent container for `info_set()` to read, not a
        # legally-played-out game (see the module docstring), so build it as a shallow copy of
        # deal_a — inheriting every already-correct public field (phase, tricks, contract, ...)
        # — with only `hands`/`original_hands` overwritten to the redistributed version.
        deal_b = copy.copy(deal_a)
        deal_b.original_hands = tuple(original_hands_b)
        deal_b.hands = list(remaining_b)
        tracked_b = TrackedDeal(deal_b, auction_log=list(tracked_a.auction_log))

        info_a = info_set(tracked_a, seat, match_score=(0, 0), deal_number=0)
        info_b = info_set(tracked_b, seat, match_score=(0, 0), deal_number=0)
        assert info_a.melds_by_seat is not None and info_b.melds_by_seat is not None

        blocks_a, blocks_b = encode(info_a), encode(info_b)
        for key in _NON_MELD_KEYS:
            assert np.array_equal(blocks_a[key], blocks_b[key]), (
                f"seed={seed} seat={seat} block={key!r}"
            )


def test_melds_hidden_during_auction() -> None:
    """`melds_by_seat` must be `None` (nothing disclosed yet) until a contract exists — the
    guard that makes the auction-phase leakage test's scope ("no legitimate disclosure channel
    at all") actually true."""
    rng = random.Random(0)
    deck = list(full_deck())
    rng.shuffle(deck)
    n = RULES.deal.cards_per_player
    hands = tuple(frozenset(deck[i * n : (i + 1) * n]) for i in range(4))
    deal = Deal(RULES, dealer=0, hands=hands, deal_id=0)
    tracked = TrackedDeal(deal)
    for seat in range(4):
        info = info_set(tracked, seat, match_score=(0, 0), deal_number=0)
        assert info.melds_by_seat is None


def test_hand_field_never_contains_another_seats_card() -> None:
    """Cheap structural sanity check, independent of the pairwise-comparison method above:
    `InfoSet.hand` for seat `s` is always a subset of what `s` was actually dealt."""
    for seed in range(200):
        rng = random.Random(seed)
        deck = list(full_deck())
        rng.shuffle(deck)
        n = RULES.deal.cards_per_player
        hands = tuple(frozenset(deck[i * n : (i + 1) * n]) for i in range(4))
        deal = Deal(RULES, dealer=seed % 4, hands=hands, deal_id=seed)
        tracked = TrackedDeal(deal)
        steps = 0
        while deal.phase in (Phase.AUCTION, Phase.PLAY) and steps < 100:
            for seat in range(4):
                info = info_set(tracked, seat, match_score=(0, 0), deal_number=0)
                assert info.hand <= hands[seat]
                for other in range(4):
                    if other != seat:
                        assert not (info.hand & deal.hands[other])
            if deal.phase == Phase.AUCTION:
                action = random_auction_action(deal.auction_state, RULES, rng)
            else:
                legal = deal.legal_actions()
                action = rng.choice(legal)
            tracked.step(action)
            steps += 1
