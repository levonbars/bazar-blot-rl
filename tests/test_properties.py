"""M1 property tests.

Two tiers, matching `pyproject.toml`'s `slow` marker:
- fast (default `pytest -q`): enough samples to catch a real bug quickly, every run.
- slow (`pytest -m slow`): the roadmap's stated "Done when" scale (10^6 playouts). Not run by
  default — it's a pre-release/nightly check, not something to wait on every commit.
"""

from __future__ import annotations

import random

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from bazarblot.core.cards import build_tables, suit_of
from bazarblot.core.deal import Phase
from bazarblot.core.play import legal_moves
from bazarblot.core.rules import load_default
from tests.helpers import play_random_legal_deal

RULES = load_default()
CONTRACT_TYPES = list(RULES.contracts.types)


# ---------------------------------------------------------------- legal_moves properties


@st.composite
def _hand_and_partial_trick(
    draw: st.DrawFn,
) -> tuple[int, frozenset[int], list[tuple[int, int]], str]:
    """A random 8-card hand for `seat`, plus 0-3 cards already played by other seats — all
    drawn from a single shuffled deck so nothing overlaps, mirroring a real reachable state."""
    seat = draw(st.integers(0, 3))
    seed = draw(st.integers(0, 2**31 - 1))
    rng = random.Random(seed)
    deck = list(range(32))
    rng.shuffle(deck)
    hand = frozenset(deck[:8])
    remaining = deck[8:]
    k = draw(st.integers(0, 3))
    other_seats = [s for s in range(4) if s != seat]
    rng.shuffle(other_seats)
    trick = list(zip(other_seats[:k], remaining[:k], strict=True))
    contract_type = draw(st.sampled_from(CONTRACT_TYPES))
    return seat, hand, trick, contract_type


@given(_hand_and_partial_trick())
@settings(max_examples=1000)
def test_legal_moves_never_empty(
    data: tuple[int, frozenset[int], list[tuple[int, int]], str],
) -> None:
    seat, hand, trick, contract_type = data
    tables = build_tables(contract_type, RULES)  # type: ignore[arg-type]
    legal = legal_moves(hand, trick, seat, contract_type, RULES, tables)
    assert len(legal) > 0
    assert set(legal) <= hand


@given(_hand_and_partial_trick())
@settings(max_examples=1000)
def test_holding_led_suit_never_produces_a_legal_move_outside_it(
    data: tuple[int, frozenset[int], list[tuple[int, int]], str],
) -> None:
    seat, hand, trick, contract_type = data
    if not trick:
        return  # leading: no led-suit constraint to check
    tables = build_tables(contract_type, RULES)  # type: ignore[arg-type]
    legal = legal_moves(hand, trick, seat, contract_type, RULES, tables)
    led_suit = suit_of(trick[0][1])
    suited = {c for c in hand if suit_of(c) == led_suit}
    if suited:
        assert set(legal) <= suited


# ---------------------------------------------------------------- full random-legal deals


def test_random_legal_deals_fast() -> None:
    """Fast tier: card conservation, invariants, and non-empty legality across many full
    auction+play deals, every test run."""
    for seed in range(300):
        rng = random.Random(seed)
        d = play_random_legal_deal(RULES, dealer=seed % 4, rng=rng, deal_id=seed)
        assert d.phase in (Phase.TERMINAL, Phase.ABORTED)
        if d.phase == Phase.TERMINAL:
            assert d.result is not None
            assert d.result.cards_attackers + d.result.cards_defenders in (162, 252)


@pytest.mark.slow
def test_random_legal_deals_slow() -> None:
    """Slow tier, closer to the roadmap's 10^6 target. 20k deals (not 1M) keeps this in the
    minutes range in pure Python; raise it for an actual pre-release run."""
    for seed in range(20_000):
        rng = random.Random(seed + 10**9)  # disjoint seed space from the fast tier
        d = play_random_legal_deal(RULES, dealer=seed % 4, rng=rng, deal_id=seed)
        assert d.phase in (Phase.TERMINAL, Phase.ABORTED)


# ---------------------------------------------------------------- replay determinism


def test_replay_determinism() -> None:
    """`(seed, dealer)` fully determines a deal, including every auction/play decision made
    by the (also seed-derived) random-legal policy — replaying it must be bit-identical."""
    for seed in range(200):
        rng1 = random.Random(seed)
        d1 = play_random_legal_deal(RULES, dealer=seed % 4, rng=rng1, deal_id=seed)

        rng2 = random.Random(seed)
        d2 = play_random_legal_deal(RULES, dealer=seed % 4, rng=rng2, deal_id=seed)

        assert d1.phase == d2.phase
        assert d1.original_hands == d2.original_hands
        if d1.phase == Phase.TERMINAL:
            assert d1.result == d2.result
            assert [t.plays for t in d1.tricks] == [t.plays for t in d2.tricks]
