"""M1: `Deal` construction validation, phase-error paths, and `check_invariants`.

The invariant-violation tests deliberately corrupt a `Deal`'s private state — that's the only
way to prove `check_invariants` actually catches corruption rather than being decorative.
"""

from __future__ import annotations

import random

import pytest

from bazarblot.core.auction import BidAction, PassAction
from bazarblot.core.cards import full_deck
from bazarblot.core.deal import Deal, DealError, DealFinishedError, IllegalActionError, Phase
from bazarblot.core.dealing import deal_hands
from bazarblot.core.play import PlayCardAction
from bazarblot.core.rules import load_default

RULES = load_default()


def _fresh_hands() -> tuple[frozenset[int], ...]:
    return deal_hands(random.Random(0), RULES)


# ---------------------------------------------------------------- construction validation


def test_rejects_wrong_number_of_hands() -> None:
    hands = _fresh_hands()
    with pytest.raises(DealError, match="expected 4 hands"):
        Deal(RULES, dealer=0, hands=hands[:3])


def test_rejects_wrong_hand_size() -> None:
    hands = list(_fresh_hands())
    hands[0] = frozenset(list(hands[0])[:7])  # only 7 cards
    with pytest.raises(DealError, match="expected 8"):
        Deal(RULES, dealer=0, hands=tuple(hands))


def test_rejects_duplicate_card_across_hands() -> None:
    hands = list(_fresh_hands())
    dupe_card = next(iter(hands[1]))
    hands[0] = (hands[0] - {next(iter(hands[0]))}) | {dupe_card}  # steal a card from hand 1
    with pytest.raises(DealError, match="more than one hand"):
        Deal(RULES, dealer=0, hands=tuple(hands))


def test_rejects_hands_with_wrong_card_ids() -> None:
    """4 correctly-sized, pairwise-disjoint hands necessarily cover all 32 cards by
    construction — there is no separate "doesn't cover the deck" failure mode. What's
    actually rejectable is overlap, already covered by `test_rejects_duplicate_card_across_hands`;
    this just exercises it via out-of-range card ids instead of a real deck's cards."""
    hands = (
        frozenset(range(0, 8)),
        frozenset(range(8, 16)),
        frozenset(range(16, 24)),
        frozenset(range(23, 31)),  # overlaps hand 2 at 23
    )
    with pytest.raises(DealError, match="more than one hand"):
        Deal(RULES, dealer=0, hands=hands)


# ---------------------------------------------------------------- phase-gated errors


def test_to_act_during_auction_delegates_to_auction_state() -> None:
    d = Deal(RULES, dealer=0, hands=_fresh_hands())
    assert d.phase == Phase.AUCTION
    assert d.to_act == d.auction_state.to_act == 1  # dealer's left


def test_to_act_raises_when_finished() -> None:
    d = Deal(RULES, dealer=0, hands=_fresh_hands())
    for _ in range(4):
        d.step(PassAction())
    assert d.phase == Phase.ABORTED
    with pytest.raises(DealFinishedError):
        _ = d.to_act


def test_legal_actions_raises_outside_play_phase() -> None:
    d = Deal(RULES, dealer=0, hands=_fresh_hands())
    assert d.phase == Phase.AUCTION
    with pytest.raises(DealFinishedError, match="only supported in PLAY phase"):
        d.legal_actions()


def test_step_wrong_action_type_in_auction_phase() -> None:
    d = Deal(RULES, dealer=0, hands=_fresh_hands())
    with pytest.raises(IllegalActionError, match="expected an auction action"):
        d.step(PlayCardAction(card=0))


def test_step_wrong_action_type_in_play_phase() -> None:
    d = Deal(RULES, dealer=0, hands=_fresh_hands())
    d.step(PassAction())
    d.step(BidAction(level=8, contract_type="H"))
    d.step(PassAction())
    d.step(PassAction())
    d.step(PassAction())
    assert d.phase == Phase.PLAY
    with pytest.raises(IllegalActionError, match="expected a PlayCardAction"):
        d.step(PassAction())


def test_step_after_terminal_raises() -> None:
    d = Deal(RULES, dealer=0, hands=_fresh_hands())
    rng = random.Random(0)
    d.step(PassAction())
    d.step(BidAction(level=8, contract_type="H"))
    d.step(PassAction())
    d.step(PassAction())
    d.step(PassAction())
    while d.phase == Phase.PLAY:
        d.step(rng.choice(d.legal_actions()))
    assert d.phase == Phase.TERMINAL
    with pytest.raises(DealFinishedError):
        d.step(PassAction())


def test_playing_a_card_not_in_hand_raises() -> None:
    d = Deal(RULES, dealer=0, hands=_fresh_hands())
    d.step(PassAction())
    d.step(BidAction(level=8, contract_type="H"))
    d.step(PassAction())
    d.step(PassAction())
    d.step(PassAction())
    assert d.phase == Phase.PLAY
    hand = d.hands[d.to_act]
    not_held = next(c for c in full_deck() if c not in hand)
    with pytest.raises(IllegalActionError, match="does not hold"):
        d.step(PlayCardAction(card=not_held))


def test_playing_a_held_but_illegal_card_raises() -> None:
    d = Deal(RULES, dealer=0, hands=_fresh_hands())
    d.step(PassAction())
    d.step(BidAction(level=8, contract_type="H"))
    d.step(PassAction())
    d.step(PassAction())
    d.step(PassAction())
    seat = d.to_act
    hand = d.hands[seat]
    d.step(PlayCardAction(card=next(iter(hand))))  # lead any card

    next_seat = d.to_act
    next_hand = d.hands[next_seat]
    legal = set(d.legal_actions())
    illegal_cards = next_hand - {a.card for a in legal}  # type: ignore[attr-defined]
    if illegal_cards:
        with pytest.raises(IllegalActionError, match="is not legal"):
            d.step(PlayCardAction(card=next(iter(illegal_cards))))


# ---------------------------------------------------------------- corrupted-state detection


def test_check_invariants_catches_card_conservation_violation() -> None:
    d = Deal(RULES, dealer=0, hands=_fresh_hands())
    d.hands[0] = d.hands[0] - {next(iter(d.hands[0]))}  # silently drop a card
    with pytest.raises(DealError, match="card conservation violated"):
        d.check_invariants()


def test_check_invariants_catches_duplicate_card() -> None:
    """Swap (not just add) so the total count stays 32 — otherwise the card-conservation
    check fires first and this branch is never reached."""
    d = Deal(RULES, dealer=0, hands=_fresh_hands())
    stolen = next(iter(d.hands[1]))
    displaced = next(iter(d.hands[0]))
    d.hands[0] = (d.hands[0] - {displaced}) | {stolen}
    with pytest.raises(DealError, match="more than once"):
        d.check_invariants()


def test_check_invariants_catches_oversized_current_trick() -> None:
    d = Deal(RULES, dealer=0, hands=_fresh_hands())
    d.step(PassAction())
    d.step(BidAction(level=8, contract_type="H"))
    d.step(PassAction())
    d.step(PassAction())
    d.step(PassAction())
    assert d.phase == Phase.PLAY
    # Manually stuff 4 plays into current_trick without going through the normal trick-
    # completion path, so the invariant check has something real to catch.
    fake_cards = (
        list(d.hands[0])[:1] + list(d.hands[1])[:1] + list(d.hands[2])[:1] + list(d.hands[3])[:1]
    )
    d.current_trick = list(zip(range(4), fake_cards, strict=True))
    for seat, card in d.current_trick:
        d.hands[seat] = d.hands[seat] - {card}
    with pytest.raises(DealError, match="current_trick has"):
        d.check_invariants()


def test_check_invariants_catches_terminal_without_result() -> None:
    d = Deal(RULES, dealer=0, hands=_fresh_hands())
    rng = random.Random(1)
    d.step(PassAction())
    d.step(BidAction(level=8, contract_type="H"))
    d.step(PassAction())
    d.step(PassAction())
    d.step(PassAction())
    while d.phase == Phase.PLAY:
        d.step(rng.choice(d.legal_actions()))
    assert d.phase == Phase.TERMINAL
    d.result = None  # corrupt it
    with pytest.raises(DealError, match="no result"):
        d.check_invariants()
