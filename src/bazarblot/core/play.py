"""Trick play. Rules §5.

`legal_moves` and `current_winner` are pure functions over a `ContractTables` (cards.py) plus
the partial trick played so far — no mutable state lives here. `Deal` (deal.py) owns the actual
turn-by-turn state machine and calls into this module.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from bazarblot.core.cards import TEAM_OF, ContractTables, suit_of

if TYPE_CHECKING:
    from bazarblot.core.rules import RuleConfig


class PlayError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class PlayCardAction:
    card: int


@dataclass(frozen=True, slots=True)
class Trick:
    leader: int
    plays: tuple[tuple[int, int], ...]  # (seat, card), in play order, length 4
    winner: int
    points: int  # raw face-value points in this trick — NOT including the last-hand bonus


def current_winner(trick_so_far: Sequence[tuple[int, int]], tables: ContractTables) -> int:
    """The seat currently holding the best position in a trick — partial or complete.

    Any trump beats any non-trump card regardless of suit (checked first); otherwise the
    highest card of the led suit wins. This degenerates correctly to "highest of led suit"
    under `NT`, since `tables.is_trump` is all-`False` there.
    """
    if not trick_so_far:
        raise PlayError("current_winner called on an empty trick")
    led_suit = suit_of(trick_so_far[0][1])
    best_seat, best_card = trick_so_far[0]
    any_trump = tables.is_trump[best_card]
    for seat, card in trick_so_far[1:]:
        if tables.is_trump[card]:
            if not any_trump or tables.strength[card] > tables.strength[best_card]:
                best_seat, best_card, any_trump = seat, card, True
        elif (
            not any_trump
            and suit_of(card) == led_suit
            and tables.strength[card] > tables.strength[best_card]
        ):
            best_seat, best_card = seat, card
        # else: a non-trump card off the led suit, or a non-trump card once trump has already
        # been played, can never win — nothing to update.
    return best_seat


def _best_trump_strength(
    trick_so_far: Sequence[tuple[int, int]], tables: ContractTables
) -> int | None:
    trumps = [c for _, c in trick_so_far if tables.is_trump[c]]
    if not trumps:
        return None
    return max(tables.strength[c] for c in trumps)


def _forced_trump_subset(
    trumps_in_hand: list[int], best_strength: int | None, tables: ContractTables
) -> list[int]:
    """Which of the player's trumps satisfy the "must beat if able" obligation.

    If no trump has been played yet in this trick, any trump beats it by definition (the
    `is_trump` check always wins over a non-trump card), so every trump in hand qualifies.
    Otherwise: trumps strictly stronger than the current best, or — if none beat it — every
    trump in hand (an unavoidable under-trump, "pisser").
    """
    if best_strength is None:
        return trumps_in_hand
    beating = [c for c in trumps_in_hand if tables.strength[c] > best_strength]
    return beating if beating else trumps_in_hand


def legal_moves(
    hand: frozenset[int],
    trick_so_far: Sequence[tuple[int, int]],
    seat: int,
    contract_type: str,
    rules: RuleConfig,
    tables: ContractTables,
) -> tuple[int, ...]:
    """The legal cards `seat` may play, given their hand and the trick in progress."""
    hand_list = sorted(hand)
    if not trick_so_far:
        return tuple(hand_list)  # leading: any card

    led_suit = suit_of(trick_so_far[0][1])
    is_nt = contract_type == "NT"

    if not is_nt and led_suit == contract_type:
        # Trump led (§5 rules 1+2): must follow with trump, and must beat if able.
        trumps_in_hand = [c for c in hand_list if tables.is_trump[c]]
        if trumps_in_hand:
            if not rules.play.must_beat_when_trump_led:
                return tuple(trumps_in_hand)
            best = _best_trump_strength(trick_so_far, tables)
            return tuple(_forced_trump_subset(trumps_in_hand, best, tables))
        return tuple(hand_list)  # void in the trump-led suit: free discard

    # Plain suit led (or any suit under NT).
    suited = [c for c in hand_list if suit_of(c) == led_suit]
    if suited:
        return tuple(suited)  # must follow suit, no "beat" requirement on plain suits

    if is_nt or not rules.play.must_follow_suit:
        return tuple(hand_list)  # NT: no trumping concept at all. Free discard.

    # Trump contract, void in a non-trump led suit — the ruff decision (§5 rules 3-5).
    winner_seat = current_winner(trick_so_far, tables)
    opponent_winning = TEAM_OF[winner_seat] != TEAM_OF[seat]
    trumps_in_hand = [c for c in hand_list if tables.is_trump[c]]

    if not trumps_in_hand:
        return tuple(hand_list)  # void in both: free discard

    must_ruff = (opponent_winning and rules.play.must_ruff_when_opponent_winning) or (
        not opponent_winning and rules.play.must_ruff_when_partner_winning
    )
    if not must_ruff:
        return tuple(hand_list)  # free discard; trumping is optional, unrestricted if chosen

    if not rules.play.must_overtrump:
        return tuple(trumps_in_hand)
    best = _best_trump_strength(trick_so_far, tables)
    return tuple(_forced_trump_subset(trumps_in_hand, best, tables))
