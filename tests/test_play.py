"""M1: trick play legality and resolution. Rules §5."""

from __future__ import annotations

import pytest

from bazarblot.core.cards import build_tables, card_id
from bazarblot.core.play import current_winner, legal_moves
from bazarblot.core.rules import RuleConfig, load_default

RULES = load_default()


def cards(*labels: str) -> frozenset[int]:
    return frozenset(card_id(label[-1], label[:-1]) for label in labels)


def c(label: str) -> int:
    return card_id(label[-1], label[:-1])


# ---------------------------------------------------------------- leading


def test_leader_may_play_anything() -> None:
    tables = build_tables("H", RULES)
    hand = cards("7C", "KH", "AS")
    assert set(legal_moves(hand, [], 0, "H", RULES, tables)) == hand


# ---------------------------------------------------------------- follow suit (plain suit led)


def test_must_follow_suit_when_holding_it() -> None:
    tables = build_tables("H", RULES)  # hearts trump, clubs led (plain)
    hand = cards("7C", "9C", "KD")
    trick = [(3, c("AC"))]  # seat 3 led a club
    assert set(legal_moves(hand, trick, 0, "H", RULES, tables)) == cards("7C", "9C")


def test_no_beat_requirement_on_plain_suits() -> None:
    """Unlike trump, following a plain suit never requires beating the best card so far."""
    tables = build_tables("H", RULES)
    hand = cards("7C", "9C")
    trick = [(3, c("AC"))]  # already unbeatable in clubs
    assert set(legal_moves(hand, trick, 0, "H", RULES, tables)) == cards("7C", "9C")


def test_void_in_plain_led_suit_under_nt_is_free_discard() -> None:
    tables = build_tables("NT", RULES)
    hand = cards("7C", "9D", "KH")  # void in spades
    trick = [(3, c("AS"))]
    assert set(legal_moves(hand, trick, 0, "NT", RULES, tables)) == hand


# ---------------------------------------------------------------- trump led


def test_trump_led_must_beat_if_able() -> None:
    tables = build_tables("H", RULES)  # trump order: J 9 A 10 K Q 8 7
    hand = cards("9H", "7H", "KC")  # holds two trumps: 9 (strong) and 7 (weak)
    trick = [(3, c("QH"))]  # Q of trump led
    legal = set(legal_moves(hand, trick, 0, "H", RULES, tables))
    assert legal == cards("9H")  # only the beating trump; 7H does not beat Q


def test_trump_led_cannot_beat_must_still_follow_with_any_trump() -> None:
    tables = build_tables("H", RULES)
    hand = cards("7H", "KC")  # only a weak trump
    trick = [(3, c("JH"))]  # the strongest trump already led
    legal = set(legal_moves(hand, trick, 0, "H", RULES, tables))
    assert legal == cards("7H")  # can't beat, but must still play the trump held


def test_void_in_trump_led_suit_is_free_discard() -> None:
    tables = build_tables("H", RULES)
    hand = cards("7C", "KD")  # no hearts at all
    trick = [(3, c("JH"))]
    assert set(legal_moves(hand, trick, 0, "H", RULES, tables)) == hand


# ---------------------------------------------------------------- ruff obligation (void, plain led)


def test_must_ruff_when_opponent_winning_and_void() -> None:
    """Seats 0/2 vs 1/3. Seat 0 is void in the led suit; seat 3 (opponent) is winning."""
    tables = build_tables("H", RULES)
    hand = cards("7H", "9H", "KD")  # void in clubs, holds trump
    trick = [(3, c("AC"))]  # opponent (seat 3) winning with an unbeaten ace of clubs
    legal = set(legal_moves(hand, trick, 0, "H", RULES, tables))
    assert legal == cards("7H", "9H")  # must trump; no trump played yet so either beats


def test_must_overtrump_when_a_trump_was_already_played() -> None:
    tables = build_tables("H", RULES)
    hand = cards("9H", "7H", "KD")
    # seat 2 (partner) led a club; seat 3 (opponent) ruffed with Q of trump and is now winning
    trick = [(2, c("AC")), (3, c("QH"))]
    legal = set(legal_moves(hand, trick, 0, "H", RULES, tables))
    assert legal == cards("9H")  # must beat the Q; only 9H does (J 9 A 10 K Q 8 7)


def test_cannot_overtrump_falls_back_to_any_trump_under_trump_obligation() -> None:
    tables = build_tables("H", RULES)
    hand = cards("7H", "KD")  # only a weak trump, cannot beat J
    trick = [(2, c("AC")), (3, c("JH"))]  # opponent (3) trumped with the highest trump
    legal = set(legal_moves(hand, trick, 0, "H", RULES, tables))
    assert legal == cards("7H")  # can't beat, but still forced to trump (under-trump / "pisser")


def test_partner_winning_exemption_free_discard_including_trump_optional() -> None:
    tables = build_tables("H", RULES)
    hand = cards("7H", "9H", "KD")
    trick = [(2, c("AC"))]  # partner (seat 2) winning
    legal = set(legal_moves(hand, trick, 0, "H", RULES, tables))
    assert legal == hand  # free discard; may still trump voluntarily if desired


def test_void_in_both_trump_and_led_suit_is_free_discard() -> None:
    tables = build_tables("H", RULES)
    hand = cards("7C", "KD")  # void in spades (led) and no trump
    trick = [(3, c("AS"))]
    assert set(legal_moves(hand, trick, 0, "H", RULES, tables)) == hand


# ---------------------------------------------------------------- trick resolution


def test_current_winner_highest_trump_wins_over_led_suit_ace() -> None:
    tables = build_tables("H", RULES)
    trick = [(0, c("AC")), (1, c("7H")), (2, c("KC")), (3, c("QC"))]
    assert current_winner(trick, tables) == 1  # any trump beats any non-trump


def test_current_winner_highest_of_led_suit_when_no_trump_played() -> None:
    tables = build_tables("H", RULES)
    trick = [(0, c("9C")), (1, c("AC")), (2, c("KC")), (3, c("7D"))]
    assert current_winner(trick, tables) == 1  # ace of clubs, clubs led, no trump involved


def test_current_winner_under_nt_is_just_highest_of_led_suit() -> None:
    tables = build_tables("NT", RULES)
    trick = [(0, c("9H")), (1, c("KH")), (2, c("AH")), (3, c("7C"))]
    assert current_winner(trick, tables) == 2


def test_current_winner_mid_trick() -> None:
    tables = build_tables("H", RULES)
    trick = [(0, c("9C")), (1, c("AC"))]
    assert current_winner(trick, tables) == 1


def test_current_winner_rejects_empty_trick() -> None:
    from bazarblot.core.play import PlayError

    tables = build_tables("H", RULES)
    with pytest.raises(PlayError):
        current_winner([], tables)


# ---------------------------------------------------------------- config-flag generality


def _with_flag(flag: str, value: bool) -> RuleConfig:
    import copy

    data = copy.deepcopy(RULES.raw)
    data["play"][flag] = value
    return RuleConfig.from_dict(data)


def test_must_beat_when_trump_led_false_allows_any_trump() -> None:
    lenient = _with_flag("must_beat_when_trump_led", False)
    tables = build_tables("H", lenient)
    hand = cards("9H", "7H", "KC")
    trick = [(3, c("QH"))]
    legal = set(legal_moves(hand, trick, 0, "H", lenient, tables))
    assert legal == cards("9H", "7H")  # both trumps legal, not just the beating one


def test_must_overtrump_false_allows_any_trump_when_ruffing() -> None:
    lenient = _with_flag("must_overtrump", False)
    tables = build_tables("H", lenient)
    hand = cards("9H", "7H", "KD")
    trick = [(2, c("AC")), (3, c("QH"))]  # opponent trumped with Q
    legal = set(legal_moves(hand, trick, 0, "H", lenient, tables))
    assert legal == cards("9H", "7H")  # no obligation to beat the Q specifically
