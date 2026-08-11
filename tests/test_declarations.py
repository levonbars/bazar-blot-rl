"""M1: combination detection and the §6.4 precedence ladder."""

from __future__ import annotations

import pytest

from bazarblot.core.cards import card_id
from bazarblot.core.declarations import (
    beats,
    detect_hand_melds,
    resolve_combinations,
)
from bazarblot.core.rules import load_default

RULES = load_default()


def _empty_hands(*filled: tuple[int, frozenset[int]]) -> tuple[frozenset[int], ...]:
    """Build a 4-hand tuple where only the given seats have cards; the rest are empty.

    Only valid for unit-testing `detect_hand_melds`/`resolve_combinations` directly (these
    don't require full 8-card hands) — never for constructing a `Deal`, which enforces
    `cards_per_player`.
    """
    hands = [frozenset[int]()] * 4
    for seat, hand in filled:
        hands[seat] = hand
    return tuple(hands)


def cards(*labels: str) -> frozenset[int]:
    return frozenset(card_id(label[-1], label[:-1]) for label in labels)


# ---------------------------------------------------------------- sequence detection


def test_sequence_is_natural_order_not_trump_order() -> None:
    """9-10-J are scattered in trump order (positions 1,3,0) but consecutive in natural
    order — this is exactly the bug Blot Star's own "Hundred" illustration (7-8-9-10-J)
    warns against."""
    hand = cards("9H", "10H", "JH")
    melds = detect_hand_melds(hand, seat=0, contract_type="H", rules=RULES)
    sequences = [m for m in melds if m.kind == "sequence"]
    assert len(sequences) == 1
    m = sequences[0]
    assert m.length == 3
    assert m.value == 20  # tierce
    assert m.rank == "J"  # top card
    assert m.is_trump is True


def test_sequence_values_by_length() -> None:
    assert detect_hand_melds(cards("7C", "8C", "9C"), 0, "H", RULES)[0].value == 20
    assert detect_hand_melds(cards("7C", "8C", "9C", "10C"), 0, "H", RULES)[0].value == 50
    assert detect_hand_melds(cards("7C", "8C", "9C", "10C", "JC"), 0, "H", RULES)[0].value == 100


@pytest.mark.parametrize("length", [6, 7, 8])
def test_long_runs_score_same_as_hundred(length: int) -> None:
    ranks = ["7", "8", "9", "10", "J", "Q", "K", "A"][:length]
    hand = cards(*(f"{r}C" for r in ranks))
    melds = detect_hand_melds(hand, 0, "H", RULES)
    sequences = [m for m in melds if m.kind == "sequence"]
    assert len(sequences) == 1
    assert sequences[0].length == length
    assert sequences[0].value == 100


def test_two_disjoint_runs_in_the_same_suit_are_separate_melds() -> None:
    """7-8-9 and Q-K-A in the same suit, with a gap — two real Tierces, not one meld."""
    hand = cards("7C", "8C", "9C", "QC", "KC", "AC")
    melds = detect_hand_melds(hand, 0, "H", RULES)
    sequences = [m for m in melds if m.kind == "sequence"]
    assert len(sequences) == 2
    assert {m.rank for m in sequences} == {"9", "A"}  # top cards of each run
    assert all(m.value == 20 for m in sequences)


def test_pairs_and_short_runs_do_not_count() -> None:
    hand = cards("7C", "8C", "QC", "KC")  # two disjoint 2-runs, neither is a combination
    melds = detect_hand_melds(hand, 0, "H", RULES)
    assert melds == []


# ---------------------------------------------------------------- carre detection & values


def test_carre_trump_values_are_blotstar_specific() -> None:
    """Trump 9=140 and A=110 — classical Belote says 150/100. These are the values most
    likely to be silently "corrected" by an implementer working from memory."""
    for rank, expected in (("J", 200), ("9", 140), ("A", 110), ("10", 100), ("K", 100), ("Q", 100)):
        hand = cards(f"{rank}C", f"{rank}D", f"{rank}H", f"{rank}S")
        melds = detect_hand_melds(hand, 0, "S", RULES)  # any trump suit; carre spans all 4
        carres = [m for m in melds if m.kind == "carre"]
        assert len(carres) == 1
        assert carres[0].value == expected, (
            f"carre of {rank} under trump: {carres[0].value} != {expected}"
        )


def test_carre_notrump_values_are_blotstar_specific() -> None:
    for rank, expected in (("A", 190), ("J", 100), ("10", 100), ("K", 100), ("Q", 100), ("9", 0)):
        hand = cards(f"{rank}C", f"{rank}D", f"{rank}H", f"{rank}S")
        melds = detect_hand_melds(hand, 0, "NT", RULES)
        carres = [m for m in melds if m.kind == "carre"]
        assert len(carres) == 1
        assert carres[0].value == expected, (
            f"carre of {rank} under NT: {carres[0].value} != {expected}"
        )


def test_carre_of_sevens_and_eights_worth_nothing() -> None:
    for rank in ("7", "8"):
        hand = cards(f"{rank}C", f"{rank}D", f"{rank}H", f"{rank}S")
        melds = detect_hand_melds(hand, 0, "S", RULES)
        assert melds[0].value == 0


# ---------------------------------------------------------------- blot-reblot


def test_blot_reblot_only_under_trump() -> None:
    hand = cards("KH", "QH")
    trump_melds = detect_hand_melds(hand, 0, "H", RULES)
    assert any(m.kind == "blot_reblot" and m.value == 20 for m in trump_melds)

    nt_melds = detect_hand_melds(hand, 0, "NT", RULES)
    assert not any(m.kind == "blot_reblot" for m in nt_melds)


def test_blot_reblot_requires_the_trump_suit_specifically() -> None:
    """K+Q of a SIDE suit under a trump contract is not Blot-Reblot."""
    hand = cards("KC", "QC")  # clubs, contract trump is hearts
    melds = detect_hand_melds(hand, 0, "H", RULES)
    assert not any(m.kind == "blot_reblot" for m in melds)


def test_blot_reblot_survives_losing_the_comparison() -> None:
    """§6.4: Blot-Reblot is exempt from the whole comparison — it is credited even when its
    holder's team loses the sequence/carre comparison outright."""
    # seat0 (team0): Blot-Reblot in trump hearts, no other combination.
    # seat1 (team1): a Fifty (50) in clubs — strictly bigger, wins the comparison.
    hands = _empty_hands(
        (0, cards("KH", "QH")),
        (1, cards("7C", "8C", "9C", "10C")),
    )
    result = resolve_combinations(hands, "H", RULES, leader_seat=0)
    assert result.winning_team == 1
    assert result.team_points[1] == 50
    assert result.team_points[0] == 20  # Blot-Reblot still credited to team 0


def test_blot_reblot_does_not_win_the_comparison_for_its_team() -> None:
    """The inferred half of "does not clash with any others": Blot-Reblot's 20 does not
    enter the ranking, so it cannot carry a smaller sequence to victory on its team's behalf."""
    # seat0 (team0): Blot-Reblot (20, exempt) + a Tierce (20) — team0's comparable total is
    # just the Tierce, worth 20, same as team1's Tierce.
    # seat1 (team1): a Fifty (50) — bigger than either Tierce.
    hands = _empty_hands(
        (0, cards("KH", "QH", "7C", "8C", "9C")),
        (1, cards("7D", "8D", "9D", "10D")),
    )
    result = resolve_combinations(hands, "H", RULES, leader_seat=0)
    assert result.winning_team == 1
    assert result.winning_meld is not None and result.winning_meld.kind == "sequence"
    assert result.team_points[0] == 20  # only the Blot-Reblot; the losing Tierce scores nothing
    assert result.team_points[1] == 50


# ---------------------------------------------------------------- precedence ladder §6.4


def test_carre_beats_sequence_by_category_not_value() -> None:
    """A carre of queens (100) beats a Hundred (100) — same value, different category."""
    hands = _empty_hands(
        (0, cards("7C", "8C", "9C", "10C", "JC")),  # Hundred, value 100
        (1, cards("QC", "QD", "QH", "QS")),  # carre of queens, value 100
    )
    result = resolve_combinations(hands, "H", RULES, leader_seat=0)
    assert result.winning_team == 1
    assert result.winning_meld is not None and result.winning_meld.kind == "carre"


def test_carre_vs_carre_by_value() -> None:
    hands = _empty_hands(
        (0, cards("10C", "10D", "10H", "10S")),  # value 100
        (1, cards("JC", "JD", "JH", "JS")),  # value 200
    )
    result = resolve_combinations(hands, "S", RULES, leader_seat=0)
    assert result.winning_team == 1


@pytest.mark.parametrize(
    ("leader_seat", "expected_winner"),
    [(0, 0), (1, 1)],  # elder hand flips depending on who leads trick 1
)
def test_carre_vs_carre_tie_goes_to_elder_hand(leader_seat: int, expected_winner: int) -> None:
    """Both carres are worth 100 (tens vs kings) — an exact value tie, broken by elder hand."""
    hands = _empty_hands(
        (0, cards("10C", "10D", "10H", "10S")),
        (1, cards("KC", "KD", "KH", "KS")),
    )
    result = resolve_combinations(hands, "S", RULES, leader_seat=leader_seat)
    assert result.winning_team == expected_winner


def test_trump_breaks_a_top_card_tie() -> None:
    """Same length, same top card (J) — the trump one wins."""
    hands = _empty_hands(
        (0, cards("9C", "10C", "JC")),  # non-trump Tierce to the Jack
        (1, cards("9H", "10H", "JH")),  # TRUMP Tierce to the Jack
    )
    result = resolve_combinations(hands, "H", RULES, leader_seat=0)
    assert result.winning_team == 1


def test_top_card_outranks_trump() -> None:
    """The owner's worked example: A-K-Q non-trump beats J-10-9 trump, despite not being
    trump — top card is checked before trump in the ladder."""
    hands = _empty_hands(
        (0, cards("QC", "KC", "AC")),  # non-trump Tierce to the Ace
        (1, cards("9H", "10H", "JH")),  # TRUMP Tierce to the Jack
    )
    result = resolve_combinations(hands, "H", RULES, leader_seat=0)
    assert result.winning_team == 0  # the non-trump Ace-topped sequence wins


def test_full_three_way_precedence_chain() -> None:
    """A-K-Q non-trump > J-10-9 trump > J-10-9 non-trump, pairwise via `beats`."""
    a_k_q_nontrump = detect_hand_melds(cards("QC", "KC", "AC"), 0, "H", RULES)[0]
    j109_trump = detect_hand_melds(cards("9H", "10H", "JH"), 1, "H", RULES)[0]
    j109_nontrump = detect_hand_melds(cards("9D", "10D", "JD"), 2, "H", RULES)[0]

    assert beats(a_k_q_nontrump, j109_trump, RULES, leader_seat=0)
    assert beats(j109_trump, j109_nontrump, RULES, leader_seat=0)
    assert beats(a_k_q_nontrump, j109_nontrump, RULES, leader_seat=0)


def test_sequence_tie_neither_trump_goes_to_elder_hand() -> None:
    hands = _empty_hands(
        (0, cards("QC", "KC", "AC")),  # Tierce to the Ace, non-trump
        (1, cards("QD", "KD", "AD")),  # Tierce to the Ace, a different non-trump suit
    )
    result = resolve_combinations(hands, "H", RULES, leader_seat=0)
    assert result.winning_team == 0  # seat 0 is elder relative to leader_seat=0

    result2 = resolve_combinations(hands, "H", RULES, leader_seat=1)
    assert result2.winning_team == 1  # now seat 1 is elder


# ---------------------------------------------------------------- team-level outcome


def test_only_winning_team_scores_its_combinations() -> None:
    """The losing team's combinations — even multiple, even from both its players — score
    nothing at all."""
    hands = _empty_hands(
        (0, cards("7C", "8C", "9C")),  # team0: Tierce, 20
        (2, cards("7D", "8D", "9D")),  # team0's partner: another Tierce, 20
        (1, cards("7H", "8H", "9H", "10H")),  # team1: a Fifty, 50 — wins
    )
    result = resolve_combinations(hands, "S", RULES, leader_seat=0)
    assert result.winning_team == 1
    assert result.team_points == (0, 50)


def test_no_combinations_at_all_scores_zero_both_sides() -> None:
    hands = _empty_hands()
    result = resolve_combinations(hands, "H", RULES, leader_seat=0)
    assert result.winning_team is None
    assert result.team_points == (0, 0)


def test_only_winning_team_scores_flag_off_lets_both_sides_score() -> None:
    """Config generality: with the flag off, each team scores its own combinations
    regardless of who has the best one — not the ruleset default, but a real code path."""
    import copy

    from bazarblot.core.rules import RuleConfig

    data = copy.deepcopy(RULES.raw)
    data["combinations"]["only_winning_team_scores"] = False
    lenient = RuleConfig.from_dict(data)

    hands = _empty_hands(
        (0, cards("7C", "8C", "9C")),  # team0: Tierce, 20 (the losing side)
        (1, cards("7H", "8H", "9H", "10H")),  # team1: a Fifty, 50 (wins)
    )
    result = resolve_combinations(hands, "S", lenient, leader_seat=0)
    assert result.winning_team == 1
    assert result.team_points == (20, 50)  # both sides scored, not just the winner
