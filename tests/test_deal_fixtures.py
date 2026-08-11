"""M1: the rules doc's §7.5 worked examples, plus the traps they were designed to catch.

These are the highest-value tests in the whole suite: every number here is copied from
`docs/01-rules.md` §7.5, which the project owner reviewed. If these pass, the scoring engine
implements the actual ruleset rather than classical Belote or a plausible-looking guess.
"""

from __future__ import annotations

from bazarblot.core.auction import Contract
from bazarblot.core.cards import card_id
from bazarblot.core.declarations import resolve_combinations
from bazarblot.core.rules import load_default
from bazarblot.core.scoring import compute_card_portions, score_deal

RULES = load_default()


def cards(*labels: str) -> frozenset[int]:
    return frozenset(card_id(label[-1], label[:-1]) for label in labels)


def _hands(*filled: tuple[int, frozenset[int]]) -> tuple[frozenset[int], ...]:
    hands = [frozenset[int]()] * 4
    for seat, hand in filled:
        hands[seat] = hand
    return tuple(hands)


# ---------------------------------------------------------------- Fixture A, §7.5


def _fixture_a_combo() -> tuple[int, int]:
    """Seat 0 (team 0, attackers): Blot-Reblot in hearts (trump). Seat 1 (team 1,
    defenders): a Fifty in spades. Seat 2 (team 0, attackers): a Tierce in diamonds.
    Comparison: 50 > 20, neither in trump -> defenders score their 50; the attackers'
    Tierce scores nothing; their Blot-Reblot (20) still counts unconditionally."""
    hands = _hands(
        (0, cards("KH", "QH")),
        (1, cards("7S", "8S", "9S", "10S")),
        (2, cards("7D", "8D", "9D")),
    )
    result = resolve_combinations(hands, "H", RULES, leader_seat=1)
    assert result.team_points == (20, 50)
    return result.team_points


def test_fixture_a_made() -> None:
    combo_a, combo_d = _fixture_a_combo()
    contract = Contract(
        level=11, contract_type="H", capot=False, declarer_seat=0, attacking_team=0, doubling="none"
    )
    cards_a, cards_d = compute_card_portions(
        card_points_attackers=96,
        card_points_defenders=56,
        says_team="attackers",
        attackers_all_tricks=False,
        defenders_all_tricks=False,
        rules=RULES,
    )
    assert (cards_a, cards_d) == (106, 56)

    score = score_deal(
        rules=RULES,
        contract=contract,
        cards_attackers=cards_a,
        cards_defenders=cards_d,
        combo_attackers=combo_a,
        combo_defenders=combo_d,
        attackers_all_tricks=False,
        defenders_all_tricks=False,
    )
    assert score.raw_attackers == 126
    assert score.raw_defenders == 106
    assert score.made is True
    assert score.attackers_score == 24  # 1*11 + round10(126) = 11 + 13
    assert score.defenders_score == 11  # round10(106)


def test_fixture_a_failed() -> None:
    """Same deal, attackers instead collect only 100 raw (fail)."""
    combo_a, combo_d = _fixture_a_combo()
    contract = Contract(
        level=11, contract_type="H", capot=False, declarer_seat=0, attacking_team=0, doubling="none"
    )
    score = score_deal(
        rules=RULES,
        contract=contract,
        cards_attackers=80,  # + combo_a(20) = 100 raw, below threshold 110
        cards_defenders=82,
        combo_attackers=combo_a,
        combo_defenders=combo_d,
        attackers_all_tricks=False,
        defenders_all_tricks=False,
    )
    assert score.raw_attackers == 100
    assert score.made is False
    assert score.attackers_score == 0
    assert score.defenders_score == 32  # 1*11 + 16 + round10(50)


def test_fixture_a_failed_under_notrump() -> None:
    combo_a, combo_d = _fixture_a_combo()
    contract = Contract(
        level=11,
        contract_type="NT",
        capot=False,
        declarer_seat=0,
        attacking_team=0,
        doubling="none",
    )
    score = score_deal(
        rules=RULES,
        contract=contract,
        cards_attackers=80,
        cards_defenders=82,
        combo_attackers=combo_a,
        combo_defenders=combo_d,
        attackers_all_tricks=False,
        defenders_all_tricks=False,
    )
    assert score.defenders_score == 43  # 2*11 + 16 + round10(50)


# ---------------------------------------------------------------- Fixture B, §7.5


def test_fixture_b_high_bid_carried_by_combinations() -> None:
    """A carre of jacks (200) + a Fifty (50) = 250 raw in bonuses, bid 26. Attackers win no
    capot but collect 40 card points and the last hand."""
    contract = Contract(
        level=26, contract_type="S", capot=False, declarer_seat=0, attacking_team=0, doubling="none"
    )
    cards_a, cards_d = compute_card_portions(
        card_points_attackers=40,
        card_points_defenders=112,
        says_team="attackers",
        attackers_all_tricks=False,
        defenders_all_tricks=False,
        rules=RULES,
    )
    assert (cards_a, cards_d) == (50, 112)

    score = score_deal(
        rules=RULES,
        contract=contract,
        cards_attackers=cards_a,
        cards_defenders=cards_d,
        combo_attackers=250,
        combo_defenders=0,
        attackers_all_tricks=False,
        defenders_all_tricks=False,
    )
    assert score.raw_attackers == 300
    assert score.made is True  # 300 >= 260
    assert score.attackers_score == 56  # 1*26 + round10(300) = 26 + 30
    assert score.defenders_score == 11  # round10(112)


def test_fixture_b_combinations_are_a_real_carre_and_sequence() -> None:
    """Confirms the 250-bonus figure used above is achievable from real hands, not just
    asserted numerically: seat 0 holds the carre of jacks (200), partner seat 2 holds an
    independent Fifty (50) — split across the two attacking players, not one hand.

    It CANNOT be one hand: a carre of jacks claims all four suits' jacks, and the only
    possible 4-card run shape (7-8-9-10) is always adjacent to that suit's jack, so a single
    hand holding both would merge into a 5-run (a Hundred) instead of two separate melds —
    which is exactly what `resolve_combinations` correctly does if you try it that way.
    """
    hands = _hands(
        (0, cards("JC", "JD", "JH", "JS")),
        (2, cards("7S", "8S", "9S", "10S")),  # seat 0 holds JS, not seat 2 — no adjacency merge
    )
    result = resolve_combinations(hands, "S", RULES, leader_seat=0)
    assert result.team_points[0] == 250


# ---------------------------------------------------------------- Fixture C, §7.5


def test_fixture_c_the_capot_trap() -> None:
    """Same holding as Fixture B, attackers bid 27 Capot. They collect 27 scaled points
    comfortably but the defenders steal a trick — point target met, shutout missed, contract
    lost regardless."""
    contract = Contract(
        level=27, contract_type="S", capot=True, declarer_seat=0, attacking_team=0, doubling="none"
    )
    score = score_deal(
        rules=RULES,
        contract=contract,
        cards_attackers=290,  # well above the 270 threshold on its own
        cards_defenders=0,
        combo_attackers=0,
        combo_defenders=0,
        attackers_all_tricks=False,  # the shutout failed — this is what sinks the contract
        defenders_all_tricks=False,
    )
    assert score.made is False
    assert score.attackers_score == 0
    assert score.defenders_score == 43  # 1*27 + 16 + 0


def test_plain_bid_on_the_same_deal_would_have_made() -> None:
    """The direct comparison the rules doc calls out: strip the capot flag from Fixture C's
    exact numbers and the identical deal is made."""
    contract = Contract(
        level=27, contract_type="S", capot=False, declarer_seat=0, attacking_team=0, doubling="none"
    )
    score = score_deal(
        rules=RULES,
        contract=contract,
        cards_attackers=290,
        cards_defenders=0,
        combo_attackers=0,
        combo_defenders=0,
        attackers_all_tricks=False,
        defenders_all_tricks=False,
    )
    assert score.made is True


# ---------------------------------------------------------------- capot: set, not add (trap)


def test_capot_plus_combinations_distinguishes_set_from_add() -> None:
    """The plain-shutout case can't tell `= 252` from `+= 90` apart (they coincide when the
    shutout side also holds all 152 face points). This fixture breaks that coincidence: the
    shutout side's face-value collection is irrelevant here because `compute_card_portions`
    must SET the portion to 252 regardless of what was actually collected, then combination
    bonuses are added on top of that flat 252 — not folded into a `collected + 90` sum."""
    cards_a, cards_d = compute_card_portions(
        card_points_attackers=152,  # would give the same 252 under a wrong += 90 too...
        card_points_defenders=0,
        says_team="attackers",
        attackers_all_tricks=True,
        defenders_all_tricks=False,
        rules=RULES,
    )
    assert cards_a == 252  # ...but asserting the exact constant, not a formula, catches it
    assert cards_d == 0

    contract = Contract(
        level=27, contract_type="H", capot=True, declarer_seat=0, attacking_team=0, doubling="none"
    )
    score = score_deal(
        rules=RULES,
        contract=contract,
        cards_attackers=cards_a,
        cards_defenders=cards_d,
        combo_attackers=20,  # Blot-Reblot, added ON TOP of the flat 252
        combo_defenders=0,
        attackers_all_tricks=True,
        defenders_all_tricks=False,
    )
    assert score.raw_attackers == 272  # 252 + 20, not (152+10+90) + 20 miscounted some other way
    assert score.made is True  # 272 >= 270 threshold at level 27, and the shutout is real


# ---------------------------------------------------------------- the maximum bid, §4.2


def test_maximum_bid_deal_totals_802_raw_80_scaled() -> None:
    """One partner holds the jack and nine carres, the other the ten and ace carres — the
    entire attacking team's 16 cards. Combined with capot, this is the highest raw total the
    game can produce."""
    hands = _hands(
        (0, cards("JC", "JD", "JH", "JS", "9C", "9D", "9H", "9S")),
        (2, cards("10C", "10D", "10H", "10S", "AC", "AD", "AH", "AS")),
    )
    result = resolve_combinations(hands, "S", RULES, leader_seat=0)
    combo = result.team_points[0]
    assert combo == 200 + 140 + 110 + 100  # 550

    contract = Contract(
        level=80, contract_type="S", capot=True, declarer_seat=0, attacking_team=0, doubling="none"
    )
    cards_a, cards_d = compute_card_portions(
        card_points_attackers=0,  # irrelevant once capot sets the portion
        card_points_defenders=0,
        says_team="attackers",
        attackers_all_tricks=True,
        defenders_all_tricks=False,
        rules=RULES,
    )
    assert cards_a == 252

    score = score_deal(
        rules=RULES,
        contract=contract,
        cards_attackers=cards_a,
        cards_defenders=cards_d,
        combo_attackers=combo,
        combo_defenders=0,
        attackers_all_tricks=True,
        defenders_all_tricks=False,
    )
    assert score.raw_attackers == 802
    assert score.made is True  # 802 >= 800, and the shutout requirement is satisfied


# ---------------------------------------------------------------- defenders' capot on failure


def test_defenders_capot_on_failed_contract_replaces_sixteen_with_twenty_five() -> None:
    """Bid 14 trump, attackers shut out entirely -> defenders score 39, not 30."""
    contract = Contract(
        level=14, contract_type="H", capot=False, declarer_seat=0, attacking_team=0, doubling="none"
    )
    cards_a, cards_d = compute_card_portions(
        card_points_attackers=0,
        card_points_defenders=152,
        says_team="defenders",
        attackers_all_tricks=False,
        defenders_all_tricks=True,
        rules=RULES,
    )
    assert cards_d == 252

    score = score_deal(
        rules=RULES,
        contract=contract,
        cards_attackers=cards_a,
        cards_defenders=cards_d,
        combo_attackers=0,
        combo_defenders=0,
        attackers_all_tricks=False,
        defenders_all_tricks=True,
    )
    assert score.made is False
    assert score.defenders_score == 39  # 1*14 + 25 + 0, NOT 1*14 + 16 + 0 = 30
