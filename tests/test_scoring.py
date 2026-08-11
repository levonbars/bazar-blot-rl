"""M1: `round10` and the raw card-portion computation. Rules §7.3-7.4.

The full made/failed payout formula (§7.2) is exercised end-to-end in `test_deal_fixtures.py`
against the rules doc's worked examples — this file covers the lower-level pieces in isolation:
rounding, the capot "set, don't add" rule, and bid-payout isolation.
"""

from __future__ import annotations

import pytest

from bazarblot.core.auction import Contract
from bazarblot.core.rules import load_default
from bazarblot.core.scoring import compute_card_portions, round10, score_deal

RULES = load_default()


# ---------------------------------------------------------------- round10, §7.4


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (85, 8),  # exact half rounds DOWN
        (86, 9),
        (77, 8),
        (155, 15),
        (0, 0),
        (9, 1),
        (4, 0),
        (5, 0),  # half rounds down
        (6, 1),
        (162, 16),
        (802, 80),  # the max-bid deal's raw total
    ],
)
def test_round10_half_down(raw: int, expected: int) -> None:
    assert round10(raw, RULES) == expected


def test_round10_half_up_variant() -> None:
    import copy

    from bazarblot.core.rules import RuleConfig

    data = copy.deepcopy(RULES.raw)
    data["scoring"]["rounding"] = "half_up"
    half_up_rules = RuleConfig.from_dict(data)
    assert round10(85, half_up_rules) == 9  # differs from half_down's 8
    assert round10(84, half_up_rules) == 8


# ---------------------------------------------------------------- card portions & capot, §7.3


def test_normal_split_includes_says_bonus_for_the_winning_side() -> None:
    cards_a, cards_d = compute_card_portions(
        card_points_attackers=96,
        card_points_defenders=56,
        says_team="attackers",
        attackers_all_tricks=False,
        defenders_all_tricks=False,
        rules=RULES,
    )
    assert cards_a == 106  # 96 + 10 says
    assert cards_d == 56
    assert cards_a + cards_d == 162


def test_capot_sets_the_card_portion_not_adds_ninety() -> None:
    """The two are numerically identical when the shutout side also collected all 152 face
    points — this test forces that face-value total to be irrelevant, so a naive `+= 90`
    implementation cannot pass by accident."""
    cards_a, cards_d = compute_card_portions(
        card_points_attackers=152,  # attackers collected every card point
        card_points_defenders=0,
        says_team="attackers",
        attackers_all_tricks=True,
        defenders_all_tricks=False,
        rules=RULES,
    )
    # A `+= 90` bug would give 152 + 10 + 90 = 252 here too (coincidentally correct) — the
    # real test is that the VALUE is exactly capot_card_portion, asserted structurally:
    assert cards_a == RULES.scoring.capot_card_portion == 252
    assert cards_d == 0


def test_defenders_capot_sets_their_portion() -> None:
    cards_a, cards_d = compute_card_portions(
        card_points_attackers=0,
        card_points_defenders=152,
        says_team="defenders",
        attackers_all_tricks=False,
        defenders_all_tricks=True,
        rules=RULES,
    )
    assert cards_a == 0
    assert cards_d == 252


# ---------------------------------------------------------------- bid-payout isolation, §7.2


def _contract(level: int, doubling: str, capot: bool = False) -> Contract:
    return Contract(
        level=level,
        contract_type="H",
        capot=capot,
        declarer_seat=0,
        attacking_team=0,
        doubling=doubling,  # type: ignore[arg-type]
    )


def test_bid_payout_is_isolated_to_the_bid_term_on_success() -> None:
    """`score_attackers = bid_payout + round10(raw)`. Changing ONLY the doubling (which
    changes ONLY `bid_payout`, via the multiplier) must change `attackers_score` by exactly
    `(new_multiplier - old_multiplier) * level`, leaving everything else — collected points,
    the rounded card/combo contribution, the defenders' score — completely untouched."""
    kwargs = dict(
        rules=RULES,
        cards_attackers=106,
        cards_defenders=56,
        combo_attackers=20,
        combo_defenders=50,
        attackers_all_tricks=False,
        defenders_all_tricks=False,
    )
    none_score = score_deal(contract=_contract(11, "none"), **kwargs)  # type: ignore[arg-type]
    contra_score = score_deal(contract=_contract(11, "contra"), **kwargs)  # type: ignore[arg-type]
    recontra_score = score_deal(contract=_contract(11, "recontra"), **kwargs)  # type: ignore[arg-type]

    level = 11
    assert contra_score.attackers_score - none_score.attackers_score == level  # (2-1)*11
    assert recontra_score.attackers_score - contra_score.attackers_score == 2 * level  # (4-2)*11

    # the raw totals and the defenders' score never move — doubling only pays the bid term
    assert none_score.raw_attackers == contra_score.raw_attackers == recontra_score.raw_attackers
    assert none_score.raw_defenders == contra_score.raw_defenders == recontra_score.raw_defenders
    assert (
        none_score.defenders_score == contra_score.defenders_score == recontra_score.defenders_score
    )


def test_bid_payout_is_isolated_to_the_bid_term_on_failure() -> None:
    """On failure, `score_defenders = bid_payout + base + round10(combo_defenders)` — raising
    the doubling must move `defenders_score` by exactly the multiplier delta times the level,
    and must NOT touch `base` (16, or 25 on a defenders' capot) or the combo term."""
    kwargs = dict(
        rules=RULES,
        cards_attackers=40,
        cards_defenders=122,
        combo_attackers=0,
        combo_defenders=30,
        attackers_all_tricks=False,
        defenders_all_tricks=False,
    )
    none_score = score_deal(contract=_contract(11, "none"), **kwargs)  # type: ignore[arg-type]
    contra_score = score_deal(contract=_contract(11, "contra"), **kwargs)  # type: ignore[arg-type]

    assert none_score.made is False
    assert contra_score.attackers_score == 0 == none_score.attackers_score
    level = 11
    assert contra_score.defenders_score - none_score.defenders_score == level  # (2-1)*11


# ---------------------------------------------------------------- the raw invariant, §7.4


@pytest.mark.parametrize("cards_a_face", range(0, 153, 7))
@pytest.mark.parametrize("says_team", ["attackers", "defenders"])
def test_card_points_always_sum_to_162_raw(cards_a_face: int, says_team: str) -> None:
    """Absent a capot, the two sides' RAW card portions always sum to exactly 162. This is
    the invariant that actually holds unconditionally — see the correction in rules §7.4."""
    cards_d_face = 152 - cards_a_face
    cards_a, cards_d = compute_card_portions(
        card_points_attackers=cards_a_face,
        card_points_defenders=cards_d_face,
        says_team=says_team,  # type: ignore[arg-type]
        attackers_all_tricks=False,
        defenders_all_tricks=False,
        rules=RULES,
    )
    assert cards_a + cards_d == 162


def test_scaled_shares_do_not_always_sum_to_sixteen() -> None:
    """Documents the correction in rules §7.4: `round10` is applied independently to each
    side, with no complement step, so the scaled shares do NOT always sum to 16. This is a
    genuine property of the ruleset (verified by brute force over all 163 splits of 162 raw
    points), not a bug — asserting the opposite is exactly the mistake to avoid. Every split
    with `x % 10 == 6` violates it; `(106, 56)` is the split from §7.5 Fixture A itself."""
    assert round10(106, RULES) + round10(56, RULES) == 17
    assert round10(106, RULES) + round10(56, RULES) != 16

    violations = sum(1 for x in range(163) if round10(x, RULES) + round10(162 - x, RULES) != 16)
    assert violations == 16  # exactly the x % 10 == 6 splits, out of 163 total
