"""M1: multi-deal match bookkeeping. Rules §8."""

from __future__ import annotations

import copy

import pytest

from bazarblot.core.auction import Contract
from bazarblot.core.deal import DealResult
from bazarblot.core.match import Match
from bazarblot.core.rules import RuleConfig, load_default

RULES = load_default()


def _result(attackers_team: int, score_a: int, score_d: int) -> DealResult:
    contract = Contract(
        level=8,
        contract_type="H",
        capot=False,
        declarer_seat=attackers_team,
        attacking_team=attackers_team,
        doubling="none",
    )
    return DealResult(
        contract=contract,
        attackers_team=attackers_team,
        cards_attackers=80,
        cards_defenders=82,
        combo_attackers=0,
        combo_defenders=0,
        raw_attackers=80,
        raw_defenders=82,
        made=True,
        score_attackers=score_a,
        score_defenders=score_d,
        attackers_took_all_tricks=False,
        defenders_took_all_tricks=False,
        winning_meld=None,
    )


def test_deal_result_team_scores_property() -> None:
    r = _result(attackers_team=1, score_a=24, score_d=11)
    assert r.team_scores == (11, 24)  # (team0, team1) regardless of attacker role

    r2 = _result(attackers_team=0, score_a=24, score_d=11)
    assert r2.team_scores == (24, 11)


def test_dealer_rotates_clockwise_each_deal() -> None:
    m = Match(rules=RULES, dealer=0)
    m.apply_deal_result(_result(0, 5, 3))
    assert m.dealer == 1
    m.apply_deal_result(_result(1, 5, 3))
    assert m.dealer == 2


def test_scores_accumulate_by_team_not_by_attacker_role() -> None:
    m = Match(rules=RULES, dealer=0)
    m.apply_deal_result(_result(0, 20, 5))  # team 0 attacks: team0 +20, team1 +5
    m.apply_deal_result(_result(1, 5, 20))  # team 1 attacks: team1 +5, team0 +20
    assert m.scores == [20 + 20, 5 + 5]  # team0=40, team1=10


def test_deal_number_and_history_tracked() -> None:
    m = Match(rules=RULES, dealer=0)
    r1 = _result(0, 5, 3)
    m.apply_deal_result(r1)
    assert m.deal_number == 1
    assert m.history == [r1]


def test_match_ends_when_a_team_crosses_target() -> None:
    data = copy.deepcopy(RULES.raw)
    data["match"]["target"] = 20
    tiny_rules = RuleConfig.from_dict(data)
    m = Match(rules=tiny_rules, dealer=0)
    assert not m.finished
    m.apply_deal_result(_result(0, 25, 0))
    assert m.finished
    assert m.winner == 0


def test_match_continues_below_target() -> None:
    data = copy.deepcopy(RULES.raw)
    data["match"]["target"] = 101
    small_rules = RuleConfig.from_dict(data)
    m = Match(rules=small_rules, dealer=0)
    m.apply_deal_result(_result(0, 24, 11))
    assert not m.finished
    assert m.winner is None


def test_double_cross_higher_score_wins_team1() -> None:
    """Both teams cross the target in the same deal -> higher total score wins."""
    data = copy.deepcopy(RULES.raw)
    data["match"]["target"] = 10
    tiny_rules = RuleConfig.from_dict(data)
    m = Match(rules=tiny_rules, dealer=0, scores=[8, 9])
    m.apply_deal_result(_result(0, 5, 5))  # both teams end up >= 10: 13 vs 14
    assert m.finished
    assert m.winner == 1  # 14 > 13


def test_double_cross_higher_score_wins_team0() -> None:
    data = copy.deepcopy(RULES.raw)
    data["match"]["target"] = 10
    tiny_rules = RuleConfig.from_dict(data)
    m = Match(rules=tiny_rules, dealer=0, scores=[9, 8])
    m.apply_deal_result(_result(0, 5, 5))  # both teams end up >= 10: 14 vs 13
    assert m.finished
    assert m.winner == 0  # 14 > 13


def test_double_cross_exact_tie_leaves_match_undecided() -> None:
    """No source specifies this case; the documented default is to continue rather than
    guess a winner."""
    data = copy.deepcopy(RULES.raw)
    data["match"]["target"] = 10
    tiny_rules = RuleConfig.from_dict(data)
    m = Match(rules=tiny_rules, dealer=0, scores=[9, 9])
    m.apply_deal_result(_result(0, 5, 5))  # both land on 14 exactly
    assert not m.finished
    assert m.winner is None


def test_apply_deal_result_after_finished_raises() -> None:
    data = copy.deepcopy(RULES.raw)
    data["match"]["target"] = 5
    tiny_rules = RuleConfig.from_dict(data)
    m = Match(rules=tiny_rules, dealer=0)
    m.apply_deal_result(_result(0, 10, 0))
    assert m.finished
    with pytest.raises(ValueError, match="already finished"):
        m.apply_deal_result(_result(0, 5, 5))


def test_redeal_does_not_advance_dealer_because_it_is_simply_not_reported() -> None:
    """There is no explicit 'abort' method — a redeal is just a Deal the caller never
    reports, so `match.dealer` is naturally unchanged for the next attempt."""
    m = Match(rules=RULES, dealer=2)
    assert m.dealer == 2  # caller would construct the next Deal with dealer=2 again
