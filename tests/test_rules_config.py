"""M0 tests: the preset loads, its arithmetic is self-consistent, and its hash is stable.

These are cheap but they guard the two things M0 exists to establish — that `RuleConfig` is
the single source of truth, and that `rules_hash` can be trusted to detect rule drift.
"""

from __future__ import annotations

import copy
import subprocess
import sys

import pytest

from bazarblot.core.rules import RuleConfig, RuleConfigError, load_default


@pytest.fixture(scope="module")
def cfg() -> RuleConfig:
    return load_default()


# ------------------------------------------------------------------ loading


def test_preset_loads(cfg: RuleConfig) -> None:
    assert cfg.preset == "blotstar"
    assert cfg.contracts.types == ("C", "D", "H", "S", "NT")


def test_no_all_trump_contract(cfg: RuleConfig) -> None:
    """Blot Star has exactly two contract families. All-trump is a different variant."""
    assert "AT" not in cfg.contracts.types
    assert len(cfg.contracts.types) == 5


# ------------------------------------------------------------------ arithmetic invariants


def test_suit_subtotals(cfg: RuleConfig) -> None:
    assert sum(cfg.contracts.points_trump_suit.values()) == 62
    assert sum(cfg.contracts.points_plain_suit.values()) == 30
    assert sum(cfg.contracts.points_notrump.values()) == 38


def test_both_contract_families_total_162(cfg: RuleConfig) -> None:
    """The check that pins the plain-suit ace at 11 rather than the no-trump 19."""
    trump = sum(cfg.contracts.points_trump_suit.values())
    plain = sum(cfg.contracts.points_plain_suit.values())
    notrump = sum(cfg.contracts.points_notrump.values())
    assert trump + 3 * plain + cfg.contracts.last_hand_bonus == 162
    assert 4 * notrump + cfg.contracts.last_hand_bonus == 162


def test_plain_suit_ace_is_eleven_not_nineteen(cfg: RuleConfig) -> None:
    assert cfg.contracts.points_plain_suit["A"] == 11
    assert cfg.contracts.points_notrump["A"] == 19


def test_rejects_notrump_ace_in_side_suits() -> None:
    """Guard the derivation: using the 19-point ace in side suits must fail loudly.

    Caught at the subtotal check (38 != 30) before it reaches the deal total, which is the
    more precise failure of the two.
    """
    data = copy.deepcopy(load_default().raw)
    data["contracts"]["points"]["plain_suit"]["A"] = 19
    with pytest.raises(RuleConfigError, match=r"plain suit totals 38"):
        RuleConfig.from_dict(data)


def test_rejects_wrong_deal_total() -> None:
    """Exercise the 162 check itself, which subtotals alone cannot reach."""
    data = copy.deepcopy(load_default().raw)
    data["contracts"]["last_hand_bonus"] = 20
    with pytest.raises(RuleConfigError, match=r"trump deal totals 172"):
        RuleConfig.from_dict(data)


# ------------------------------------------------------------------ Blot Star specific values


def test_carre_values_are_blotstar_not_classical(cfg: RuleConfig) -> None:
    """Classical Belote says 9s=150 and A=100. Blot Star says 140 and 110."""
    assert cfg.combinations.carre_trump["9"] == 140
    assert cfg.combinations.carre_trump["A"] == 110
    assert cfg.combinations.carre_trump["J"] == 200
    assert cfg.combinations.carre_notrump["A"] == 190
    assert cfg.combinations.carre_notrump["9"] == 0


def test_bid_multiplier_is_a_table_not_a_formula(cfg: RuleConfig) -> None:
    """No-trump is trump+1 at each doubling level — never trump x2."""
    assert cfg.scoring.multiplier("H", "none") == 1
    assert cfg.scoring.multiplier("NT", "none") == 2
    assert cfg.scoring.multiplier("H", "contra") == 2
    assert cfg.scoring.multiplier("NT", "contra") == 3
    assert cfg.scoring.multiplier("H", "recontra") == 4
    assert cfg.scoring.multiplier("NT", "recontra") == 5


def test_capot_sets_rather_than_adds(cfg: RuleConfig) -> None:
    assert cfg.scoring.capot_card_portion == 252
    assert cfg.scoring.capot_bid_requires_shutout is True


def test_defender_capot_replaces_the_sixteen(cfg: RuleConfig) -> None:
    assert cfg.scoring.failed_contract_base == 16
    assert cfg.scoring.failed_contract_base_on_defender_capot == 25


# ------------------------------------------------------------------ auction shape


def test_ladder_is_not_capped_at_sixteen(cfg: RuleConfig) -> None:
    """Combinations count toward fulfilment, so bids run far above the deal's 16."""
    assert cfg.auction.min_bid == 8
    assert cfg.auction.max_bid == 80  # capot 25 + carres 20 + 14 + 11 + 10
    assert cfg.n_bid_levels == 73


def test_seniority_is_purely_numeric(cfg: RuleConfig) -> None:
    assert cfg.auction.nt_outranks_suit_at_equal_level is False
    assert cfg.auction.capot_outranks_at_equal_level is False


def test_capot_is_sticky_and_passes_are_not_binding(cfg: RuleConfig) -> None:
    assert cfg.auction.capot_sticky_upward is True
    assert cfg.auction.pass_is_binding is False


def test_bid_action_count(cfg: RuleConfig) -> None:
    assert cfg.n_bid_actions == 73 * 5 * 2 == 730


# ------------------------------------------------------------------ combinations


def test_top_card_outranks_trump(cfg: RuleConfig) -> None:
    """A-K-Q non-trump > J-10-9 trump > J-10-9 non-trump."""
    assert cfg.combinations.sequence_precedence == ("length", "top_card", "trump", "elder_hand")


def test_rejects_trump_before_top_card() -> None:
    data = copy.deepcopy(load_default().raw)
    data["combinations"]["sequence_precedence"] = ["length", "trump", "top_card", "elder_hand"]
    with pytest.raises(RuleConfigError, match="top card outranks trump"):
        RuleConfig.from_dict(data)


def test_questioning_never_discloses_the_suit(cfg: RuleConfig) -> None:
    disclosed = cfg.combinations.question_discloses["sequence"]
    assert "suit" not in disclosed
    assert set(disclosed) == {"top_card", "is_trump"}


def test_rejects_suit_disclosure() -> None:
    data = copy.deepcopy(load_default().raw)
    data["combinations"]["question_discloses"]["sequence"] = ["suit"]
    with pytest.raises(RuleConfigError, match="must not disclose the suit"):
        RuleConfig.from_dict(data)


def test_bluffing_is_legal(cfg: RuleConfig) -> None:
    assert cfg.combinations.announce_requires_holding is False
    assert cfg.combinations.answers_may_be_false is True


def test_announce_classes_are_classes_not_values(cfg: RuleConfig) -> None:
    """You announce 4x, not '140' — the value stays hidden until questioned."""
    assert cfg.combinations.announce_classes == ("tierce", "fifty", "hundred", "4x")


# ------------------------------------------------------------------ rules_hash


def test_rules_hash_is_stable_within_process(cfg: RuleConfig) -> None:
    assert cfg.rules_hash == load_default().rules_hash
    assert len(cfg.rules_hash) == 12


def test_rules_hash_is_stable_across_processes(cfg: RuleConfig) -> None:
    """PYTHONHASHSEED must not leak into the fingerprint."""
    code = "from bazarblot.core.rules import load_default; print(load_default().rules_hash)"
    seen = {
        subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        ).stdout.strip()
        for _ in range(3)
    }
    assert seen == {cfg.rules_hash}


def test_rules_hash_changes_when_a_rule_changes(cfg: RuleConfig) -> None:
    data = copy.deepcopy(cfg.raw)
    data["match"]["target"] = 301
    assert RuleConfig.from_dict(data).rules_hash != cfg.rules_hash


def test_rules_hash_ignores_key_order(cfg: RuleConfig) -> None:
    data = {k: cfg.raw[k] for k in reversed(list(cfg.raw))}
    assert RuleConfig.from_dict(data).rules_hash == cfg.rules_hash


def test_staging_is_in_the_hash(cfg: RuleConfig) -> None:
    """An agent trained with auto-announced combinations plays a different game."""
    data = copy.deepcopy(cfg.raw)
    data["staging"]["declarations_are_actions"] = True
    assert RuleConfig.from_dict(data).rules_hash != cfg.rules_hash


def test_staging_deviations_are_reported(cfg: RuleConfig) -> None:
    deviations = cfg.staging.deviations()
    assert len(deviations) == 1
    assert "announce/question/show" in deviations[0]
