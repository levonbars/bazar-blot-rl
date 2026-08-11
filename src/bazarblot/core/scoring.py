"""Deal scoring. Rules §7 — the part of this ruleset that differs most from classical Belote.

The payout is `M x bid + collected + bonuses` on a made contract, and `M x bid + 16 + bonuses`
to the *defenders* on a failed one — not "each team banks what it collected". Capot *sets* the
winning side's card portion to 252 raw rather than adding 90 to it, and a capot *bid* additionally
requires the shutout: hitting the point target with the contract still fails if a single trick
leaked. See rules §7.5 for three worked fixtures (A/B/C) that pin this down exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from bazarblot.core.auction import Contract
    from bazarblot.core.rules import RuleConfig

SaysTeam = Literal["attackers", "defenders"]


def round10(x: int, rules: RuleConfig) -> int:
    """Divide by 10 and round per `rules.scoring.rounding`.

    `half_down` (the default — OPEN-10): a remainder of exactly 5 rounds DOWN. This is the
    rule that makes two teams' card-point shares of a bonus-free deal always sum to exactly 16;
    `half_up` breaks that invariant. §7.4
    """
    q, r = divmod(x, 10)
    if rules.scoring.rounding == "half_down":
        return q + (1 if r > 5 else 0)
    return q + (1 if r >= 5 else 0)  # half_up


@dataclass(frozen=True, slots=True)
class DealScore:
    made: bool
    attackers_score: int
    defenders_score: int
    raw_attackers: int
    raw_defenders: int


def compute_card_portions(
    *,
    card_points_attackers: int,
    card_points_defenders: int,
    says_team: SaysTeam,
    attackers_all_tricks: bool,
    defenders_all_tricks: bool,
    rules: RuleConfig,
) -> tuple[int, int]:
    """Raw card-point portion per side, including the last-hand bonus and the capot override.

    Capot is an assignment (`= capot_card_portion`), never `+= 90` — the two coincide only when
    the shutout side also happens to hold every combination bonus, which is exactly the case a
    naive `+=` implementation would pass while getting the general rule wrong. §7.3
    """
    if attackers_all_tricks:
        return rules.scoring.capot_card_portion, 0
    if defenders_all_tricks:
        return 0, rules.scoring.capot_card_portion
    bonus = rules.contracts.last_hand_bonus
    cards_a = card_points_attackers + (bonus if says_team == "attackers" else 0)
    cards_d = card_points_defenders + (bonus if says_team == "defenders" else 0)
    return cards_a, cards_d


def score_deal(
    *,
    rules: RuleConfig,
    contract: Contract,
    cards_attackers: int,
    cards_defenders: int,
    combo_attackers: int,
    combo_defenders: int,
    attackers_all_tricks: bool,
    defenders_all_tricks: bool,
) -> DealScore:
    """Score one completed deal. `cards_*` are raw card-point portions from
    `compute_card_portions` (already including says/capot); `combo_*` are raw combination
    bonuses from `declarations.resolve_combinations`.
    """
    raw_a = cards_attackers + combo_attackers
    raw_d = cards_defenders + combo_defenders

    fulfilment_basis = raw_a if rules.scoring.combinations_count_for_fulfilment else cards_attackers
    threshold = 10 * contract.level
    point_target_met = (
        fulfilment_basis > threshold
        if rules.scoring.fulfilment_strict
        else fulfilment_basis >= threshold
    )
    # A capot BID has two independent failure modes: the point target, and the shutout itself.
    # Meeting one without the other still loses the contract. §7.2
    shutout_ok = attackers_all_tricks if contract.capot else True
    made = point_target_met and shutout_ok

    multiplier = rules.scoring.multiplier(contract.contract_type, contract.doubling)
    bid_payout = multiplier * contract.level

    if made:
        score_a = bid_payout + round10(raw_a, rules)
        score_d = round10(raw_d, rules)  # defenders DO bank their own collected points
    else:
        score_a = 0
        # A defenders' shutout on a failed contract REPLACES the 16 with 25 — it does not
        # stack on top of it. §7.3
        base = (
            rules.scoring.failed_contract_base_on_defender_capot
            if defenders_all_tricks
            else rules.scoring.failed_contract_base
        )
        score_d = bid_payout + base + round10(combo_defenders, rules)

    return DealScore(
        made=made,
        attackers_score=score_a,
        defenders_score=score_d,
        raw_attackers=raw_a,
        raw_defenders=raw_d,
    )
