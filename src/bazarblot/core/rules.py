"""`RuleConfig` — the machine-readable contract for a Bazar Blot ruleset.

This module is deliberately the *first* thing built. Every downstream component derives its
shape from a `RuleConfig`: the action space from the bid ladder, the legal-move generator from
the play obligations, the reward from the scoring model. Hard-coding any of these elsewhere is
how a project ends up with results it cannot reproduce.

See `docs/01-rules.md` for provenance of every value, and `presets/blotstar.yaml` for the
canonical ruleset.

Two invariants are enforced at load time rather than trusted:

* a trump deal and a no-trump deal must both total 162 raw points, and
* the arithmetic that makes the "+16" in the scoring formula the same number for both.

These are the checks that caught the plain-suit ace being worth 11 and not 19.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Self

import yaml

from bazarblot.core.cards import RANKS, SUITS, ContractType

Doubling = Literal["none", "contra", "recontra"]
Rounding = Literal["half_down", "half_up"]
ThreePassRule = Literal["close", "last_bidder_may_raise"]
EpisodeUnit = Literal["deal", "match"]

_TRUMP_SUIT_TOTAL = 62
_PLAIN_SUIT_TOTAL = 30
_NOTRUMP_SUIT_TOTAL = 38
_LAST_HAND = 10
_DEAL_TOTAL = 162


class RuleConfigError(ValueError):
    """A preset is internally inconsistent. Never swallow this."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuleConfigError(message)


def _points_total(table: dict[str, int]) -> int:
    return sum(table.values())


def _check_rank_table(table: dict[str, int], name: str) -> None:
    missing = set(RANKS) - set(table)
    extra = set(table) - set(RANKS)
    _require(not missing, f"{name}: missing ranks {sorted(missing)}")
    _require(not extra, f"{name}: unknown ranks {sorted(extra)}")


@dataclass(frozen=True, slots=True)
class DealRules:
    cards_per_player: int
    deal_all_before_auction: bool
    dealer_rotation: str
    opener_and_leader: str
    redeal_rotates_dealer: bool


@dataclass(frozen=True, slots=True)
class ContractRules:
    types: tuple[ContractType, ...]
    points_trump_suit: dict[str, int]
    points_plain_suit: dict[str, int]
    points_notrump: dict[str, int]
    order_trump: tuple[str, ...]
    order_plain: tuple[str, ...]
    last_hand_bonus: int
    deal_card_points: int
    full_deal_scaled: int

    def validate(self) -> None:
        _check_rank_table(self.points_trump_suit, "points.trump_suit")
        _check_rank_table(self.points_plain_suit, "points.plain_suit")
        _check_rank_table(self.points_notrump, "points.notrump")

        trump = _points_total(self.points_trump_suit)
        plain = _points_total(self.points_plain_suit)
        notrump = _points_total(self.points_notrump)

        _require(trump == _TRUMP_SUIT_TOTAL, f"trump suit totals {trump}, expected 62")
        _require(plain == _PLAIN_SUIT_TOTAL, f"plain suit totals {plain}, expected 30")
        _require(notrump == _NOTRUMP_SUIT_TOTAL, f"no-trump suit totals {notrump}, expected 38")

        # The load-bearing check. Both contract families must total 162 raw, which is what
        # makes the "+16" in the scoring formula the same number for both. If the plain-suit
        # ace were given the 19-point no-trump value, this fails at 186.
        trump_deal = trump + 3 * plain + self.last_hand_bonus
        notrump_deal = 4 * notrump + self.last_hand_bonus
        _require(
            trump_deal == _DEAL_TOTAL,
            f"trump deal totals {trump_deal} raw, expected 162 "
            f"(62 + 3x30 + 10) — check the plain-suit column",
        )
        _require(
            notrump_deal == _DEAL_TOTAL,
            f"no-trump deal totals {notrump_deal} raw, expected 162 (4x38 + 10)",
        )
        _require(
            self.deal_card_points == _DEAL_TOTAL,
            f"deal_card_points is {self.deal_card_points}, expected 162",
        )

        for name, order in (("order_trump", self.order_trump), ("order_plain", self.order_plain)):
            _require(sorted(order) == sorted(RANKS), f"{name} is not a permutation of the ranks")


@dataclass(frozen=True, slots=True)
class AuctionRules:
    min_bid: int
    max_bid: int
    bid_increment: int
    bid_axis: str
    nt_outranks_suit_at_equal_level: bool
    capot_outranks_at_equal_level: bool
    capot_is_a_bid_modifier: bool
    capot_sticky_upward: bool
    pass_is_binding: bool
    four_passes_redeal: bool
    three_passes_after_bid: ThreePassRule
    max_auction_steps: int
    contra_out_of_turn: bool

    def validate(self) -> None:
        _require(self.min_bid >= 1, "min_bid must be positive")
        _require(
            self.max_bid > self.min_bid,
            f"max_bid ({self.max_bid}) must exceed min_bid ({self.min_bid})",
        )
        # 80 = capot 25 + carres J 20 + 9 14 + A 11 + 10 10. Bidding above the largest
        # makeable total is pointless; below it makes legal auctions unrepresentable.
        _require(self.bid_increment >= 1, "bid_increment must be at least 1")
        _require(self.max_auction_steps > 0, "max_auction_steps must be positive")

    @property
    def n_levels(self) -> int:
        return self.max_bid - self.min_bid + 1


@dataclass(frozen=True, slots=True)
class PlayRules:
    must_follow_suit: bool
    must_beat_when_trump_led: bool
    must_ruff_when_opponent_winning: bool
    must_overtrump: bool
    must_ruff_when_partner_winning: bool


@dataclass(frozen=True, slots=True)
class CombinationRules:
    sequences: dict[str, int]
    long_run_value: int
    carre_trump: dict[str, int]
    carre_notrump: dict[str, int]
    blot_reblot: int
    blot_reblot_in_notrump: bool
    blot_reblot_exempt_from_comparison: bool
    carre_always_beats_sequence: bool
    sequence_precedence: tuple[str, ...]
    tie_resolution: str
    only_winning_team_scores: bool
    announce_classes: tuple[str, ...]
    announce_requires_holding: bool
    questioning_enabled: bool
    questioner_must_hold_combination: bool
    question_discloses: dict[str, tuple[str, ...]]
    answers_may_be_false: bool
    lapsed_claim_passes_to_runner_up: bool

    def validate(self) -> None:
        _check_rank_table(self.carre_trump, "carre_trump")
        _check_rank_table(self.carre_notrump, "carre_notrump")
        _require(
            set(self.sequences) == {"tierce", "fifty", "hundred"},
            "sequences must define exactly tierce, fifty and hundred",
        )
        _require(
            self.sequences["tierce"] < self.sequences["fifty"] < self.sequences["hundred"],
            "sequence values must be strictly increasing by length",
        )
        # Top card ranks above trump — the owner's A-K-Q non-trump > J-10-9 trump example.
        _require(
            list(self.sequence_precedence) == ["length", "top_card", "trump", "elder_hand"],
            "sequence_precedence must be length -> top_card -> trump -> elder_hand; "
            "top card outranks trump (docs/01-rules.md §6.4)",
        )
        # Questioning discloses the top card and a trump bit, never the suit. Revealing the
        # suit would hand over which non-trump suit holds a run — real play-phase information
        # the rules withhold.
        seq_disclosure = set(self.question_discloses.get("sequence", ()))
        _require(
            "suit" not in seq_disclosure,
            "questioning must not disclose the suit (docs/01-rules.md §6.5)",
        )


@dataclass(frozen=True, slots=True)
class ScoringRules:
    bid_multiplier: dict[str, dict[str, int]]
    fulfilment_strict: bool
    combinations_count_for_fulfilment: bool
    defenders_score_collected_on_success: bool
    capot_card_portion: int
    capot_bid_requires_shutout: bool
    failed_contract_base: int
    failed_contract_base_on_defender_capot: int
    rounding: Rounding

    def validate(self) -> None:
        for family in ("trump", "notrump"):
            _require(family in self.bid_multiplier, f"bid_multiplier missing '{family}'")
            for doubling in ("none", "contra", "recontra"):
                _require(
                    doubling in self.bid_multiplier[family],
                    f"bid_multiplier.{family} missing '{doubling}'",
                )
        # No single formula reproduces this: no-trump is trump+1 at each level, not trump x2.
        # Keep it a lookup table and assert the shape so nobody "simplifies" it later.
        t, n = self.bid_multiplier["trump"], self.bid_multiplier["notrump"]
        for doubling in ("none", "contra", "recontra"):
            _require(
                n[doubling] == t[doubling] + 1,
                f"bid_multiplier: expected notrump[{doubling}] == trump[{doubling}] + 1",
            )

    def multiplier(self, contract_type: ContractType, doubling: Doubling) -> int:
        family = "notrump" if contract_type == "NT" else "trump"
        return self.bid_multiplier[family][doubling]


@dataclass(frozen=True, slots=True)
class MatchRules:
    target: int
    tiebreak_on_double_cross: str
    episode_unit: EpisodeUnit


@dataclass(frozen=True, slots=True)
class StagingRules:
    """Implementation choices that deviate from the real game.

    Not rules claims. Anything set here narrows the strategy space relative to how Bazar Blot
    is actually played, and must be reported as such rather than quietly assumed.
    """

    declarations_are_actions: bool

    def deviations(self) -> list[str]:
        out: list[str] = []
        if not self.declarations_are_actions:
            out.append(
                "combinations auto-announced and auto-shown: removes the announce/question/"
                "show sub-game (costless bluffing, suppression, counter-announcement)"
            )
        return out


@dataclass(frozen=True, slots=True)
class RuleConfig:
    preset: str
    deal: DealRules
    contracts: ContractRules
    auction: AuctionRules
    play: PlayRules
    combinations: CombinationRules
    scoring: ScoringRules
    match: MatchRules
    staging: StagingRules
    raw: dict[str, Any] = field(repr=False, compare=False, default_factory=dict)

    # ---------------------------------------------------------------- loading

    @classmethod
    def load(cls, path: str | Path) -> Self:
        with Path(path).open("r", encoding="utf-8") as fh:
            data: dict[str, Any] = yaml.safe_load(fh)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        c = data["contracts"]
        cm = data["combinations"]
        cfg = cls(
            preset=data["preset"],
            deal=DealRules(**data["deal"]),
            contracts=ContractRules(
                types=tuple(c["types"]),
                points_trump_suit=dict(c["points"]["trump_suit"]),
                points_plain_suit=dict(c["points"]["plain_suit"]),
                points_notrump=dict(c["points"]["notrump"]),
                order_trump=tuple(c["order_trump"]),
                order_plain=tuple(c["order_plain"]),
                last_hand_bonus=c["last_hand_bonus"],
                deal_card_points=c["deal_card_points"],
                full_deal_scaled=c["full_deal_scaled"],
            ),
            auction=AuctionRules(**data["auction"]),
            play=PlayRules(**data["play"]),
            combinations=CombinationRules(
                sequences=dict(cm["sequences"]),
                long_run_value=cm["long_run_value"],
                carre_trump=dict(cm["carre_trump"]),
                carre_notrump=dict(cm["carre_notrump"]),
                blot_reblot=cm["blot_reblot"],
                blot_reblot_in_notrump=cm["blot_reblot_in_notrump"],
                blot_reblot_exempt_from_comparison=cm["blot_reblot_exempt_from_comparison"],
                carre_always_beats_sequence=cm["carre_always_beats_sequence"],
                sequence_precedence=tuple(cm["sequence_precedence"]),
                tie_resolution=cm["tie_resolution"],
                only_winning_team_scores=cm["only_winning_team_scores"],
                announce_classes=tuple(cm["announce_classes"]),
                announce_requires_holding=cm["announce_requires_holding"],
                questioning_enabled=cm["questioning_enabled"],
                questioner_must_hold_combination=cm["questioner_must_hold_combination"],
                question_discloses={k: tuple(v) for k, v in cm["question_discloses"].items()},
                answers_may_be_false=cm["answers_may_be_false"],
                lapsed_claim_passes_to_runner_up=cm["lapsed_claim_passes_to_runner_up"],
            ),
            scoring=ScoringRules(**data["scoring"]),
            match=MatchRules(**data["match"]),
            staging=StagingRules(**data["staging"]),
            raw=data,
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        self.contracts.validate()
        self.auction.validate()
        self.combinations.validate()
        self.scoring.validate()
        _require(
            set(self.contracts.types) <= set(SUITS) | {"NT"},
            f"unknown contract types in {self.contracts.types}",
        )
        _require(self.match.target > 0, "match target must be positive")

    # ---------------------------------------------------------------- identity

    def canonical_json(self) -> str:
        """Order-independent, comment-free serialization. The basis of `rules_hash`."""
        return json.dumps(self.raw, sort_keys=True, separators=(",", ":"), ensure_ascii=True)

    @property
    def rules_hash(self) -> str:
        """Stable 12-hex-char fingerprint of the resolved ruleset.

        Stamp this into every checkpoint and every result table, and refuse to load a
        checkpoint whose hash differs. Rule drift between runs is the most likely way this
        project produces numbers it cannot defend.

        `staging` is included deliberately: an agent trained with combinations auto-announced
        is not comparable to one trained against the real protocol.
        """
        digest = hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()
        return digest[:12]

    # ---------------------------------------------------------------- derived

    @property
    def n_bid_levels(self) -> int:
        return self.auction.n_levels

    @property
    def n_bid_actions(self) -> int:
        """`levels x contract types x capot flag` — the flat engine bid space."""
        capot_states = 2 if self.auction.capot_is_a_bid_modifier else 1
        return self.n_bid_levels * len(self.contracts.types) * capot_states


DEFAULT_PRESET = Path(__file__).resolve().parents[3] / "presets" / "blotstar.yaml"


def load_default() -> RuleConfig:
    return RuleConfig.load(DEFAULT_PRESET)
