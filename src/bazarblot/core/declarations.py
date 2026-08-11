"""Combination (meld) detection and the cross-team precedence comparison. Rules §6.

M1 targets `staging.declarations_are_actions = False`: every combination a player holds is
auto-detected, auto-announced and auto-shown at trick 1 (rules §6.5). Concretely, this module's
`resolve_combinations` is called with the four *dealt hands* directly, which is equivalent to
every seat truthfully showing everything it holds.

When `declarations_are_actions` is turned on later, the caller instead assembles a `shown`
frozenset per seat from the announce/question/show protocol (only the melds a player actually
revealed) and passes those in place of the dealt hands. `resolve_combinations` itself does not
change — the comparison logic is identical either way, only its input changes. This is why the
function takes plain hand-like iterables rather than a `DealState`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from bazarblot.core.cards import RANK_INDEX, RANKS, SUITS, TEAM_OF, ContractType, card_id, suit_of

if TYPE_CHECKING:
    from bazarblot.core.rules import RuleConfig

MeldKind = Literal["sequence", "carre", "blot_reblot"]


@dataclass(frozen=True, slots=True)
class Meld:
    """One detected combination. `suit` is `None` for a carré (it spans all four suits)."""

    kind: MeldKind
    owner_seat: int
    suit: str | None
    rank: str | None  # sequence: top card's rank. carre: the four-of-a-kind's rank.
    length: int  # sequence: 3..8. carre: 4. blot_reblot: 2.
    value: int
    is_trump: bool
    cards: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class CombinationResult:
    """`team_points` is indexed by absolute team (0/1), not attacker/defender-relative."""

    winning_team: int | None
    winning_meld: Meld | None
    team_points: tuple[int, int]
    melds_by_seat: tuple[tuple[Meld, ...], ...]


# ---------------------------------------------------------------- detection


def _sequence_value(length: int, rules: RuleConfig) -> int:
    if length == 3:
        return rules.combinations.sequences["tierce"]
    if length == 4:
        return rules.combinations.sequences["fifty"]
    if length == 5:
        return rules.combinations.sequences["hundred"]
    return rules.combinations.long_run_value  # 6, 7 or 8-card runs — still 100 §6.1


def _find_sequences(
    hand: frozenset[int], suit: str, contract_type: ContractType, rules: RuleConfig, seat: int
) -> list[Meld]:
    """Maximal consecutive runs of held cards in `suit`, in natural rank order.

    A single 6/7/8-card run is one meld (via `long_run_value`), not a decomposition into
    overlapping smaller ones — "each card belongs to at most one declared sequence" (§6.1).
    Two *disjoint* runs in the same suit (e.g. 7-8-9 and Q-K-A, with a gap) are both real,
    separate melds.
    """
    # `cid % 8` is the rank index directly, by the `card_id = 8*suit + rank_index` encoding.
    ranks_present = sorted({cid % 8 for cid in hand if suit_of(cid) == suit})
    if not ranks_present:
        return []

    melds: list[Meld] = []
    is_trump = contract_type != "NT" and suit == contract_type

    def flush(start_idx: int, end_idx: int) -> None:
        length = end_idx - start_idx + 1
        if length < 3:
            return
        value = _sequence_value(length, rules)
        top_rank = RANKS[end_idx]
        cards = tuple(card_id(suit, RANKS[i]) for i in range(start_idx, end_idx + 1))
        melds.append(
            Meld(
                kind="sequence",
                owner_seat=seat,
                suit=suit,
                rank=top_rank,
                length=length,
                value=value,
                is_trump=is_trump,
                cards=cards,
            )
        )

    start = prev = ranks_present[0]
    for idx in ranks_present[1:]:
        if idx == prev + 1:
            prev = idx
            continue
        flush(start, prev)
        start = prev = idx
    flush(start, prev)
    return melds


def _find_carres(
    hand: frozenset[int], contract_type: ContractType, rules: RuleConfig, seat: int
) -> list[Meld]:
    melds: list[Meld] = []
    table = (
        rules.combinations.carre_trump
        if contract_type != "NT"
        else rules.combinations.carre_notrump
    )
    for rank in RANKS:
        cards = tuple(card_id(s, rank) for s in SUITS)
        if all(c in hand for c in cards):
            melds.append(
                Meld(
                    kind="carre",
                    owner_seat=seat,
                    suit=None,
                    rank=rank,
                    length=4,
                    value=table[rank],
                    is_trump=False,
                    cards=cards,
                )
            )
    return melds


def _find_blot_reblot(
    hand: frozenset[int], contract_type: ContractType, rules: RuleConfig, seat: int
) -> Meld | None:
    """K+Q of the trump suit. Never exists under `NT` — there is no trump suit. §6.2"""
    if contract_type == "NT":
        return None
    k, q = card_id(contract_type, "K"), card_id(contract_type, "Q")
    if k in hand and q in hand:
        return Meld(
            kind="blot_reblot",
            owner_seat=seat,
            suit=contract_type,
            rank=None,
            length=2,
            value=rules.combinations.blot_reblot,
            is_trump=True,
            cards=(k, q),
        )
    return None


def detect_hand_melds(
    hand: frozenset[int], seat: int, contract_type: ContractType, rules: RuleConfig
) -> list[Meld]:
    """All melds held by one player's hand: sequences per suit, carrés, Blot-Reblot."""
    melds: list[Meld] = []
    for suit in SUITS:
        melds.extend(_find_sequences(hand, suit, contract_type, rules, seat))
    melds.extend(_find_carres(hand, contract_type, rules, seat))
    br = _find_blot_reblot(hand, contract_type, rules, seat)
    if br is not None:
        melds.append(br)
    return melds


def detect_all_melds(
    hands: tuple[frozenset[int], ...], contract_type: ContractType, rules: RuleConfig
) -> list[Meld]:
    out: list[Meld] = []
    for seat, hand in enumerate(hands):
        out.extend(detect_hand_melds(hand, seat, contract_type, rules))
    return out


# ---------------------------------------------------------------- precedence ladder §6.4


def _more_elder(seat_a: int, seat_b: int, leader_seat: int) -> bool:
    """True if `seat_a` acts before `seat_b` in the trick-1 play order from `leader_seat`."""
    return (seat_a - leader_seat) % 4 < (seat_b - leader_seat) % 4


def beats(a: Meld, b: Meld, rules: RuleConfig, leader_seat: int) -> bool:
    """True if meld `a` strictly outranks meld `b` under the §6.4 ladder.

    Order: any carré beats any sequence (by category) -> carrés by value -> sequences by
    length -> top card -> trump -> elder hand. Top card outranks trump — see the worked
    example in §6.4: A-K-Q non-trump > J-10-9 trump > J-10-9 non-trump.
    """
    if a.kind != b.kind and rules.combinations.carre_always_beats_sequence:
        return a.kind == "carre"

    if a.kind == "carre" and b.kind == "carre":
        if a.value != b.value:
            return a.value > b.value
        return _more_elder(a.owner_seat, b.owner_seat, leader_seat)

    # both sequences (or carre_always_beats_sequence is off and kinds mismatch — fall through
    # to length/value comparison, which for a carre uses length=4 and is otherwise consistent)
    if a.length != b.length:
        return a.length > b.length
    a_top = RANK_INDEX[a.rank] if a.rank is not None else -1
    b_top = RANK_INDEX[b.rank] if b.rank is not None else -1
    if a_top != b_top:
        return a_top > b_top
    if a.is_trump != b.is_trump:
        return a.is_trump
    return _more_elder(a.owner_seat, b.owner_seat, leader_seat)


def resolve_combinations(
    hands: tuple[frozenset[int], ...],
    contract_type: ContractType,
    rules: RuleConfig,
    leader_seat: int,
) -> CombinationResult:
    """Cross-team comparison over every meld in `hands`.

    `hands` should be the melds actually *shown* — under M1 staging that is simply the dealt
    hands (auto-show), per this module's docstring.

    Blot-Reblot is excluded from the comparison pool entirely and credited unconditionally to
    its holder's team (§6.2, §6.4) — it neither wins nor is suppressed by the comparison.
    """
    all_melds = detect_all_melds(hands, contract_type, rules)
    comparable = [m for m in all_melds if m.kind != "blot_reblot"]
    blot_reblots = [m for m in all_melds if m.kind == "blot_reblot"]

    winning_team: int | None = None
    winning_meld: Meld | None = None
    if comparable:
        best = comparable[0]
        for m in comparable[1:]:
            if beats(m, best, rules, leader_seat):
                best = m
        winning_team = TEAM_OF[best.owner_seat]
        winning_meld = best

    team_points = [0, 0]
    if winning_team is not None and rules.combinations.only_winning_team_scores:
        for m in comparable:
            if TEAM_OF[m.owner_seat] == winning_team:
                team_points[winning_team] += m.value
    elif winning_team is not None:
        for m in comparable:
            team_points[TEAM_OF[m.owner_seat]] += m.value

    for br in blot_reblots:
        team_points[TEAM_OF[br.owner_seat]] += br.value

    melds_by_seat = tuple(tuple(m for m in all_melds if m.owner_seat == s) for s in range(4))
    return CombinationResult(
        winning_team=winning_team,
        winning_meld=winning_meld,
        team_points=(team_points[0], team_points[1]),
        melds_by_seat=melds_by_seat,
    )
