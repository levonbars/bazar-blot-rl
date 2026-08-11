"""Card primitives and per-contract lookup tables.

`ContractType` lives here (not in `rules.py`) so that `rules.py` can import it from `cards.py`
without creating a cycle back the other way — `build_tables` below needs `RuleConfig` only for
type-checking, via the `TYPE_CHECKING` guard, never at runtime.

The encoding fixed here is load-bearing and must not change: `card_id = 8 * suit + rank_index`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal

if TYPE_CHECKING:
    from bazarblot.core.rules import RuleConfig

ContractType = Literal["C", "D", "H", "S", "NT"]

# Natural rank order, ascending. Sequences (tierce/fifty/hundred) are consecutive in THIS
# order — never in trump order. Blot Star's own "Hundred" illustration is 7-8-9-10-J.
RANKS: Final[tuple[str, ...]] = ("7", "8", "9", "10", "J", "Q", "K", "A")

SUITS: Final[tuple[str, ...]] = ("C", "D", "H", "S")

RANK_INDEX: Final[dict[str, int]] = {r: i for i, r in enumerate(RANKS)}
SUIT_INDEX: Final[dict[str, int]] = {s: i for i, s in enumerate(SUITS)}

N_CARDS: Final[int] = 32
N_PLAYERS: Final[int] = 4
CARDS_PER_HAND: Final[int] = 8
TRICKS_PER_DEAL: Final[int] = 8

# Seats 0/2 are partners (team 0, "Blue"), seats 1/3 are partners (team 1, "Red"). §2.
TEAM_OF: Final[tuple[int, int, int, int]] = (0, 1, 0, 1)
PARTNER_OF: Final[tuple[int, int, int, int]] = (2, 3, 0, 1)


def card_id(suit: str, rank: str) -> int:
    """Canonical card encoding: `8 * suit + rank_index`."""
    return 8 * SUIT_INDEX[suit] + RANK_INDEX[rank]


def rank_of(cid: int) -> str:
    return RANKS[cid % 8]


def suit_of(cid: int) -> str:
    return SUITS[cid // 8]


def card_label(cid: int) -> str:
    """Debug-friendly label, e.g. `card_label(0) == "7C"`."""
    return f"{rank_of(cid)}{suit_of(cid)}"


def parse_card(label: str) -> int:
    """Inverse of `card_label`, e.g. `parse_card("10S")`. Test convenience only."""
    return card_id(label[-1], label[:-1])


def full_deck() -> tuple[int, ...]:
    return tuple(range(N_CARDS))


@dataclass(frozen=True, slots=True)
class ContractTables:
    """Per-card-id lookup tables for one contract, built once and reused.

    `strength` is only meaningful for comparisons *within* the same suit (or within the trump
    group) — trick resolution always checks `is_trump` first, exactly mirroring the "any trump
    beats any non-trump" rule (§5), so cross-suit strength values are never compared directly.
    """

    contract_type: ContractType
    points: tuple[int, ...]
    strength: tuple[int, ...]
    is_trump: tuple[bool, ...]


def build_tables(contract_type: ContractType, rules: RuleConfig) -> ContractTables:
    points = [0] * N_CARDS
    strength = [0] * N_CARDS
    is_trump = [False] * N_CARDS
    for cid in range(N_CARDS):
        suit = suit_of(cid)
        rank = rank_of(cid)
        if contract_type == "NT":
            points[cid] = rules.contracts.points_notrump[rank]
            order = rules.contracts.order_plain
        elif suit == contract_type:
            points[cid] = rules.contracts.points_trump_suit[rank]
            order = rules.contracts.order_trump
            is_trump[cid] = True
        else:
            points[cid] = rules.contracts.points_plain_suit[rank]
            order = rules.contracts.order_plain
        # order is strongest-first (e.g. trump: J 9 A 10 K Q 8 7), so index 0 -> highest strength.
        strength[cid] = len(order) - order.index(rank)
    return ContractTables(
        contract_type=contract_type,
        points=tuple(points),
        strength=tuple(strength),
        is_trump=tuple(is_trump),
    )
