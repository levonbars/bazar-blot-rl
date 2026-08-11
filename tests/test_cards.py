"""M1: card primitives and per-contract lookup tables."""

from __future__ import annotations

from bazarblot.core.cards import (
    N_CARDS,
    PARTNER_OF,
    RANKS,
    SUITS,
    TEAM_OF,
    build_tables,
    card_id,
    card_label,
    full_deck,
    parse_card,
    rank_of,
    suit_of,
)
from bazarblot.core.rules import load_default

RULES = load_default()


def test_card_id_roundtrip_for_every_card() -> None:
    for suit in SUITS:
        for rank in RANKS:
            cid = card_id(suit, rank)
            assert rank_of(cid) == rank
            assert suit_of(cid) == suit


def test_full_deck_is_32_distinct_cards() -> None:
    deck = full_deck()
    assert len(deck) == N_CARDS
    assert len(set(deck)) == N_CARDS
    assert set(deck) == set(range(32))


def test_card_label_and_parse_card_roundtrip() -> None:
    for cid in full_deck():
        label = card_label(cid)
        assert parse_card(label) == cid
    assert card_label(card_id("C", "7")) == "7C"
    assert card_label(card_id("S", "10")) == "10S"
    assert parse_card("AH") == card_id("H", "A")


def test_team_and_partner_constants() -> None:
    assert TEAM_OF == (0, 1, 0, 1)
    assert PARTNER_OF == (2, 3, 0, 1)
    for seat in range(4):
        assert TEAM_OF[PARTNER_OF[seat]] == TEAM_OF[seat]


# ---------------------------------------------------------------- build_tables


def test_build_tables_trump_suit_point_values() -> None:
    tables = build_tables("H", RULES)
    assert tables.points[card_id("H", "J")] == 20
    assert tables.points[card_id("H", "9")] == 14
    assert tables.points[card_id("H", "A")] == 11
    assert tables.is_trump[card_id("H", "7")] is True


def test_build_tables_side_suit_uses_plain_table_not_notrump_ace() -> None:
    """Side suits in a trump contract use A=11, not the NT ace value of 19."""
    tables = build_tables("H", RULES)
    assert tables.points[card_id("C", "A")] == 11
    assert tables.is_trump[card_id("C", "A")] is False


def test_build_tables_notrump_has_no_trump_cards_at_all() -> None:
    tables = build_tables("NT", RULES)
    assert not any(tables.is_trump)
    assert tables.points[card_id("H", "A")] == 19


def test_build_tables_suit_totals() -> None:
    for contract_type in RULES.contracts.types:
        tables = build_tables(contract_type, RULES)
        if contract_type == "NT":
            assert sum(tables.points) == 4 * 38
        else:
            trump_total = sum(tables.points[c] for c in range(32) if tables.is_trump[c])
            plain_total = sum(tables.points[c] for c in range(32) if not tables.is_trump[c])
            assert trump_total == 62
            assert plain_total == 3 * 30


def test_strength_orders_trump_correctly() -> None:
    """Trump order: J 9 A 10 K Q 8 7 (strongest first)."""
    tables = build_tables("S", RULES)
    ranks_by_strength = sorted(RANKS, key=lambda r: tables.strength[card_id("S", r)], reverse=True)
    assert ranks_by_strength == ["J", "9", "A", "10", "K", "Q", "8", "7"]


def test_strength_orders_plain_suit_correctly() -> None:
    """Plain order: A 10 K Q J 9 8 7 (strongest first)."""
    tables = build_tables("S", RULES)
    ranks_by_strength = sorted(RANKS, key=lambda r: tables.strength[card_id("C", r)], reverse=True)
    assert ranks_by_strength == ["A", "10", "K", "Q", "J", "9", "8", "7"]
