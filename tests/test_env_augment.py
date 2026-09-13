"""M3: suit-permutation augmentation. The load-bearing property is DD-value preservation — the
roadmap's own "Done when" line: "augmentation verified to preserve the game value (a suit
permutation applied to a solved deal gives the same DD result)."""

from __future__ import annotations

import random

import pytest

from bazarblot.core.cards import SUIT_INDEX, full_deck
from bazarblot.core.rules import load_default
from bazarblot.env.augment import (
    IDENTITY,
    invert,
    permute_card,
    permute_contract_type,
    permute_hand,
    permute_hands,
    random_suit_permutation,
)
from bazarblot.solver.dd import solve

RULES = load_default()


def _full_hands(rng: random.Random) -> tuple[frozenset[int], ...]:
    deck = list(full_deck())
    rng.shuffle(deck)
    return tuple(frozenset(deck[i * 8 : (i + 1) * 8]) for i in range(4))


def test_identity_permutation_is_a_noop() -> None:
    assert permute_card(17, IDENTITY) == 17
    hand = frozenset({0, 5, 17, 31})
    assert permute_hand(hand, IDENTITY) == hand


def test_invert_undoes_permutation() -> None:
    rng = random.Random(0)
    for _ in range(50):
        perm = random_suit_permutation("NT", rng)
        inv = invert(perm)
        for card in range(32):
            assert permute_card(permute_card(card, perm), inv) == card


def test_trump_suit_fixed_under_trump_contract() -> None:
    rng = random.Random(0)
    for contract_type in ("C", "D", "H", "S"):
        for _ in range(30):
            perm = random_suit_permutation(contract_type, rng)
            assert perm[SUIT_INDEX[contract_type]] == SUIT_INDEX[contract_type]
            assert permute_contract_type(contract_type, perm) == contract_type


def test_nt_permutation_is_a_genuine_permutation() -> None:
    rng = random.Random(0)
    for _ in range(50):
        perm = random_suit_permutation("NT", rng)
        assert sorted(perm) == [0, 1, 2, 3]


def test_permute_hands_preserves_card_conservation() -> None:
    rng = random.Random(0)
    hands = _full_hands(rng)
    for contract_type in ("C", "D", "H", "S", "NT"):
        perm = random_suit_permutation(contract_type, rng)
        permuted = permute_hands(hands, perm)
        all_original = set().union(*hands)
        all_permuted = set().union(*permuted)
        assert all_original == set(range(32))
        assert all_permuted == set(range(32))
        for h, p in zip(hands, permuted, strict=True):
            assert len(h) == len(p)


@pytest.mark.slow
def test_dd_value_preserved_under_random_suit_permutation() -> None:
    """The roadmap's stated M3 done-when criterion. Marked slow — each sample is a full 8-trick
    DD solve (seconds, per M2's documented performance), so this stays out of the fast tier."""
    n = 12
    for seed in range(n):
        rng = random.Random(seed)
        hands = _full_hands(rng)
        contract_type = rng.choice(list(RULES.contracts.types))
        leader = rng.randrange(4)
        declaring_team = rng.randrange(2)

        base = solve(hands, contract_type, leader, declaring_team, RULES)

        perm = random_suit_permutation(contract_type, rng)
        permuted_hands = permute_hands(hands, perm)
        permuted_type = permute_contract_type(contract_type, perm)
        permuted = solve(permuted_hands, permuted_type, leader, declaring_team, RULES)

        assert base.declarer_points == permuted.declarer_points, (
            f"seed={seed} contract={contract_type}->{permuted_type} perm={perm}: "
            f"{base.declarer_points} != {permuted.declarer_points}"
        )
