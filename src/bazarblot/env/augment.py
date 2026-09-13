"""Suit canonicalization / permutation augmentation (environment spec §4.2).

Suits are interchangeable except for their role in the contract: under a trump contract, the
trump suit is structurally special (it beats everything, melds detect it specially) but the
other three are pure labels; under `NT`, all four are pure labels. Relabeling suits consistently
across an entire deal — every hand, the contract, every played card — produces a *different deal
that is exactly as good for whoever's exactly as good*: the DD-solved value is unchanged, because
nothing in the rules engine ever refers to a suit by name, only by "is it the trump suit" and
"is it the same suit as this other card". `tests/test_env_augment.py`'s DD-value-preservation
test is the check that this claim is actually true of `core/`, not just of this module's
intentions.

This gives **6x augmentation under a trump contract** (trump fixed, 3 others freely permuted:
`3! = 6`) and **24x under `NT`** (`4! = 24`, all four free) — "do it in M3, not as an
afterthought" per the spec, because it is close to free sample-efficiency.

Only the permutation utilities live here; `encode()` in `env/obs.py` deliberately does NOT apply
one internally — augmentation is meant to be applied to the underlying game objects *before*
`info_set()`/`encode()` run (once per sampled permutation), not baked into the encoder itself.
"""

from __future__ import annotations

import random

from bazarblot.core.cards import SUIT_INDEX, SUITS, ContractType

SuitPermutation = tuple[int, int, int, int]  # perm[old_suit_index] = new_suit_index

IDENTITY: SuitPermutation = (0, 1, 2, 3)


def random_suit_permutation(contract_type: ContractType, rng: random.Random) -> SuitPermutation:
    """A uniformly random suit relabeling consistent with `contract_type`: trump suit fixed
    under a trump contract (it cannot be relabeled without changing what "trump" means), all
    four suits free under `NT`."""
    if contract_type == "NT":
        free = list(range(4))
        rng.shuffle(free)
        return (free[0], free[1], free[2], free[3])

    trump_idx = SUIT_INDEX[contract_type]
    others = [s for s in range(4) if s != trump_idx]
    shuffled = others.copy()
    rng.shuffle(shuffled)
    perm = [0, 0, 0, 0]
    perm[trump_idx] = trump_idx
    for old, new in zip(others, shuffled, strict=True):
        perm[old] = new
    return (perm[0], perm[1], perm[2], perm[3])


def permute_card(card: int, perm: SuitPermutation) -> int:
    suit_idx, rank_idx = divmod(card, 8)
    return 8 * perm[suit_idx] + rank_idx


def permute_hand(hand: frozenset[int], perm: SuitPermutation) -> frozenset[int]:
    return frozenset(permute_card(c, perm) for c in hand)


def permute_hands(
    hands: tuple[frozenset[int], ...], perm: SuitPermutation
) -> tuple[frozenset[int], ...]:
    return tuple(permute_hand(h, perm) for h in hands)


def permute_contract_type(contract_type: ContractType, perm: SuitPermutation) -> ContractType:
    if contract_type == "NT":
        return "NT"
    new_idx = perm[SUIT_INDEX[contract_type]]
    return SUITS[new_idx]  # type: ignore[return-value]


def invert(perm: SuitPermutation) -> SuitPermutation:
    inv = [0, 0, 0, 0]
    for old, new in enumerate(perm):
        inv[new] = old
    return (inv[0], inv[1], inv[2], inv[3])
