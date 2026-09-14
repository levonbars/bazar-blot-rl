"""Shuffling and dealing. Rules §2.1, §9.

Deterministic from a seed: `deal_hands(random.Random(seed), rules)` is fully reproducible, which
is what lets a `(match_seed, deal_number, action_seq)` triple reconstruct any deal without
storing observations (see the environment spec §10). Python's stdlib `random.Random` is used
rather than numpy — `core/` has no numpy dependency by design (see `docs/02-environment-spec.md`
§1).
"""

from __future__ import annotations

import random

from bazarblot.core.cards import full_deck
from bazarblot.core.rules import RuleConfig


def deal_hands(rng: random.Random, rules: RuleConfig) -> tuple[frozenset[int], ...]:
    deck = list(full_deck())
    rng.shuffle(deck)
    n = rules.deal.cards_per_player
    return tuple(frozenset(deck[i * n : (i + 1) * n]) for i in range(4))


def deal_seed(seed: int, deal_number: int, redeal_attempt: int = 0) -> int:
    """The per-deal RNG seed for deal `deal_number` of a match seeded with `seed`, on its
    `redeal_attempt`-th shuffle (0 for the first attempt, incrementing on each 4-pass abort).

    `redeal_attempt` is load-bearing, not decorative — see M1.5's fixed bug (README, "Redeals
    reused the exact same shuffle"): `deal_number` alone doesn't change across a 4-pass abort (by
    design — an aborted deal is never reported to `Match`), so a seed that ignored the attempt
    number would hand a deterministic policy the identical hand forever. Shared by `env/aec.py`
    (live play) and `eval/duplicate.py` (paired evaluation) so both redeal identically for the
    same `(seed, deal_number, redeal_attempt)` — needed for `eval/duplicate.py`'s "same shuffle,
    seats swapped" guarantee to actually hold across a deal that happens to abort.
    """
    return (seed * 1_000_003 + deal_number * 97 + redeal_attempt) & 0x7FFFFFFF
