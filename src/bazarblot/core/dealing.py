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
