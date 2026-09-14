"""`core.dealing.deal_seed` — shared by `env/aec.py` (live play) and `eval/duplicate.py` (paired
evaluation), so both redeal identically for the same `(seed, deal_number, redeal_attempt)`."""

from __future__ import annotations

from bazarblot.core.dealing import deal_seed


def test_deal_seed_is_deterministic() -> None:
    assert deal_seed(1, 2, 3) == deal_seed(1, 2, 3)


def test_deal_seed_differs_by_redeal_attempt() -> None:
    """The exact bug M1.5 caught: without this, a 4-pass abort would redeal into the identical
    hand forever, since `deal_number` alone doesn't advance on an abort."""
    seeds = {deal_seed(1, 2, attempt) for attempt in range(10)}
    assert len(seeds) == 10


def test_deal_seed_differs_by_deal_number_and_base_seed() -> None:
    seeds = set()
    for seed in range(5):
        for deal_number in range(5):
            seeds.add(deal_seed(seed, deal_number, 0))
    assert len(seeds) == 25


def test_deal_seed_is_a_valid_random_seed() -> None:
    import random

    random.Random(deal_seed(0, 0, 0))  # must not raise
    random.Random(deal_seed(999_999, 999, 999))
