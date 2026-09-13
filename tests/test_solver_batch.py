"""M2.5 item 1: `solve_many`. The correctness bar is simple — parallel results must be bit-for-
bit identical to calling `solve`/`solve_from` directly, in the same order — since `solve_many`
adds no new solving logic, only a process pool around the already-verified `solve_from`."""

from __future__ import annotations

import random

import pytest

from bazarblot.core.cards import build_tables, full_deck
from bazarblot.core.play import legal_moves
from bazarblot.core.rules import load_default
from bazarblot.solver.batch import SolveSpec, solve_many
from bazarblot.solver.dd import Hands, solve_from

RULES = load_default()
CONTRACT_TYPES = list(RULES.contracts.types)


def _reduced_hands(rng: random.Random, cards_per_player: int) -> Hands:
    deck = list(full_deck())
    rng.shuffle(deck)
    n = cards_per_player
    return (
        frozenset(deck[0:n]),
        frozenset(deck[n : 2 * n]),
        frozenset(deck[2 * n : 3 * n]),
        frozenset(deck[3 * n : 4 * n]),
    )


def test_solve_many_empty_specs() -> None:
    assert solve_many([], RULES, workers=2) == []


def test_solve_many_matches_solve_from_directly_reduced_scale() -> None:
    """Reduced (4 cards/player) deals — fast enough to run every build, not slow-tier only."""
    specs = []
    for seed in range(20):
        rng = random.Random(seed)
        hands = _reduced_hands(rng, 4)
        specs.append(
            SolveSpec(
                hands=hands,
                contract_type=rng.choice(CONTRACT_TYPES),
                to_act=rng.randrange(4),
                declaring_team=rng.randrange(2),
            )
        )

    parallel = solve_many(specs, RULES, workers=2)
    assert len(parallel) == len(specs)
    for spec, result in zip(specs, parallel, strict=True):
        direct = solve_from(
            spec.hands,
            spec.contract_type,
            spec.to_act,
            spec.trick_so_far,
            spec.declaring_team,
            RULES,
        )
        assert result.declarer_points == direct.declarer_points
        assert result.principal_variation == direct.principal_variation


def test_solve_many_respects_trick_so_far() -> None:
    """Same property, but exercising the mid-trick (`trick_so_far` non-empty) path — the whole
    reason `solve_many` builds on `solve_from` rather than the narrower `solve`."""
    rng = random.Random(0)
    hands = _reduced_hands(rng, 4)
    contract_type = "H"
    tables = build_tables(contract_type, RULES)
    leader = 0
    first_card = next(iter(legal_moves(hands[leader], (), leader, contract_type, RULES, tables)))
    remaining = (hands[0] - {first_card}, hands[1], hands[2], hands[3])
    trick_so_far = ((leader, first_card),)

    spec = SolveSpec(
        hands=remaining,
        contract_type=contract_type,
        to_act=1,
        declaring_team=0,
        trick_so_far=trick_so_far,
    )
    (result,) = solve_many([spec], RULES, workers=1)
    direct = solve_from(remaining, contract_type, 1, trick_so_far, 0, RULES)
    assert result.declarer_points == direct.declarer_points


@pytest.mark.slow
def test_solve_many_matches_solve_at_full_scale() -> None:
    """The roadmap's stated M2.5 use case: full 8-trick deals, at a scale that would be
    impractical one at a time (slow tier only — see solver/dd.py's documented per-solve cost)."""

    def full_hands(rng: random.Random) -> Hands:
        return _reduced_hands(rng, 8)

    specs = []
    for seed in range(8):
        rng = random.Random(seed + 8_000_000)
        hands = full_hands(rng)
        specs.append(
            SolveSpec(
                hands=hands,
                contract_type=rng.choice(CONTRACT_TYPES),
                to_act=rng.randrange(4),
                declaring_team=rng.randrange(2),
            )
        )

    parallel = solve_many(specs, RULES, workers=4)
    for spec, result in zip(specs, parallel, strict=True):
        direct = solve_from(
            spec.hands,
            spec.contract_type,
            spec.to_act,
            spec.trick_so_far,
            spec.declaring_team,
            RULES,
        )
        assert result.declarer_points == direct.declarer_points
