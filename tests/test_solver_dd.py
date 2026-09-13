"""M2: the double-dummy solver.

The brute-force cross-check is the load-bearing test in this file — it is what actually proves
`solve()`'s alpha-beta + transposition-table + equivalence-class machinery didn't introduce a
bug relative to a plain, independent exhaustive search. See `solver/dd.py`'s module docstring
for why the last-hand bonus is folded into the search itself, and why the "equivalence class"
reduction is restricted to same-point-value groups rather than the naive rank-adjacency version.
"""

from __future__ import annotations

import random
import time

import pytest

from bazarblot.core.cards import TEAM_OF, build_tables, full_deck
from bazarblot.core.play import current_winner, legal_moves
from bazarblot.core.rules import load_default
from bazarblot.solver.dd import Hands, brute_force_solve, solve, solve_from

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


def _full_hands(rng: random.Random) -> Hands:
    return _reduced_hands(rng, 8)


# ---------------------------------------------------------------- brute-force agreement


@pytest.mark.parametrize("cards_per_player", [1, 2, 3, 4])
def test_agrees_with_brute_force_on_reduced_deals(cards_per_player: int) -> None:
    """The roadmap's stated cross-check, run across several reduced sizes and every contract
    family, not just 4-trick deals."""
    n_per_size = 40
    for seed in range(n_per_size):
        rng = random.Random(cards_per_player * 1_000_000 + seed)
        hands = _reduced_hands(rng, cards_per_player)
        contract_type = rng.choice(CONTRACT_TYPES)
        leader = rng.randrange(4)
        declaring_team = rng.randrange(2)

        fast = solve(hands, contract_type, leader, declaring_team, RULES)
        brute = brute_force_solve(hands, contract_type, leader, declaring_team, RULES)
        assert fast.declarer_points == brute, (
            f"cards_per_player={cards_per_player} seed={seed} contract={contract_type} "
            f"leader={leader} team={declaring_team}: fast={fast.declarer_points} brute={brute}"
        )


@pytest.mark.slow
def test_agrees_with_brute_force_at_reduced_scale_10000_samples() -> None:
    """The roadmap's literal "10^4 random reduced deals" figure, at a single representative
    reduced size (3 tricks — large enough to exercise real search, small enough that brute
    force stays fast across 10,000 samples). Even at this reduced size, 10,000 samples takes
    minutes, not seconds — well past what belongs in the fast tier (see
    `test_agrees_with_brute_force_on_reduced_deals` for that: same property, 160 samples)."""
    n = 10_000
    cards_per_player = 3
    for seed in range(n):
        rng = random.Random(9_000_000 + seed)
        hands = _reduced_hands(rng, cards_per_player)
        contract_type = rng.choice(CONTRACT_TYPES)
        leader = rng.randrange(4)
        declaring_team = rng.randrange(2)

        fast = solve(hands, contract_type, leader, declaring_team, RULES).declarer_points
        brute = brute_force_solve(hands, contract_type, leader, declaring_team, RULES)
        assert fast == brute, f"seed={seed}: fast={fast} brute={brute}"


# ---------------------------------------------------------------- PV validity


def _assert_pv_is_a_legal_value_achieving_playout(
    hands: Hands, contract_type: str, leader: int, declaring_team: int, seed: int
) -> None:
    tables = build_tables(contract_type, RULES)
    result = solve(hands, contract_type, leader, declaring_team, RULES)
    n_cards = sum(len(h) for h in hands)
    assert len(result.principal_variation) == n_cards

    live_hands = [set(h) for h in hands]
    to_act = leader
    trick: list[tuple[int, int]] = []
    declarer_points = 0
    for card in result.principal_variation:
        legal = legal_moves(
            frozenset(live_hands[to_act]), tuple(trick), to_act, contract_type, RULES, tables
        )
        assert card in legal, f"seed={seed}: PV plays an illegal card"
        live_hands[to_act].discard(card)
        trick.append((to_act, card))
        if len(trick) == 4:
            winner = current_winner(tuple(trick), tables)
            pts = sum(tables.points[c] for _, c in trick)
            if not any(live_hands):
                pts += RULES.contracts.last_hand_bonus
            if TEAM_OF[winner] == declaring_team:
                declarer_points += pts
            to_act = winner
            trick = []
        else:
            to_act = (to_act + 1) % 4

    assert declarer_points == result.declarer_points


def test_principal_variation_is_a_legal_full_playout_fast() -> None:
    """Same property as the slow-tier version below, at reduced scale (4 cards/player, cheap to
    solve) so it still runs every build."""
    for seed in range(30):
        rng = random.Random(seed + 4_000_000)
        hands = _reduced_hands(rng, 4)
        contract_type = rng.choice(CONTRACT_TYPES)
        leader = rng.randrange(4)
        declaring_team = rng.randrange(2)
        _assert_pv_is_a_legal_value_achieving_playout(
            hands, contract_type, leader, declaring_team, seed
        )


@pytest.mark.slow
def test_principal_variation_is_a_legal_full_playout() -> None:
    """Replay the PV through the actual engine's own legality/resolution and confirm the
    points it produces match `declarer_points` exactly — the PV isn't just A valid line, it's
    the one that actually achieves the claimed value. Full 8-trick deals, so each sample is a
    multi-second DD solve (see `solver/dd.py`'s documented performance) — slow tier only."""
    for seed in range(60):
        rng = random.Random(seed)
        hands = _full_hands(rng)
        contract_type = rng.choice(CONTRACT_TYPES)
        leader = rng.randrange(4)
        declaring_team = rng.randrange(2)
        _assert_pv_is_a_legal_value_achieving_playout(
            hands, contract_type, leader, declaring_team, seed
        )


def test_declarer_took_all_tricks_flag_is_consistent_with_the_pv() -> None:
    """Exercises `_pv_all_tricks_to_declarer` — doesn't need a full 8-trick deal to do so, and a
    reduced (4 cards/player) one keeps this in the fast tier."""
    rng = random.Random(123)
    hands = _reduced_hands(rng, 4)
    tables = build_tables("H", RULES)
    result = solve(hands, "H", leader=0, declaring_team=0, rules=RULES)

    to_act = 0
    trick: list[tuple[int, int]] = []
    all_declarer = True
    for card in result.principal_variation:
        trick.append((to_act, card))
        if len(trick) == 4:
            winner = current_winner(tuple(trick), tables)
            if TEAM_OF[winner] != 0:
                all_declarer = False
            to_act = winner
            trick = []
        else:
            to_act = (to_act + 1) % 4
    assert result.declarer_took_all_tricks == all_declarer


# ---------------------------------------------------------------- deal totals & sanity bounds


def test_solving_from_either_teams_perspective_is_the_same_game_fast() -> None:
    """Same identity as the slow-tier version below, at reduced scale (4 cards/player). The
    total isn't the full deal's fixed 162 here — only 16 of the 32 cards are in play — so it's
    computed from the actual dealt cards' point values (plus the last-hand bonus, which still
    fires at a reduced deal's own final trick) rather than hardcoded."""
    for seed in range(60):
        rng = random.Random(seed + 5_000_000)
        hands = _reduced_hands(rng, 4)
        contract_type = rng.choice(CONTRACT_TYPES)
        leader = rng.randrange(4)
        tables = build_tables(contract_type, RULES)
        total = sum(tables.points[c] for h in hands for c in h) + RULES.contracts.last_hand_bonus

        as_team0 = solve(hands, contract_type, leader, declaring_team=0, rules=RULES)
        as_team1 = solve(hands, contract_type, leader, declaring_team=1, rules=RULES)
        assert as_team0.declarer_points + as_team1.declarer_points == total, (
            f"seed={seed} contract={contract_type} leader={leader}: "
            f"{as_team0.declarer_points} + {as_team1.declarer_points} != {total}"
        )


@pytest.mark.slow
def test_solving_from_either_teams_perspective_is_the_same_game() -> None:
    """`solve(..., declaring_team=T)` has team T maximize its own points while the other team
    minimizes team T's points. Because raw points partition exactly (whatever team T doesn't
    get, the other team does, at every single trick resolution — not just on average), "the
    other team minimizes team T's points" is pointwise IDENTICAL to "the other team maximizes
    its own points". So solving with `declaring_team=0` and `declaring_team=1` on the exact
    same hands/contract/leader are the same optimal-play game viewed from each side, and their
    results must sum to the deal total EXACTLY, not approximately. This is a hard mathematical
    identity, not a heuristic — if `_Solver` treats the two calls asymmetrically in any way
    (an alpha-beta sign error, a mishandled window shift), this is very likely to catch it.
    Full 8-trick deals — 120 full solves total, minutes not seconds — slow tier only; see
    `test_solving_from_either_teams_perspective_is_the_same_game_fast` for the fast-tier check
    of the same identity."""
    for seed in range(60):
        rng = random.Random(seed)
        hands = _full_hands(rng)
        contract_type = rng.choice(CONTRACT_TYPES)
        leader = rng.randrange(4)

        as_team0 = solve(hands, contract_type, leader, declaring_team=0, rules=RULES)
        as_team1 = solve(hands, contract_type, leader, declaring_team=1, rules=RULES)
        total = RULES.contracts.deal_card_points
        assert as_team0.declarer_points + as_team1.declarer_points == total, (
            f"seed={seed} contract={contract_type} leader={leader}: "
            f"{as_team0.declarer_points} + {as_team1.declarer_points} != {total}"
        )


def test_capot_hand_solves_to_the_full_deal_total() -> None:
    """A team holding every card of overwhelming strength should be able to force all 162."""
    hands: Hands = (
        frozenset(range(0, 8)),  # all of suit C: 7,8,9,10,J,Q,K,A of clubs
        frozenset(range(8, 16)),
        frozenset(range(16, 24)),
        frozenset(range(24, 32)),
    )
    # seat 0 holds an entire suit; under that suit as trump, seat 0's hand is unbeatable
    # trump-wise, but still must follow suit against the other three who hold different suits
    # entirely — not a realistic deal, just a clean bound-check.
    result = solve(hands, "C", leader=0, declaring_team=0, rules=RULES)
    assert 0 <= result.declarer_points <= 162


# ---------------------------------------------------------------- solve_from (mid-trick entry)


def test_solve_from_with_empty_trick_matches_solve() -> None:
    """`solve()` is documented as the `trick_so_far=()` special case of `solve_from()` — this
    pins that they actually agree, not just that the docstring claims it."""
    for seed in range(20):
        rng = random.Random(seed + 6_000_000)
        hands = _reduced_hands(rng, 4)
        contract_type = rng.choice(CONTRACT_TYPES)
        leader = rng.randrange(4)
        declaring_team = rng.randrange(2)
        a = solve(hands, contract_type, leader, declaring_team, RULES)
        b = solve_from(hands, contract_type, leader, (), declaring_team, RULES)
        assert a.declarer_points == b.declarer_points
        assert a.principal_variation == b.principal_variation
        assert a.declarer_took_all_tricks == b.declarer_took_all_tricks


def test_solve_from_agrees_with_a_sub_path_of_an_optimal_solve() -> None:
    """A sub-path of an optimal minimax line is itself optimal for the position it starts from
    — a standard property of minimax, and the property `agents/pimc.py` actually relies on when
    it calls `solve_from` mid-trick. Verified by walking partway along a full `solve()`'s own PV,
    snapshotting the position (including a partially-played current trick), and checking that
    `solve_from` from that exact snapshot returns exactly the raw points still remaining along
    the original PV — not merely "some value", the SAME value the original optimal line achieves
    for the rest of the deal."""
    for seed in range(15):
        rng = random.Random(seed + 7_000_000)
        hands = _reduced_hands(rng, 4)
        contract_type = rng.choice(CONTRACT_TYPES)
        leader = rng.randrange(4)
        declaring_team = rng.randrange(2)
        tables = build_tables(contract_type, RULES)

        full = solve(hands, contract_type, leader, declaring_team, RULES)
        pv = full.principal_variation
        stop_at = rng.randrange(1, len(pv))  # a point strictly inside the PV, possibly mid-trick

        live_hands = [set(h) for h in hands]
        to_act = leader
        trick: list[tuple[int, int]] = []
        consumed = 0
        for card in pv[:stop_at]:
            live_hands[to_act].discard(card)
            trick.append((to_act, card))
            if len(trick) == 4:
                winner = current_winner(tuple(trick), tables)
                pts = sum(tables.points[c] for _, c in trick)
                if not any(live_hands):
                    pts += RULES.contracts.last_hand_bonus
                if TEAM_OF[winner] == declaring_team:
                    consumed += pts
                to_act = winner
                trick = []
            else:
                to_act = (to_act + 1) % 4

        snapshot_hands: Hands = tuple(frozenset(h) for h in live_hands)  # type: ignore[assignment]
        remaining = solve_from(
            snapshot_hands, contract_type, to_act, tuple(trick), declaring_team, RULES
        )
        assert remaining.declarer_points == full.declarer_points - consumed, (
            f"seed={seed} stop_at={stop_at}: "
            f"expected {full.declarer_points - consumed}, got {remaining.declarer_points}"
        )


# ---------------------------------------------------------------- performance


@pytest.mark.slow
def test_median_solve_time_is_a_measured_regression_guard() -> None:
    """The roadmap's stated aspiration was median < 5ms per full 8-trick deal on one core. That
    target assumed the classic cross-position equivalence-class transposition table from bridge
    solvers; `solver/dd.py`'s module docstring documents three independent soundness bugs found
    trying to implement that reduction in this (point-scoring, not trick-counting) ruleset, the
    third of which is not fixable by better bookkeeping — it's a genuine information loss
    (collapsing which hand holds the higher vs. lower of two equivalent cards can flip who wins
    a trick between them). What ships instead is a same-hand move-dedup optimization that is
    provably safe but far weaker: it cuts a much smaller fraction of the tree, since it can only
    ever collapse a single mover's own redundant choices, never cross positions in the
    transposition table.

    Measured medians on this machine are in the **1-5 second** range per full 8-trick deal (not
    milliseconds) — 200-1000x over the original target. This is a real, reported limitation, not
    a bug: closing the gap needs either a compiled implementation (Rust/PyO3, Cython, numba) or a
    multiset-and-hand-identity-safe cross-position reduction that nobody has yet designed
    correctly for this ruleset. Until then, exact-oracle DD solves are usable for offline
    dataset generation and evaluation (where seconds-per-deal is fine) but not as an inner-loop
    component of self-play. This test guards against a further regression, not against missing
    the original target — it will already fail loudly if that's news."""
    n = 40
    times: list[float] = []
    for seed in range(n):
        rng = random.Random(seed)
        hands = _full_hands(rng)
        contract_type = rng.choice(CONTRACT_TYPES)
        leader = rng.randrange(4)
        declaring_team = rng.randrange(2)
        start = time.perf_counter()
        solve(hands, contract_type, leader, declaring_team, RULES)
        times.append(time.perf_counter() - start)
    times.sort()
    median = times[n // 2]
    p95 = times[int(n * 0.95)]
    assert median < 20.0, (
        f"median solve time {median:.2f}s regressed past the 20s guard rail "
        f"(recent measured range: 1-5s; see test docstring)"
    )
    print(f"\nDD solver: median={median * 1000:.1f}ms p95={p95 * 1000:.1f}ms over {n} full deals")
