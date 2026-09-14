"""M5: `eval.duplicate` — the paired/duplicate primitives everything else in `eval/` builds on."""

from __future__ import annotations

import random

from bazarblot.agents.heuristic import HeuristicAgent
from bazarblot.agents.random_agent import RandomAgent
from bazarblot.core.rules import load_default
from bazarblot.env.actions import ActionSpace, build_action_space
from bazarblot.env.infoset import InfoSet
from bazarblot.eval.duplicate import play_deal_once, play_duplicate_match, play_duplicate_pair

RULES = load_default()
SPACE = build_action_space(RULES)


class _AlwaysPass:
    """A pathological agent that only ever passes when possible — used to force a real 4-pass
    abort so `play_deal_once`'s `None` path is actually exercised, not just assumed rare."""

    def act(self, info: InfoSet, space: ActionSpace, legal_mask: object) -> int:  # type: ignore[override]
        del info
        return 0 if legal_mask[0] else int(legal_mask.nonzero()[0][0])  # type: ignore[union-attr]


class _SeatHandRecorder:
    """Wraps a real agent, sharing one `recorded` dict across every wrapper used in a single
    trial — records the first (hence full, original) hand seen for EACH SEAT, keyed by seat
    rather than by which agent object happened to be controlling it. Lets a test compare
    per-seat hands across two runs even though the two policies swap which seats they control
    between those runs."""

    def __init__(self, inner: object, recorded: dict[int, frozenset[int]]) -> None:
        self.inner = inner
        self.recorded = recorded

    def act(self, info: InfoSet, space: ActionSpace, legal_mask: object) -> int:  # type: ignore[override]
        self.recorded.setdefault(info.seat, info.hand)
        return self.inner.act(info, space, legal_mask)  # type: ignore[attr-defined]


def test_play_deal_once_aborts_return_none() -> None:
    agents = {s: _AlwaysPass() for s in range(4)}
    result = play_deal_once(RULES, SPACE, seed=0, deal_number=0, dealer=0, agents=agents)
    assert result is None


def test_play_deal_once_produces_a_valid_result() -> None:
    h = HeuristicAgent(RULES, random.Random(0))
    agents = {s: h for s in range(4)}
    found_result = False
    for seed in range(20):
        result = play_deal_once(
            RULES, SPACE, seed=seed, deal_number=0, dealer=seed % 4, agents=agents
        )
        if result is None:
            continue
        found_result = True
        team0, team1 = result.team_scores
        assert team0 >= 0 and team1 >= 0
        assert (team0 == result.score_attackers) or (team1 == result.score_attackers)
    assert found_result


def test_play_duplicate_pair_deals_the_same_cards_to_each_seat_both_ways() -> None:
    """The core correctness property this whole module depends on: every seat gets the same 8
    cards in both of a pair's runs, regardless of which policy is assigned to that seat in each —
    replicates `play_duplicate_pair`'s own seat-assignment pattern directly against
    `play_deal_once` so per-seat hands can be inspected from both runs."""
    for seed in range(30):
        heur_x = HeuristicAgent(RULES, random.Random(seed))
        heur_y = HeuristicAgent(RULES, random.Random(seed + 1))
        recorded_a: dict[int, frozenset[int]] = {}
        recorded_b: dict[int, frozenset[int]] = {}
        agents_a = {
            0: _SeatHandRecorder(heur_x, recorded_a),
            2: _SeatHandRecorder(heur_x, recorded_a),
            1: _SeatHandRecorder(heur_y, recorded_a),
            3: _SeatHandRecorder(heur_y, recorded_a),
        }
        agents_b = {
            0: _SeatHandRecorder(heur_y, recorded_b),
            2: _SeatHandRecorder(heur_y, recorded_b),
            1: _SeatHandRecorder(heur_x, recorded_b),
            3: _SeatHandRecorder(heur_x, recorded_b),
        }
        result_a = play_deal_once(RULES, SPACE, seed, 0, seed % 4, agents_a)
        result_b = play_deal_once(RULES, SPACE, seed, 0, seed % 4, agents_b)
        if result_a is None or result_b is None:
            continue
        assert recorded_a == recorded_b, f"seed={seed}: per-seat hands differ between the runs"
        return
    raise AssertionError("no non-aborting seed found in range(30)")


def test_play_duplicate_pair_is_reproducible_given_fresh_identical_agents() -> None:
    """Re-running a pair with fresh, identically-seeded agent instances (same construction, not
    the same mutable objects — these agents carry an internal `self.rng` that advances as they
    act, so reusing one object across two "repeat" calls is not actually a repeat) reproduces
    the exact same result. This is the property evaluation reproducibility actually depends on;
    argument order is not (see `eval/duplicate.py`'s `play_duplicate_match` docstring for why
    "same deal regardless of X/Y argument order" isn't promised for either function — the
    abort-driven resampling can legitimately land on a different trial depending on which
    seat assignment happens to get checked in what role)."""
    for seed in range(10):
        r1 = play_duplicate_pair(
            RULES,
            SPACE,
            seed + 90000,
            dealer=seed % 4,
            policy_x=HeuristicAgent(RULES, random.Random(1)),
            policy_y=RandomAgent(random.Random(2)),
        )
        r2 = play_duplicate_pair(
            RULES,
            SPACE,
            seed + 90000,
            dealer=seed % 4,
            policy_x=HeuristicAgent(RULES, random.Random(1)),
            policy_y=RandomAgent(random.Random(2)),
        )
        assert r1 is not None and r2 is not None
        assert r1.margin_x == r2.margin_x
        assert r1.seed == r2.seed


def test_play_duplicate_match_terminates_and_reports_a_result() -> None:
    heur = HeuristicAgent(RULES, random.Random(3))
    rnd = RandomAgent(random.Random(4))
    result = play_duplicate_match(RULES, SPACE, seed=1, policy_x=heur, policy_y=rnd)
    assert result is not None
    assert result.match_x_as_team0.finished
    assert result.match_x_as_team1.finished
    assert result.x_win is None or 0.0 <= result.x_win <= 1.0
