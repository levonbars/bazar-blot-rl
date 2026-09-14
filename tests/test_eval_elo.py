"""M5: `eval.elo` — round-robin Elo on paired deals."""

from __future__ import annotations

import random

import pytest

from bazarblot.agents.heuristic import HeuristicAgent
from bazarblot.agents.random_agent import RandomAgent
from bazarblot.core.rules import load_default
from bazarblot.env.actions import build_action_space
from bazarblot.eval.elo import EloPool, margin_to_score, run_round_robin

RULES = load_default()
SPACE = build_action_space(RULES)


def test_margin_to_score() -> None:
    assert margin_to_score(10.0) == 1.0
    assert margin_to_score(-10.0) == 0.0
    assert margin_to_score(0.0) == 0.5


def test_elo_pool_expected_score_symmetric() -> None:
    pool = EloPool(ratings={"a": 1500.0, "b": 1500.0})
    assert pool.expected_score("a", "b") == pytest.approx(0.5)
    assert pool.expected_score("a", "b") + pool.expected_score("b", "a") == pytest.approx(1.0)


def test_elo_pool_update_conserves_total_rating() -> None:
    """A zero-sum property any correct Elo update must have: what one side gains, the other
    loses, exactly — this is the standard formula's own invariant, not particular to this
    implementation, but worth pinning."""
    pool = EloPool(ratings={"a": 1600.0, "b": 1400.0})
    total_before = sum(pool.ratings.values())
    pool.update("a", "b", score_a=1.0)
    total_after = sum(pool.ratings.values())
    assert total_after == pytest.approx(total_before)


def test_elo_pool_winner_rating_increases() -> None:
    pool = EloPool(ratings={"a": 1500.0, "b": 1500.0})
    pool.update("a", "b", score_a=1.0)
    assert pool.ratings["a"] > 1500.0
    assert pool.ratings["b"] < 1500.0


def test_run_round_robin_needs_at_least_two_agents() -> None:
    with pytest.raises(ValueError):
        run_round_robin(RULES, SPACE, {"solo": RandomAgent()}, n_pairs_per_matchup=1)


def test_run_round_robin_heuristic_rated_above_random() -> None:
    agents = {
        "random": RandomAgent(random.Random(1)),
        "heuristic": HeuristicAgent(RULES, random.Random(2)),
    }
    pool = run_round_robin(RULES, SPACE, agents, n_pairs_per_matchup=40, seed=0)
    assert pool.ratings["heuristic"] > pool.ratings["random"]
    # A wide, unambiguous separation, matching the wide margin already established by
    # `test_heuristic_vs_random_paired_margin_is_tight_and_excludes_zero`.
    assert pool.ratings["heuristic"] - pool.ratings["random"] > 200


def test_run_round_robin_covers_every_matchup() -> None:
    """A 3-agent pool must play every distinct pair, not just the first two."""
    agents = {
        "random_a": RandomAgent(random.Random(1)),
        "random_b": RandomAgent(random.Random(2)),
        "heuristic": HeuristicAgent(RULES, random.Random(3)),
    }
    pool = run_round_robin(RULES, SPACE, agents, n_pairs_per_matchup=10, seed=100)
    assert set(pool.ratings) == set(agents)
    # heuristic should separate from BOTH random agents, not just whichever it played first
    assert pool.ratings["heuristic"] > pool.ratings["random_a"]
    assert pool.ratings["heuristic"] > pool.ratings["random_b"]
