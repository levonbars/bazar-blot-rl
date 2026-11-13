"""The obvious-mistake probe (`learn/mistakes.py`)."""

from __future__ import annotations

import random

from bazarblot.agents.heuristic import HeuristicAgent
from bazarblot.agents.random_agent import RandomAgent
from bazarblot.core.rules import load_default
from bazarblot.env.actions import build_action_space
from bazarblot.learn.mistakes import measure_mistakes

RULES = load_default()
SPACE = build_action_space(RULES)


def test_heuristic_never_throws_points_and_random_play_does() -> None:
    """The heuristic discards its lowest card and wins valuable last-hand tricks; random play does
    neither — the probe must tell them apart."""
    h = measure_mistakes(RULES, SPACE, HeuristicAgent(RULES, random.Random(1)), n_deals=60)
    r = measure_mistakes(RULES, SPACE, RandomAgent(random.Random(1)), n_deals=60)
    assert h.rate("throws_points") == 0.0
    assert h.rate("ducks_valuable_trick") == 0.0
    r_throw, r_duck = r.rate("throws_points"), r.rate("ducks_valuable_trick")
    assert r_throw is not None and r_throw > 0.1
    assert r_duck is not None and r_duck > 0.1
    assert "contracts failed" in h.summary()
