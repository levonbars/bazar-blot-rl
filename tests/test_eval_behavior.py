"""`evaluate_behavior` reports what an agent DID, which a margin cannot: the property that matters
is that it tells apart an agent that bids from one that concedes every auction."""

from __future__ import annotations

import random

from bazarblot.agents.composed import ComposedAgent
from bazarblot.agents.heuristic import HeuristicAgent
from bazarblot.agents.pass_bidder import PassBidder
from bazarblot.core.rules import load_default
from bazarblot.env.actions import build_action_space
from bazarblot.eval.behavior import evaluate_behavior

RULES = load_default()
SPACE = build_action_space(RULES)


def _seeds(n: int) -> tuple[list[int], list[int]]:
    rng = random.Random(0)
    return [rng.randrange(1 << 30) for _ in range(n)], [rng.randrange(4) for _ in range(n)]


def test_always_pass_agent_never_declares_and_never_completes_selfplay_auctions() -> None:
    heuristic = HeuristicAgent(RULES)
    passer = ComposedAgent(PassBidder(), heuristic)
    seeds, dealers = _seeds(25)
    report = evaluate_behavior(RULES, SPACE, passer, heuristic, seeds, dealers)
    assert report.declare_rate == 0.0
    assert report.selfplay_contract_rate == 0.0
    assert report.made_rate is None  # it declared nothing, so there is nothing to have made
    assert "declare_rate=0.00" in report.summary()


def test_heuristic_declares_and_reaches_contracts() -> None:
    heuristic = HeuristicAgent(RULES)
    seeds, dealers = _seeds(25)
    report = evaluate_behavior(RULES, SPACE, heuristic, heuristic, seeds, dealers)
    assert report.contract_rate > 0.5
    assert report.declare_rate is not None and 0.2 < report.declare_rate < 0.8
    assert report.selfplay_contract_rate > 0.5
    assert report.made_rate is not None and report.made_rate > 0.5
