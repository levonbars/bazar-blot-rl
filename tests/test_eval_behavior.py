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


class _DoublesWheneverItCan:
    def __init__(self, fallback) -> None:  # type: ignore[no-untyped-def]
        self.fallback = fallback

    def act(self, info, space, legal_mask):  # type: ignore[no-untyped-def]
        from bazarblot.env.actions import CONTRA_IDX, RECONTRA_IDX

        for idx in (CONTRA_IDX, RECONTRA_IDX):
            if info.phase.name == "AUCTION" and legal_mask[idx]:
                return idx
        return self.fallback.act(info, space, legal_mask)


def test_contra_rate_reports_how_often_the_agent_doubles() -> None:
    heuristic = HeuristicAgent(RULES)
    seeds, dealers = _seeds(25)
    never = evaluate_behavior(RULES, SPACE, heuristic, heuristic, seeds, dealers)
    assert never.contra_rate == 0.0  # the heuristic has chances and never takes them
    always = evaluate_behavior(
        RULES, SPACE, _DoublesWheneverItCan(heuristic), heuristic, seeds, dealers
    )
    assert always.contra_rate == 1.0
    assert "contra_rate=1.00" in always.summary()
