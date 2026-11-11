"""Behaviour checks that accompany every margin (`eval/metrics.py`'s paired margins say who wins,
not HOW).

A paired margin cannot tell a good bidder from an agent that concedes every auction against an
opponent who over-bids: both can show a large positive number. This report states what the agent
actually did, over the same shuffles with the agent on each team in turn:

- `contract_rate`: how often a deal reaches a contract at all (a pass-everything agent drives this
  to ~0 against itself).
- `declare_rate`: of the deals that reached a contract, how often the agent's team declared.
- `made_rate`: of the contracts the agent's team declared, how many were made. A bidder that
  overbids shows up here immediately (the original heuristic made 11%).
- `selfplay_contract_rate`: contract rate when the agent plays all four seats.
"""

from __future__ import annotations

from dataclasses import dataclass

from bazarblot.agents.base import Agent
from bazarblot.core.rules import RuleConfig
from bazarblot.env.actions import ActionSpace
from bazarblot.eval.duplicate import play_deal_once


@dataclass(frozen=True, slots=True)
class BehaviorReport:
    n_deals: int
    contract_rate: float
    declare_rate: float | None
    made_rate: float | None
    selfplay_contract_rate: float

    def summary(self) -> str:
        def fmt(x: float | None) -> str:
            return "n/a" if x is None else f"{x:.2f}"

        return (
            f"contract_rate={fmt(self.contract_rate)} declare_rate={fmt(self.declare_rate)} "
            f"made_rate={fmt(self.made_rate)} selfplay_contract_rate="
            f"{fmt(self.selfplay_contract_rate)} (n={self.n_deals} shuffles x 2 seatings)"
        )


def evaluate_behavior(
    rules: RuleConfig,
    space: ActionSpace,
    agent: Agent,
    opponent: Agent,
    seeds: list[int],
    dealers: list[int],
) -> BehaviorReport:
    """Play every `(seed, dealer)` twice against `opponent` (agent as team 0, then team 1) plus
    once in all-four-seats self-play, and report what the agent did."""
    if len(seeds) != len(dealers):
        raise ValueError("seeds and dealers must be the same length")
    games = contracts = declared = made = selfplay_contracts = 0
    for seed, dealer in zip(seeds, dealers, strict=True):
        for agent_team in (0, 1):
            seats = {s: (agent if s % 2 == agent_team else opponent) for s in range(4)}
            result = play_deal_once(rules, space, seed, 0, dealer, seats)
            games += 1
            if result is None:
                continue
            contracts += 1
            if result.attackers_team == agent_team:
                declared += 1
                made += int(result.made)
        if play_deal_once(rules, space, seed, 0, dealer, {s: agent for s in range(4)}) is not None:
            selfplay_contracts += 1
    return BehaviorReport(
        n_deals=len(seeds),
        contract_rate=contracts / max(1, games),
        declare_rate=declared / contracts if contracts else None,
        made_rate=made / declared if declared else None,
        selfplay_contract_rate=selfplay_contracts / max(1, len(seeds)),
    )
