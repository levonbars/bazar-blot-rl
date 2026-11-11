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
- `contra_rate`: of the auction decisions where the agent COULD double (contra, or recontra in the
  reply window), the share on which it did. A policy cloned from a teacher that never doubles sits
  at exactly 0, and stays there unless something teaches it the value of doubling.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from bazarblot.agents.base import Agent
from bazarblot.core.deal import Phase
from bazarblot.core.rules import RuleConfig
from bazarblot.env.actions import CONTRA_IDX, RECONTRA_IDX, ActionSpace
from bazarblot.env.infoset import InfoSet
from bazarblot.eval.duplicate import play_deal_once


@dataclass(frozen=True, slots=True)
class BehaviorReport:
    n_deals: int
    contract_rate: float
    declare_rate: float | None
    made_rate: float | None
    selfplay_contract_rate: float
    contra_rate: float | None = None

    def summary(self) -> str:
        def fmt(x: float | None) -> str:
            return "n/a" if x is None else f"{x:.2f}"

        return (
            f"contract_rate={fmt(self.contract_rate)} declare_rate={fmt(self.declare_rate)} "
            f"made_rate={fmt(self.made_rate)} selfplay_contract_rate="
            f"{fmt(self.selfplay_contract_rate)} contra_rate={fmt(self.contra_rate)} "
            f"(n={self.n_deals} shuffles x 2 seatings)"
        )


class _DoublingCounter:
    """Delegates to `agent` and counts its chances to double and the times it took them."""

    def __init__(self, agent: Agent) -> None:
        self.agent = agent
        self.chances = 0
        self.taken = 0

    def act(self, info: InfoSet, space: ActionSpace, legal_mask: npt.NDArray[np.bool_]) -> int:
        action = self.agent.act(info, space, legal_mask)
        if info.phase == Phase.AUCTION and (legal_mask[CONTRA_IDX] or legal_mask[RECONTRA_IDX]):
            self.chances += 1
            self.taken += int(action in (CONTRA_IDX, RECONTRA_IDX))
        return action


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
    counter = _DoublingCounter(agent)
    for seed, dealer in zip(seeds, dealers, strict=True):
        for agent_team in (0, 1):
            seats = {s: (counter if s % 2 == agent_team else opponent) for s in range(4)}
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
        contra_rate=counter.taken / counter.chances if counter.chances else None,
    )
