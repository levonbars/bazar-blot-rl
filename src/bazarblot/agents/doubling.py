"""`RuleDoubler`: wraps any agent and adds one hand-written behaviour, doubling (contra) an
opponent's standing bid when "the opponents can't reach their number".

The rule: estimate what my side could make under the bidder's contract with the same hand
estimator the heuristic bids with (`agents/heuristic.py`), so the bidder's side is left with about
`16 - mine` scaled points (the deal holds 162 raw, ~16 scaled); double when that is still below the
bid even after `slack` extra levels of doubt. Larger `slack` means doubling only when I am more
sure.

This is a PROBE and a possible teacher, not the learned behaviour: it measures how much doubling is
worth (`cli/doubling_probe.py`) and could supply labels to bootstrap a policy that never tries it.
The trained agents in this project never double (the heuristic they were cloned from never did),
and this rule is what showed the action is worth a lot against over-bidders.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from bazarblot.agents.base import Agent
from bazarblot.agents.heuristic import _estimate_scaled_value
from bazarblot.core.deal import Phase
from bazarblot.core.rules import RuleConfig
from bazarblot.env.actions import CONTRA_IDX, ActionSpace
from bazarblot.env.infoset import InfoSet

_DEAL_SCALED_POINTS = 16  # 162 raw points on the table, in bid units


class RuleDoubler:
    def __init__(self, inner: Agent, rules: RuleConfig, slack: int = 3) -> None:
        self.inner = inner
        self.rules = rules
        self.slack = slack

    def act(self, info: InfoSet, space: ActionSpace, legal_mask: npt.NDArray[np.bool_]) -> int:
        if info.phase == Phase.AUCTION and legal_mask[CONTRA_IDX]:
            standing = info.auction_state.standing_bid
            assert standing is not None  # contra is only legal against a standing bid
            mine = _estimate_scaled_value(info.hand, standing.contract_type, self.rules)
            if (_DEAL_SCALED_POINTS - mine) + self.slack < standing.level:
                return CONTRA_IDX
        return self.inner.act(info, space, legal_mask)
