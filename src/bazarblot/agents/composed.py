"""`ComposedAgent`: one `Agent` whose auction decisions come from `bidder` and whose card play
comes from `player`. This is the building block for the two-phase experiments (roadmap M8): a
learned player under a fixed heuristic bidder isolates play skill from bidding, and a learned
bidder over a fixed player does the reverse -- without either phase's quality hiding behind the
other's, which is exactly what went wrong when joint self-play was judged by a single margin."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from bazarblot.agents.base import Agent
from bazarblot.core.deal import Phase
from bazarblot.env.actions import ActionSpace
from bazarblot.env.infoset import InfoSet


class ComposedAgent:
    def __init__(self, bidder: Agent, player: Agent) -> None:
        self.bidder = bidder
        self.player = player

    def act(self, info: InfoSet, space: ActionSpace, legal_mask: npt.NDArray[np.bool_]) -> int:
        if info.phase == Phase.AUCTION:
            return self.bidder.act(info, space, legal_mask)
        return self.player.act(info, space, legal_mask)
