"""`PassBidder`: passes at every auction decision, never bids. A standard reference opponent, used
inside a `ComposedAgent` with a real card player.

It exists because "beats the baseline" can be true for the wrong reason: against a bidder that
over-bids and fails its contracts, simply conceding every auction and defending earns a large
positive margin without any bidding skill (that is how a never-bidding agent scored +66 against
the heuristic before the heuristic's bidding bug was found, `docs/03-implementation-roadmap.md`
M7). Any claim of bidding skill should be checked against this opponent, and a healthy baseline
bidder should beat it.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from bazarblot.core.deal import Phase
from bazarblot.env.actions import PASS_IDX, ActionSpace
from bazarblot.env.infoset import InfoSet


class PassBidder:
    """Auction-only agent: compose it with a card player (`ComposedAgent(PassBidder(), player)`)."""

    def act(self, info: InfoSet, space: ActionSpace, legal_mask: npt.NDArray[np.bool_]) -> int:
        del space
        if info.phase != Phase.AUCTION:
            raise ValueError("PassBidder only makes auction decisions; compose it with a player")
        return PASS_IDX if legal_mask[PASS_IDX] else int(legal_mask.nonzero()[0][0])
