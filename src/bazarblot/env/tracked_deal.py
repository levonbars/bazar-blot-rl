"""`TrackedDeal`: a `core.deal.Deal` plus the one piece of public history `Deal` itself doesn't
keep — the full auction action log with seats.

`Deal`/`AuctionState` deliberately track only the *current* standing bid, doubling state and
pass count, not the sequence of actions that produced them (the same economy `core/` applies
everywhere: store what the rules need to decide legality, not what a UI or observation encoder
might want later). The environment spec's `DealState.bid_history` needs that sequence for the
`bid_summary`/`bid_recent` observation blocks, so this thin wrapper records it as the deal is
driven, without touching `core/` at all — every field it adds is already public (seat + action),
nothing that isn't visible to all four players anyway.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from bazarblot.core.auction import AuctionAction
from bazarblot.core.deal import Action, Deal, Phase
from bazarblot.core.play import PlayCardAction


@dataclass(frozen=True, slots=True)
class AuctionEvent:
    seat: int
    action: AuctionAction


@dataclass(slots=True)
class TrackedDeal:
    deal: Deal
    auction_log: list[AuctionEvent] = field(default_factory=list)

    def step(self, action: Action) -> None:
        if self.deal.phase == Phase.AUCTION and not isinstance(action, PlayCardAction):
            self.auction_log.append(AuctionEvent(seat=self.deal.to_act, action=action))
        self.deal.step(action)
