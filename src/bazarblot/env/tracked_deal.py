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
from bazarblot.core.declarations import Meld
from bazarblot.core.play import PlayCardAction


@dataclass(frozen=True, slots=True)
class AuctionEvent:
    seat: int
    action: AuctionAction


@dataclass(slots=True)
class TrackedDeal:
    deal: Deal
    auction_log: list[AuctionEvent] = field(default_factory=list)
    melds_cache: tuple[tuple[Meld, ...], ...] | None = field(
        default=None, repr=False, compare=False
    )
    """Memoizes `env/infoset.py::info_set()`'s per-seat meld detection — `detect_all_melds` is
    a pure function of `deal.original_hands` and `deal.contract.contract_type`, both fixed for
    the entire PLAY phase once the auction closes, so recomputing it on every single decision (as
    `info_set()` used to) was pure waste — a profiling pass (M6) found it was the single largest
    cost in a full self-play step, ~45% of `encode()`+`legal_mask()`'s combined time. Populated
    lazily by `info_set()` the first time it's needed for this deal, not eagerly here — `Deal`
    doesn't know its own contract yet at `TrackedDeal.__init__` time (the auction hasn't run)."""

    def step(self, action: Action) -> None:
        if self.deal.phase == Phase.AUCTION and not isinstance(action, PlayCardAction):
            self.auction_log.append(AuctionEvent(seat=self.deal.to_act, action=action))
        self.deal.step(action)
