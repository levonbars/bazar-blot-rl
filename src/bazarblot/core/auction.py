"""The auction (*bazar*). Rules §4.

A bid is a triple `(level, contract_type, capot)` — see rules §3.4. `AuctionState` is treated
as immutable: `apply` returns a new state rather than mutating in place, which keeps replay and
search reuse straightforward later.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal

from bazarblot.core.cards import TEAM_OF, ContractType

if TYPE_CHECKING:
    from bazarblot.core.rules import RuleConfig

Doubling = Literal["none", "contra", "recontra"]


class AuctionError(ValueError):
    """An illegal auction action was attempted, or a safety net fired."""


# ---------------------------------------------------------------- actions


@dataclass(frozen=True, slots=True)
class PassAction:
    pass


@dataclass(frozen=True, slots=True)
class BidAction:
    level: int
    contract_type: ContractType
    capot: bool = False


@dataclass(frozen=True, slots=True)
class ContraAction:
    pass


@dataclass(frozen=True, slots=True)
class RecontraAction:
    pass


AuctionAction = PassAction | BidAction | ContraAction | RecontraAction


# ---------------------------------------------------------------- state


@dataclass(frozen=True, slots=True)
class Bid:
    seat: int
    level: int
    contract_type: ContractType
    capot: bool


@dataclass(frozen=True, slots=True)
class Contract:
    level: int
    contract_type: ContractType
    capot: bool
    declarer_seat: int
    attacking_team: int
    doubling: Doubling


@dataclass(frozen=True, slots=True)
class AuctionState:
    dealer: int
    to_act: int
    standing_bid: Bid | None = None
    doubling: Doubling = "none"
    consecutive_passes: int = 0
    steps: int = 0
    finished: bool = False
    aborted: bool = False


def opener_seat(dealer: int, rules: RuleConfig) -> int:
    """The player who opens the auction — and, per §2, also leads trick 1."""
    del rules  # `opener_and_leader` is currently always "dealer_left"; kept for config parity.
    return (dealer + 1) % 4


def new_auction(dealer: int, rules: RuleConfig) -> AuctionState:
    return AuctionState(dealer=dealer, to_act=opener_seat(dealer, rules))


# ---------------------------------------------------------------- legality


def is_legal(state: AuctionState, action: AuctionAction, rules: RuleConfig) -> bool:
    if state.finished:
        return False

    # Contra opens a forced two-ply reply window (§4.3): once `doubling == "contra"` and the
    # auction is not yet finished, we are IN that window. The only two legal replies belong
    # to the declarer alone (never their partner, never the defenders) — accept via PASS, or
    # RECONTRA. This check must come before the general PassAction branch below, or an
    # ordinary pass would incorrectly be allowed from the wrong seat during the window.
    if state.doubling == "contra":
        assert state.standing_bid is not None  # only reachable via a just-applied ContraAction
        if state.to_act != state.standing_bid.seat:
            return False
        return isinstance(action, PassAction | RecontraAction)

    if isinstance(action, PassAction):
        return True

    if isinstance(action, BidAction):
        if action.contract_type not in rules.contracts.types:
            return False
        if not (rules.auction.min_bid <= action.level <= rules.auction.max_bid):
            return False
        if state.standing_bid is not None and action.level <= state.standing_bid.level:
            return False
        # Sticky upward (§4.4): once capot is bid, every higher bid must also be capot.
        return not (
            state.standing_bid is not None and state.standing_bid.capot and not action.capot
        )

    if isinstance(action, ContraAction):
        if state.standing_bid is None:
            return False
        return TEAM_OF[state.to_act] != TEAM_OF[state.standing_bid.seat]

    return False  # a bare RecontraAction outside the contra reply window is never legal


# ---------------------------------------------------------------- transition


def apply(state: AuctionState, action: AuctionAction, rules: RuleConfig) -> AuctionState:
    if not is_legal(state, action, rules):
        raise AuctionError(f"illegal auction action {action!r} in state {state!r}")

    steps = state.steps + 1
    if steps > rules.auction.max_auction_steps:
        # Passing is non-binding, so nothing in `is_legal` alone bounds the auction's length —
        # this is the safety net. It should never fire under sane raising policies: the ladder
        # is finite (`max_bid`), so a normal auction converges long before this trips.
        raise AuctionError(
            f"max_auction_steps ({rules.auction.max_auction_steps}) exceeded — "
            "safety net triggered, this indicates a policy that never converges"
        )

    # Resolving the contra reply window (§4.3): PASS here means "accept the contra", not an
    # ordinary auction pass — it does not touch `consecutive_passes` and it always finishes.
    if state.doubling == "contra":
        if isinstance(action, RecontraAction):
            return replace(state, doubling="recontra", finished=True, steps=steps)
        return replace(state, finished=True, steps=steps)  # PASS: contra stands

    if isinstance(action, PassAction):
        consecutive = state.consecutive_passes + 1
        aborted = state.standing_bid is None and consecutive == 4
        threshold = 3 if rules.auction.three_passes_after_bid == "close" else 4
        finished = aborted or (state.standing_bid is not None and consecutive == threshold)
        return replace(
            state,
            to_act=(state.to_act + 1) % 4,
            consecutive_passes=consecutive,
            finished=finished,
            aborted=aborted,
            steps=steps,
        )

    if isinstance(action, BidAction):
        bid = Bid(
            seat=state.to_act,
            level=action.level,
            contract_type=action.contract_type,
            capot=action.capot,
        )
        return replace(
            state,
            to_act=(state.to_act + 1) % 4,
            standing_bid=bid,
            consecutive_passes=0,
            steps=steps,
        )

    if isinstance(action, ContraAction):
        # Opens the reply window (§4.3): hands the turn to the declarer specifically, with
        # exactly two legal replies. Not finished yet — `is_legal` routes subsequent actions
        # through the `doubling == "contra"` branch above until the declarer replies.
        assert state.standing_bid is not None  # guaranteed by is_legal
        return replace(state, doubling="contra", to_act=state.standing_bid.seat, steps=steps)

    raise AssertionError(f"unreachable: unknown action type {action!r}")  # pragma: no cover


def final_contract(state: AuctionState) -> Contract | None:
    """The settled contract, or `None` if the auction is still open, aborted, or never bid."""
    if not state.finished or state.aborted or state.standing_bid is None:
        return None
    b = state.standing_bid
    return Contract(
        level=b.level,
        contract_type=b.contract_type,
        capot=b.capot,
        declarer_seat=b.seat,
        attacking_team=TEAM_OF[b.seat],
        doubling=state.doubling,
    )
