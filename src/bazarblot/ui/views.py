"""The information-hiding boundary between the engine and the browser.

`player_view()` and `full_state_view()` are deliberately two separate functions with two
separate return types, not one function with a "reveal everything" flag — a single function
guarded by a boolean is one wrong call away from leaking all four hands to a human `play`
session. `play` mode may only ever call `player_view()`; `watch` and `replay` modes may only ever
call `full_state_view()`. There is no third path.

This is a UI-layer stand-in for the formal `InfoSet` the M3 environment layer will define —
narrower in scope (no observation encoding, no augmentation), but the same boundary. When M3
lands, this should be reconciled with it rather than left as a second, independently-maintained
notion of "what can seat N see."
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from bazarblot.core.auction import (
    BidAction,
    ContraAction,
    PassAction,
    RecontraAction,
    is_legal,
)
from bazarblot.core.cards import card_label
from bazarblot.core.deal import Deal, Phase
from bazarblot.core.play import PlayCardAction

BidKind = Literal["pass", "bid", "contra", "recontra"]


@dataclass(frozen=True, slots=True)
class PublicBid:
    seat: int
    kind: BidKind
    level: int | None = None
    contract_type: str | None = None
    capot: bool | None = None


@dataclass(frozen=True, slots=True)
class ContractView:
    level: int
    contract_type: str
    capot: bool
    declarer_seat: int
    attacking_team: int
    doubling: str


@dataclass(frozen=True, slots=True)
class TrickView:
    leader: int
    plays: tuple[tuple[int, int], ...]  # (seat, card_id)
    winner: int
    points: int


@dataclass(frozen=True, slots=True)
class ResultView:
    attackers_team: int
    made: bool
    cards_attackers: int
    cards_defenders: int
    combo_attackers: int
    combo_defenders: int
    raw_attackers: int
    raw_defenders: int
    score_attackers: int
    score_defenders: int
    attackers_took_all_tricks: bool
    defenders_took_all_tricks: bool


def _result_view(d: Deal) -> ResultView | None:
    if d.result is None:
        return None
    r = d.result
    return ResultView(
        attackers_team=r.attackers_team,
        made=r.made,
        cards_attackers=r.cards_attackers,
        cards_defenders=r.cards_defenders,
        combo_attackers=r.combo_attackers,
        combo_defenders=r.combo_defenders,
        raw_attackers=r.raw_attackers,
        raw_defenders=r.raw_defenders,
        score_attackers=r.score_attackers,
        score_defenders=r.score_defenders,
        attackers_took_all_tricks=r.attackers_took_all_tricks,
        defenders_took_all_tricks=r.defenders_took_all_tricks,
    )


def _contract_view(d: Deal) -> ContractView | None:
    if d.contract is None:
        return None
    c = d.contract
    return ContractView(
        level=c.level,
        contract_type=c.contract_type,
        capot=c.capot,
        declarer_seat=c.declarer_seat,
        attacking_team=c.attacking_team,
        doubling=c.doubling,
    )


def _tricks_view(d: Deal) -> tuple[TrickView, ...]:
    return tuple(
        TrickView(leader=t.leader, plays=t.plays, winner=t.winner, points=t.points)
        for t in d.tricks
    )


def _legal_action_summaries(d: Deal, seat: int) -> list[dict[str, Any]]:
    """A compact description of what `seat` may currently do — enough for the frontend to
    build a form, without materializing all ~700 bid actions. The server re-validates every
    submitted action via `core.auction.is_legal` / `Deal.legal_actions()` regardless; this is
    a hint, never the authority.
    """
    if d.phase == Phase.PLAY:
        if d.to_act != seat:
            return []
        return [
            {"type": "play", "card": a.card, "label": card_label(a.card)}
            for a in d.legal_actions()
            if isinstance(a, PlayCardAction)
        ]

    if d.phase != Phase.AUCTION or d.to_act != seat:
        return []

    state = d.auction_state
    out: list[dict[str, Any]] = []
    if is_legal(state, PassAction(), d.rules):
        out.append({"type": "pass"})
    if is_legal(state, ContraAction(), d.rules):
        out.append({"type": "contra"})
    if is_legal(state, RecontraAction(), d.rules):
        out.append({"type": "recontra"})

    # Bidding is only possible outside the contra reply window; summarize the legal range
    # rather than enumerating every (level, type, capot) triple.
    if state.doubling != "contra":
        min_level = (
            (state.standing_bid.level + 1) if state.standing_bid else d.rules.auction.min_bid
        )
        forced_capot = bool(state.standing_bid and state.standing_bid.capot)
        out.append(
            {
                "type": "bid",
                "min_level": min_level,
                "max_level": d.rules.auction.max_bid,
                "contract_types": list(d.rules.contracts.types),
                "forced_capot": forced_capot,
            }
        )
    return out


@dataclass(frozen=True, slots=True)
class PlayerView:
    """What ONE seat is allowed to see. Never construct this with another seat's hand."""

    seat: int
    dealer: int
    deal_id: int
    deal_number: int
    phase: str
    to_act: int | None
    my_hand: tuple[int, ...]
    hand_sizes: tuple[int, int, int, int]
    contract: ContractView | None
    current_trick: tuple[tuple[int, int], ...]
    completed_tricks: tuple[TrickView, ...]
    result: ResultView | None
    match_score: tuple[int, int]
    match_target: int
    legal_actions: list[dict[str, Any]] = field(default_factory=list)


def player_view(d: Deal, seat: int, match_score: tuple[int, int], match_target: int) -> PlayerView:
    """The ONLY function `play` mode may call. Deliberately takes `seat` as a required,
    non-optional int — there is no "give me everything" call shape available here.
    """
    to_act = d.to_act if d.phase in (Phase.AUCTION, Phase.PLAY) else None
    return PlayerView(
        seat=seat,
        dealer=d.dealer,
        deal_id=d.deal_id,
        deal_number=d.deal_id,
        phase=d.phase.name,
        to_act=to_act,
        my_hand=tuple(sorted(d.hands[seat])),
        hand_sizes=(len(d.hands[0]), len(d.hands[1]), len(d.hands[2]), len(d.hands[3])),
        contract=_contract_view(d),
        current_trick=tuple(d.current_trick),
        completed_tricks=_tricks_view(d),
        result=_result_view(d),
        match_score=match_score,
        match_target=match_target,
        legal_actions=_legal_action_summaries(d, seat) if to_act == seat else [],
    )


@dataclass(frozen=True, slots=True)
class FullStateView:
    """The debug/spectator view: all four hands. `watch` and `replay` only — `play` must
    never reach this."""

    dealer: int
    deal_id: int
    deal_number: int
    phase: str
    to_act: int | None
    hands: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]
    contract: ContractView | None
    current_trick: tuple[tuple[int, int], ...]
    completed_tricks: tuple[TrickView, ...]
    result: ResultView | None
    match_score: tuple[int, int]
    match_target: int
    legal_actions: list[dict[str, Any]] = field(default_factory=list)


def full_state_view(d: Deal, match_score: tuple[int, int], match_target: int) -> FullStateView:
    """The ONLY function `watch`/`replay` mode may call. Not reachable from `player_view`."""
    to_act = d.to_act if d.phase in (Phase.AUCTION, Phase.PLAY) else None
    hands = tuple(tuple(sorted(h)) for h in d.hands)
    return FullStateView(
        dealer=d.dealer,
        deal_id=d.deal_id,
        deal_number=d.deal_id,
        phase=d.phase.name,
        to_act=to_act,
        hands=hands,  # type: ignore[arg-type]
        contract=_contract_view(d),
        current_trick=tuple(d.current_trick),
        completed_tricks=_tricks_view(d),
        result=_result_view(d),
        match_score=match_score,
        match_target=match_target,
        legal_actions=_legal_action_summaries(d, to_act) if to_act is not None else [],
    )


# ---------------------------------------------------------------- action decoding


class ActionDecodeError(ValueError):
    pass


def decode_action(
    payload: dict[str, Any],
) -> PassAction | BidAction | ContraAction | RecontraAction | PlayCardAction:
    """Turn a JSON action payload from the browser into an engine action. The engine's own
    `is_legal`/`legal_actions` remain the authority — this only parses shape."""
    kind = payload.get("type")
    if kind == "pass":
        return PassAction()
    if kind == "contra":
        return ContraAction()
    if kind == "recontra":
        return RecontraAction()
    if kind == "bid":
        try:
            return BidAction(
                level=int(payload["level"]),
                contract_type=payload["contract_type"],
                capot=bool(payload.get("capot", False)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ActionDecodeError(f"malformed bid payload: {payload}") from exc
    if kind == "play":
        try:
            return PlayCardAction(card=int(payload["card"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ActionDecodeError(f"malformed play payload: {payload}") from exc
    raise ActionDecodeError(f"unknown action type: {kind!r}")


__all__ = [
    "ContractView",
    "FullStateView",
    "PlayerView",
    "PublicBid",
    "ResultView",
    "TrickView",
    "decode_action",
    "full_state_view",
    "player_view",
]
