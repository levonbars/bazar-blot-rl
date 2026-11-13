"""What has been announced about combinations so far in a deal (rules §6.2, §6.5).

The engine runs with `staging.declarations_are_actions = False`: nobody chooses whether to
announce, show or bluff, every combination is claimed truthfully. Scoring does not depend on
timing (`declarations.resolve_combinations` over the dealt hands), but what the table KNOWS does,
and this module gives that timeline, derived only from the deal's play history:

- **Hand 1** — before playing their card to trick 1, each seat holding a comparable combination
  announces its class: `tierce`, `fifty`, `hundred` or `4x` (its best combination; no suit,
  rank or cards).
- **Hand 2** — before playing their card to trick 2, each seat of the team that won the
  comparison shows its comparable combinations (the actual cards). The losing team has nothing
  to collect and shows nothing.
- **Blot / Reblot** — the holder of the trump K+Q says `blot` when playing the first of the two
  and `reblot` when playing the second. It is outside the comparison (§6.2) and is revealed only
  as the cards are played.

Each announcement is made together with the card it precedes, so it becomes public when that
card is played. Questioning (§6.5 stage 1b) is not modelled: with truthful auto-announcing it is
settled by the hand-2 show.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from bazarblot.core.auction import opener_seat
from bazarblot.core.cards import TEAM_OF
from bazarblot.core.declarations import Meld, detect_all_melds, resolve_combinations_from_melds

if TYPE_CHECKING:
    from bazarblot.core.deal import Deal

AnnounceKind = Literal["tierce", "fifty", "hundred", "4x", "show", "blot", "reblot"]

_SEQUENCE_CLASS: dict[int, AnnounceKind] = {3: "tierce", 4: "fifty"}  # 5+ -> "hundred"


@dataclass(frozen=True, slots=True)
class Announcement:
    seat: int
    trick: int  # 0-based index of the trick whose card it accompanied
    kind: AnnounceKind
    melds: tuple[Meld, ...] = ()  # "show": the combinations shown
    card: int | None = None  # "blot"/"reblot": the trump K or Q it was said with


def announced_class(melds: tuple[Meld, ...] | list[Meld]) -> AnnounceKind | None:
    """The class a seat announces at hand 1: its best comparable combination, or `None`."""
    comparable = [m for m in melds if m.kind != "blot_reblot"]
    if any(m.kind == "carre" for m in comparable):
        return "4x"  # any carre beats any sequence (§6.4 step 1)
    lengths = [m.length for m in comparable if m.kind == "sequence"]
    if not lengths:
        return None
    return _SEQUENCE_CLASS.get(max(lengths), "hundred")


def _plays(deal: Deal) -> Iterator[tuple[int, int, int]]:
    """(trick index, seat, card) for every card played so far, in order."""
    for t, trick in enumerate(deal.tricks):
        for seat, card in trick.plays:
            yield t, seat, card
    for seat, card in deal.current_trick:
        yield len(deal.tricks), seat, card


def deal_melds(deal: Deal) -> tuple[tuple[Meld, ...], int | None]:
    """Every meld dealt (all seats, under the final contract) and the team that wins the
    comparison. Fixed for the whole play phase, so callers may cache it per deal."""
    assert deal.contract is not None
    rules = deal.rules
    melds = detect_all_melds(deal.original_hands, deal.contract.contract_type, rules)
    winning_team = resolve_combinations_from_melds(
        melds, rules, opener_seat(deal.dealer, rules)
    ).winning_team
    return tuple(melds), winning_team


def announcements(
    deal: Deal, cached: tuple[tuple[Meld, ...], int | None] | None = None
) -> list[Announcement]:
    """Every combination announcement made so far, in the order made. Empty before play.
    `cached` is `deal_melds(deal)`, when the caller keeps it."""
    contract = deal.contract
    if contract is None or not (deal.tricks or deal.current_trick):
        return []
    melds, winning_team = cached if cached is not None else deal_melds(deal)
    by_seat = [[m for m in melds if m.owner_seat == s] for s in range(4)]
    blot_cards = {
        m.owner_seat: set(m.cards) for m in melds if m.kind == "blot_reblot"
    }  # seat -> its trump K and Q

    out: list[Announcement] = []
    played_blot: set[int] = set()
    for t, seat, card in _plays(deal):
        if t == 0:
            cls = announced_class(by_seat[seat])
            if cls is not None:
                out.append(Announcement(seat, t, cls))
        if t == 1 and TEAM_OF[seat] == winning_team:
            shown = tuple(m for m in by_seat[seat] if m.kind != "blot_reblot")
            if shown:
                out.append(Announcement(seat, t, "show", melds=shown))
        if card in blot_cards.get(seat, ()):
            kind: AnnounceKind = "reblot" if seat in played_blot else "blot"
            played_blot.add(seat)
            out.append(Announcement(seat, t, kind, card=card))
    return out
