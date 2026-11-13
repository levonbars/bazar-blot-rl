"""A standard suit order for every decision, so what the network learns about one suit applies to
all four.

The rules never favour a suit: relabeling the suits of a whole deal consistently (hands, bids,
the contract, every played card) gives an equivalent deal. Random deals already show every suit
equally often, but nothing stops a network from learning, say, clubs and hearts differently (the
long-baseline seeds each over-bid one arbitrary suit and did worst in it). Here every `InfoSet` is
relabeled into a standard order before the network sees it:

- slot 0 is the trump suit once a trump contract exists;
- the other suits follow by the deciding seat's own original holding: more cards first, then
  stronger cards (sum of natural rank indices), then the original suit order as a last tie-break.

The order depends only on the deciding seat's own hand and the public contract, so it leaks
nothing. The network chooses in the relabeled frame; `to_env_action` maps the choice back.
"""

from __future__ import annotations

import dataclasses
from functools import lru_cache

import numpy as np
import numpy.typing as npt

from bazarblot.core.announcements import Announcement
from bazarblot.core.auction import Bid, BidAction, Contract
from bazarblot.core.cards import SUIT_INDEX, SUITS, ContractType, build_tables
from bazarblot.core.declarations import Meld
from bazarblot.env.actions import ActionSpace
from bazarblot.env.augment import SuitPermutation, invert, permute_card, permute_contract_type
from bazarblot.env.infoset import InfoSet
from bazarblot.env.tracked_deal import AuctionEvent


def canonical_permutation(info: InfoSet) -> SuitPermutation:
    """`perm[old_suit_index] = slot` for this decision."""
    trump = (
        info.contract.contract_type
        if info.contract is not None and info.contract.contract_type != "NT"
        else None
    )

    def key(si: int) -> tuple[int, int, int, int]:
        suit = SUITS[si]
        held = [c % 8 for c in info.original_hand if c // 8 == si]
        return (int(suit == trump), len(held), sum(held), -si)

    order = sorted(range(4), key=key, reverse=True)  # order[slot] = old suit index
    perm = [0, 0, 0, 0]
    for slot, si in enumerate(order):
        perm[si] = slot
    return (perm[0], perm[1], perm[2], perm[3])


def _cards(cards: frozenset[int], perm: SuitPermutation) -> frozenset[int]:
    return frozenset(permute_card(c, perm) for c in cards)


Plays = tuple[tuple[int, int], ...]


def _plays(plays: Plays, perm: SuitPermutation) -> Plays:
    return tuple((s, permute_card(c, perm)) for s, c in plays)


def _type(t: ContractType, perm: SuitPermutation) -> ContractType:
    return permute_contract_type(t, perm)


def _meld(m: Meld, perm: SuitPermutation) -> Meld:
    suit = SUITS[perm[SUIT_INDEX[m.suit]]] if m.suit is not None else None
    return dataclasses.replace(m, suit=suit, cards=tuple(permute_card(c, perm) for c in m.cards))


def _announcement(a: Announcement, perm: SuitPermutation) -> Announcement:
    return dataclasses.replace(
        a,
        melds=tuple(_meld(m, perm) for m in a.melds),
        card=permute_card(a.card, perm) if a.card is not None else None,
    )


def _event(e: AuctionEvent, perm: SuitPermutation) -> AuctionEvent:
    if isinstance(e.action, BidAction):
        return AuctionEvent(
            e.seat, dataclasses.replace(e.action, contract_type=_type(e.action.contract_type, perm))
        )
    return e


def permute_infoset(info: InfoSet, perm: SuitPermutation) -> InfoSet:
    """`info` with every suit relabeled by `perm` (hands, bids, contract, tables, tricks, melds,
    announcements). The deal result is left as is: no decision is made after the deal ends."""
    if perm == (0, 1, 2, 3):
        return info
    state = info.auction_state
    standing: Bid | None = state.standing_bid
    if standing is not None:
        standing = dataclasses.replace(standing, contract_type=_type(standing.contract_type, perm))
    contract: Contract | None = info.contract
    tables = info.tables
    if contract is not None:
        contract = dataclasses.replace(contract, contract_type=_type(contract.contract_type, perm))
        tables = build_tables(contract.contract_type, info.rules)
    return dataclasses.replace(
        info,
        hand=_cards(info.hand, perm),
        original_hand=_cards(info.original_hand, perm),
        auction_log=tuple(_event(e, perm) for e in info.auction_log),
        auction_state=dataclasses.replace(state, standing_bid=standing),
        contract=contract,
        tables=tables,
        tricks=tuple(_plays(t, perm) for t in info.tricks),
        current_trick=_plays(info.current_trick, perm),
        own_melds=tuple(_meld(m, perm) for m in info.own_melds)
        if info.own_melds is not None
        else None,
        announcements=tuple(_announcement(a, perm) for a in info.announcements),
    )


@lru_cache(maxsize=64)
def _index_map(space: ActionSpace, perm: SuitPermutation) -> npt.NDArray[np.int64]:
    """`m[env_index] = relabeled_index` over the whole action space."""
    m = np.arange(space.action_dim, dtype=np.int64)
    for idx in range(space.action_dim):
        action = space.decode(idx)
        if idx >= space.play_start:
            m[idx] = space.play_start + permute_card(idx - space.play_start, perm)
        elif isinstance(action, BidAction):
            m[idx] = space.bid_index(action.level, _type(action.contract_type, perm), action.capot)
    return m


def to_canonical_mask(
    mask: npt.NDArray[np.bool_], space: ActionSpace, perm: SuitPermutation
) -> npt.NDArray[np.bool_]:
    out = np.zeros_like(mask)
    out[_index_map(space, perm)] = mask
    return out


def to_canonical_action(idx: int, space: ActionSpace, perm: SuitPermutation) -> int:
    """A real action index -> the relabeled-frame index."""
    return int(_index_map(space, perm)[idx])


def to_env_action(idx: int, space: ActionSpace, perm: SuitPermutation) -> int:
    """A relabeled-frame action index -> the real action index."""
    return int(_index_map(space, invert(perm))[idx])


def canonical_belief_target(
    target: npt.NDArray[np.int8], perm: SuitPermutation
) -> npt.NDArray[np.int8]:
    """A per-card label array (`learn/belief.py::belief_target`) in the relabeled frame."""
    out = np.empty_like(target)
    for c in range(len(target)):
        out[permute_card(c, perm)] = target[c]
    return out
