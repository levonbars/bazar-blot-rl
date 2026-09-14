"""Translates `env/actions.py`'s flat bid segment into the `(Δ, type, capot)` factored
representation the policy network's bid head samples over (environment spec §3.2), and back to a
flat action index. Pure numpy, no torch -- usable and testable without the `learn` extra.

**Why factor at all:** the flat bid segment is `n_levels x n_types x capot_states` (730 for the
shipped preset) actions for what is really three small, mostly-independent choices. Encoding the
level as a *raise* `Delta` over the current standing bid (rather than an absolute level) lets the
same parameters fire whether the standing bid is 12 or 62, and factoring the head into three
conditional sub-heads makes the cost additive (`n_delta + n_types + capot_states`) instead of
multiplicative. See the spec section for the full argument; this module only implements the
index/legality bookkeeping the network needs to sample and score under that factoring.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from bazarblot.env.actions import ActionSpace
from bazarblot.env.infoset import InfoSet

BoolArray = npt.NDArray[np.bool_]


def base_level(info: InfoSet) -> int:
    """The level a `Delta = 0` raise would refer to: the standing bid's level, or `min_bid - 1`
    for an opening bid (spec §3.2) -- so `Delta = level - base_level` is always `>= 1` for any
    legal bid, opening included, and the opening bid needs no special-cased head."""
    standing = info.auction_state.standing_bid
    return standing.level if standing is not None else info.rules.auction.min_bid - 1


def n_delta_slots(space: ActionSpace) -> int:
    """The number of `Delta` head logits needed: `max_bid - (min_bid - 1)`.

    The spec's own worked arithmetic ("72 + 5 + 2 = 79 logits ... the largest possible raise is
    80 - 8 = 72") is one short: that measures the raise from `min_bid` itself, but the very
    paragraph before it fixes the opening bid's virtual base at `min_bid - 1`, not `min_bid`. An
    opening bid AT `max_bid` is legal and real (rules §4.2's own worked capot-plus-three-carrés
    hand reaches exactly `max_bid`), and needs `Delta = max_bid - (min_bid - 1)`, one more than
    the spec's stated bound. Sized correctly here (73 for the shipped preset) rather than
    reproducing the off-by-one and silently making the top of the ladder unopenable in one bid.
    """
    return space.max_bid - (space.min_bid - 1)


@dataclass(frozen=True, slots=True)
class BidLegality:
    """The bid segment of a `legal_mask` result, reshaped into `(Delta, type, capot)` and
    re-based from absolute level onto `Delta` relative to one decision's `base_level` -- exactly
    the shape the factored head's hierarchical sampling (`Delta`, then `type | Delta`, then
    `capot | Delta, type`) needs at each conditioning step."""

    delta_mask: BoolArray  # (n_delta,) -- legal at some (type, capot) for this Delta
    type_mask_given_delta: BoolArray  # (n_delta, n_types)
    capot_mask_given_delta_type: BoolArray  # (n_delta, n_types, capot_states)

    def any_bid_legal(self) -> bool:
        return bool(self.delta_mask.any())


def bid_legality(mask: BoolArray, space: ActionSpace, base: int) -> BidLegality:
    """Build a `BidLegality` from one decision's full flat `legal_mask` output and `base_level`.

    `ActionSpace.bid_index`'s flat layout is already `(level, type, capot)` in row-major order
    (`bid_start + (level-min_bid)*n_types*capot_states + type_idx*capot_states + capot_idx`), so
    reshaping the bid segment directly reproduces that cube with no reordering; only the level
    axis needs shifting from absolute level to `Delta = level - base`.
    """
    n_types = len(space.contract_types)
    n_levels = space.max_bid - space.min_bid + 1
    level_cube = mask[space.bid_start : space.play_start].reshape(
        n_levels, n_types, space.capot_states
    )
    n_delta = n_delta_slots(space)
    delta_cube = np.zeros((n_delta, n_types, space.capot_states), dtype=bool)
    for level in range(space.min_bid, space.max_bid + 1):
        delta = level - base
        if 1 <= delta <= n_delta:
            delta_cube[delta - 1] = level_cube[level - space.min_bid]
    type_mask = delta_cube.any(axis=2)
    delta_mask = type_mask.any(axis=1)
    return BidLegality(
        delta_mask=delta_mask,
        type_mask_given_delta=type_mask,
        capot_mask_given_delta_type=delta_cube,
    )


def flat_bid_index(space: ActionSpace, base: int, delta: int, type_idx: int, capot_idx: int) -> int:
    """Map one `(Delta, type_idx, capot_idx)` choice (as sampled by the factored head, `delta`
    1-indexed matching `BidLegality`'s `delta - 1` array convention) back to the flat engine
    action index `ActionSpace.decode` understands."""
    level = base + delta
    contract_type = space.contract_types[type_idx]
    capot = bool(capot_idx) if space.capot_states == 2 else False
    return space.bid_index(level, contract_type, capot)
