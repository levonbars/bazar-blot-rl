"""`encode(info_set) -> dict[str, np.ndarray]`: the tensor encoding of an `InfoSet` (environment
spec §4). `OBS_VERSION` is stamped into every checkpoint alongside `rules_hash`; bump it and
refuse to load a checkpoint encoded under a different version whenever this module's output
shape or semantics change — a silent version drift is a silent, undetectable training bug.

**Seat-relative throughout (§4.1).** Every per-seat block is indexed by `rel = (seat - my_seat)
% 4`, never by absolute seat: `0 = me, 1 = left opponent, 2 = partner, 3 = right opponent`. One
network then serves all four seats.

**Suit canonicalization is deliberately NOT done here (§4.2).** It's an augmentation applied to
the underlying game objects (a permutation of card ids) *before* `info_set()`/`encode()` ever
run — see `env/augment.py`. `encode()` just encodes whatever suits the cards it's given nominally
have; baking the permutation into the encoder would make it impossible to reuse the same encoder
for both the identity case and every augmented sample.

Every block's shape is fixed by config-derived constants (`MAX_BID`, `N_CARDS`, ...), computed
once at import time for the shipped preset — see `BLOCK_SHAPES`. A rules preset that changes
`max_bid` or the contract type list changes these shapes; `OBS_VERSION` does not encode the
preset, `rules_hash` does (via `InfoSet.rules_hash`), so a training run must check both.

Documented simplifications relative to the spec's prose (each is a real design choice under real
ambiguity, not an oversight — see the inline comment at each site):
- `own_combinations` and `melds_public` pack a small, fixed set of concrete features into their
  block rather than every conceivably-relevant quantity the spec's one-paragraph description
  could support; both leave explicit, documented zero-padding as headroom for `v2`.
- `voids` tracks the 3 non-"primary" suits only (primary = trump under a trump contract, or a
  fixed arbitrary suit under `NT`, where the concept doesn't really apply) — see `_primary_suit`.
- Combination-derived features (`own_combinations`, `melds_public`) are valid only under the
  shipped preset's `staging.declarations_are_actions = False`; see `env/infoset.py`.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from bazarblot.core.auction import BidAction, ContraAction, PassAction, RecontraAction
from bazarblot.core.cards import N_CARDS, RANKS, SUITS, TEAM_OF, ContractType, card_id, suit_of
from bazarblot.core.declarations import Meld, detect_hand_melds, resolve_combinations
from bazarblot.core.play import current_winner
from bazarblot.core.scoring import round10
from bazarblot.env.infoset import InfoSet
from bazarblot.env.tracked_deal import AuctionEvent

OBS_VERSION = "v1"

Array = npt.NDArray[np.float32]

_CONTRACT_TYPES_ALL: tuple[str, ...] = ("none", "C", "D", "H", "S", "NT")  # fixed, absolute
_LEVEL_ONEHOT_MIN = 8
_LEVEL_ONEHOT_MAX = 32  # spec §4.3: clipped one-hot for 8..32, scalar carries the rest
_LEVEL_ONEHOT_N = _LEVEL_ONEHOT_MAX - _LEVEL_ONEHOT_MIN + 1  # 25
_BID_RECENT_LEN = 8
_BID_KIND = ("pass", "bid", "contra", "recontra")

BLOCK_SHAPES: dict[str, tuple[int, ...]] = {
    "hand": (N_CARDS,),
    "card_state": (N_CARDS, 6),
    "current_trick": (N_CARDS, 4),
    "trick_leader": (4,),
    "trick_index": (8,),
    "contract_type": (6,),
    "contract_level": (_LEVEL_ONEHOT_N + 1,),
    "capot_flag": (1,),
    "declarer_rel": (4,),
    "doubling": (3,),
    "bid_payout": (1,),
    "bid_summary": (4, 10),
    "headroom": (2,),
    "own_combinations": (12,),
    "voids": (3, 4),
    "cards_left": (4,),
    "suit_unseen": (4,),
    # Spec §4.3 states this block as "4 x 25 = 100", but its own itemized channel list for one
    # seat — class one-hot(5) + top-card one-hot(9) + trump bit(2) + carre-rank one-hot(7) +
    # was_questioned/has_shown/announced_but_withheld/belote/rebelote(5x1) — sums to 28, not 25
    # (5+9+2+7+5=28). The itemized list is unambiguous and directly checkable; the "25"/"100"
    # summary figures are not, and the spec's own totals are already stated as approximate
    # ("≈480 floats") elsewhere. Trusting the itemization: (4, 28), not (4, 25).
    "melds_public": (4, 28),
    "running_points": (4,),
    "contract_progress": (5,),
    "match_state": (6,),
    "bid_recent": (_BID_RECENT_LEN, 16),  # auction phase only
}


def _rel(other_seat: int, my_seat: int) -> int:
    return (other_seat - my_seat) % 4


def _primary_suit(contract_type: str | None) -> str:
    """The suit `voids` treats as "not one of the 3 tracked" — trump under a trump contract,
    or a fixed arbitrary suit (`C`) under `NT`/no contract yet, where there is no real trump to
    exclude. Fixing it rather than tracking all 4 keeps the block at the spec's stated 3x4=12."""
    if contract_type is not None and contract_type != "NT":
        return contract_type
    return "C"


def _effective_contract(
    info: InfoSet,
) -> tuple[ContractType | None, int | None, bool, int | None, str]:
    """The contract relevant to THIS decision: the final `Contract` once the auction has closed,
    or the current standing bid while still bidding (needed for the auction net — a bidder must
    see what they'd be raising over). Returns (type, level, capot, declarer_seat, doubling)."""
    if info.contract is not None:
        c = info.contract
        return c.contract_type, c.level, c.capot, c.declarer_seat, c.doubling
    sb = info.auction_state.standing_bid
    if sb is not None:
        return sb.contract_type, sb.level, sb.capot, sb.seat, info.auction_state.doubling
    return None, None, False, None, info.auction_state.doubling


def _card_state_block(info: InfoSet) -> Array:
    out = np.zeros(BLOCK_SHAPES["card_state"], dtype=np.float32)
    played_by: dict[int, int] = {}
    for trick in info.tricks:
        for seat, c in trick:
            played_by[c] = seat
    for seat, c in info.current_trick:
        played_by[c] = seat
    for c in range(N_CARDS):
        if c in info.hand:
            out[c, 0] = 1.0
        elif c in played_by:
            out[c, 1 + _rel(played_by[c], info.seat)] = 1.0
        else:
            out[c, 5] = 1.0
    return out


def _current_trick_block(info: InfoSet) -> Array:
    out = np.zeros(BLOCK_SHAPES["current_trick"], dtype=np.float32)
    for seat, c in info.current_trick:
        out[c, _rel(seat, info.seat)] = 1.0
    return out


def _trick_leader_block(info: InfoSet) -> Array:
    out = np.zeros(BLOCK_SHAPES["trick_leader"], dtype=np.float32)
    if info.trick_leader is not None:
        out[_rel(info.trick_leader, info.seat)] = 1.0
    return out


def _trick_index_block(info: InfoSet) -> Array:
    out = np.zeros(BLOCK_SHAPES["trick_index"], dtype=np.float32)
    out[min(len(info.tricks), 7)] = 1.0
    return out


def _contract_type_block(contract_type: str | None) -> Array:
    out = np.zeros(BLOCK_SHAPES["contract_type"], dtype=np.float32)
    out[_CONTRACT_TYPES_ALL.index(contract_type if contract_type is not None else "none")] = 1.0
    return out


def _contract_level_block(level: int | None) -> Array:
    out = np.zeros(BLOCK_SHAPES["contract_level"], dtype=np.float32)
    if level is not None:
        clipped = max(_LEVEL_ONEHOT_MIN, min(_LEVEL_ONEHOT_MAX, level))
        out[clipped - _LEVEL_ONEHOT_MIN] = 1.0
        out[-1] = level / _LEVEL_ONEHOT_MAX
    return out


def _declarer_rel_block(declarer_seat: int | None, my_seat: int) -> Array:
    out = np.zeros(BLOCK_SHAPES["declarer_rel"], dtype=np.float32)
    if declarer_seat is not None:
        out[_rel(declarer_seat, my_seat)] = 1.0
    return out


def _doubling_block(doubling: str) -> Array:
    out = np.zeros(BLOCK_SHAPES["doubling"], dtype=np.float32)
    out[("none", "contra", "recontra").index(doubling)] = 1.0
    return out


def _bid_summary_block(info: InfoSet) -> Array:
    out = np.zeros(BLOCK_SHAPES["bid_summary"], dtype=np.float32)
    max_bid = info.rules.auction.max_bid
    types_with_none = ("none", *info.rules.contracts.types)
    for ev in info.auction_log:
        rel = _rel(ev.seat, info.seat)
        if isinstance(ev.action, PassAction):
            out[rel, 0] = 1.0
        elif isinstance(ev.action, BidAction):
            out[rel, 1] = min(1.0, out[rel, 1] + 0.1)  # n_bids, saturating count/10
            out[rel, 2] = ev.action.level / max_bid
            out[rel, 3] = float(ev.action.capot)
            type_onehot = np.zeros(6, dtype=np.float32)
            type_onehot[types_with_none.index(ev.action.contract_type)] = 1.0
            out[rel, 4:10] = type_onehot
    return out


def _headroom_block(info: InfoSet, level: int | None) -> Array:
    max_bid = info.rules.auction.max_bid
    standing = level if level is not None else info.rules.auction.min_bid - 1  # spec §3.2's
    # virtual standing level for "no bid yet"
    return np.array([(max_bid - standing) / max(1, max_bid - 1), float(standing)], dtype=np.float32)


def _bid_payout_block(
    info: InfoSet, contract_type: ContractType | None, level: int | None, doubling: str
) -> Array:
    if contract_type is None or level is None:
        return np.zeros(BLOCK_SHAPES["bid_payout"], dtype=np.float32)
    m = info.rules.scoring.multiplier(contract_type, doubling)  # type: ignore[arg-type]
    return np.array([m * level / 32], dtype=np.float32)


def _own_melds(info: InfoSet, contract_type: ContractType) -> list[Meld]:
    return detect_hand_melds(info.original_hand, info.seat, contract_type, info.rules)


def _own_combinations_block(info: InfoSet) -> Array:
    out = np.zeros(BLOCK_SHAPES["own_combinations"], dtype=np.float32)
    contract_type, _level, _capot, _declarer, _doubling = _effective_contract(info)
    # Sequence value doesn't depend on contract type at all (only its is_trump flag, which
    # affects precedence, not value) — but carre/blot-reblot values do. Before any bid exists
    # there is no real contract type to evaluate carre/blot-reblot against; NT is used as a
    # documented, arbitrary stand-in for that one rare edge (the opening bidder's very first
    # decision) rather than leaving the whole block zero.
    melds = _own_melds(info, contract_type if contract_type is not None else "NT")
    seqs = sorted((m.value for m in melds if m.kind == "sequence"), reverse=True)
    carres = sorted((m.value for m in melds if m.kind == "carre"), reverse=True)
    blot_reblot = next((m for m in melds if m.kind == "blot_reblot"), None)
    best_seq = seqs[0] if seqs else 0
    second_seq = seqs[1] if len(seqs) > 1 else 0
    best_carre = carres[0] if carres else 0
    blot_reblot_value = blot_reblot.value if blot_reblot is not None else 0
    total = best_seq + sum(carres) + blot_reblot_value
    out[0] = best_seq / 100
    out[1] = best_carre / 200
    out[2] = blot_reblot_value / 20 if blot_reblot_value else 0.0
    out[3] = min(1.0, total / info.rules.auction.max_bid)
    out[4] = len(seqs) / 4
    out[5] = len(carres) / 4
    out[6] = float(blot_reblot is not None)
    out[7] = second_seq / 100
    # out[8:12] reserved, documented headroom for a future v2 — see module docstring.
    return out


def _voids_block(info: InfoSet, contract_type: str | None) -> Array:
    out = np.zeros(BLOCK_SHAPES["voids"], dtype=np.float32)
    primary = _primary_suit(contract_type)
    tracked_suits = [s for s in SUITS if s != primary]
    slot_of = {s: i for i, s in enumerate(tracked_suits)}

    def _mark(trick: tuple[tuple[int, int], ...]) -> None:
        if not trick:
            return
        led_suit = suit_of(trick[0][1])
        if led_suit not in slot_of:
            return
        for seat, c in trick[1:]:
            if suit_of(c) != led_suit:
                out[slot_of[led_suit], _rel(seat, info.seat)] = 1.0

    for trick in info.tricks:
        _mark(trick)
    _mark(info.current_trick)
    return out


def _cards_left_block(info: InfoSet) -> Array:
    out = np.zeros(BLOCK_SHAPES["cards_left"], dtype=np.float32)
    n = info.rules.deal.cards_per_player
    for seat in range(4):
        out[_rel(seat, info.seat)] = info.hand_sizes[seat] / n
    return out


def _suit_unseen_block(info: InfoSet) -> Array:
    out = np.zeros(BLOCK_SHAPES["suit_unseen"], dtype=np.float32)
    seen: set[int] = set(info.hand)
    for trick in info.tricks:
        seen |= {c for _, c in trick}
    seen |= {c for _, c in info.current_trick}
    n_ranks = len(RANKS)
    for si, suit in enumerate(SUITS):
        unseen = sum(1 for r in RANKS if card_id(suit, r) not in seen)
        out[si] = unseen / n_ranks
    return out


_MELD_CLASS = ("none", "tierce", "fifty", "hundred", "4x")  # carre -> "4x", per spec §3.1


def _meld_class_of(meld: Meld) -> str:
    if meld.kind == "carre":
        return "4x"
    if meld.kind == "blot_reblot":
        return "none"  # blot-reblot has its own dedicated channels, not a class slot
    return {3: "tierce", 4: "fifty"}.get(meld.length, "hundred")


_CARRE_ANSWER_RANKS = ("J", "9", "A", "10", "K", "Q")  # spec §3.1's SAY_CARRE_RANK set — a carre
# of 7s or 8s (worth 0 under every carre table) has no disclosure action and is never "the best
# meld to announce" for this purpose, even though `detect_hand_melds` still detects it structurally.


def _melds_public_block(info: InfoSet) -> Array:
    out = np.zeros(BLOCK_SHAPES["melds_public"], dtype=np.float32)
    if info.melds_by_seat is None:
        return out
    top_rank_onehot_n, trump_bit_n, carre_rank_onehot_n = 9, 2, 1 + len(_CARRE_ANSWER_RANKS)
    for seat in range(4):
        rel = _rel(seat, info.seat)
        melds = info.melds_by_seat[seat]
        comparable = [
            m
            for m in melds
            if m.kind != "blot_reblot" and not (m.kind == "carre" and m.rank in ("7", "8"))
        ]
        blot_reblot = next((m for m in melds if m.kind == "blot_reblot"), None)
        class_onehot = np.zeros(5, dtype=np.float32)
        top_onehot = np.zeros(top_rank_onehot_n, dtype=np.float32)
        trump_bit = np.zeros(trump_bit_n, dtype=np.float32)
        carre_rank_onehot = np.zeros(carre_rank_onehot_n, dtype=np.float32)
        if comparable:
            best = max(comparable, key=lambda m: m.value)
            class_onehot[_MELD_CLASS.index(_meld_class_of(best))] = 1.0
            if best.kind == "sequence":
                assert best.rank is not None
                top_onehot[1 + RANKS.index(best.rank)] = 1.0
                trump_bit[int(best.is_trump)] = 1.0
            else:
                top_onehot[0] = 1.0  # "none" — carre discloses rank, not a top card
                trump_bit[0] = 1.0  # "unknown" slot doubles as "n/a" here
            if best.kind == "carre":
                assert best.rank is not None
                carre_rank_onehot[1 + _CARRE_ANSWER_RANKS.index(best.rank)] = 1.0
            else:
                carre_rank_onehot[0] = 1.0
        else:
            class_onehot[0] = 1.0
            top_onehot[0] = 1.0
            trump_bit[0] = 1.0
            carre_rank_onehot[0] = 1.0
        was_questioned = 0.0  # no staged protocol exists yet (declarations_are_actions=False)
        has_shown = float(bool(comparable))  # auto-show: true the instant a meld is held
        announced_but_withheld = 0.0
        belote = float(blot_reblot is not None)
        rebelote = float(blot_reblot is not None)  # single combined Meld — see module docstring

        row = np.concatenate(
            [
                class_onehot,
                top_onehot,
                trump_bit,
                carre_rank_onehot,
                [was_questioned, has_shown, announced_but_withheld, belote, rebelote],
            ]
        )
        assert row.shape == (28,)
        out[rel] = row
    return out


def _running_points_block(info: InfoSet) -> Array:
    my_team = TEAM_OF[info.seat]
    pts = info.running_points
    my_pts, opp_pts = pts[my_team], pts[1 - my_team]
    my_tricks = 0
    if info.tables is not None:
        my_tricks = sum(
            1 for t in info.tricks if TEAM_OF[current_winner(t, info.tables)] == my_team
        )
    opp_tricks = len(info.tricks) - my_tricks
    n_tricks = info.rules.deal.cards_per_player
    return np.array(
        [my_pts / 162, opp_pts / 162, my_tricks / n_tricks, opp_tricks / n_tricks], dtype=np.float32
    )


def _shown_cards_by_seat(info: InfoSet) -> tuple[frozenset[int], ...]:
    """`resolve_combinations` only needs a hand-like iterable of the cards each seat showed —
    `info.melds_by_seat` (each seat's already-detected melds) already carries exactly that, so
    this just flattens each seat's meld cards back into one frozenset per seat rather than
    re-deriving anything from ground truth."""
    assert info.melds_by_seat is not None
    out = []
    for seat_melds in info.melds_by_seat:
        cards: set[int] = set()
        for m in seat_melds:
            cards |= set(m.cards)
        out.append(frozenset(cards))
    return tuple(out)


def _contract_progress_block(info: InfoSet) -> Array:
    out = np.zeros(BLOCK_SHAPES["contract_progress"], dtype=np.float32)
    if info.contract is None or info.tables is None:
        return out
    attacking_team = info.contract.attacking_team
    raw_a = info.running_points[attacking_team]
    played_points = sum(sum(info.tables.points[c] for _, c in t) for t in info.tricks)
    points_still_available = 162 - played_points

    combo_a = 0
    if info.melds_by_seat is not None:
        # Elder hand for meld tie-breaking is the trick-1 leader — reconstructible as whoever
        # led the very first completed trick, or the current trick's leader before any trick
        # has completed yet (both are the same seat in practice: the trick-1 leader never
        # changes mid-deal).
        leader_seat = info.tricks[0][0][0] if info.tricks else (info.trick_leader or 0)
        combo = resolve_combinations(
            _shown_cards_by_seat(info), info.contract.contract_type, info.rules, leader_seat
        )
        combo_a = combo.team_points[attacking_team]

    total_scaled_a = round10(raw_a, info.rules) + combo_a
    level = info.contract.level
    margin = total_scaled_a - level
    need_met_sign = float((margin > 0) - (margin < 0))

    attackers_all_so_far = all(
        TEAM_OF[current_winner(t, info.tables)] == attacking_team for t in info.tricks
    )
    capot_still_live = attackers_all_so_far
    capot_already_broken = bool(info.contract.capot and not capot_still_live)

    out[0] = raw_a / max(1, 10 * level)
    out[1] = points_still_available / 162
    out[2] = need_met_sign
    out[3] = float(capot_still_live)
    out[4] = float(capot_already_broken)
    return out


def _match_state_block(info: InfoSet) -> Array:
    my_team = TEAM_OF[info.seat]
    target = info.rules.match.target
    my_score, opp_score = info.match_score[my_team], info.match_score[1 - my_team]
    return np.array(
        [
            my_score / target,
            opp_score / target,
            max(0, target - my_score) / target,
            max(0, target - opp_score) / target,
            info.deal_number / 20,  # loose normalization; matches tend to run a handful of deals
            _rel(info.dealer, info.seat) / 4,
        ],
        dtype=np.float32,
    )


def _bid_recent_block(info: InfoSet) -> Array:
    out = np.zeros(BLOCK_SHAPES["bid_recent"], dtype=np.float32)
    max_bid = info.rules.auction.max_bid
    types_with_none = ("none", *info.rules.contracts.types)
    recent = info.auction_log[-_BID_RECENT_LEN:]
    pad = _BID_RECENT_LEN - len(recent)
    for i, ev in enumerate(recent):
        row = pad + i  # left-pad with zero rows; most recent action is always the last row
        out[row, :] = _encode_auction_event(ev, info.seat, max_bid, types_with_none)
    return out


def _encode_auction_event(
    ev: AuctionEvent, my_seat: int, max_bid: int, types_with_none: tuple[str, ...]
) -> Array:
    kind = np.zeros(4, dtype=np.float32)
    type_oh = np.zeros(6, dtype=np.float32)
    level = 0.0
    capot = 0.0
    if isinstance(ev.action, PassAction):
        kind[0] = 1.0
        type_oh[0] = 1.0
    elif isinstance(ev.action, BidAction):
        kind[1] = 1.0
        type_oh[types_with_none.index(ev.action.contract_type)] = 1.0
        level = ev.action.level / max_bid
        capot = float(ev.action.capot)
    elif isinstance(ev.action, ContraAction):
        kind[2] = 1.0
        type_oh[0] = 1.0
    elif isinstance(ev.action, RecontraAction):
        kind[3] = 1.0
        type_oh[0] = 1.0
    seat_rel = np.zeros(4, dtype=np.float32)
    seat_rel[_rel(ev.seat, my_seat)] = 1.0
    return np.concatenate([kind, type_oh, [level, capot], seat_rel])


def encode(info: InfoSet) -> dict[str, Array]:
    """`InfoSet` -> the `v1` feature blocks (spec §4.3), seat-relative throughout. Deterministic
    and pure — same `InfoSet` always encodes to the same tensors, which is exactly what the
    leakage test in `tests/test_env_infoset.py` relies on."""
    contract_type, level, capot, declarer_seat, doubling = _effective_contract(info)

    blocks: dict[str, Array] = {
        "hand": np.array([float(c in info.hand) for c in range(N_CARDS)], dtype=np.float32),
        "card_state": _card_state_block(info),
        "current_trick": _current_trick_block(info),
        "trick_leader": _trick_leader_block(info),
        "trick_index": _trick_index_block(info),
        "contract_type": _contract_type_block(contract_type),
        "contract_level": _contract_level_block(level),
        "capot_flag": np.array([float(capot)], dtype=np.float32),
        "declarer_rel": _declarer_rel_block(declarer_seat, info.seat),
        "doubling": _doubling_block(doubling),
        "bid_payout": _bid_payout_block(info, contract_type, level, doubling),
        "bid_summary": _bid_summary_block(info),
        "headroom": _headroom_block(info, level),
        "own_combinations": _own_combinations_block(info),
        "voids": _voids_block(info, contract_type),
        "cards_left": _cards_left_block(info),
        "suit_unseen": _suit_unseen_block(info),
        "melds_public": _melds_public_block(info),
        "running_points": _running_points_block(info),
        "contract_progress": _contract_progress_block(info),
        "match_state": _match_state_block(info),
        # Spec §4.3 marks this block "(auction phase only)" and the two block-total figures
        # (~480 play / ~610 auction) assume it's dropped outside the auction. It's computed
        # unconditionally here instead: `info.auction_log` never gets cleared once play starts,
        # so during PLAY this block naturally shows the tail of the auction that produced the
        # contract — genuinely useful context, not filler — and, more importantly, a
        # phase-dependent flat shape is incompatible with Gym/PettingZoo's fixed-shape
        # `observation_space` (§5.1), which every agent framework downstream assumes. `flatten`
        # is therefore a pure function of `BLOCK_SHAPES` with no phase-conditional branch.
        "bid_recent": _bid_recent_block(info),
    }
    return blocks


def flatten(blocks: dict[str, Array]) -> Array:
    """Concatenate every block into one flat float32 vector, in `BLOCK_SHAPES`'s fixed key
    order — `encode()` always returns every key, so this shape never varies (see its docstring
    on why `bid_recent` is unconditional)."""
    return np.concatenate([blocks[k].reshape(-1) for k in BLOCK_SHAPES])
