"""The "club player" baseline (roadmap M4): hand-evaluation bidding plus classic trick-taking
play conventions. Deliberately not tuned to be strong — the roadmap's own bar is "must be
beatable but not trivial," a believable amateur, not a target to chase.

**Bidding** evaluates every contract type the ruleset offers against the bidder's OWN hand only —
there is no model of partner's or opponents' hands beyond a crude "some points are still out
there, assume a fixed fraction end up on my side" adjustment (`_PARTNER_SHARE`). It makes no
attempt at bidding conventions (asking bids, signaling through the sequence of bids itself),
which is a genuinely harder problem than card play and squarely out of scope for a first
baseline — real conventions coordinate a partnership's *private* information through a *public*
channel under an information constraint, which is close to the paper's actual research question,
not something a hand-picked heuristic should attempt to pre-empt.

**Play** follows textbook trick-taking conventions: second hand plays low (conserve strength
when you can't yet tell if it matters), third hand plays high only when needed to win, fourth
hand wins as cheaply as possible or discards its lowest-value card, and NT leads prefer cashing
an ace early (it only gets weaker relative to the field as the deal goes on). What the roadmap
calls "signal partner" is only implemented as the passive, one-directional half — discarding from
your weakest suit rather than an active two-way convention with a matching interpretation model
on the receiving end, which (like bidding conventions) is out of scope here.
"""

from __future__ import annotations

import random

import numpy as np
import numpy.typing as npt

from bazarblot.core.cards import (
    TEAM_OF,
    ContractTables,
    ContractType,
    build_tables,
    rank_of,
    suit_of,
)
from bazarblot.core.deal import Phase
from bazarblot.core.declarations import detect_hand_melds
from bazarblot.core.play import current_winner
from bazarblot.core.rules import RuleConfig
from bazarblot.env.actions import ActionSpace
from bazarblot.env.infoset import InfoSet

_TRUMP_LENGTH_BONUS_PER_CARD = 8  # raw points per trump card beyond 3 — long trump suits let you
# ruff opponents' winners, worth more in play than their face value alone suggests
_PARTNER_SHARE = 0.15  # crude assumed fraction of "points not in my hand" landing on my side
_PASS_IDX = 0


def _estimate_scaled_value(
    hand: frozenset[int], contract_type: ContractType, rules: RuleConfig
) -> int:
    """A rough "what could my team plausibly make in this contract" estimate, in the same
    scaled units as a bid level (8..80) — own hand's raw points, scaled down, plus a length
    bonus for a long trump suit, plus a crude partner-share estimate, plus my own detected
    combination value (which counts toward fulfilment for real, unlike the rest of this
    estimate)."""
    tables = build_tables(contract_type, rules)
    own_raw = sum(tables.points[c] for c in hand)
    n_trump = sum(1 for c in hand if tables.is_trump[c]) if contract_type != "NT" else 0
    trump_bonus = max(0, n_trump - 3) * _TRUMP_LENGTH_BONUS_PER_CARD
    combo_value = sum(m.value for m in detect_hand_melds(hand, 0, contract_type, rules))
    deal_total = rules.contracts.deal_card_points
    remaining_raw = max(0, deal_total - own_raw)
    partner_estimate = int(remaining_raw * _PARTNER_SHARE)
    return (own_raw + trump_bonus + partner_estimate) // 10 + combo_value


class HeuristicAgent:
    def __init__(self, rules: RuleConfig, rng: random.Random | None = None) -> None:
        self.rules = rules
        self.rng = rng if rng is not None else random.Random()

    def act(self, info: InfoSet, space: ActionSpace, legal_mask: npt.NDArray[np.bool_]) -> int:
        if info.phase == Phase.AUCTION:
            return self._bid(info, space, legal_mask)
        return self._play(info, space, legal_mask)

    # ---------------------------------------------------------------- bidding

    def _best_contract(self, hand: frozenset[int]) -> tuple[ContractType, int]:
        best_type: ContractType | None = None
        best_value = -1
        for ct in self.rules.contracts.types:
            v = _estimate_scaled_value(hand, ct, self.rules)
            if v > best_value:
                best_type, best_value = ct, v
        assert best_type is not None
        return best_type, best_value

    def _bid(self, info: InfoSet, space: ActionSpace, legal_mask: npt.NDArray[np.bool_]) -> int:
        if info.auction_state.doubling == "contra":
            # Forced reply window (rules §4.3): always accept, never redouble — no model here of
            # when a contra is bad enough to warrant fighting back.
            return self._safe(_PASS_IDX, legal_mask)

        standing = info.auction_state.standing_bid
        min_bid, max_bid = self.rules.auction.min_bid, self.rules.auction.max_bid
        best_type, best_value = self._best_contract(info.hand)
        target = min(max_bid, best_value)

        if standing is None:
            if target >= min_bid:
                return self._safe(space.bid_index(target, best_type, False), legal_mask)
            return self._safe(_PASS_IDX, legal_mask)

        if TEAM_OF[info.seat] == TEAM_OF[standing.seat]:
            return self._safe(_PASS_IDX, legal_mask)  # never raise over partner — see docstring

        if target > standing.level:
            return self._safe(space.bid_index(target, best_type, False), legal_mask)
        return self._safe(_PASS_IDX, legal_mask)

    def _safe(self, idx: int, legal_mask: npt.NDArray[np.bool_]) -> int:
        """Fall back to PASS, or to any legal action if even PASS somehow isn't legal, rather
        than ever returning an action the mask disagrees with."""
        if legal_mask[idx]:
            return idx
        if legal_mask[_PASS_IDX]:
            return _PASS_IDX
        return int(legal_mask.nonzero()[0][0])

    # ---------------------------------------------------------------- play

    def _play(self, info: InfoSet, space: ActionSpace, legal_mask: npt.NDArray[np.bool_]) -> int:
        assert info.contract is not None and info.tables is not None
        tables = info.tables
        legal_cards = [c for c in range(32) if legal_mask[space.play_index(c)]]
        assert legal_cards, "play phase always has at least one legal card"

        position = len(info.current_trick)  # 0 = leading, 1/2/3 = 2nd/3rd/4th to act
        if position == 0:
            card = self._choose_lead(info, tables, legal_cards)
        else:
            my_team = TEAM_OF[info.seat]
            winner_so_far = current_winner(info.current_trick, tables)
            partner_winning = TEAM_OF[winner_so_far] == my_team
            can_beat = [
                c
                for c in legal_cards
                if current_winner((*info.current_trick, (info.seat, c)), tables) == info.seat
            ]
            if partner_winning:
                # Second/third/fourth hand, partner already winning: conserve strength.
                card = min(legal_cards, key=lambda c: tables.strength[c])
            elif can_beat:
                # Win as cheaply as possible — no point spending a strong card when a weaker one
                # already wins it (classic "third/fourth hand high, but only as high as needed").
                card = min(can_beat, key=lambda c: tables.strength[c])
            else:
                # Can't win: discard the least valuable card, protecting points for later.
                card = min(legal_cards, key=lambda c: tables.points[c])
        return space.play_index(card)

    def _choose_lead(self, info: InfoSet, tables: ContractTables, legal_cards: list[int]) -> int:
        assert info.contract is not None
        if info.contract.contract_type == "NT":
            # Cash aces in NT: they only get relatively weaker as the deal goes on.
            aces = [c for c in legal_cards if rank_of(c) == "A"]
            if aces:
                return max(aces, key=lambda c: tables.points[c])
        elif TEAM_OF[info.seat] == info.contract.attacking_team:
            # Declaring side leading trump: draw out the defenders' trump early.
            trumps = [c for c in legal_cards if tables.is_trump[c]]
            if len(trumps) >= 2:
                return max(trumps, key=lambda c: tables.strength[c])

        by_suit: dict[str, list[int]] = {}
        for c in legal_cards:
            by_suit.setdefault(suit_of(c), []).append(c)
        longest_suit = max(by_suit.values(), key=len)
        return max(longest_suit, key=lambda c: tables.strength[c])
