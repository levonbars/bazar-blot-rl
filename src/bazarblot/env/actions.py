"""The flat, phase-masked discrete action space (environment spec §3).

`ACTION_DIM` and every index mapping are **derived from `RuleConfig`**, never hard-coded — the
bid ladder width, the contract type list and whether capot is a bid modifier at all are config,
and a preset that changes any of them must not silently desync the action space from the engine.
`tests/test_env_actions.py` pins the exact derived value for the shipped preset so a config change
that moves it is a visible, deliberate test update, not a silent drift.

The combination-protocol slots from spec §3.1 (31 actions: announce/question/answer/show) are
**not implemented here** — `core/` has no staged announce/question/show state machine to make
them legal or meaningful against (see `declarations.py`'s and `env/infoset.py`'s docstrings), and
the shipped preset has `staging.declarations_are_actions = False` anyway. `build_action_space`
raises `NotImplementedError` if that flag is ever turned on for exactly this reason, rather than
quietly exposing 31 action slots nothing can legally use. `ACTION_DIM` for the shipped preset is
therefore `3 + n_bid_actions + 32` (bids exactly per §3, no combination range), not the spec's
worked `796` — that number already assumes the combination protocol is live.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from bazarblot.core.auction import (
    AuctionAction,
    BidAction,
    ContraAction,
    PassAction,
    RecontraAction,
    is_legal,
)
from bazarblot.core.cards import N_CARDS, ContractType
from bazarblot.core.deal import Action, Phase
from bazarblot.core.play import PlayCardAction, legal_moves
from bazarblot.core.rules import RuleConfig
from bazarblot.env.infoset import InfoSet

BoolArray = npt.NDArray[np.bool_]

PASS_IDX = 0
CONTRA_IDX = 1
RECONTRA_IDX = 2
_N_FIXED = 3  # PASS, CONTRA, RECONTRA


@dataclass(frozen=True, slots=True)
class ActionSpace:
    """Derived, immutable index layout for one `RuleConfig`. Build once per rules preset and
    reuse — it's pure arithmetic over the config, safe to share across an entire training run."""

    min_bid: int
    max_bid: int
    contract_types: tuple[ContractType, ...]
    capot_states: int  # 2 if capot is a bid modifier, else 1 (capot is always False)
    bid_start: int
    n_bid_actions: int
    play_start: int
    action_dim: int

    def bid_index(self, level: int, contract_type: ContractType, capot: bool) -> int:
        combos_per_level = len(self.contract_types) * self.capot_states
        type_idx = self.contract_types.index(contract_type)
        capot_idx = int(capot) if self.capot_states == 2 else 0
        return (
            self.bid_start
            + (level - self.min_bid) * combos_per_level
            + type_idx * self.capot_states
            + capot_idx
        )

    def play_index(self, card: int) -> int:
        return self.play_start + card

    def decode(self, idx: int) -> Action:
        """Index -> the concrete `core` action it represents. Does not check legality — pair
        with `legal_mask` before using the result to step a `Deal`."""
        if idx == PASS_IDX:
            return PassAction()
        if idx == CONTRA_IDX:
            return ContraAction()
        if idx == RECONTRA_IDX:
            return RecontraAction()
        if self.bid_start <= idx < self.play_start:
            offset = idx - self.bid_start
            combos_per_level = len(self.contract_types) * self.capot_states
            level = self.min_bid + offset // combos_per_level
            rem = offset % combos_per_level
            type_idx = rem // self.capot_states
            capot_idx = rem % self.capot_states
            return BidAction(
                level=level,
                contract_type=self.contract_types[type_idx],
                capot=bool(capot_idx) if self.capot_states == 2 else False,
            )
        if self.play_start <= idx < self.action_dim:
            return PlayCardAction(card=idx - self.play_start)
        raise ValueError(f"action index {idx} out of range [0, {self.action_dim})")

    def encode(self, action: Action) -> int:
        """The inverse of `decode` — the flat index for a concrete `core` action."""
        if isinstance(action, PassAction):
            return PASS_IDX
        if isinstance(action, ContraAction):
            return CONTRA_IDX
        if isinstance(action, RecontraAction):
            return RECONTRA_IDX
        if isinstance(action, BidAction):
            return self.bid_index(action.level, action.contract_type, action.capot)
        if isinstance(action, PlayCardAction):
            return self.play_index(action.card)
        raise TypeError(f"unrecognized action type {type(action)!r}")  # pragma: no cover


def build_action_space(rules: RuleConfig) -> ActionSpace:
    """Derive an `ActionSpace` from a `RuleConfig`."""
    r = rules
    if r.staging.declarations_are_actions:
        raise NotImplementedError(
            "build_action_space: staging.declarations_are_actions=True has no combination-"
            "protocol action space implemented yet — see this module's docstring."
        )
    min_bid, max_bid = r.auction.min_bid, r.auction.max_bid
    contract_types = tuple(r.contracts.types)
    capot_states = 2 if r.auction.capot_is_a_bid_modifier else 1
    n_bid_actions = r.n_bid_actions
    bid_start = _N_FIXED
    play_start = bid_start + n_bid_actions
    action_dim = play_start + N_CARDS
    return ActionSpace(
        min_bid=min_bid,
        max_bid=max_bid,
        contract_types=contract_types,
        capot_states=capot_states,
        bid_start=bid_start,
        n_bid_actions=n_bid_actions,
        play_start=play_start,
        action_dim=action_dim,
    )


def legal_mask(info: InfoSet, space: ActionSpace) -> BoolArray:
    """`legal_mask(info_set) -> bool[ACTION_DIM]` (spec §3.2), computed strictly from the
    `InfoSet` — never from full state. This is the mask's entire security property: a mask
    computed from `Deal` instead could differ from the one an agent training on `InfoSet` would
    compute, and the agent would learn to read the difference as a side channel. Delegates
    legality to `core.auction.is_legal` / `core.play.legal_moves` rather than re-deriving the
    rules here — the same "don't re-implement, reuse the tested engine" discipline as
    `solver/dd.py`."""
    mask = np.zeros(space.action_dim, dtype=bool)

    if info.phase == Phase.AUCTION:
        auction_action: AuctionAction
        for auction_action, idx in (
            (PassAction(), PASS_IDX),
            (ContraAction(), CONTRA_IDX),
            (RecontraAction(), RECONTRA_IDX),
        ):
            mask[idx] = is_legal(info.auction_state, auction_action, info.rules)
        for level in range(space.min_bid, space.max_bid + 1):
            for contract_type in space.contract_types:
                capot_options = (False, True) if space.capot_states == 2 else (False,)
                for capot in capot_options:
                    bid = BidAction(level=level, contract_type=contract_type, capot=capot)
                    mask[space.bid_index(level, contract_type, capot)] = is_legal(
                        info.auction_state, bid, info.rules
                    )
        return mask

    if info.phase == Phase.PLAY:
        assert info.contract is not None and info.tables is not None
        legal_cards = legal_moves(
            info.hand,
            info.current_trick,
            info.to_act,
            info.contract.contract_type,
            info.rules,
            info.tables,
        )
        for card in legal_cards:
            mask[space.play_index(card)] = True
        return mask

    return mask  # TERMINAL / ABORTED: nobody has a legal action
