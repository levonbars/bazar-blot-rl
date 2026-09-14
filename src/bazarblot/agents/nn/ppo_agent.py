"""`NNAgent`: wraps a `PolicyValueNet` to satisfy `agents.base.Agent` (so it drops directly into
`eval/duplicate.py`, `eval/elo.py`, and any other agent-consuming code unchanged), plus the richer
`act_train` entry point `learn/rollout.py` needs (log-prob and value alongside the action, for the
PPO update).

**Masked hierarchical sampling, matching `agents/nn/bid_factor.py`'s decomposition exactly:**
during the auction, sample `meta in {PASS, CONTRA, RECONTRA, BID}` under the meta mask; if `BID`,
sample `Delta`, then `type | Delta`, then `capot | Delta, type`, each under the legality slice
`bid_factor.bid_legality` computed for that `Delta`/`(Delta, type)`. During play, sample directly
over the 32 card logits under the play segment of `legal_mask`. The joint log-probability of a bid
is the sum of its three conditional log-probabilities -- exactly the chain rule for the
hierarchical decomposition, so PPO's importance ratio (`exp(new_log_prob - old_log_prob)`) is
correct without any special-casing at update time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor
from torch.distributions import Categorical

from bazarblot.agents.nn.bid_factor import base_level, bid_legality, flat_bid_index
from bazarblot.agents.nn.network import (
    META_BID,
    META_CONTRA,
    META_PASS,
    META_RECONTRA,
    PolicyValueNet,
)
from bazarblot.core.deal import Phase
from bazarblot.env.actions import PASS_IDX, ActionSpace
from bazarblot.env.infoset import InfoSet
from bazarblot.env.obs import encode, flatten

BoolArray = npt.NDArray[np.bool_]

_NEG_INF = -1e9


def _masked_categorical(logits: Tensor, mask: Tensor) -> Categorical:
    """`Categorical` over `logits`, with every position `mask` marks illegal driven to (near)
    zero probability -- `-1e9` rather than literal `-inf` so a row that is all-illegal-but-one
    still normalizes cleanly and gradients don't `nan` out on a fully-masked row that should never
    occur in practice but must not crash training if it does."""
    return Categorical(logits=logits.masked_fill(~mask, _NEG_INF))


@dataclass(frozen=True, slots=True)
class Decision:
    """Everything about one sampled decision needed to (a) turn it into a flat env action and (b)
    recompute its log-probability later, under updated network parameters, for the PPO ratio."""

    action_idx: int
    log_prob: float
    value: float
    obs: npt.NDArray[np.float32]
    mask: BoolArray
    phase: Phase
    base_level: int


def _auction_decision(
    info: InfoSet, space: ActionSpace, mask: BoolArray, out: dict[str, Tensor], sample: bool
) -> tuple[int, Tensor]:
    base = base_level(info)
    legality = bid_legality(mask, space, base)

    meta_mask = torch.zeros(4, dtype=torch.bool)
    meta_mask[META_PASS] = bool(mask[PASS_IDX])
    meta_mask[META_CONTRA] = bool(mask[1])
    meta_mask[META_RECONTRA] = bool(mask[2])
    meta_mask[META_BID] = legality.any_bid_legal()
    meta_dist = _masked_categorical(out["meta"][0], meta_mask)
    meta = meta_dist.sample() if sample else meta_dist.probs.argmax()
    log_prob = meta_dist.log_prob(meta)

    if int(meta.item()) != META_BID:
        action_idx = {META_PASS: PASS_IDX, META_CONTRA: 1, META_RECONTRA: 2}[int(meta.item())]
        return action_idx, log_prob

    delta_mask = torch.from_numpy(legality.delta_mask)
    delta_dist = _masked_categorical(out["delta"][0, : delta_mask.shape[0]], delta_mask)
    delta_idx = delta_dist.sample() if sample else delta_dist.probs.argmax()
    log_prob = log_prob + delta_dist.log_prob(delta_idx)

    type_mask = torch.from_numpy(legality.type_mask_given_delta[int(delta_idx.item())])
    type_dist = _masked_categorical(out["type"][0], type_mask)
    type_idx = type_dist.sample() if sample else type_dist.probs.argmax()
    log_prob = log_prob + type_dist.log_prob(type_idx)

    capot_mask = torch.from_numpy(
        legality.capot_mask_given_delta_type[int(delta_idx.item()), int(type_idx.item())]
    )
    capot_dist = _masked_categorical(out["capot"][0], capot_mask)
    capot_idx = capot_dist.sample() if sample else capot_dist.probs.argmax()
    log_prob = log_prob + capot_dist.log_prob(capot_idx)

    action_idx = flat_bid_index(
        space, base, int(delta_idx.item()) + 1, int(type_idx.item()), int(capot_idx.item())
    )
    return action_idx, log_prob


def _play_decision(
    space: ActionSpace, mask: BoolArray, out: dict[str, Tensor], sample: bool
) -> tuple[int, Tensor]:
    card_mask = torch.from_numpy(mask[space.play_start : space.action_dim])
    card_dist = _masked_categorical(out["card"][0], card_mask)
    card_idx = card_dist.sample() if sample else card_dist.probs.argmax()
    return space.play_start + int(card_idx.item()), card_dist.log_prob(card_idx)


def decide(
    net: PolicyValueNet, info: InfoSet, space: ActionSpace, mask: BoolArray, sample: bool
) -> Decision:
    """One full decision: encode `info`, run the network, sample (or, if `sample=False`, take the
    greedy action) under the phase-appropriate hierarchical mask, and package everything
    `learn/rollout.py` needs to later recompute this same log-probability under updated
    parameters."""
    obs_np = flatten(encode(info)).astype(np.float32)
    obs = torch.from_numpy(obs_np).unsqueeze(0)
    with torch.no_grad():
        out = net(obs)
        value = float(out["value"][0].item())
        if info.phase == Phase.AUCTION:
            action_idx, log_prob_t = _auction_decision(info, space, mask, out, sample)
        elif info.phase == Phase.PLAY:
            action_idx, log_prob_t = _play_decision(space, mask, out, sample)
        else:
            raise ValueError(f"no decision to make in phase {info.phase}")
    return Decision(
        action_idx=action_idx,
        log_prob=float(log_prob_t.item()),
        value=value,
        obs=obs_np,
        mask=mask,
        phase=info.phase,
        base_level=base_level(info) if info.phase == Phase.AUCTION else 0,
    )


def recompute_log_prob(
    net: PolicyValueNet,
    space: ActionSpace,
    obs: Tensor,
    mask: BoolArray,
    phase: Phase,
    base: int,
    action_idx: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Re-run the SAME hierarchical decomposition `decide()` used, but scoring the action that was
    actually taken (`action_idx`) under the CURRENT network parameters, with gradients -- what the
    PPO update needs for the importance ratio and the value/entropy losses. Returns
    `(log_prob, entropy, value)`, each a 0-d tensor."""
    out = net(obs.unsqueeze(0))
    value = out["value"][0]

    if phase == Phase.PLAY:
        card_mask = torch.from_numpy(mask[space.play_start : space.action_dim])
        card_dist = _masked_categorical(out["card"][0], card_mask)
        card_idx = torch.tensor(action_idx - space.play_start)
        return card_dist.log_prob(card_idx), card_dist.entropy(), value

    # AUCTION
    action = space.decode(action_idx)
    legality = bid_legality(mask, space, base)
    meta_mask = torch.zeros(4, dtype=torch.bool)
    meta_mask[META_PASS] = bool(mask[PASS_IDX])
    meta_mask[META_CONTRA] = bool(mask[1])
    meta_mask[META_RECONTRA] = bool(mask[2])
    meta_mask[META_BID] = legality.any_bid_legal()
    meta_dist = _masked_categorical(out["meta"][0], meta_mask)

    if action_idx == PASS_IDX:
        meta = torch.tensor(META_PASS)
        return meta_dist.log_prob(meta), meta_dist.entropy(), value
    if action_idx == 1:
        meta = torch.tensor(META_CONTRA)
        return meta_dist.log_prob(meta), meta_dist.entropy(), value
    if action_idx == 2:
        meta = torch.tensor(META_RECONTRA)
        return meta_dist.log_prob(meta), meta_dist.entropy(), value

    from bazarblot.core.auction import BidAction

    assert isinstance(action, BidAction)
    delta = action.level - base
    type_idx = space.contract_types.index(action.contract_type)
    capot_idx = int(action.capot) if space.capot_states == 2 else 0

    meta = torch.tensor(META_BID)
    log_prob = meta_dist.log_prob(meta)
    entropy = meta_dist.entropy()

    delta_mask = torch.from_numpy(legality.delta_mask)
    delta_dist = _masked_categorical(out["delta"][0, : delta_mask.shape[0]], delta_mask)
    delta_t = torch.tensor(delta - 1)
    log_prob = log_prob + delta_dist.log_prob(delta_t)
    entropy = entropy + delta_dist.entropy()

    type_mask = torch.from_numpy(legality.type_mask_given_delta[delta - 1])
    type_dist = _masked_categorical(out["type"][0], type_mask)
    type_t = torch.tensor(type_idx)
    log_prob = log_prob + type_dist.log_prob(type_t)
    entropy = entropy + type_dist.entropy()

    capot_mask = torch.from_numpy(legality.capot_mask_given_delta_type[delta - 1, type_idx])
    capot_dist = _masked_categorical(out["capot"][0], capot_mask)
    capot_t = torch.tensor(capot_idx)
    log_prob = log_prob + capot_dist.log_prob(capot_t)
    entropy = entropy + capot_dist.entropy()

    return log_prob, entropy, value


class NNAgent:
    """`Agent`-protocol wrapper: greedy by default (`sample=False`), matching what an evaluation
    harness or a live game wants (a fixed, deterministic policy), not what training wants (which
    calls `decide(..., sample=True)` directly via `learn/rollout.py` instead of going through
    this class)."""

    def __init__(self, net: PolicyValueNet, sample: bool = False) -> None:
        self.net = net
        self.sample = sample

    def act(self, info: InfoSet, space: ActionSpace, legal_mask: BoolArray) -> int:
        return decide(self.net, info, space, legal_mask, sample=self.sample).action_idx
