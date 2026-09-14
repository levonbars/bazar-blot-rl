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
    info: InfoSet,
    space: ActionSpace,
    mask: BoolArray,
    out: dict[str, Tensor],
    sample: bool,
    forced_bid_prob: float = 0.0,
) -> tuple[int, Tensor]:
    base = base_level(info)
    legality = bid_legality(mask, space, base)

    meta_mask = torch.zeros(4, dtype=torch.bool)
    meta_mask[META_PASS] = bool(mask[PASS_IDX])
    meta_mask[META_CONTRA] = bool(mask[1])
    meta_mask[META_RECONTRA] = bool(mask[2])
    meta_mask[META_BID] = legality.any_bid_legal()
    meta_dist = _masked_categorical(out["meta"][0], meta_mask)
    force_bid = (
        sample
        and meta_mask[META_BID]
        and forced_bid_prob > 0.0
        and torch.rand(()) < forced_bid_prob
    )
    if force_bid:
        # Forced-exploration override (annealed by `learn/train.py` over the run): sample from a
        # BIASED behavior distribution during rollout collection only, but score the resulting
        # action under `meta_dist` -- the network's own TRUE (unbiased) distribution -- exactly
        # like every other path here. PPO's importance ratio only needs `log_prob` to be an
        # accurate probability of the action under the policy being optimized, not under whatever
        # distribution generated it; recording the true `meta_dist.log_prob` keeps the math
        # correct despite the biased sampling. See `learn/train.py`'s docstring on why this
        # exists: without it, self-play converges to a pure pass-and-defend policy that never
        # once declares even against a real opponent (`docs/03-implementation-roadmap.md` M7) --
        # entropy bonuses alone, even large and slowly-annealed ones, did not prevent it.
        meta = torch.tensor(META_BID)
    else:
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
    net: PolicyValueNet,
    info: InfoSet,
    space: ActionSpace,
    mask: BoolArray,
    sample: bool,
    forced_bid_prob: float = 0.0,
) -> Decision:
    """One full decision: encode `info`, run the network, sample (or, if `sample=False`, take the
    greedy action) under the phase-appropriate hierarchical mask, and package everything
    `learn/rollout.py` needs to later recompute this same log-probability under updated
    parameters.

    `forced_bid_prob` (rollout collection only, see `_auction_decision`'s docstring) biases
    auction-phase sampling toward `BID` without affecting the recorded log-probability's
    correctness."""
    obs_np = flatten(encode(info)).astype(np.float32)
    obs = torch.from_numpy(obs_np).unsqueeze(0)
    with torch.no_grad():
        out = net(obs)
        value = float(out["value"][0].item())
        if info.phase == Phase.AUCTION:
            action_idx, log_prob_t = _auction_decision(
                info, space, mask, out, sample, forced_bid_prob
            )
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


def _recompute_log_prob_from_output(
    space: ActionSpace,
    out: dict[str, Tensor],
    mask: BoolArray,
    phase: Phase,
    base: int,
    action_idx: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """The scoring half of `recompute_log_prob`, taking an ALREADY-COMPUTED, single-sample
    (no batch dimension) network output `out` rather than running the network itself -- lets
    `learn/ppo.py` batch the expensive forward pass across a whole minibatch (one `net(obs_batch)`
    call) and then call this cheap, network-free function once per sample to do the per-sample
    hierarchical bookkeeping, instead of one full forward pass per transition."""
    value = out["value"]

    if phase == Phase.PLAY:
        card_mask = torch.from_numpy(mask[space.play_start : space.action_dim])
        card_dist = _masked_categorical(out["card"], card_mask)
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
    meta_dist = _masked_categorical(out["meta"], meta_mask)

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
    delta_dist = _masked_categorical(out["delta"][: delta_mask.shape[0]], delta_mask)
    delta_t = torch.tensor(delta - 1)
    log_prob = log_prob + delta_dist.log_prob(delta_t)
    entropy = entropy + delta_dist.entropy()

    type_mask = torch.from_numpy(legality.type_mask_given_delta[delta - 1])
    type_dist = _masked_categorical(out["type"], type_mask)
    type_t = torch.tensor(type_idx)
    log_prob = log_prob + type_dist.log_prob(type_t)
    entropy = entropy + type_dist.entropy()

    capot_mask = torch.from_numpy(legality.capot_mask_given_delta_type[delta - 1, type_idx])
    capot_dist = _masked_categorical(out["capot"], capot_mask)
    capot_t = torch.tensor(capot_idx)
    log_prob = log_prob + capot_dist.log_prob(capot_t)
    entropy = entropy + capot_dist.entropy()

    return log_prob, entropy, value


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
    `(log_prob, entropy, value)`, each a 0-d tensor.

    Single-sample convenience wrapper around `_recompute_log_prob_from_output` -- `learn/ppo.py`
    calls that directly on a batched forward pass instead, for throughput."""
    out = net(obs.unsqueeze(0))
    out_i = {k: v[0] for k, v in out.items()}
    return _recompute_log_prob_from_output(space, out_i, mask, phase, base, action_idx)


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
