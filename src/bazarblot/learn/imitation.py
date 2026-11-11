"""Behavior cloning of a teacher's bidding and card play, plus the belief head (roadmap M7).

**Why.** Learning from the single end-of-deal margin with PPO did not get off the ground, and the
exploration it needs is only meaningful if the first bids it tries are sensible. Cloning the fixed
heuristic first gives a policy that already bids and plays like a club player, so everything after
it (joint self-play, signaling) starts from "reasonable" instead of from noise. The belief head is
fitted at the same time because it is a dense, per-decision supervised signal (who holds each
unseen card) that shapes the shared trunk, and it is the instrument for measuring how much a bid
tells a partner.

**Bids use the network's own factored decomposition.** An auction action is `meta in {PASS, CONTRA,
RECONTRA, BID}`, then for a bid `Delta`, then `type | Delta`, then `capot | Delta, type`
(`agents/nn/bid_factor.py`). Each factor is trained with a cross-entropy under the same legality
mask the sampler and PPO use, so the cloned policy's log-probabilities mean the same thing later.

**Noise injection.** Cloning only on the teacher's own trajectories leaves the student blind to
positions the teacher would never reach, so it drifts off-distribution the first time it deviates.
With some probability the EXECUTED action is a random legal one, while the recorded label is still
the teacher's choice at that position. The dataset then covers the neighbourhood of the teacher's
play with the teacher's corrections (DART-style), at no extra cost. Auction noise is deliberately
mild: a uniformly random legal auction action is almost always an absurd high bid, which would
flood the dataset with states the teacher never faces, so it is limited to passes and raises of a
few levels.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from torch.nn import functional as F

from bazarblot.agents.base import Agent
from bazarblot.agents.nn.bid_factor import base_level, bid_legality
from bazarblot.agents.nn.network import (
    META_BID,
    META_CONTRA,
    META_PASS,
    META_RECONTRA,
    N_META,
    PolicyValueNet,
)
from bazarblot.core.auction import BidAction, ContraAction, PassAction, RecontraAction
from bazarblot.core.deal import Deal, Phase
from bazarblot.core.dealing import deal_hands, deal_seed
from bazarblot.core.rules import RuleConfig
from bazarblot.env.actions import CONTRA_IDX, PASS_IDX, RECONTRA_IDX, ActionSpace, legal_mask
from bazarblot.env.infoset import InfoSet, info_set
from bazarblot.env.obs import encode, flatten
from bazarblot.env.tracked_deal import TrackedDeal
from bazarblot.learn.belief import belief_accuracy, belief_loss, belief_target

_NEG_INF = -1e9
_NOISE_MAX_RAISE = 4  # auction noise: raises of at most this many levels over the standing bid
NO_LABEL = -1


@dataclass(slots=True)
class ImitationData:
    # Card play
    obs: npt.NDArray[np.float16]  # (N, OBS_DIM), float16 to halve memory; observations are 0/small
    card_mask: npt.NDArray[np.bool_]  # (N, n_cards) legal cards at the decision
    action: npt.NDArray[np.int64]  # (N,) the teacher's card index (0..n_cards-1)
    belief: npt.NDArray[np.int8]  # (N, n_cards) holder labels, IGNORE where unlabelled
    # Auction (M samples). Masks are the legality slices the sampler uses, conditioned on the
    # TEACHER's earlier factors (type mask given the teacher's delta, capot mask given delta+type).
    a_obs: npt.NDArray[np.float16]
    a_meta_mask: npt.NDArray[np.bool_]  # (M, N_META)
    a_delta_mask: npt.NDArray[np.bool_]  # (M, n_delta)
    a_type_mask: npt.NDArray[np.bool_]  # (M, n_types)
    a_capot_mask: npt.NDArray[np.bool_]  # (M, capot_states)
    a_meta: npt.NDArray[np.int64]  # (M,) PASS/CONTRA/RECONTRA/BID
    a_delta: npt.NDArray[np.int64]  # (M,) 0-based Delta index, NO_LABEL unless meta == BID
    a_type: npt.NDArray[np.int64]
    a_capot: npt.NDArray[np.int64]
    a_belief: npt.NDArray[np.int8]

    def __len__(self) -> int:
        """Number of card-play samples."""
        return int(self.action.shape[0])

    @property
    def n_auction(self) -> int:
        return int(self.a_meta.shape[0])


@dataclass(frozen=True, slots=True)
class _AuctionLabel:
    meta_mask: npt.NDArray[np.bool_]
    delta_mask: npt.NDArray[np.bool_]
    type_mask: npt.NDArray[np.bool_]
    capot_mask: npt.NDArray[np.bool_]
    meta: int
    delta: int = NO_LABEL
    type_idx: int = NO_LABEL
    capot: int = NO_LABEL


def _auction_label(
    info: InfoSet, space: ActionSpace, mask: npt.NDArray[np.bool_], action_idx: int
) -> _AuctionLabel:
    """Masks and factor labels for one auction decision, built with the same `bid_legality` the
    sampler and PPO use."""
    base = base_level(info)
    legality = bid_legality(mask, space, base)
    meta_mask = np.zeros(N_META, dtype=bool)
    meta_mask[META_PASS] = bool(mask[PASS_IDX])
    meta_mask[META_CONTRA] = bool(mask[CONTRA_IDX])
    meta_mask[META_RECONTRA] = bool(mask[RECONTRA_IDX])
    meta_mask[META_BID] = legality.any_bid_legal()

    delta_mask = legality.delta_mask.copy()
    type_mask = np.zeros(len(space.contract_types), dtype=bool)
    capot_mask = np.zeros(space.capot_states, dtype=bool)

    def label(meta: int) -> _AuctionLabel:
        return _AuctionLabel(meta_mask, delta_mask, type_mask, capot_mask, meta)

    action = space.decode(action_idx)
    if isinstance(action, PassAction):
        return label(META_PASS)
    if isinstance(action, ContraAction):
        return label(META_CONTRA)
    if isinstance(action, RecontraAction):
        return label(META_RECONTRA)
    assert isinstance(action, BidAction)
    delta = action.level - base - 1  # 0-based, matching `BidLegality`
    type_idx = space.contract_types.index(action.contract_type)
    capot_idx = int(action.capot) if space.capot_states == 2 else 0
    return _AuctionLabel(
        meta_mask,
        delta_mask,
        legality.type_mask_given_delta[delta].copy(),
        legality.capot_mask_given_delta_type[delta, type_idx].copy(),
        META_BID,
        delta,
        type_idx,
        capot_idx,
    )


def _noisy_auction_action(
    info: InfoSet, space: ActionSpace, mask: npt.NDArray[np.bool_], rng: random.Random
) -> int:
    """A random legal auction action restricted to passes/doubles and modest raises."""
    base = base_level(info)
    candidates = [PASS_IDX, CONTRA_IDX, RECONTRA_IDX]
    candidates = [i for i in candidates if mask[i]]
    for idx in mask[space.bid_start : space.play_start].nonzero()[0]:
        action = space.decode(space.bid_start + int(idx))
        assert isinstance(action, BidAction)
        if action.level - base <= _NOISE_MAX_RAISE:
            candidates.append(space.bid_start + int(idx))
    return candidates[rng.randrange(len(candidates))]


def collect_imitation_data(
    rules: RuleConfig,
    space: ActionSpace,
    teacher: Agent,
    n_deals: int,
    seed: int,
    noise_prob: float = 0.1,
    auction_noise_prob: float = 0.05,
    opponents: Sequence[Agent] | None = None,
) -> ImitationData:
    """Play `n_deals` shuffles, recording one sample per card-play decision and one per auction
    decision made by the teacher. Aborted deals still contribute their auction samples (the
    teacher's passes are real, informative examples).

    By default `teacher` makes every decision (all four seats are recorded). With `opponents`, the
    teacher holds one team and an opponent from the list holds the other (cycling through the list,
    alternating which team is the teacher's), and only the TEACHER's decisions are recorded; an
    opponent that is the teacher itself (`opponent is teacher`) is the all-teacher case. Needed to
    clone behaviour that only comes up against particular opponents, e.g. doubling an over-bidder:
    a teacher playing only itself rarely meets a bid worth doubling."""
    rng = random.Random(seed)
    obs_rows: list[npt.NDArray[np.float16]] = []
    mask_rows: list[npt.NDArray[np.bool_]] = []
    action_rows: list[int] = []
    belief_rows: list[npt.NDArray[np.int8]] = []
    a_rows: dict[str, list[Any]] = {
        k: []
        for k in (
            "obs",
            "meta_mask",
            "delta_mask",
            "type_mask",
            "capot_mask",
            "meta",
            "delta",
            "type",
            "capot",
            "belief",
        )
    }

    for i in range(n_deals):
        hands = deal_hands(random.Random(deal_seed(seed + i, 0, 0)), rules)
        deal = Deal(rules, dealer=i % 4, hands=hands, deal_id=0)
        tracked = TrackedDeal(deal)
        opponent: Agent = teacher
        teacher_team = 0
        if opponents:
            # Not tied to the dealer's parity (`i % 4`), or the teacher would always sit on the
            # dealer's team.
            teacher_team = (i // 4) % 2
            opponent = opponents[(i // 8) % len(opponents)]
        record_all = opponent is teacher
        while deal.phase in (Phase.AUCTION, Phase.PLAY):
            seat = deal.to_act
            info = info_set(tracked, seat, match_score=(0, 0), deal_number=0)
            mask = legal_mask(info, space)
            is_teacher_seat = record_all or seat % 2 == teacher_team
            if not is_teacher_seat:
                tracked.step(space.decode(opponent.act(info, space, mask)))
                continue
            action_idx = teacher.act(info, space, mask)
            if info.phase == Phase.PLAY:
                obs_rows.append(flatten(encode(info)).astype(np.float16))
                mask_rows.append(mask[space.play_start : space.action_dim].copy())
                action_rows.append(action_idx - space.play_start)
                belief_rows.append(belief_target(deal.hands, seat))
                if noise_prob > 0.0 and rng.random() < noise_prob:
                    legal = mask.nonzero()[0]
                    action_idx = int(legal[rng.randrange(len(legal))])
            else:
                lab = _auction_label(info, space, mask, action_idx)
                a_rows["obs"].append(flatten(encode(info)).astype(np.float16))
                a_rows["meta_mask"].append(lab.meta_mask)
                a_rows["delta_mask"].append(lab.delta_mask)
                a_rows["type_mask"].append(lab.type_mask)
                a_rows["capot_mask"].append(lab.capot_mask)
                a_rows["meta"].append(lab.meta)
                a_rows["delta"].append(lab.delta)
                a_rows["type"].append(lab.type_idx)
                a_rows["capot"].append(lab.capot)
                a_rows["belief"].append(belief_target(deal.hands, seat))
                if auction_noise_prob > 0.0 and rng.random() < auction_noise_prob:
                    action_idx = _noisy_auction_action(info, space, mask, rng)
            tracked.step(space.decode(action_idx))

    return ImitationData(
        obs=np.stack(obs_rows),
        card_mask=np.stack(mask_rows),
        action=np.asarray(action_rows, dtype=np.int64),
        belief=np.stack(belief_rows),
        a_obs=np.stack(a_rows["obs"]),
        a_meta_mask=np.stack(a_rows["meta_mask"]),
        a_delta_mask=np.stack(a_rows["delta_mask"]),
        a_type_mask=np.stack(a_rows["type_mask"]),
        a_capot_mask=np.stack(a_rows["capot_mask"]),
        a_meta=np.asarray(a_rows["meta"], dtype=np.int64),
        a_delta=np.asarray(a_rows["delta"], dtype=np.int64),
        a_type=np.asarray(a_rows["type"], dtype=np.int64),
        a_capot=np.asarray(a_rows["capot"], dtype=np.int64),
        a_belief=np.stack(a_rows["belief"]),
    )


@dataclass(frozen=True, slots=True)
class CloneStats:
    train_loss: float
    holdout_card_acc: float
    holdout_belief_acc: float
    holdout_belief_loss: float
    holdout_meta_acc: float = 0.0
    holdout_delta_acc: float = 0.0
    holdout_type_acc: float = 0.0


def _card_loss(card_logits: torch.Tensor, mask: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(card_logits.masked_fill(~mask, _NEG_INF), action)


def _bid_loss(out: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> torch.Tensor:
    """Sum of the factor cross-entropies under the sampler's masks: meta for every row, and
    delta/type/capot for the rows whose teacher action was a bid."""
    loss = F.cross_entropy(out["meta"].masked_fill(~b["meta_mask"], _NEG_INF), b["meta"])
    is_bid = b["meta"] == META_BID
    if bool(is_bid.any()):
        n_delta = b["delta_mask"].shape[1]
        loss = loss + F.cross_entropy(
            out["delta"][is_bid][:, :n_delta].masked_fill(~b["delta_mask"][is_bid], _NEG_INF),
            b["delta"][is_bid],
        )
        loss = loss + F.cross_entropy(
            out["type"][is_bid].masked_fill(~b["type_mask"][is_bid], _NEG_INF), b["type"][is_bid]
        )
        loss = loss + F.cross_entropy(
            out["capot"][is_bid].masked_fill(~b["capot_mask"][is_bid], _NEG_INF), b["capot"][is_bid]
        )
    return loss


def behavior_clone(
    net: PolicyValueNet,
    data: ImitationData,
    *,
    epochs: int = 10,
    batch_size: int = 512,
    lr: float = 1e-3,
    belief_coef: float = 0.5,
    bid_coef: float = 1.0,
    holdout_frac: float = 0.1,
    seed: int = 0,
    verbose: bool = True,
) -> CloneStats:
    """Fit the card head, bid heads and belief head of `net` to `data` by supervised learning.
    The holdout is the TAIL of each dataset: samples are stored deal by deal, so a tail split keeps
    whole deals out of training instead of leaking neighbouring positions from the same deal."""
    n = len(data)
    n_hold = max(1, int(n * holdout_frac))
    n_train = n - n_hold
    m = data.n_auction
    m_hold = max(1, int(m * holdout_frac)) if m else 0
    m_train = m - m_hold
    gen = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    def batch(idx: npt.NDArray[np.int64]) -> tuple[torch.Tensor, ...]:
        return (
            torch.from_numpy(data.obs[idx].astype(np.float32)),
            torch.from_numpy(data.card_mask[idx]),
            torch.from_numpy(data.action[idx]),
            torch.from_numpy(data.belief[idx].astype(np.int64)),
        )

    def abatch(idx: npt.NDArray[np.int64]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        b = {
            "meta_mask": torch.from_numpy(data.a_meta_mask[idx]),
            "delta_mask": torch.from_numpy(data.a_delta_mask[idx]),
            "type_mask": torch.from_numpy(data.a_type_mask[idx]),
            "capot_mask": torch.from_numpy(data.a_capot_mask[idx]),
            "meta": torch.from_numpy(data.a_meta[idx]),
            "delta": torch.from_numpy(data.a_delta[idx]),
            "type": torch.from_numpy(data.a_type[idx]),
            "capot": torch.from_numpy(data.a_capot[idx]),
            "belief": torch.from_numpy(data.a_belief[idx].astype(np.int64)),
        }
        return torch.from_numpy(data.a_obs[idx].astype(np.float32)), b

    def evaluate() -> tuple[float, float, float, float, float, float]:
        net.eval()
        correct = 0
        belief_acc_sum = belief_loss_sum = 0.0
        n_batches = 0
        with torch.no_grad():
            for start in range(n_train, n, 4096):
                idx = np.arange(start, min(n, start + 4096))
                obs, mask, action, belief = batch(idx)
                out = net(obs)
                pred = out["card"].masked_fill(~mask, _NEG_INF).argmax(dim=-1)
                correct += int((pred == action).sum().item())
                belief_acc_sum += belief_accuracy(out["belief"], belief)
                belief_loss_sum += float(belief_loss(out["belief"], belief).item())
                n_batches += 1
            meta_ok = delta_ok = type_ok = n_meta = n_bid = 0
            if m_hold:
                idx = np.arange(m_train, m)
                aobs, b = abatch(idx)
                out = net(aobs)
                meta_pred = out["meta"].masked_fill(~b["meta_mask"], _NEG_INF).argmax(dim=-1)
                meta_ok = int((meta_pred == b["meta"]).sum().item())
                n_meta = len(idx)
                is_bid = b["meta"] == META_BID
                n_bid = int(is_bid.sum().item())
                if n_bid:
                    nd = b["delta_mask"].shape[1]
                    dpred = (
                        out["delta"][is_bid][:, :nd]
                        .masked_fill(~b["delta_mask"][is_bid], _NEG_INF)
                        .argmax(dim=-1)
                    )
                    delta_ok = int((dpred == b["delta"][is_bid]).sum().item())
                    tpred = (
                        out["type"][is_bid]
                        .masked_fill(~b["type_mask"][is_bid], _NEG_INF)
                        .argmax(dim=-1)
                    )
                    type_ok = int((tpred == b["type"][is_bid]).sum().item())
        net.train()
        return (
            correct / n_hold,
            belief_acc_sum / n_batches,
            belief_loss_sum / n_batches,
            meta_ok / n_meta if n_meta else 0.0,
            delta_ok / n_bid if n_bid else 0.0,
            type_ok / n_bid if n_bid else 0.0,
        )

    steps_per_epoch = (n_train + batch_size - 1) // batch_size
    a_bs = max(32, (m_train + steps_per_epoch - 1) // steps_per_epoch) if m_train else 0
    last_train_loss = 0.0
    for epoch in range(epochs):
        order = torch.randperm(n_train, generator=gen).numpy()
        a_order = torch.randperm(m_train, generator=gen).numpy() if m_train else None
        total = 0.0
        steps = 0
        for step, start in enumerate(range(0, n_train, batch_size)):
            obs, mask, action, belief = batch(order[start : start + batch_size])
            out = net(obs)
            loss = _card_loss(out["card"], mask, action)
            if belief_coef > 0.0:
                loss = loss + belief_coef * belief_loss(out["belief"], belief)
            if a_order is not None:
                a_idx = a_order[step * a_bs : (step + 1) * a_bs]
                if len(a_idx):
                    aobs, b = abatch(a_idx)
                    aout = net(aobs)
                    loss = loss + bid_coef * _bid_loss(aout, b)
                    if belief_coef > 0.0:
                        loss = loss + belief_coef * belief_loss(aout["belief"], b["belief"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.item())
            steps += 1
        last_train_loss = total / max(1, steps)
        card_acc, b_acc, b_loss, meta_acc, delta_acc, type_acc = evaluate()
        if verbose:
            print(
                f"  clone epoch {epoch + 1:2d}/{epochs}  train_loss={last_train_loss:.4f}  "
                f"card_acc={card_acc:.3f}  meta_acc={meta_acc:.3f}  delta_acc={delta_acc:.3f}  "
                f"type_acc={type_acc:.3f}  belief_acc={b_acc:.3f}",
                flush=True,
            )

    card_acc, b_acc, b_loss, meta_acc, delta_acc, type_acc = evaluate()
    return CloneStats(
        train_loss=last_train_loss,
        holdout_card_acc=card_acc,
        holdout_belief_acc=b_acc,
        holdout_belief_loss=b_loss,
        holdout_meta_acc=meta_acc,
        holdout_delta_acc=delta_acc,
        holdout_type_acc=type_acc,
    )
