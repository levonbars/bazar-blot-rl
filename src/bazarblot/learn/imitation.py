"""Behavior cloning of a teacher's card play, plus the belief head (roadmap M7, warm start).

**Why.** Training card play from the single end-of-deal margin with PPO did not get off the ground:
with bidding held fixed to the heuristic, 800 iterations (~3M decisions) left the learned player
statistically indistinguishable from random play and slightly worse than the heuristic's play
(`docs/03-implementation-roadmap.md` M7). Cloning the heuristic first gives PPO a policy that
already plays sensible cards, so fine-tuning starts from "club player" instead of from noise.
The belief head is fitted at the same time because it is a dense, per-decision supervised signal
(who holds each unseen card) that shapes the shared trunk, and it is the instrument for measuring
how much a bid tells partner.

**Noise injection.** Cloning only on the teacher's own trajectories leaves the student blind to
positions the teacher would never reach, so it drifts off-distribution the first time it deviates.
With probability `noise_prob` the EXECUTED card is a random legal one, while the recorded label is
still the teacher's choice at that position. The dataset then covers the neighbourhood of the
teacher's play with the teacher's corrections (DART-style), at no extra cost.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import torch
from torch.nn import functional as F

from bazarblot.agents.base import Agent
from bazarblot.agents.nn.network import PolicyValueNet
from bazarblot.core.deal import Deal, Phase
from bazarblot.core.dealing import deal_hands, deal_seed
from bazarblot.core.rules import RuleConfig
from bazarblot.env.actions import ActionSpace, legal_mask
from bazarblot.env.infoset import info_set
from bazarblot.env.obs import encode, flatten
from bazarblot.env.tracked_deal import TrackedDeal
from bazarblot.learn.belief import belief_accuracy, belief_loss, belief_target

_NEG_INF = -1e9


@dataclass(slots=True)
class ImitationData:
    obs: npt.NDArray[np.float16]  # (N, OBS_DIM), float16 to halve memory; observations are 0/small
    card_mask: npt.NDArray[np.bool_]  # (N, n_cards) legal cards at the decision
    action: npt.NDArray[np.int64]  # (N,) the teacher's card index (0..n_cards-1)
    belief: npt.NDArray[np.int8]  # (N, n_cards) holder labels, IGNORE where unlabelled

    def __len__(self) -> int:
        return int(self.action.shape[0])


def collect_imitation_data(
    rules: RuleConfig,
    space: ActionSpace,
    teacher: Agent,
    n_deals: int,
    seed: int,
    noise_prob: float = 0.1,
) -> ImitationData:
    """Play `n_deals` shuffles with `teacher` making every decision (auction included, so the
    contracts are the ones the teacher actually bids), recording one sample per card-play decision.
    Aborted deals contribute nothing."""
    rng = random.Random(seed)
    obs_rows: list[npt.NDArray[np.float16]] = []
    mask_rows: list[npt.NDArray[np.bool_]] = []
    action_rows: list[int] = []
    belief_rows: list[npt.NDArray[np.int8]] = []

    for i in range(n_deals):
        hands = deal_hands(random.Random(deal_seed(seed + i, 0, 0)), rules)
        deal = Deal(rules, dealer=i % 4, hands=hands, deal_id=0)
        tracked = TrackedDeal(deal)
        while deal.phase in (Phase.AUCTION, Phase.PLAY):
            seat = deal.to_act
            info = info_set(tracked, seat, match_score=(0, 0), deal_number=0)
            mask = legal_mask(info, space)
            action_idx = teacher.act(info, space, mask)
            if info.phase == Phase.PLAY:
                obs_rows.append(flatten(encode(info)).astype(np.float16))
                mask_rows.append(mask[space.play_start : space.action_dim].copy())
                action_rows.append(action_idx - space.play_start)
                belief_rows.append(belief_target(deal.hands, seat))
                if noise_prob > 0.0 and rng.random() < noise_prob:
                    legal = mask.nonzero()[0]
                    action_idx = int(legal[rng.randrange(len(legal))])
            tracked.step(space.decode(action_idx))

    return ImitationData(
        obs=np.stack(obs_rows),
        card_mask=np.stack(mask_rows),
        action=np.asarray(action_rows, dtype=np.int64),
        belief=np.stack(belief_rows),
    )


@dataclass(frozen=True, slots=True)
class CloneStats:
    train_loss: float
    holdout_card_acc: float
    holdout_belief_acc: float
    holdout_belief_loss: float


def _card_loss(card_logits: torch.Tensor, mask: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(card_logits.masked_fill(~mask, _NEG_INF), action)


def behavior_clone(
    net: PolicyValueNet,
    data: ImitationData,
    *,
    epochs: int = 10,
    batch_size: int = 512,
    lr: float = 1e-3,
    belief_coef: float = 0.5,
    holdout_frac: float = 0.1,
    seed: int = 0,
    verbose: bool = True,
) -> CloneStats:
    """Fit the card head (and belief head) of `net` to `data` by supervised learning. The holdout is
    the TAIL of the dataset: samples are stored deal by deal, so a tail split keeps whole deals out
    of training instead of leaking neighbouring positions from the same deal."""
    n = len(data)
    n_hold = max(1, int(n * holdout_frac))
    n_train = n - n_hold
    gen = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    def batch(idx: npt.NDArray[np.int64]) -> tuple[torch.Tensor, ...]:
        return (
            torch.from_numpy(data.obs[idx].astype(np.float32)),
            torch.from_numpy(data.card_mask[idx]),
            torch.from_numpy(data.action[idx]),
            torch.from_numpy(data.belief[idx].astype(np.int64)),
        )

    def evaluate() -> tuple[float, float, float]:
        net.eval()
        correct = 0
        belief_acc_sum = 0.0
        belief_loss_sum = 0.0
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
        net.train()
        return correct / n_hold, belief_acc_sum / n_batches, belief_loss_sum / n_batches

    last_train_loss = 0.0
    for epoch in range(epochs):
        order = torch.randperm(n_train, generator=gen).numpy()
        total = 0.0
        steps = 0
        for start in range(0, n_train, batch_size):
            obs, mask, action, belief = batch(order[start : start + batch_size])
            out = net(obs)
            loss = _card_loss(out["card"], mask, action)
            if belief_coef > 0.0:
                loss = loss + belief_coef * belief_loss(out["belief"], belief)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.item())
            steps += 1
        last_train_loss = total / max(1, steps)
        card_acc, b_acc, b_loss = evaluate()
        if verbose:
            print(
                f"  clone epoch {epoch + 1:2d}/{epochs}  train_loss={last_train_loss:.4f}  "
                f"holdout card_acc={card_acc:.3f}  belief_acc={b_acc:.3f}  "
                f"belief_loss={b_loss:.3f}",
                flush=True,
            )

    card_acc, b_acc, b_loss = evaluate()
    return CloneStats(
        train_loss=last_train_loss,
        holdout_card_acc=card_acc,
        holdout_belief_acc=b_acc,
        holdout_belief_loss=b_loss,
    )
