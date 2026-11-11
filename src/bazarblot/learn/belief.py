"""Belief-head labels and loss (environment spec §7.2): for each card still in play, who holds it.

The label for a decision by `seat` is, per card, the seat-relative index (0 = left opponent,
1 = partner, 2 = right opponent, matching `rel = (other - seat) % 4` minus one) of the seat holding
that card in its UNPLAYED hand, or -1 where there is nothing to predict -- the acting seat's own
cards and cards already played, both of which the observation already shows.

These labels come from the ground-truth deal, so they are computed in training code only
(`learn/`), never inside `env/`'s information-set boundary. They supervise an auxiliary head; the
policy's input stays the clean information-set observation, so nothing hidden reaches the policy at
play time.

**Why it matters beyond a regulariser.** Conditioned on the auction, this head measures how much a
seat can infer about its partner's hand from the bidding -- i.e. how much information the bids
convey. That is the quantity the signaling work needs to track (see
`docs/03-implementation-roadmap.md` M7).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor

from bazarblot.core.cards import N_CARDS

IGNORE = -1


def belief_target(
    hands: list[frozenset[int]] | tuple[frozenset[int], ...], seat: int
) -> npt.NDArray[np.int8]:
    """`(N_CARDS,)` int8: relative holder index 0..2 for every card in another seat's unplayed
    hand, `IGNORE` (-1) for everything else. `hands` is the deal's current unplayed hands."""
    target = np.full(N_CARDS, IGNORE, dtype=np.int8)
    for other in range(4):
        if other == seat:
            continue
        rel = (other - seat) % 4  # 1..3
        for card in hands[other]:
            target[card] = rel - 1
    return target


def privileged_features(targets: npt.NDArray[np.int8]) -> npt.NDArray[np.float32]:
    """The hidden hands as critic input: `(..., N_CARDS)` holder labels -> `(..., N_CARDS * 3)`
    one-hot (per card, which of the three other seats holds it; all zeros for unlabelled cards).

    This is exactly the information a centralized critic adds to the actor's observation (spec
    §7.4): together with the information-set observation it determines the whole deal state. It is
    used for training-time value estimates only and never reaches the policy."""
    out = np.zeros((*targets.shape, 3), dtype=np.float32)
    valid = targets != IGNORE
    idx = np.nonzero(valid)
    out[(*idx, targets[valid].astype(np.int64))] = 1.0
    return out.reshape(*targets.shape[:-1], targets.shape[-1] * 3)


def belief_loss(logits: Tensor, targets: Tensor) -> Tensor:
    """Mean cross-entropy over the labelled cards of a batch. `logits` is `(B, N_CARDS, 3)`,
    `targets` is `(B, N_CARDS)` int64 with `IGNORE` where unlabelled. Returns 0 (with grad
    graph intact) if the batch has no labelled card at all."""
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_targets = targets.reshape(-1)
    if not bool((flat_targets != IGNORE).any()):
        return logits.sum() * 0.0
    return torch.nn.functional.cross_entropy(flat_logits, flat_targets, ignore_index=IGNORE)


@torch.no_grad()
def belief_accuracy(logits: Tensor, targets: Tensor) -> float:
    """Top-1 accuracy over labelled cards (chance is about 1/3)."""
    valid = targets != IGNORE
    if not bool(valid.any()):
        return 0.0
    pred = logits.argmax(dim=-1)
    return float((pred[valid] == targets[valid]).float().mean().item())
