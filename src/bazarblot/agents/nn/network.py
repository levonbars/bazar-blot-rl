"""The M7 model-free baseline network: a shared MLP trunk over `v1` observation features
(`env/obs.py::flatten`) feeding a value head plus the phase-appropriate policy head(s) --
factored `(Delta, type, capot)` for a bid (spec §3.2), flat over `PASS`/`CONTRA`/`RECONTRA`/`BID`
during the auction, flat over the 32 cards during play.

**Belief and contract-outcome heads (spec §7.2) are built; DD-value is not.** The outcome head
predicts how the deal ends (`OUTCOME_*`) from the deciding seat's view. The belief head
predicts, per card, which of the three other seats holds it, trained from the ground-truth deal
(`learn/belief.py`) as a dense auxiliary signal, and doubles as the measurement of how much
information the bidding conveys to a partner. The other two are real, documented future work
(see `docs/03-implementation-roadmap.md` M7), not an oversight: contract-outcome needs its own
label construction, and the DD-value head's "free" labels are only free in the sense
that `solver/dd.py` already exists -- each label is still a multi-second-to-tens-of-seconds full
solve (M2/M2.5), which would make DD-value labels the dominant cost of every training batch and
undo M6's throughput work. Shipping the plain policy+value baseline first (the spec's own
"family 1, ship this first" recommendation, §7.1) keeps the initial training loop honest and
inexpensive; auxiliary heads are a clearly separable follow-up, not a blocker for M7's own
"Done when" bar (beats `heuristic` / `PIMC-20` on a paired evaluation).

Everything here operates on the SAME shared parameters for all four seats (self-play): one
network serves every seat, matching `env/obs.py`'s seat-relative encoding, which is exactly what
makes that sharing sound.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from bazarblot.agents.nn.bid_factor import n_delta_slots
from bazarblot.core.rules import RuleConfig
from bazarblot.env.actions import ActionSpace, build_action_space
from bazarblot.env.obs import BLOCK_SHAPES

OBS_DIM = sum(math.prod(shape) for shape in BLOCK_SHAPES.values())

# Meta head order for the auction phase: which of the four top-level auction choices this is.
META_PASS, META_CONTRA, META_RECONTRA, META_BID = range(4)
N_META = 4

# Belief head: for each of the 32 cards, a 3-way distribution over WHO holds it, among the three
# other seats in seat-relative order (rel 1 = left opponent, 2 = partner, 3 = right opponent).
# Only cards that are still in some other seat's hand carry a label (`learn/belief.py`).
N_BELIEF_SEATS = 3

# Contract-outcome head (spec §7.2, "*the* bidding signal"): how the deal will end, from the
# deciding seat's point of view. Learned from outcomes (free labels from the finished deal), so it
# carries no hand-written knowledge; it gives the trunk a dense reason to represent "can my side
# make a contract here" during the auction, where the terminal reward alone is a thin signal.
OUTCOME_ABORT, OUTCOME_WE_MAKE, OUTCOME_WE_FAIL, OUTCOME_THEY_MAKE, OUTCOME_THEY_FAIL = range(5)
N_OUTCOMES = 5


@dataclass(frozen=True, slots=True)
class NetworkHeads:
    """Head widths derived from one `ActionSpace` -- never hard-coded, for the same reason
    `ActionSpace` itself is derived from `RuleConfig` (a preset change must not silently desync
    the network from the environment)."""

    obs_dim: int
    n_delta: int
    n_types: int
    capot_states: int
    n_cards: int

    @classmethod
    def from_space(cls, space: ActionSpace) -> NetworkHeads:
        return cls(
            obs_dim=OBS_DIM,
            n_delta=n_delta_slots(space),
            n_types=len(space.contract_types),
            capot_states=space.capot_states,
            n_cards=space.action_dim - space.play_start,
        )


def _trunk(in_dim: int, hidden: tuple[int, ...]) -> nn.Sequential:
    """`Linear -> ReLU` repeated over `hidden`, ending in a ReLU -- a plain feature trunk with no
    output layer of its own; heads are separate `Linear`s applied to its final activation."""
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden:
        layers.append(nn.Linear(prev, h))
        layers.append(nn.ReLU())
        prev = h
    return nn.Sequential(*layers)


class PolicyValueNet(nn.Module):
    """`obs (B, OBS_DIM) -> logits for every head + value (B,)`. Masking (setting illegal logits
    to -inf) and the hierarchical bid sampling/scoring are the caller's job
    (`agents/nn/ppo_agent.py`) -- this module only produces raw, unmasked logits, so it stays a
    plain, easily-unit-tested function of the observation."""

    def __init__(self, heads: NetworkHeads, hidden: tuple[int, ...] = (256, 256)) -> None:
        super().__init__()
        self.heads = heads
        self.trunk = _trunk(heads.obs_dim, hidden)
        trunk_out = hidden[-1]
        self.meta_head = nn.Linear(trunk_out, N_META)
        self.delta_head = nn.Linear(trunk_out, heads.n_delta)
        self.type_head = nn.Linear(trunk_out, heads.n_types)
        self.capot_head = nn.Linear(trunk_out, heads.capot_states)
        self.card_head = nn.Linear(trunk_out, heads.n_cards)
        self.value_head = nn.Linear(trunk_out, 1)
        self.belief_head = nn.Linear(trunk_out, heads.n_cards * N_BELIEF_SEATS)
        self.outcome_head = nn.Linear(trunk_out, N_OUTCOMES)

    def forward(self, obs: Tensor) -> dict[str, Tensor]:
        trunk = self.trunk(obs)
        return {
            "meta": self.meta_head(trunk),
            "delta": self.delta_head(trunk),
            "type": self.type_head(trunk),
            "capot": self.capot_head(trunk),
            "card": self.card_head(trunk),
            "value": self.value_head(trunk).squeeze(-1),
            "belief": self.belief_head(trunk).view(-1, self.heads.n_cards, N_BELIEF_SEATS),
            "outcome": self.outcome_head(trunk),
        }


def apply_action_prior(
    net: PolicyValueNet,
    raise_decay: float = 0.0,
    capot_bias: float = 0.0,
    double_bias: float = 0.0,
) -> None:
    """Set the starting BIASES of the auction heads of an untrained network (weights untouched).

    This is a prior on the shape of the action space, not on strategy: it says nothing about
    which hands to bid on, which suit, or when to double. It exists because the factored bid head
    makes a uniform start absurd -- the raise head is uniform over 73 raise sizes, so an untrained
    network opens near level 40 on average and almost never makes a contract, and half of its bids
    are capot (all eight tricks). PPO then learns mostly that bidding loses.

    - `raise_decay`: logit of a raise of `k` levels is lowered by `raise_decay * (k - 1)`, so small
      raises are likelier; with 0.5, about 40% of raises are +1 and ~85% are at most +4.
    - `capot_bias`: added to the capot logit (index 1 of the capot head).
    - `double_bias`: added to the contra and recontra logits of the meta head.
    """
    with torch.no_grad():
        n_delta = net.delta_head.bias.shape[0]
        net.delta_head.bias.copy_(-raise_decay * torch.arange(n_delta, dtype=torch.float32))
        if net.capot_head.bias.shape[0] > 1:
            net.capot_head.bias.zero_()
            net.capot_head.bias[1] = capot_bias
        net.meta_head.bias.zero_()
        net.meta_head.bias[META_CONTRA] = double_bias
        net.meta_head.bias[META_RECONTRA] = double_bias


PRIVILEGED_DIM = 32 * N_BELIEF_SEATS


class CentralCritic(nn.Module):
    """Centralized value function (spec §7.4): `V(information-set observation, hidden hands)`.

    Takes the actor's observation PLUS the other three seats' unplayed hands (`PRIVILEGED_DIM`
    one-hot features, `learn/belief.py::privileged_features`). Because it knows the deal, its
    estimate absorbs most of the luck of the cards dealt, so a decision's advantage reflects the
    decision and not the deal -- the main variance reduction available when one terminal reward is
    shared by ~9 decisions. It is a SEPARATE network from the actor and exists for training only:
    the actor never receives hidden hands, by construction."""

    def __init__(self, obs_dim: int = OBS_DIM, hidden: tuple[int, ...] = (256, 256)) -> None:
        super().__init__()
        self.hidden = hidden
        self.trunk = _trunk(obs_dim + PRIVILEGED_DIM, hidden)
        self.head = nn.Linear(hidden[-1], 1)

    def forward(self, obs: Tensor, privileged: Tensor) -> Tensor:
        value: Tensor = self.head(self.trunk(torch.cat([obs, privileged], dim=-1))).squeeze(-1)
        return value


def build_network(rules: RuleConfig, hidden: tuple[int, ...] = (256, 256)) -> PolicyValueNet:
    space = build_action_space(rules)
    return PolicyValueNet(NetworkHeads.from_space(space), hidden=hidden)
