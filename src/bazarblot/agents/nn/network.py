"""The M7 model-free baseline network: a shared MLP trunk over `v1` observation features
(`env/obs.py::flatten`) feeding a value head plus the phase-appropriate policy head(s) --
factored `(Delta, type, capot)` for a bid (spec §3.2), flat over `PASS`/`CONTRA`/`RECONTRA`/`BID`
during the auction, flat over the 32 cards during play.

**Auxiliary heads (belief, DD-value, contract-outcome, spec §7.2) are deliberately not built
here.** They are real, documented future work (see `docs/03-implementation-roadmap.md` M7), not
an oversight: belief/contract-outcome need their own label-construction and loss-weighting code
that nothing here yet exercises, and the DD-value head's "free" labels are only free in the sense
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

from torch import Tensor, nn

from bazarblot.agents.nn.bid_factor import n_delta_slots
from bazarblot.core.rules import RuleConfig
from bazarblot.env.actions import ActionSpace, build_action_space
from bazarblot.env.obs import BLOCK_SHAPES

OBS_DIM = sum(math.prod(shape) for shape in BLOCK_SHAPES.values())

# Meta head order for the auction phase: which of the four top-level auction choices this is.
META_PASS, META_CONTRA, META_RECONTRA, META_BID = range(4)
N_META = 4


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

    def forward(self, obs: Tensor) -> dict[str, Tensor]:
        trunk = self.trunk(obs)
        return {
            "meta": self.meta_head(trunk),
            "delta": self.delta_head(trunk),
            "type": self.type_head(trunk),
            "capot": self.capot_head(trunk),
            "card": self.card_head(trunk),
            "value": self.value_head(trunk).squeeze(-1),
        }


def build_network(rules: RuleConfig, hidden: tuple[int, ...] = (256, 256)) -> PolicyValueNet:
    space = build_action_space(rules)
    return PolicyValueNet(NetworkHeads.from_space(space), hidden=hidden)
