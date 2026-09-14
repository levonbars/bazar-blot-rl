"""A minimal historical-checkpoint opponent pool (environment spec §7.1: "League/opponent pool
with historical checkpoints to avoid self-play cycling"). Pure self-play against an
always-identical, always-current copy of itself lets a policy's bidding and play drift into a
private, mutually-consistent-but-exploitable convention (classic self-play cycling/collusion) --
periodically freezing a snapshot and mixing it into training as an opponent gives the learner
something that does not shift out from under it every update, at essentially no extra cost."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

from bazarblot.agents.nn.network import PolicyValueNet


@dataclass(slots=True)
class OpponentPool:
    max_size: int = 8
    snapshots: list[PolicyValueNet] = field(default_factory=list)

    def add(self, net: PolicyValueNet) -> None:
        """Freeze a deep copy of `net`'s current parameters (`eval()`, no grad needed since
        `agents/nn/ppo_agent.decide` already wraps its forward pass in `torch.no_grad()`) and
        keep at most the `max_size` most recent -- oldest evicted first, so the pool tracks a
        recent window of the learner's own history rather than growing unbounded."""
        snapshot = copy.deepcopy(net)
        snapshot.eval()
        for p in snapshot.parameters():
            p.requires_grad_(False)
        self.snapshots.append(snapshot)
        if len(self.snapshots) > self.max_size:
            self.snapshots.pop(0)

    def __len__(self) -> int:
        return len(self.snapshots)
