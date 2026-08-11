"""Multi-deal match bookkeeping. Rules §8.

`Match` only tracks scores, dealer rotation and target-crossing — it does not create `Deal`
objects or generate hands itself. That separation keeps this module agnostic of dealing/RNG
concerns: the caller drives a `Deal` to completion (or an abort) and reports the outcome via
`apply_deal_result`.

Redeals need no special handling here: `dealer` only advances inside `apply_deal_result`, which
is only ever called for a genuine (non-aborted) result. An aborted deal simply isn't reported,
so the caller's next `Deal` naturally uses the same `match.dealer` — "redeal with the same
dealer" falls out of the design rather than being implemented as a branch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bazarblot.core.deal import DealResult
    from bazarblot.core.rules import RuleConfig


@dataclass
class Match:
    rules: RuleConfig
    dealer: int = 0
    scores: list[int] = field(default_factory=lambda: [0, 0])
    deal_number: int = 0
    finished: bool = False
    winner: int | None = None
    history: list[DealResult] = field(default_factory=list)

    def apply_deal_result(self, result: DealResult) -> None:
        if self.finished:
            raise ValueError("match is already finished")
        team_a, team_b = result.attackers_team, 1 - result.attackers_team
        self.scores[team_a] += result.score_attackers
        self.scores[team_b] += result.score_defenders
        self.history.append(result)
        self.deal_number += 1
        self.dealer = (self.dealer + 1) % 4
        self._check_end()

    def _check_end(self) -> None:
        target = self.rules.match.target
        crossed = [i for i in (0, 1) if self.scores[i] >= target]
        if not crossed:
            return
        self.finished = True
        if len(crossed) == 1:
            self.winner = crossed[0]
            return
        # Both teams crossed the target in the same deal -> higher score wins (§8, [OWNER]).
        if self.scores[0] > self.scores[1]:
            self.winner = 0
        elif self.scores[1] > self.scores[0]:
            self.winner = 1
        else:
            # Exact tie while both cross simultaneously is unaddressed by any source. Rather
            # than guess, treat the match as undecided and let play continue — a defensible,
            # documented default for a case the rules do not cover.
            self.finished = False
