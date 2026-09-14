"""Round-robin Elo over a pool of agents (M5, environment spec §9's "Elo over a round-robin
pool" bullet), computed entirely on paired (duplicate) deals — never a bare unpaired game, for
the same reason `eval.duplicate` exists at all: Belote's per-deal variance would otherwise
dominate the signal a handful of games can extract.

A duplicate pair's raw-point margin is converted to a standard Elo win/loss/draw score (1/0/0.5)
by its SIGN, not its magnitude — the same way duplicate bridge turns a paired score into IMPs/VPs
rather than reporting raw point differences directly. This keeps the rating update as the
well-understood, standard Elo formula; a magnitude-weighted variant is a reasonable extension but
not attempted here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from bazarblot.agents.base import Agent
from bazarblot.core.rules import RuleConfig
from bazarblot.env.actions import ActionSpace
from bazarblot.eval.duplicate import play_duplicate_pair

_DEFAULT_RATING = 1500.0
_DEFAULT_K = 32.0


@dataclass(slots=True)
class EloPool:
    ratings: dict[str, float] = field(default_factory=dict)
    k: float = _DEFAULT_K

    def expected_score(self, a: str, b: str) -> float:
        return 1.0 / (1.0 + 10 ** ((self.ratings[b] - self.ratings[a]) / 400.0))

    def update(self, a: str, b: str, score_a: float) -> None:
        """`score_a`: 1.0 if `a` won the pair, 0.0 if `b` won, 0.5 for a tie."""
        exp_a = self.expected_score(a, b)
        self.ratings[a] += self.k * (score_a - exp_a)
        self.ratings[b] += self.k * ((1.0 - score_a) - (1.0 - exp_a))


def margin_to_score(margin: float, draw_margin: float = 0.0) -> float:
    """A duplicate pair's raw-point margin, converted to an Elo score in `{0, 0.5, 1}` by sign."""
    if margin > draw_margin:
        return 1.0
    if margin < -draw_margin:
        return 0.0
    return 0.5


def run_round_robin(
    rules: RuleConfig,
    space: ActionSpace,
    agents: dict[str, Agent],
    n_pairs_per_matchup: int,
    seed: int = 0,
    k: float = _DEFAULT_K,
    initial_rating: float = _DEFAULT_RATING,
) -> EloPool:
    """Play `n_pairs_per_matchup` duplicate pairs between every distinct pair of named agents in
    `agents`, updating Elo ratings incrementally after each pair. Deterministic given `seed` and
    `agents`'s iteration order (Python dicts preserve insertion order) — good enough for ranking
    a research pool, not a claim of a rigorously order-invariant rating system."""
    if len(agents) < 2:
        raise ValueError("run_round_robin needs at least 2 agents")
    pool = EloPool(ratings=dict.fromkeys(agents, initial_rating), k=k)
    names = list(agents)
    trial_seed = seed
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            name_a, name_b = names[i], names[j]
            for _ in range(n_pairs_per_matchup):
                dealer = trial_seed % 4
                result = play_duplicate_pair(
                    rules, space, trial_seed, dealer, agents[name_a], agents[name_b]
                )
                trial_seed += 1
                if result is None:
                    continue
                pool.update(name_a, name_b, margin_to_score(result.margin_x))
    return pool
