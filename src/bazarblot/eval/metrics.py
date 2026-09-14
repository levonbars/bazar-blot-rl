"""Aggregate evaluation metrics (M5, environment spec §9), built on `eval.duplicate`'s paired
deals — every number here comes from paired comparisons, and every reported number carries a
bootstrap confidence interval rather than a bare point estimate.

**Two clearly separated cost tiers.** `evaluate_pairs` is cheap — no DD-solving anywhere in it —
and affordable at hundreds or thousands of deals: the paired margin, and the contract-make rate
for whichever policy is under test. `evaluate_dd_oracle_metrics` is expensive: bid-accuracy-vs-
oracle and points-lost-vs-DD-optimal both need at least one full 8-trick DD solve per deal (the
oracle-bid metric needs five, one per candidate contract type), at the 1-40+ second per-solve cost
`solver/dd.py` documents. It's built on `solver.batch.solve_many` and meant for a much smaller
sample — tens to low hundreds of deals, not the thousands `evaluate_pairs` can afford.

**What "DD oracle bid" means here, and what it doesn't.** For a real deal, the true oracle bid
would need the auction's actual competitive dynamics — could the other team have outbid a higher
declaration? — which the environment spec (§8) acknowledges is itself only approximable ("sampled
partner/opponent distributions"). This module computes something narrower and exact instead:
given the REAL four hands from this specific deal (no sampling — the deal already happened, so
its hands are ground truth) and the team that actually ended up declaring, what's the highest
value that team could have reached across every candidate contract type? That's "could this team
have bid smarter, given how this exact deal really lay" — not "was the final contract level
optimal against a rational opponent," which is a genuinely harder problem out of scope here.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from bazarblot.agents.base import Agent
from bazarblot.core.auction import opener_seat
from bazarblot.core.deal import DealResult
from bazarblot.core.dealing import deal_hands, deal_seed
from bazarblot.core.declarations import resolve_combinations
from bazarblot.core.rules import RuleConfig
from bazarblot.core.scoring import round10
from bazarblot.env.actions import ActionSpace
from bazarblot.eval.duplicate import play_deal_once, play_duplicate_pair
from bazarblot.solver.batch import SolveSpec, solve_many
from bazarblot.solver.dd import Hands


@dataclass(frozen=True, slots=True)
class ConfidenceInterval:
    mean: float
    lo: float
    hi: float

    def contains(self, value: float) -> bool:
        return self.lo <= value <= self.hi


def bootstrap_ci(
    samples: list[float], n_resamples: int = 10_000, ci: float = 0.95, seed: int | None = None
) -> ConfidenceInterval:
    """A percentile bootstrap CI on the mean of `samples`. Every metric this module reports uses
    one — no point estimate goes in a table without it."""
    if not samples:
        raise ValueError("bootstrap_ci: samples must be non-empty")
    arr = np.asarray(samples, dtype=np.float64)
    rng = np.random.default_rng(seed)
    resample_idx = rng.integers(0, len(arr), size=(n_resamples, len(arr)))
    resample_means = arr[resample_idx].mean(axis=1)
    alpha = (1 - ci) / 2
    lo, hi = np.percentile(resample_means, [100 * alpha, 100 * (1 - alpha)])
    return ConfidenceInterval(mean=float(arr.mean()), lo=float(lo), hi=float(hi))


@dataclass(frozen=True, slots=True)
class DuplicateEvalReport:
    n_pairs: int
    margin: ConfidenceInterval
    """Policy X's paired raw-score margin over Y (spec §9's "raw points per deal, paired")."""
    make_rate: ConfidenceInterval | None
    """Fraction of instances where X's team was declaring and made its contract, across both
    seat assignments in every pair — `None` if X never ended up declaring in the sample."""


def evaluate_pairs(
    rules: RuleConfig,
    space: ActionSpace,
    seeds: list[int],
    dealers: list[int],
    policy_x: Agent,
    policy_y: Agent,
    n_resamples: int = 10_000,
    ci_seed: int | None = None,
) -> DuplicateEvalReport:
    """Run one duplicate pair per `(seed, dealer)` and aggregate. No DD-solving — affordable at
    however many pairs `seeds`/`dealers` ask for."""
    if len(seeds) != len(dealers):
        raise ValueError("seeds and dealers must be the same length")
    margins: list[float] = []
    make_flags: list[float] = []
    for s, d in zip(seeds, dealers, strict=True):
        result = play_duplicate_pair(rules, space, s, d, policy_x, policy_y)
        if result is None:
            continue
        margins.append(result.margin_x)
        for deal_result, x_team in (
            (result.result_x_as_team0, 0),
            (result.result_x_as_team1, 1),
        ):
            if deal_result.attackers_team == x_team:
                make_flags.append(1.0 if deal_result.made else 0.0)

    if not margins:
        raise RuntimeError("every requested pair aborted on every resample attempt")
    margin_ci = bootstrap_ci(margins, n_resamples=n_resamples, seed=ci_seed)
    make_ci = (
        bootstrap_ci(make_flags, n_resamples=n_resamples, seed=ci_seed) if make_flags else None
    )
    return DuplicateEvalReport(n_pairs=len(margins), margin=margin_ci, make_rate=make_ci)


@dataclass(frozen=True, slots=True)
class DDOracleReport:
    n_deals: int
    bid_exact_agreement: ConfidenceInterval
    """Fraction of deals where the actual final bid level exactly equals the oracle's best
    achievable level for the team that ended up declaring."""
    bid_within_one_level: ConfidenceInterval
    """Same, but counting agreement within +/-1 level as a match too."""
    points_lost_vs_optimal: ConfidenceInterval
    """DD-optimal play in the ACTUAL contract, from the exact leader/hands this deal really had,
    minus what the declaring team actually took. Despite the name, this is NOT bounded at zero:
    the DD value assumes optimal play from BOTH sides, so if the real defense in a given deal
    played worse than double-dummy-optimal, the real declarer can exceed that value and this
    comes out negative. That's expected, not a bug — it's the standard double-dummy "par"
    comparison (spec §9: "how much worse than a cheating player"), not a fixed-opponent
    recomputation that only re-optimizes the declaring side."""


def evaluate_dd_oracle_metrics(
    rules: RuleConfig,
    space: ActionSpace,
    seeds: list[int],
    dealers: list[int],
    policy: Agent,
    n_resamples: int = 10_000,
    ci_seed: int | None = None,
    workers: int | None = None,
) -> DDOracleReport:
    """DD-oracle bid accuracy and trick-play efficiency for `policy` playing all four seats
    (self-play), one deal per `(seed, dealer)`. Expensive — see module docstring; keep `seeds`
    short (tens to low hundreds) relative to what `evaluate_pairs` can afford."""
    if len(seeds) != len(dealers):
        raise ValueError("seeds and dealers must be the same length")

    agents = {s: policy for s in range(4)}
    deals: list[tuple[DealResult, Hands, int]] = []
    for s, d in zip(seeds, dealers, strict=True):
        result = play_deal_once(rules, space, s, 0, d, agents)
        if result is None:
            continue
        # `play_deal_once` doesn't hand back the original hands directly — cheaply re-derive
        # them from the identical (seed, deal_number) rather than threading extra state through
        # its return type just for this one, expensive-tier caller. Re-dealing is just a
        # shuffle, negligible next to the DD solves this function is about to do.
        rng = random.Random(deal_seed(s, 0, 0))
        dealt = deal_hands(rng, rules)
        hands: Hands = (dealt[0], dealt[1], dealt[2], dealt[3])
        leader = opener_seat(d, rules)
        deals.append((result, hands, leader))

    if not deals:
        raise RuntimeError("every requested deal aborted")

    contract_types = list(rules.contracts.types)

    oracle_specs: list[SolveSpec] = []
    for result, hands, leader in deals:
        for ct in contract_types:
            oracle_specs.append(
                SolveSpec(
                    hands=hands,
                    contract_type=ct,
                    to_act=leader,
                    declaring_team=result.attackers_team,
                )
            )
    oracle_solved = solve_many(oracle_specs, rules, workers=workers)

    actual_specs = [
        SolveSpec(
            hands=hands,
            contract_type=result.contract.contract_type,
            to_act=leader,
            declaring_team=result.attackers_team,
        )
        for result, hands, leader in deals
    ]
    actual_solved = solve_many(actual_specs, rules, workers=workers)

    exact_flags: list[float] = []
    within_one_flags: list[float] = []
    points_lost: list[float] = []

    for i, (result, hands, leader) in enumerate(deals):
        best_level = 0
        for j, ct in enumerate(contract_types):
            dd_points = oracle_solved[i * len(contract_types) + j].declarer_points
            combo_result = resolve_combinations(hands, ct, rules, leader)
            combo = combo_result.team_points[result.attackers_team]
            scaled = round10(dd_points, rules) + combo
            best_level = max(best_level, scaled)

        actual_level = result.contract.level
        exact_flags.append(1.0 if actual_level == best_level else 0.0)
        within_one_flags.append(1.0 if abs(actual_level - best_level) <= 1 else 0.0)

        # `declaring_team=result.attackers_team` was fixed when `actual_specs` was built above,
        # so the attackers' raw points are always the right side of this comparison — no
        # conditional needed, the declaring team here is the attacking team by definition.
        optimal_raw = actual_solved[i].declarer_points
        actual_raw = result.raw_attackers
        points_lost.append(float(optimal_raw - actual_raw))

    return DDOracleReport(
        n_deals=len(deals),
        bid_exact_agreement=bootstrap_ci(exact_flags, n_resamples=n_resamples, seed=ci_seed),
        bid_within_one_level=bootstrap_ci(within_one_flags, n_resamples=n_resamples, seed=ci_seed),
        points_lost_vs_optimal=bootstrap_ci(points_lost, n_resamples=n_resamples, seed=ci_seed),
    )
