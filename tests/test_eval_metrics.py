"""M5: `eval.metrics`. `test_random_vs_random_paired_margin_ci_contains_zero` and
`test_heuristic_vs_random_paired_margin_is_tight_and_excludes_zero` together ARE the roadmap's
stated M5 "done when" bar."""

from __future__ import annotations

import random

import pytest

from bazarblot.agents.heuristic import HeuristicAgent
from bazarblot.agents.random_agent import RandomAgent
from bazarblot.core.rules import load_default
from bazarblot.env.actions import build_action_space
from bazarblot.eval.metrics import bootstrap_ci, evaluate_dd_oracle_metrics, evaluate_pairs

RULES = load_default()
SPACE = build_action_space(RULES)


def test_bootstrap_ci_on_a_constant_sample() -> None:
    ci = bootstrap_ci([5.0] * 50, n_resamples=1000, seed=0)
    assert ci.mean == 5.0
    assert ci.lo == pytest.approx(5.0)
    assert ci.hi == pytest.approx(5.0)
    assert ci.contains(5.0)
    assert not ci.contains(4.0)


def test_bootstrap_ci_empty_raises() -> None:
    with pytest.raises(ValueError):
        bootstrap_ci([])


def test_bootstrap_ci_is_reproducible_given_a_seed() -> None:
    samples = [1.0, 2.0, 3.0, -1.0, 5.0, 0.0, 2.5]
    ci1 = bootstrap_ci(samples, n_resamples=2000, seed=42)
    ci2 = bootstrap_ci(samples, n_resamples=2000, seed=42)
    assert ci1 == ci2


def test_random_vs_random_paired_margin_ci_contains_zero() -> None:
    """M5 done-when, part 1: two independent RandomAgent instances should show no reliable skill
    difference — the paired mean should be noise, and its CI should contain 0."""
    rnd_x = RandomAgent(random.Random(1))
    rnd_y = RandomAgent(random.Random(2))
    n = 300
    seeds = list(range(n))
    dealers = [s % 4 for s in seeds]
    report = evaluate_pairs(RULES, SPACE, seeds, dealers, rnd_x, rnd_y, ci_seed=0)
    assert report.n_pairs > n * 0.9
    assert report.margin.contains(0.0), f"random vs random margin CI excludes 0: {report.margin}"


def test_heuristic_vs_random_paired_margin_is_tight_and_excludes_zero() -> None:
    """M5 done-when, part 2: a real skill gap should show up as a tight, reproducible,
    non-zero-excluding margin."""
    heur = HeuristicAgent(RULES, random.Random(3))
    rnd = RandomAgent(random.Random(4))
    n = 300
    seeds = [s + 50_000 for s in range(n)]
    dealers = [s % 4 for s in range(n)]
    report = evaluate_pairs(RULES, SPACE, seeds, dealers, heur, rnd, ci_seed=0)
    assert report.n_pairs > n * 0.9
    assert not report.margin.contains(0.0)
    assert report.margin.mean > 50.0  # a wide margin, not a marginal one
    ci_width = report.margin.hi - report.margin.lo
    assert ci_width < report.margin.mean  # "tight" relative to the size of the effect


def test_evaluate_pairs_raises_on_mismatched_lengths() -> None:
    rnd = RandomAgent(random.Random(0))
    with pytest.raises(ValueError):
        evaluate_pairs(RULES, SPACE, [0, 1], [0], rnd, rnd)


@pytest.mark.slow
def test_evaluate_dd_oracle_metrics_smoke() -> None:
    """Expensive (full 8-trick DD solves) — see module docstring. Just checks it runs, returns
    well-formed bounds, and that points-lost-vs-optimal is allowed to be negative (a real
    property of this metric, not a bug — see `DDOracleReport.points_lost_vs_optimal`)."""
    rnd = RandomAgent(random.Random(5))
    seeds = list(range(4))
    dealers = [0, 1, 2, 3]
    report = evaluate_dd_oracle_metrics(RULES, SPACE, seeds, dealers, rnd, workers=4, ci_seed=0)
    assert report.n_deals > 0
    assert 0.0 <= report.bid_exact_agreement.mean <= 1.0
    assert 0.0 <= report.bid_within_one_level.mean <= 1.0
    assert report.bid_exact_agreement.mean <= report.bid_within_one_level.mean
