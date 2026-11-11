"""Measure the fixed reference agents against each other: `python -m bazarblot.cli.baselines`.

Prints (1) how the heuristic behaves when all four seats are the heuristic -- how often a contract
is reached and made, the median bid, and what declaring is worth -- and (2) paired margins between
the standard reference agents (random, heuristic, and an always-pass bidder with heuristic play).

These are the yardsticks every learned agent is compared with, so they are worth re-running
whenever a baseline changes. The always-pass row is the important sanity check: a healthy
bidder must beat it; if conceding every auction ever wins, the baseline's bidding is broken
(which is how the heuristic's combination-units bug was found).
"""

from __future__ import annotations

import argparse
import random
import sys

import numpy as np

from bazarblot.agents.base import Agent
from bazarblot.agents.composed import ComposedAgent
from bazarblot.agents.heuristic import HeuristicAgent
from bazarblot.agents.pass_bidder import PassBidder
from bazarblot.agents.random_agent import RandomAgent
from bazarblot.core.rules import load_default
from bazarblot.env.actions import build_action_space
from bazarblot.eval.duplicate import play_deal_once
from bazarblot.eval.metrics import evaluate_pairs


def main(argv: list[str]) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selfplay-deals", type=int, default=2000)
    p.add_argument("--pairs", type=int, default=400)
    p.add_argument("--seed", type=int, default=5)
    args = p.parse_args(argv[1:])

    rules = load_default()
    space = build_action_space(rules)
    heuristic = HeuristicAgent(rules)
    random_agent = RandomAgent(random.Random(1))
    passer = ComposedAgent(PassBidder(), heuristic)

    results = [
        r
        for r in (
            play_deal_once(rules, space, i, 0, i % 4, dict.fromkeys(range(4), heuristic))
            for i in range(args.selfplay_deals)
        )
        if r is not None
    ]
    levels = [r.contract.level for r in results]
    worth = np.mean([r.score_attackers - r.score_defenders for r in results])
    print(f"heuristic in all four seats ({args.selfplay_deals} deals):")
    print(f"  deals reaching a contract: {len(results) / args.selfplay_deals:.2f}")
    print(f"  contracts made:            {np.mean([r.made for r in results]):.2f}")
    print(f"  median / p90 bid level:    {np.median(levels):.0f} / {np.percentile(levels, 90):.0f}")
    print(f"  declaring is worth:        {worth:+.1f} points per deal (attackers minus defenders)")

    rng = random.Random(args.seed)
    seeds = [rng.randrange(1 << 30) for _ in range(args.pairs)]
    dealers = [rng.randrange(4) for _ in range(args.pairs)]
    print(f"\npaired margins, X over Y ({args.pairs} pairs; positive means X is better):")
    matchups: list[tuple[str, Agent, Agent]] = [
        ("heuristic        vs random", heuristic, random_agent),
        ("heuristic        vs always-pass + heuristic play", heuristic, passer),
        ("always-pass+heur vs random", passer, random_agent),
        (
            "heuristic play   vs random play (same heuristic bidding)",
            heuristic,
            ComposedAgent(heuristic, RandomAgent(random.Random(2))),
        ),
    ]
    for name, x, y in matchups:
        rep = evaluate_pairs(rules, space, seeds, dealers, x, y, n_resamples=2000)
        print(
            f"  {name:<58} {rep.margin.mean:+7.1f}  CI ({rep.margin.lo:+.1f}, {rep.margin.hi:+.1f})"
        )


if __name__ == "__main__":
    main(sys.argv)
