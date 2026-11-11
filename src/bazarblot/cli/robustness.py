"""Held-out opponent test: `python -m bazarblot.cli.robustness CHECKPOINT [CHECKPOINT ...]`.

A learned agent's margin over the heuristic it trained against can mean "better bidder" or just
"tuned to that opponent's style". This plays each checkpoint against heuristics whose bidding is
shifted timid (negative `bid_offset`) or aggressive (positive), which it never trained against,
and reports self-play behaviour (contract rate, made rate, bid level, what declaring is worth).

Read the margins as a profile, not a single number: a genuinely stronger bidder should hold up
across the whole range, while an opponent-tuned one wins against some styles and loses against
others (this is how the first joint-training run, which beat the training heuristic by +4.9, showed
it was only better against timid bidders).
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np

from bazarblot.agents.heuristic import HeuristicAgent
from bazarblot.agents.nn.ppo_agent import NNAgent
from bazarblot.core.rules import load_default
from bazarblot.env.actions import build_action_space
from bazarblot.eval.duplicate import play_deal_once
from bazarblot.eval.metrics import evaluate_pairs
from bazarblot.learn.checkpoint import load_checkpoint


def main(argv: list[str]) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoints", nargs="+", type=Path)
    p.add_argument("--offsets", default="-3,-1,0,2,4,8", help="comma-separated bid offsets")
    p.add_argument("--pairs", type=int, default=300)
    p.add_argument("--selfplay-deals", type=int, default=1500)
    p.add_argument("--seed", type=int, default=11)
    args = p.parse_args(argv[1:])

    rules = load_default()
    space = build_action_space(rules)
    rng = random.Random(args.seed)
    seeds = [rng.randrange(1 << 30) for _ in range(args.pairs)]
    dealers = [rng.randrange(4) for _ in range(args.pairs)]
    offsets = [int(x) for x in args.offsets.split(",")]

    for path in args.checkpoints:
        net, _ = load_checkpoint(path, rules)
        net.eval()
        agent = NNAgent(net, sample=False)
        results = [
            r
            for r in (
                play_deal_once(rules, space, i, 0, i % 4, dict.fromkeys(range(4), agent))
                for i in range(args.selfplay_deals)
            )
            if r is not None
        ]
        levels = [r.contract.level for r in results]
        worth = np.mean([r.score_attackers - r.score_defenders for r in results])
        print(f"\n{path}")
        print(
            f"  self-play, all four seats: contract {len(results) / args.selfplay_deals:.2f}, "
            f"made {np.mean([r.made for r in results]):.2f}, bid level median "
            f"{np.median(levels):.0f} / p90 {np.percentile(levels, 90):.0f}, "
            f"declaring worth {worth:+.1f} per deal"
        )
        for k in offsets:
            opponent = HeuristicAgent(rules, bid_offset=k)
            rep = evaluate_pairs(rules, space, seeds, dealers, agent, opponent, n_resamples=2000)
            print(
                f"  vs heuristic bid_offset={k:+d}: {rep.margin.mean:+7.2f}  "
                f"CI ({rep.margin.lo:+.1f}, {rep.margin.hi:+.1f})",
                flush=True,
            )


if __name__ == "__main__":
    main(sys.argv)
