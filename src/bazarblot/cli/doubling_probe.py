"""How much is doubling worth? `python -m bazarblot.cli.doubling_probe CHECKPOINT`.

Plays a trained agent against heuristics of different bidding styles twice: as it is, and with
`RuleDoubler` (a hand-written "double when they can't reach their number" rule) added on top. The
trained agents never double on their own, so a large gain here means the action is valuable and
the learning method is what is missing.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

from bazarblot.agents.base import Agent
from bazarblot.agents.doubling import RuleDoubler
from bazarblot.agents.heuristic import HeuristicAgent
from bazarblot.agents.nn.ppo_agent import NNAgent
from bazarblot.core.rules import load_default
from bazarblot.env.actions import build_action_space
from bazarblot.eval.behavior import evaluate_behavior
from bazarblot.eval.metrics import evaluate_pairs
from bazarblot.learn.checkpoint import load_checkpoint


def main(argv: list[str]) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--offsets", default="0,4,8,12", help="opponent bid offsets")
    p.add_argument("--slacks", default="0,3,6", help="doubts to try for the rule")
    p.add_argument("--pairs", type=int, default=300)
    p.add_argument("--seed", type=int, default=21)
    args = p.parse_args(argv[1:])

    rules = load_default()
    space = build_action_space(rules)
    net, _ = load_checkpoint(args.checkpoint, rules)
    net.eval()
    base = NNAgent(net, sample=False)
    rng = random.Random(args.seed)
    seeds = [rng.randrange(1 << 30) for _ in range(args.pairs)]
    dealers = [rng.randrange(4) for _ in range(args.pairs)]
    offsets = [int(x) for x in args.offsets.split(",")]

    agents: list[tuple[str, Agent]] = [("as trained", base)]
    agents += [
        (f"+ rule, slack={s}", RuleDoubler(base, rules, int(s))) for s in args.slacks.split(",")
    ]
    print(
        f"{'agent':<20}" + "".join(f"  vs offset {k:+d}".rjust(24) for k in offsets) + "  doubles"
    )
    for name, agent in agents:
        cells = []
        for k in offsets:
            rep = evaluate_pairs(
                rules,
                space,
                seeds,
                dealers,
                agent,
                HeuristicAgent(rules, bid_offset=k),
                n_resamples=1000,
            )
            cells.append(f"{rep.margin.mean:+7.1f} ({rep.margin.lo:+.0f},{rep.margin.hi:+.0f})")
        probe = evaluate_behavior(
            rules,
            space,
            agent,
            HeuristicAgent(rules, bid_offset=offsets[-1]),
            seeds[:60],
            dealers[:60],
        )
        rate = "n/a" if probe.contra_rate is None else f"{probe.contra_rate:.2f}"
        print(f"{name:<20}" + "".join(c.rjust(24) for c in cells) + f"  {rate}", flush=True)


if __name__ == "__main__":
    main(sys.argv)
