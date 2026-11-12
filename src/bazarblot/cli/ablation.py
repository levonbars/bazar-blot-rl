"""Bidding-vs-play ablation (roadmap M8(d)): `python -m bazarblot.cli.ablation CHECKPOINT [...]`.

A learned agent's margin over the heuristic mixes two skills. Swapping one phase at a time for the
heuristic's separates them. Each line is a paired (duplicate) margin, X over Y, same shuffles:

- `full`:            learned bid + learned play   vs  heuristic
- `bidding only`:    learned bid + heuristic play vs  heuristic  (play held at heuristic level)
- `play only`:       heuristic bid + learned play vs  heuristic  (identical auctions both sides)
- `play, own bids`:  learned bid + learned play   vs  learned bid + heuristic play
                     (identical auctions again, but the agent's OWN auctions -- where reading
                     partner's bids could matter, unlike under the heuristic's auctions)

The heuristic plays from its hand and the contract and ignores the rest of the auction, so the
last line is learned play against a reference that cannot use bidding information at all.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

from bazarblot.agents.base import Agent
from bazarblot.agents.composed import ComposedAgent
from bazarblot.agents.heuristic import HeuristicAgent
from bazarblot.agents.nn.ppo_agent import NNAgent
from bazarblot.core.rules import load_default
from bazarblot.env.actions import build_action_space
from bazarblot.eval.metrics import evaluate_pairs
from bazarblot.learn.checkpoint import load_checkpoint


def main(argv: list[str]) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoints", nargs="+", type=Path)
    p.add_argument("--pairs", type=int, default=300)
    p.add_argument("--seed", type=int, default=23)
    args = p.parse_args(argv[1:])

    rules = load_default()
    space = build_action_space(rules)
    rng = random.Random(args.seed)
    seeds = [rng.randrange(1 << 30) for _ in range(args.pairs)]
    dealers = [rng.randrange(4) for _ in range(args.pairs)]
    heuristic = HeuristicAgent(rules)

    for path in args.checkpoints:
        net, _ = load_checkpoint(path, rules)
        net.eval()
        nn: Agent = NNAgent(net, sample=False)
        comparisons: list[tuple[str, Agent, Agent]] = [
            ("full", nn, heuristic),
            ("bidding only", ComposedAgent(nn, heuristic), heuristic),
            ("play only", ComposedAgent(heuristic, nn), heuristic),
            ("play, own bids", nn, ComposedAgent(nn, heuristic)),
        ]
        print(path)
        for label, x, y in comparisons:
            report = evaluate_pairs(rules, space, seeds, dealers, x, y, n_resamples=2000)
            m = report.margin
            print(
                f"  {label:15s} {m.mean:+6.2f}  CI ({m.lo:+.1f}, {m.hi:+.1f})  "
                f"n_pairs={report.n_pairs}",
                flush=True,
            )


if __name__ == "__main__":
    main(sys.argv)
