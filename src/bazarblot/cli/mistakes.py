"""Obvious-mistake counts (`learn/mistakes.py`) for checkpoints and/or the heuristic.

python -m bazarblot.cli.mistakes checkpoints/run/final.pt heuristic --deals 400
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

from bazarblot.agents.base import Agent
from bazarblot.agents.heuristic import HeuristicAgent
from bazarblot.core.rules import load_default
from bazarblot.env.actions import build_action_space
from bazarblot.learn.mistakes import measure_mistakes


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("agents", nargs="+", help="checkpoint paths, or 'heuristic'")
    p.add_argument("--deals", type=int, default=400)
    p.add_argument("--seed", type=int, default=31337)
    args = p.parse_args(argv)
    rules = load_default()
    space = build_action_space(rules)
    for name in args.agents:
        agent: Agent
        if name == "heuristic":
            agent = HeuristicAgent(rules, random.Random(1))
        else:
            from bazarblot.agents.nn.ppo_agent import NNAgent
            from bazarblot.learn.checkpoint import load_checkpoint

            net, _ = load_checkpoint(Path(name), rules)
            net.eval()
            agent = NNAgent(net, sample=False)
        report = measure_mistakes(rules, space, agent, n_deals=args.deals, seed=args.seed)
        print(f"{name}\n  {report.summary()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
