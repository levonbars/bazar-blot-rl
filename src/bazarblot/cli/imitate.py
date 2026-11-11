"""Behavior-clone the heuristic's card play (and fit the belief head), then report how the cloned
player compares with the heuristic under identical bidding.

`python -m bazarblot.cli.imitate --out checkpoints/m7_bc/bc.pt`

The result is the starting point for PPO fine-tuning (`cli.train --init-checkpoint ...`); see
`learn/imitation.py` for why this exists.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import torch

from bazarblot.agents.composed import ComposedAgent
from bazarblot.agents.heuristic import HeuristicAgent
from bazarblot.agents.nn.network import build_network
from bazarblot.agents.nn.ppo_agent import NNAgent
from bazarblot.agents.random_agent import RandomAgent
from bazarblot.core.rules import load_default
from bazarblot.env.actions import build_action_space
from bazarblot.eval.metrics import evaluate_pairs
from bazarblot.learn.checkpoint import save_checkpoint
from bazarblot.learn.imitation import behavior_clone, collect_imitation_data


def main(argv: list[str]) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n-deals", type=int, default=8000)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--belief-coef", type=float, default=0.5)
    p.add_argument("--noise-prob", type=float, default=0.1)
    p.add_argument("--eval-pairs", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv[1:])

    rules = load_default()
    space = build_action_space(rules)
    torch.manual_seed(args.seed)
    teacher = HeuristicAgent(rules)

    t0 = time.perf_counter()
    data = collect_imitation_data(
        rules, space, teacher, args.n_deals, seed=args.seed, noise_prob=args.noise_prob
    )
    print(
        f"collected {len(data)} card-play samples from {args.n_deals} deals "
        f"in {time.perf_counter() - t0:.0f}s",
        flush=True,
    )

    net = build_network(rules)
    t0 = time.perf_counter()
    stats = behavior_clone(
        net,
        data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        belief_coef=args.belief_coef,
        seed=args.seed,
    )
    print(
        f"cloned in {time.perf_counter() - t0:.0f}s; holdout card accuracy vs teacher "
        f"{stats.holdout_card_acc:.3f}, belief accuracy {stats.holdout_belief_acc:.3f}",
        flush=True,
    )

    save_checkpoint(args.out, net, rules, extra={"iteration": 0, "stage": "behavior_clone"})
    print(f"saved {args.out}", flush=True)

    net.eval()
    rng = random.Random(args.seed + 1)
    seeds = [rng.randrange(1 << 30) for _ in range(args.eval_pairs)]
    dealers = [rng.randrange(4) for _ in range(args.eval_pairs)]
    learned = ComposedAgent(teacher, NNAgent(net, sample=False))
    random_play = ComposedAgent(teacher, RandomAgent(random.Random(args.seed + 2)))
    for name, opponent in (("heuristic play", teacher), ("random play", random_play)):
        report = evaluate_pairs(rules, space, seeds, dealers, learned, opponent, n_resamples=2000)
        print(
            f"cloned player vs {name} (same heuristic bidding): margin {report.margin.mean:+.2f} "
            f"CI ({report.margin.lo:+.2f}, {report.margin.hi:+.2f}) over {report.n_pairs} pairs",
            flush=True,
        )


if __name__ == "__main__":
    main(sys.argv)
