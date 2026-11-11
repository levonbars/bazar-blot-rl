"""Behavior-clone the heuristic's bidding AND card play (and fit the belief head), then report how
the cloned network behaves as a full agent against the heuristic.

`python -m bazarblot.cli.imitate --out checkpoints/m7_bc/bc.pt`

The result is the starting point for joint self-play (`cli.train --init-checkpoint ...`); see
`learn/imitation.py` for why this exists. Success is judged on BEHAVIOUR, not just agreement with
the teacher's labels: the clone should reach contracts, declare and make them at about the
heuristic's own rates, and sit level with the heuristic in paired play.
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
from bazarblot.agents.pass_bidder import PassBidder
from bazarblot.agents.random_agent import RandomAgent
from bazarblot.core.rules import load_default
from bazarblot.env.actions import build_action_space
from bazarblot.eval.behavior import evaluate_behavior
from bazarblot.eval.metrics import evaluate_pairs
from bazarblot.learn.checkpoint import save_checkpoint
from bazarblot.learn.imitation import behavior_clone, collect_imitation_data


def main(argv: list[str]) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n-deals", type=int, default=12000)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--belief-coef", type=float, default=0.5)
    p.add_argument("--bid-coef", type=float, default=1.0)
    p.add_argument("--noise-prob", type=float, default=0.1)
    p.add_argument("--auction-noise-prob", type=float, default=0.05)
    p.add_argument("--eval-pairs", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv[1:])

    rules = load_default()
    space = build_action_space(rules)
    torch.manual_seed(args.seed)
    teacher = HeuristicAgent(rules)

    t0 = time.perf_counter()
    data = collect_imitation_data(
        rules,
        space,
        teacher,
        args.n_deals,
        seed=args.seed,
        noise_prob=args.noise_prob,
        auction_noise_prob=args.auction_noise_prob,
    )
    print(
        f"collected {len(data)} card-play and {data.n_auction} auction samples from "
        f"{args.n_deals} deals in {time.perf_counter() - t0:.0f}s",
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
        bid_coef=args.bid_coef,
        seed=args.seed,
    )
    print(
        f"cloned in {time.perf_counter() - t0:.0f}s; holdout vs teacher: "
        f"card {stats.holdout_card_acc:.3f}, meta {stats.holdout_meta_acc:.3f}, "
        f"bid level {stats.holdout_delta_acc:.3f}, "
        f"bid suit {stats.holdout_type_acc:.3f}, belief {stats.holdout_belief_acc:.3f}",
        flush=True,
    )

    save_checkpoint(args.out, net, rules, extra={"iteration": 0, "stage": "behavior_clone"})
    print(f"saved {args.out}", flush=True)

    net.eval()
    rng = random.Random(args.seed + 1)
    seeds = [rng.randrange(1 << 30) for _ in range(args.eval_pairs)]
    dealers = [rng.randrange(4) for _ in range(args.eval_pairs)]
    clone = NNAgent(net, sample=False)

    print("\nbehaviour (60% of the pairs; teacher row is the heuristic against itself):")
    n_b = max(20, args.eval_pairs // 3)
    for name, agent in (("clone    ", clone), ("heuristic", teacher)):
        report = evaluate_behavior(rules, space, agent, teacher, seeds[:n_b], dealers[:n_b])
        print(f"  {name} vs heuristic: {report.summary()}", flush=True)

    print(f"\npaired margins, clone over Y ({args.eval_pairs} pairs):")
    opponents = (
        ("heuristic", teacher),
        ("always-pass bidder + heuristic play", ComposedAgent(PassBidder(), teacher)),
        ("random", RandomAgent(random.Random(args.seed + 2))),
    )
    for name, opponent in opponents:
        pairs = evaluate_pairs(rules, space, seeds, dealers, clone, opponent, n_resamples=2000)
        print(
            f"  vs {name:<38} {pairs.margin.mean:+7.1f}  "
            f"CI ({pairs.margin.lo:+.1f}, {pairs.margin.hi:+.1f})",
            flush=True,
        )


if __name__ == "__main__":
    main(sys.argv)
