"""Distill a teacher agent into a copy of its own network: `python -m bazarblot.cli.distill ...`.

The teacher is a trained checkpoint, optionally with a hand-written `RuleDoubler` on top. The
student starts as an exact copy of the teacher's network and is trained (behaviour cloning, same
loss as `cli.imitate`) on the teacher's decisions against a mix of opponents. Because the student
starts AS the teacher, everything the teacher's network learned is kept, and training only has to
add what the rule changes. This is how the doubling rule gets into the network without overwriting
what reinforcement learning taught it (`docs/03-implementation-roadmap.md` M7, step 3d).

The teacher's centralized critic is carried into the output checkpoint so PPO can resume from it.
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np
import torch

from bazarblot.agents.base import Agent
from bazarblot.agents.composed import ComposedAgent
from bazarblot.agents.doubling import RuleDoubler
from bazarblot.agents.heuristic import HeuristicAgent
from bazarblot.agents.nn.network import META_CONTRA, META_RECONTRA
from bazarblot.agents.nn.ppo_agent import NNAgent
from bazarblot.agents.pass_bidder import PassBidder
from bazarblot.core.rules import RuleConfig, load_default
from bazarblot.env.actions import build_action_space
from bazarblot.learn.checkpoint import critic_from_payload, load_checkpoint, save_checkpoint
from bazarblot.learn.imitation import behavior_clone, collect_imitation_data


def _opponent(name: str, rules: RuleConfig, teacher: Agent) -> Agent:
    if name == "self":
        return teacher
    if name == "heuristic":
        return HeuristicAgent(rules)
    if name == "passer":
        return ComposedAgent(PassBidder(), HeuristicAgent(rules))
    if name.startswith("heuristic@"):
        return HeuristicAgent(rules, bid_offset=int(name.split("@", 1)[1]))
    raise ValueError(f"unknown opponent {name!r}; expected self, heuristic, heuristic@K or passer")


def main(argv: list[str]) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--teacher-checkpoint", type=Path, required=True)
    p.add_argument("--rule-slack", type=int, default=None, help="add RuleDoubler with this slack")
    p.add_argument(
        "--opponents",
        default="heuristic@-4,heuristic@-2,heuristic,heuristic@1,heuristic@3,heuristic@6,passer,self",
    )
    p.add_argument("--n-deals", type=int, default=24000)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--belief-coef", type=float, default=0.2)
    p.add_argument("--noise-prob", type=float, default=0.1)
    p.add_argument("--auction-noise-prob", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv[1:])

    rules = load_default()
    space = build_action_space(rules)
    torch.manual_seed(args.seed)

    teacher_net, payload = load_checkpoint(args.teacher_checkpoint, rules)
    teacher_net.eval()
    teacher: Agent = NNAgent(teacher_net, sample=False)
    if args.rule_slack is not None:
        teacher = RuleDoubler(teacher, rules, args.rule_slack)
    opponents = [_opponent(n, rules, teacher) for n in args.opponents.split(",")]

    t0 = time.perf_counter()
    data = collect_imitation_data(
        rules,
        space,
        teacher,
        args.n_deals,
        seed=args.seed,
        noise_prob=args.noise_prob,
        auction_noise_prob=args.auction_noise_prob,
        opponents=opponents,
    )
    doubles = int(np.isin(data.a_meta, (META_CONTRA, META_RECONTRA)).sum())
    print(
        f"collected {len(data)} card-play and {data.n_auction} auction samples "
        f"({doubles} doubling decisions) from {args.n_deals} deals in "
        f"{time.perf_counter() - t0:.0f}s",
        flush=True,
    )

    student = copy.deepcopy(teacher_net)
    student.train()
    t0 = time.perf_counter()
    stats = behavior_clone(
        student,
        data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        belief_coef=args.belief_coef,
        seed=args.seed,
    )
    print(
        f"distilled in {time.perf_counter() - t0:.0f}s; holdout vs teacher: card "
        f"{stats.holdout_card_acc:.3f}, meta {stats.holdout_meta_acc:.3f}, bid level "
        f"{stats.holdout_delta_acc:.3f}, suit {stats.holdout_type_acc:.3f}",
        flush=True,
    )
    save_checkpoint(
        args.out,
        student,
        rules,
        extra={"iteration": 0, "stage": "distilled", "source": str(args.teacher_checkpoint)},
        critic=critic_from_payload(payload),
    )
    print(f"saved {args.out}", flush=True)


if __name__ == "__main__":
    main(sys.argv)
