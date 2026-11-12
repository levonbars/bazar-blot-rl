"""Signaling probe: `python -m bazarblot.cli.signaling CHECKPOINT [...]`.

Reports, per checkpoint, how much partner's bids tell a seat about partner's hand beyond the final
contract, and whether its first card changes when partner's bids are hidden
(`learn/signaling.py`). An untrained network (with the from-scratch starting biases) is measured
first as the control, since noise alone changes a few percent of cards.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from bazarblot.agents.nn.network import apply_action_prior, build_network
from bazarblot.core.rules import load_default
from bazarblot.env.actions import build_action_space
from bazarblot.learn.checkpoint import load_checkpoint
from bazarblot.learn.signaling import measure_signaling


def main(argv: list[str]) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoints", nargs="*", type=Path)
    p.add_argument("--deals", type=int, default=400)
    p.add_argument("--seed", type=int, default=5)
    p.add_argument("--no-control", action="store_true", help="skip the untrained control")
    args = p.parse_args(argv[1:])

    rules = load_default()
    space = build_action_space(rules)
    if not args.no_control:
        torch.manual_seed(0)
        control = build_network(rules)
        # An untrained network with no bias passes every greedy auction; the from-scratch biases
        # make it reach card play, so the control actually has decisions to probe.
        apply_action_prior(control, raise_decay=0.5, capot_bias=-4.0, double_bias=-3.0)
        report = measure_signaling(rules, space, control, n_deals=args.deals, seed=args.seed)
        print("untrained control")
        print(f"  {report.summary() if report else 'no deal reached card play'}", flush=True)
    for path in args.checkpoints:
        net, _ = load_checkpoint(path, rules)
        report = measure_signaling(rules, space, net, n_deals=args.deals, seed=args.seed)
        print(path)
        print(f"  {report.summary() if report else 'no deal reached card play'}", flush=True)


if __name__ == "__main__":
    main(sys.argv)
