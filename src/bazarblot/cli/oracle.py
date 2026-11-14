"""`python -m bazarblot.cli.oracle ORACLE_CKPT BASE_CKPT [--pairs N] [--blind]`: the team that sees
partner's hand (a network trained with the oracle input) against a baseline network
(`learn/oracle_eval.py`)."""

from __future__ import annotations

import argparse
from pathlib import Path

from bazarblot.core.rules import load_default
from bazarblot.env.actions import build_action_space
from bazarblot.learn.checkpoint import load_checkpoint
from bazarblot.learn.oracle_eval import oracle_match


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("oracle_checkpoint", type=Path)
    p.add_argument("base_checkpoint", type=Path)
    p.add_argument("--pairs", type=int, default=300)
    p.add_argument("--seed", type=int, default=4242)
    p.add_argument("--blind", action="store_true", help="oracle team does not see partner")
    args = p.parse_args(argv)
    rules = load_default()
    space = build_action_space(rules)
    oracle_net, _ = load_checkpoint(args.oracle_checkpoint, rules)
    base_net, _ = load_checkpoint(args.base_checkpoint, rules)
    report = oracle_match(
        rules, space, oracle_net, base_net, args.pairs, args.seed, see=not args.blind
    )
    print(("blind  " if args.blind else "sees   ") + report.summary())


if __name__ == "__main__":
    main()
