"""M7 follow-up diagnostic: DD-oracle bid-accuracy and points-lost-vs-optimal for a trained
checkpoint (`eval.metrics.evaluate_dd_oracle_metrics`, M5's expensive tier). Answers the open
question the M7 status writeup left unresolved -- is the trained agent's bidding well-calibrated,
or merely beating baselines that aren't close to optimal themselves (`docs/03-implementation-
roadmap.md` M7).

Expensive: 6 full double-dummy solves per deal (5 candidate contract types + the actual one), each
1-40+ seconds serial (M2). Run with a modest deal count -- see M5's own "tens to low hundreds"
guidance -- and `--workers` to parallelize across processes (`solver/batch.py`, M2.5).

Run directly:
`python -m bazarblot.cli.dd_oracle_eval --checkpoint <path> [--n-deals N] [--workers W]`
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

from bazarblot.agents.nn.ppo_agent import NNAgent
from bazarblot.core.rules import load_default
from bazarblot.env.actions import build_action_space
from bazarblot.eval.metrics import evaluate_dd_oracle_metrics
from bazarblot.learn.checkpoint import load_checkpoint


def main(argv: list[str]) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--n-deals", type=int, default=30)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv[1:])

    rules = load_default()
    space = build_action_space(rules)
    net, payload = load_checkpoint(args.checkpoint, rules)
    net.eval()
    agent = NNAgent(net, sample=False)

    rng = random.Random(args.seed)
    seeds = [rng.randrange(1 << 30) for _ in range(args.n_deals)]
    dealers = [rng.randrange(4) for _ in range(args.n_deals)]

    print(
        f"checkpoint: {args.checkpoint} (trained to iteration {payload.get('iteration')})\n"
        f"running DD-oracle diagnostic on {args.n_deals} deals "
        f"({args.n_deals * 6} full solves), workers={args.workers or 'auto'}..."
    )
    t0 = time.perf_counter()
    report = evaluate_dd_oracle_metrics(rules, space, seeds, dealers, agent, workers=args.workers)
    elapsed = time.perf_counter() - t0

    print(f"\ndone in {elapsed:.1f}s ({report.n_deals} deals actually solved)\n")
    print(
        f"bid exact agreement:    mean={report.bid_exact_agreement.mean:.3f}  "
        f"CI=({report.bid_exact_agreement.lo:.3f}, {report.bid_exact_agreement.hi:.3f})"
    )
    print(
        f"bid within one level:   mean={report.bid_within_one_level.mean:.3f}  "
        f"CI=({report.bid_within_one_level.lo:.3f}, {report.bid_within_one_level.hi:.3f})"
    )
    print(
        f"points lost vs optimal: mean={report.points_lost_vs_optimal.mean:.2f}  "
        f"CI=({report.points_lost_vs_optimal.lo:.2f}, {report.points_lost_vs_optimal.hi:.2f})"
    )


if __name__ == "__main__":
    main(sys.argv)
