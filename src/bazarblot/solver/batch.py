"""Batch/parallel DD solving via a process pool (M2.5 item 1) — makes offline oracle metrics and
label generation feasible without touching `solve()`'s own per-call speed at all.

`solve_from` (and hence `solve`) is a pure function of its arguments — no shared state, no I/O —
which is exactly what makes it trivially parallelizable across a `multiprocessing.Pool`: each
worker does its own independent solve with no coordination needed. This module adds nothing to
the solving *algorithm*; it only removes the "one solve at a time" bottleneck for whatever needs
many of them (M5's DD-oracle metrics — bid-accuracy-vs-oracle, points-lost-vs-DD-optimal — and any
offline dataset/label generation). At the measured ~1-5s median per full-deal solve (M2), a plain
single-process loop over 1,000 deals is 15-80+ minutes; spread across 8 cores that's roughly
2-10 minutes — the difference between "impractical to run" and "an ordinary batch job."
"""

from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import Pool
from typing import TYPE_CHECKING

from bazarblot.solver.dd import DDResult, Hands, Trick, solve_from

if TYPE_CHECKING:
    from bazarblot.core.cards import ContractType
    from bazarblot.core.rules import RuleConfig


@dataclass(frozen=True, slots=True)
class SolveSpec:
    """One `solve_from` call's worth of arguments, bundled so `solve_many` can hand them across
    the process boundary — `multiprocessing` pickles task arguments, so this needs to be a plain,
    picklable value rather than a bound method or a closure. `trick_so_far` defaults to `()` and
    `to_act` to the leader, matching `solve()`'s common case; pass both explicitly for a
    `solve_from`-style mid-trick query."""

    hands: Hands
    contract_type: ContractType
    to_act: int
    declaring_team: int
    trick_so_far: Trick = ()


_worker_rules: RuleConfig | None = None


def _init_worker(rules: RuleConfig) -> None:
    # Sent once per worker process via `initargs`, not once per task — avoids re-pickling the
    # (potentially non-trivial, since it carries the full parsed preset in `raw`) RuleConfig for
    # every single spec.
    global _worker_rules
    _worker_rules = rules


def _solve_one(spec: SolveSpec) -> DDResult:
    assert _worker_rules is not None, "solve_many's pool workers must be initialized with rules"
    return solve_from(
        spec.hands,
        spec.contract_type,
        spec.to_act,
        spec.trick_so_far,
        spec.declaring_team,
        _worker_rules,
    )


def solve_many(
    specs: list[SolveSpec], rules: RuleConfig, workers: int | None = None
) -> list[DDResult]:
    """Solve every spec in `specs`, in parallel across `workers` processes (default: all
    available cores, via `multiprocessing.Pool`'s own default). Order-preserving:
    `result[i]` corresponds to `specs[i]`.

    Correct under both `fork` and `spawn` process-start methods (`spawn` is the default on macOS
    and Windows) because `rules` is threaded through explicitly via the pool's initializer rather
    than relied upon as inherited global state, which `spawn`'d workers wouldn't have.

    `chunksize=1`: per-deal solve time is heavily right-skewed (median ~6s, p95 ~43s, measured
    in M2) — `Pool.map`'s default chunking hands each worker several specs at once, so a worker
    unlucky enough to draw two or three slow deals in its own chunk becomes the whole batch's
    bottleneck while other workers sit idle with nothing left to do. `chunksize=1` makes every
    worker pull one spec at a time as it finishes the last, which spreads the slow outliers
    across all workers instead of concentrating them — the standard fix for uneven per-task cost
    in a process pool, at the cost of slightly more IPC overhead that's negligible next to
    multi-second solves.
    """
    if not specs:
        return []
    with Pool(processes=workers, initializer=_init_worker, initargs=(rules,)) as pool:
        return pool.map(_solve_one, specs, chunksize=1)
