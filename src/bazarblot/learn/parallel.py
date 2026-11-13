"""Parallel rollout collection: the deal-playing half of `learn/rollout.py` run in worker processes.

Playing deals (game engine + observation encoding + one small forward pass per decision) is pure
Python on one core and was ~56% of training time before the update was vectorised, and nearly all
of it after. Each worker holds its own copy of the learner and of the opponent-pool snapshots,
plays a disjoint range of shuffle indices (`rollout.play_rollout_deals`), and returns the
trajectories; the main process then assigns critic values and advantages
(`rollout.finalize_rollout`), exactly as the serial path does.

Weights are sent per collection: the learner every time, a pool snapshot only the first time a
worker sees it (snapshots are frozen, identified by a version number the caller assigns). Fixed
opponents are rebuilt in each worker from their names (`heuristic`, `heuristic@K`, `random`,
`passer`). Each worker seeds its own torch and opponent-sampling RNGs from the collection seed and
its worker index, so a collection is reproducible for a given worker count.
"""

from __future__ import annotations

import contextlib
import multiprocessing as mp
import random
import time
from collections.abc import Sequence
from dataclasses import dataclass
from multiprocessing.connection import Connection
from typing import Any

import numpy as np
import numpy.typing as npt
import torch

from bazarblot.agents.base import Agent
from bazarblot.agents.composed import ComposedAgent
from bazarblot.agents.heuristic import HeuristicAgent
from bazarblot.agents.nn.network import CentralCritic, PolicyValueNet, build_network
from bazarblot.agents.nn.ppo_agent import Decision
from bazarblot.agents.pass_bidder import PassBidder
from bazarblot.agents.random_agent import RandomAgent
from bazarblot.core.deal import Phase
from bazarblot.core.rules import RuleConfig
from bazarblot.env.actions import build_action_space
from bazarblot.learn.joint_search import JointSearchConfig
from bazarblot.learn.reward import SQUASH_SCALE
from bazarblot.learn.rollout import (
    Opponent,
    RolloutStats,
    Transition,
    finalize_rollout,
    merge_stats,
    play_rollout_deals,
)


def build_fixed_opponent(name: str, rules: RuleConfig, seed: int) -> Agent:
    """A fixed (non-network) training opponent from its name."""
    heuristic = HeuristicAgent(rules, random.Random(seed + 3))
    if name == "heuristic":
        return heuristic
    if name == "random":
        return RandomAgent(random.Random(seed + 2))
    if name == "passer":
        return ComposedAgent(PassBidder(), heuristic)
    if name.startswith("heuristic@"):
        try:
            offset = int(name.split("@", 1)[1])
        except ValueError:
            raise ValueError(f"bad heuristic offset in opponent {name!r}") from None
        return HeuristicAgent(rules, bid_offset=offset)
    raise ValueError(
        f"opponent {name!r} cannot be used with parallel rollouts; expected heuristic, "
        "heuristic@K, random or passer"
    )


@dataclass(frozen=True, slots=True)
class PackedTrajectories:
    """Trajectories as a few stacked arrays: sending ~40k `Transition` objects (each holding its
    own small observation and mask arrays) through a pipe cost about as much as playing the deals,
    while a handful of large arrays pickle at memory-copy speed."""

    lengths: npt.NDArray[np.int64]  # (n_trajectories,)
    obs: npt.NDArray[np.float32]  # (N, obs_dim)
    mask: npt.NDArray[np.bool_]  # (N, action_dim)
    belief: npt.NDArray[np.int8]  # (N, n_cards)
    action_idx: npt.NDArray[np.int64]
    log_prob: npt.NDArray[np.float64]
    value: npt.NDArray[np.float64]
    is_play: npt.NDArray[np.bool_]
    base_level: npt.NDArray[np.int64]
    forced: npt.NDArray[np.bool_]
    reward: npt.NDArray[np.float64]
    outcome: npt.NDArray[np.int64]


def pack_trajectories(trajectories: list[list[Transition]]) -> PackedTrajectories:
    flat = [t for traj in trajectories for t in traj]
    if not flat:
        raise ValueError("nothing to pack")
    if any(t.belief_target is None for t in flat):
        raise ValueError("rollout transitions always carry belief targets")
    return PackedTrajectories(
        lengths=np.array([len(traj) for traj in trajectories], dtype=np.int64),
        obs=np.stack([t.decision.obs for t in flat]),
        mask=np.stack([t.decision.mask for t in flat]),
        belief=np.stack([t.belief_target for t in flat]),  # type: ignore[misc]
        action_idx=np.array([t.decision.action_idx for t in flat], dtype=np.int64),
        log_prob=np.array([t.decision.log_prob for t in flat], dtype=np.float64),
        value=np.array([t.decision.value for t in flat], dtype=np.float64),
        is_play=np.array([t.decision.phase == Phase.PLAY for t in flat], dtype=bool),
        base_level=np.array([t.decision.base_level for t in flat], dtype=np.int64),
        forced=np.array([t.decision.forced for t in flat], dtype=bool),
        reward=np.array([t.reward for t in flat], dtype=np.float64),
        outcome=np.array([t.outcome_target for t in flat], dtype=np.int64),
    )


def unpack_trajectories(packed: PackedTrajectories | None) -> list[list[Transition]]:
    """Inverse of `pack_trajectories`; observations and masks are views into the stacked arrays."""
    if packed is None:
        return []
    p = packed
    out: list[list[Transition]] = []
    i = 0
    for length in p.lengths.tolist():
        traj = []
        for j in range(i, i + length):
            decision = Decision(
                action_idx=int(p.action_idx[j]),
                log_prob=float(p.log_prob[j]),
                value=float(p.value[j]),
                obs=p.obs[j],
                mask=p.mask[j],
                phase=Phase.PLAY if p.is_play[j] else Phase.AUCTION,
                base_level=int(p.base_level[j]),
                forced=bool(p.forced[j]),
            )
            traj.append(
                Transition(
                    decision=decision,
                    belief_target=p.belief[j],
                    outcome_target=int(p.outcome[j]),
                    value=float(p.value[j]),
                    reward=float(p.reward[j]),
                )
            )
        out.append(traj)
        i += length
    return out


@dataclass(frozen=True, slots=True)
class _Task:
    learner_state: dict[str, torch.Tensor]
    new_snapshots: dict[int, dict[str, torch.Tensor]]
    active_versions: list[int]
    deal_indices: list[int]
    seed: int
    worker_seed: int
    opponent_prob: float
    forced_bid_prob: float
    forced_contra_prob: float
    reward_scale: float
    trick_credit: bool = False
    search_config: JointSearchConfig | None = None
    critic_state: dict[str, torch.Tensor] | None = None
    critic_hidden: tuple[int, ...] = (256, 256)


def _worker(
    conn: Connection,
    rules: RuleConfig,
    hidden: tuple[int, ...],
    fixed_names: tuple[str, ...],
    bid_agent_name: str | None,
    base_seed: int,
) -> None:
    torch.set_num_threads(1)
    space = build_action_space(rules)
    learner = build_network(rules, hidden=hidden)
    learner.eval()
    bid_agent = HeuristicAgent(rules) if bid_agent_name == "heuristic" else None
    snapshots: dict[int, PolicyValueNet] = {}
    while True:
        task: _Task | None = conn.recv()
        if task is None:
            return
        learner.load_state_dict(task.learner_state)
        for version, state in task.new_snapshots.items():
            net = build_network(rules, hidden=hidden)
            net.load_state_dict(state)
            net.eval()
            snapshots[version] = net
        for version in [v for v in snapshots if v not in task.active_versions]:
            del snapshots[version]
        # Fixed opponents are rebuilt per task (they are cheap) so none carries RNG state from
        # one collection to the next: a collection depends only on its own seeds.
        fixed: list[Opponent] = [
            build_fixed_opponent(n, rules, base_seed + task.worker_seed) for n in fixed_names
        ]
        pool: list[Opponent] = [*fixed, *(snapshots[v] for v in task.active_versions)]
        critic: CentralCritic | None = None
        if task.critic_state is not None:
            critic = CentralCritic(hidden=task.critic_hidden)
            critic.load_state_dict(task.critic_state)
            critic.eval()
        torch.manual_seed(task.worker_seed)
        t0 = time.perf_counter()
        trajectories, stats = play_rollout_deals(
            rules,
            space,
            learner,
            task.deal_indices,
            task.seed,
            random.Random(task.worker_seed),
            opponent_pool=pool,
            opponent_prob=task.opponent_prob,
            forced_bid_prob=task.forced_bid_prob,
            bid_agent=bid_agent,
            forced_contra_prob=task.forced_contra_prob,
            reward_scale=task.reward_scale,
            trick_credit=task.trick_credit,
            search_config=task.search_config,
            critic=critic,
        )
        packed = pack_trajectories(trajectories) if trajectories else None
        conn.send((packed, stats, time.perf_counter() - t0))


class ParallelRollout:
    """A fixed set of worker processes that play rollout deals for the learner. Use as a context
    manager, or call `close()`."""

    def __init__(
        self,
        rules: RuleConfig,
        n_workers: int,
        hidden: tuple[int, ...],
        fixed_opponents: Sequence[str] = (),
        bid_agent: str | None = None,
        seed: int = 0,
    ) -> None:
        if n_workers < 1:
            raise ValueError("n_workers must be >= 1")
        for name in fixed_opponents:  # fail fast, in the main process
            build_fixed_opponent(name, rules, seed)
        ctx = mp.get_context("spawn")
        self._conns: list[Connection] = []
        self._procs: list[Any] = []
        for _ in range(n_workers):
            parent, child = ctx.Pipe()
            proc = ctx.Process(
                target=_worker,
                args=(child, rules, hidden, tuple(fixed_opponents), bid_agent, seed),
                daemon=True,
            )
            proc.start()
            child.close()
            self._conns.append(parent)
            self._procs.append(proc)
        self._sent_versions: set[int] = set()
        self.last_worker_seconds: list[float] = []
        """Time each worker spent playing deals in the last `collect` (for diagnosing overhead)."""

    @property
    def n_workers(self) -> int:
        return len(self._conns)

    def collect(
        self,
        learner: PolicyValueNet,
        snapshots: Sequence[tuple[int, PolicyValueNet]],
        n_deals: int,
        seed: int,
        opponent_prob: float = 0.0,
        gamma: float = 1.0,
        lam: float = 0.95,
        forced_bid_prob: float = 0.0,
        critic: CentralCritic | None = None,
        forced_contra_prob: float = 0.0,
        reward_scale: float = SQUASH_SCALE,
        trick_credit: bool = False,
        lam_auction: float | None = None,
        search_config: JointSearchConfig | None = None,
    ) -> tuple[list[Transition], RolloutStats]:
        """Same contract as `rollout.collect_rollout` (shuffle `i` uses deal seed `seed + i`),
        with `snapshots` given as `(version, net)` pairs; a version's weights are sent once."""
        learner_state = {k: v.detach().clone() for k, v in learner.state_dict().items()}
        new = {
            v: {k: t.detach().clone() for k, t in net.state_dict().items()}
            for v, net in snapshots
            if v not in self._sent_versions
        }
        active = [v for v, _ in snapshots]
        # The critic is needed in the workers only to search with critic scoring.
        critic_search = search_config is not None and search_config.evaluate == "critic"
        critic_state = (
            {k: v.detach().clone() for k, v in critic.state_dict().items()}
            if critic is not None and critic_search
            else None
        )
        chunks = [list(range(n_deals))[k :: self.n_workers] for k in range(self.n_workers)]
        for k, (conn, chunk) in enumerate(zip(self._conns, chunks, strict=True)):
            conn.send(
                _Task(
                    learner_state=learner_state,
                    new_snapshots=new,
                    active_versions=active,
                    deal_indices=chunk,
                    seed=seed,
                    worker_seed=seed * 1_000 + k,
                    opponent_prob=opponent_prob,
                    forced_bid_prob=forced_bid_prob,
                    forced_contra_prob=forced_contra_prob,
                    reward_scale=reward_scale,
                    trick_credit=trick_credit,
                    search_config=search_config,
                    critic_state=critic_state,
                    critic_hidden=tuple(critic.hidden) if critic is not None else (256, 256),
                )
            )
        self._sent_versions.update(new)
        self._sent_versions.intersection_update(active)
        trajectories: list[list[Transition]] = []
        parts: list[RolloutStats] = []
        self.last_worker_seconds = []
        for conn in self._conns:
            packed, stats, seconds = conn.recv()
            trajectories.extend(unpack_trajectories(packed))
            parts.append(stats)
            self.last_worker_seconds.append(seconds)
        return finalize_rollout(trajectories, critic, gamma, lam, lam_auction), merge_stats(parts)

    def close(self) -> None:
        for conn in self._conns:
            with contextlib.suppress(BrokenPipeError, OSError):
                conn.send(None)
        for proc in self._procs:
            proc.join(timeout=10)
            if proc.is_alive():
                proc.terminate()
        self._conns.clear()
        self._procs.clear()

    def __enter__(self) -> ParallelRollout:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
