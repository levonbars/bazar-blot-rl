"""Which agent fills the bot seats in `watch`/`play` mode.

Three kinds, chosen per game:
- `trained:<path>` — a trained network checkpoint, playing greedily (its most likely action);
- `heuristic` — the M4 "club player" baseline (`agents/heuristic.py`);
- `simple` — the original toy UI bot (`ui/bots.py`).

Trained checkpoints are listed from `checkpoints/*/final.pt` (or `$BAZARBLOT_CHECKPOINTS`, a
glob), keeping only those the current code can load (same observation version and rules). Loaded
networks are cached for the life of the server.
"""

from __future__ import annotations

import glob
import os
import random
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bazarblot.agents.base import Agent
from bazarblot.agents.heuristic import HeuristicAgent
from bazarblot.core.rules import RuleConfig

DEFAULT_GLOB = "checkpoints/*/final.pt"
PREFERRED = "checkpoints/m7_long_s1/final.pt"  # best of the three long-baseline seeds


@dataclass(frozen=True, slots=True)
class BotOption:
    id: str
    label: str


_lock = threading.Lock()
_nets: dict[str, Any] = {}
_listing: list[BotOption] | None = None


def _compatible(path: str, rules: RuleConfig) -> bool:
    import torch

    from bazarblot.env.obs import OBS_VERSION

    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return False
    return bool(
        payload.get("obs_version") == OBS_VERSION and payload.get("rules_hash") == rules.rules_hash
    )


def available_bots(rules: RuleConfig) -> list[BotOption]:
    """Bot choices for the UI, the preferred trained checkpoint first when present."""
    global _listing
    with _lock:
        if _listing is None:
            options: list[BotOption] = []
            try:
                import torch  # noqa: F401

                paths = sorted(glob.glob(os.environ.get("BAZARBLOT_CHECKPOINTS", DEFAULT_GLOB)))
                paths.sort(key=lambda p: p != PREFERRED)
                for p in paths:
                    if _compatible(p, rules):
                        name = Path(p).parent.name
                        options.append(BotOption(f"trained:{p}", f"Trained agent ({name})"))
            except ImportError:
                pass  # the `learn` extra is not installed: no trained bots
            options.append(BotOption("heuristic", "Heuristic (club player)"))
            options.append(BotOption("simple", "Simple bot"))
            _listing = options
        return list(_listing)


def reset_cache() -> None:
    """Forget the checkpoint listing and loaded networks (tests, or after new checkpoints)."""
    global _listing
    with _lock:
        _listing = None
        _nets.clear()


def default_bot(rules: RuleConfig) -> str:
    return available_bots(rules)[0].id


def build_bot(kind: str, rules: RuleConfig, seed: int) -> Agent | None:
    """The agent for `kind`, or `None` for the simple bot (which has its own code path)."""
    if kind == "simple":
        return None
    if kind == "heuristic":
        return HeuristicAgent(rules, random.Random(seed))
    if kind.startswith("trained:"):
        if kind not in {o.id for o in available_bots(rules)}:
            raise ValueError(f"unknown or incompatible checkpoint: {kind[len('trained:') :]}")
        from bazarblot.agents.nn.ppo_agent import NNAgent
        from bazarblot.learn.checkpoint import load_checkpoint

        path = kind[len("trained:") :]
        with _lock:
            net = _nets.get(path)
            if net is None:
                net, _ = load_checkpoint(Path(path), rules)
                net.eval()
                _nets[path] = net
        return NNAgent(net, sample=False)
    raise ValueError(f"unknown bot kind {kind!r}")
