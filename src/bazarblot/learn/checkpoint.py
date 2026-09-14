"""Checkpoint save/load, versioned the same way `env/obs.py` versions observations: a checkpoint
records the `OBS_VERSION` and `rules_hash` it was trained under, and loading refuses a mismatch
rather than silently restoring weights whose input features no longer mean what the checkpoint
thinks they mean (spec §4's own rule for `OBS_VERSION`, applied here to the consumer side)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from bazarblot.agents.nn.network import PolicyValueNet, build_network
from bazarblot.core.rules import RuleConfig
from bazarblot.env.obs import OBS_VERSION


class CheckpointVersionMismatch(RuntimeError):
    pass


def save_checkpoint(
    path: Path, net: PolicyValueNet, rules: RuleConfig, extra: dict[str, Any]
) -> None:
    payload = {
        "obs_version": OBS_VERSION,
        "rules_hash": rules.rules_hash,
        "hidden": _hidden_sizes(net),
        "state_dict": net.state_dict(),
        **extra,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: Path, rules: RuleConfig) -> tuple[PolicyValueNet, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload["obs_version"] != OBS_VERSION:
        raise CheckpointVersionMismatch(
            f"checkpoint was trained under obs_version={payload['obs_version']!r}, "
            f"current code is {OBS_VERSION!r}"
        )
    if payload["rules_hash"] != rules.rules_hash:
        raise CheckpointVersionMismatch(
            f"checkpoint was trained under rules_hash={payload['rules_hash']!r}, "
            f"current preset is {rules.rules_hash!r}"
        )
    net = build_network(rules, hidden=tuple(payload["hidden"]))
    net.load_state_dict(payload["state_dict"])
    return net, payload


def _hidden_sizes(net: PolicyValueNet) -> tuple[int, ...]:
    return tuple(layer.out_features for layer in net.trunk if isinstance(layer, torch.nn.Linear))
