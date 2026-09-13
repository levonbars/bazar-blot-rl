"""The common agent interface every baseline in this package implements."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import numpy.typing as npt

from bazarblot.env.actions import ActionSpace
from bazarblot.env.infoset import InfoSet


class Agent(Protocol):
    def act(self, info: InfoSet, space: ActionSpace, legal_mask: npt.NDArray[np.bool_]) -> int:
        """Return one legal flat action index. `legal_mask` is passed in rather than
        recomputed — the caller (an eval harness, `env/aec.py`, or a test) already has it from
        `observe()`, and every agent needing it this way keeps them all cheap to call in a loop
        without redundant `legal_mask()` calls."""
        ...
