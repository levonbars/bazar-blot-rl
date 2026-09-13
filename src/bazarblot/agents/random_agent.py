"""Uniform-over-legal baseline. The floor everything else in M4 is measured against — beating
this is a very low bar by design, so a heuristic or PIMC agent clearing it tells you little on
its own; it's the `heuristic`-beats-`random`-by-a-wide-margin comparison that matters."""

from __future__ import annotations

import random

import numpy as np
import numpy.typing as npt

from bazarblot.env.actions import ActionSpace
from bazarblot.env.infoset import InfoSet


class RandomAgent:
    def __init__(self, rng: random.Random | None = None) -> None:
        self.rng = rng if rng is not None else random.Random()

    def act(self, info: InfoSet, space: ActionSpace, legal_mask: npt.NDArray[np.bool_]) -> int:
        del info, space
        legal = legal_mask.nonzero()[0]
        return int(self.rng.choice(legal.tolist()))
