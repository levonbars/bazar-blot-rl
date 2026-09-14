"""The `deal_margin` training reward (environment spec §6): `tanh(margin / SQUASH_SCALE)`, never
a plain linear `margin / NORM` -- the raw margin distribution is heavy-tailed (a recontra'd
no-trump failure can be two orders of magnitude above a quiet made contract, per the spec's own
worked table), and a single such deal would otherwise dominate a batch's gradient. `SQUASH_SCALE`
is fixed from the spec's own recommendation (set from the empirical median absolute margin, not
the maximum) and frozen -- changing it later silently rescales the value function and invalidates
cross-run comparisons, so it is a named constant here, not a training hyperparameter to sweep."""

from __future__ import annotations

import math

SQUASH_SCALE = 24.0


def deal_margin_reward(margin: float, scale: float = SQUASH_SCALE) -> float:
    """`margin` is `my_team_score - opponent_team_score` (raw, unsquashed) -- antisymmetric by
    construction, so `deal_margin_reward(m) == -deal_margin_reward(-m)` always holds; this is
    what makes the reward zero-sum between the two teams even though raw deal scores are not
    (spec §6)."""
    return math.tanh(margin / scale)
