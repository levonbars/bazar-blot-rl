"""M1: golden-file regression test.

1000 seeded deals, played by the fixed random-legal policy in `tests/helpers.py`, hashed into a
single fingerprint. Any change to the rules engine that alters this hash is either a bug or a
deliberate rules change — if deliberate, regenerate the constant below (run this file's
`__main__` block) and say so in the commit message. This is the "did I just silently change the
game" tripwire the roadmap asks for.
"""

from __future__ import annotations

import hashlib
import random

from bazarblot.core.deal import Deal, Phase
from bazarblot.core.rules import load_default
from tests.helpers import play_random_legal_deal

RULES = load_default()
N_DEALS = 1000

# Regenerate by running: python -m tests.test_golden
GOLDEN_HASH = "4b37097ebc8b9e28335ea1c3bc03803be2fe121ab1dca1a94ea2d3f1f9aabdf6"


def _deal_fingerprint(d: Deal) -> str:
    """A compact, order-sensitive text summary of one finished deal."""
    if d.phase == Phase.ABORTED:
        return f"ABORTED dealer={d.dealer}"
    assert d.result is not None
    r = d.result
    c = r.contract
    tricks = ";".join(
        f"{t.leader}:{[card for _, card in t.plays]}->{t.winner}({t.points})" for t in d.tricks
    )
    return (
        f"dealer={d.dealer} contract=({c.level},{c.contract_type},capot={c.capot},"
        f"declarer={c.declarer_seat},doubling={c.doubling}) "
        f"cards=({r.cards_attackers},{r.cards_defenders}) "
        f"combo=({r.combo_attackers},{r.combo_defenders}) "
        f"made={r.made} score=({r.score_attackers},{r.score_defenders}) "
        f"tricks=[{tricks}]"
    )


def _compute_golden_hash(n_deals: int = N_DEALS) -> str:
    lines = []
    for seed in range(n_deals):
        rng = random.Random(seed)
        d = play_random_legal_deal(
            RULES, dealer=seed % 4, rng=rng, deal_id=seed, check_invariants=False
        )
        lines.append(f"seed={seed} " + _deal_fingerprint(d))
    blob = "\n".join(lines).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def test_golden_hash_is_stable() -> None:
    actual = _compute_golden_hash()
    assert actual == GOLDEN_HASH, (
        "Golden hash mismatch: the engine's behaviour over 1000 seeded deals changed. "
        "If this was a deliberate rules change, regenerate GOLDEN_HASH by running "
        "`python -m tests.test_golden` and update the constant in this file. "
        f"Got: {actual}"
    )


if __name__ == "__main__":
    print(_compute_golden_hash())
