"""Evaluation harness (M5, environment spec §9): `duplicate.py` (paired/duplicate deals — the
ONLY evaluation entry point), `metrics.py` (aggregate metrics with bootstrap CIs), `elo.py`
(round-robin Elo on paired deals). Depends on `core/`, `env/`, `agents/`, and `solver/` — never
on `ui/`.
"""
