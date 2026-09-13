"""Baseline agents (M4): `random_agent.RandomAgent`, `heuristic.HeuristicAgent`, and
`pimc.PIMCAgent`. Depends on `core/`, `env/`, and `solver/` — never on `ui/`.

Every agent implements the same shape (`base.Agent`): given an `InfoSet` and the legal_mask
computed for it, return one legal flat action index (`env/actions.py`'s numbering). None of them
receive a `Deal`/`TrackedDeal` — same security boundary the environment layer itself enforces.
"""
