"""In-memory game sessions. This is a local single-user tool — no database, no auth, no
multi-process concerns. State lives in a process-global dict for the lifetime of the server.
"""

from __future__ import annotations

import random
import threading
import uuid
from dataclasses import dataclass
from typing import Any, Literal

from bazarblot.core.auction import BidAction, ContraAction, PassAction, RecontraAction
from bazarblot.core.deal import Action, Deal, Phase
from bazarblot.core.dealing import deal_hands
from bazarblot.core.match import Match
from bazarblot.core.rules import RuleConfig, load_default
from bazarblot.ui.bots import bot_action

Mode = Literal["watch", "play"]


class SessionError(ValueError):
    pass


class NotYourTurnError(SessionError):
    pass


@dataclass
class BidLogEntry:
    deal_number: int
    seat: int
    kind: str
    level: int | None = None
    contract_type: str | None = None
    capot: bool | None = None


def _bid_log_entry(deal_number: int, seat: int, action: Action) -> BidLogEntry | None:
    if isinstance(action, PassAction):
        return BidLogEntry(deal_number, seat, "pass")
    if isinstance(action, BidAction):
        return BidLogEntry(
            deal_number, seat, "bid", action.level, action.contract_type, action.capot
        )
    if isinstance(action, ContraAction):
        return BidLogEntry(deal_number, seat, "contra")
    if isinstance(action, RecontraAction):
        return BidLogEntry(deal_number, seat, "recontra")
    return None  # PlayCardAction — not part of the bid log


def _serialize_action(action: Action) -> dict[str, Any]:
    if isinstance(action, PassAction):
        return {"type": "pass"}
    if isinstance(action, BidAction):
        return {
            "type": "bid",
            "level": action.level,
            "contract_type": action.contract_type,
            "capot": action.capot,
        }
    if isinstance(action, ContraAction):
        return {"type": "contra"}
    if isinstance(action, RecontraAction):
        return {"type": "recontra"}
    return {"type": "play", "card": action.card}  # type: ignore[attr-defined]


class GameSession:
    """One `watch` or `play` session: a `Match` plus the current `Deal`, with bots filling
    every seat not controlled by the human (or all four seats, in `watch` mode)."""

    def __init__(
        self,
        session_id: str,
        rules: RuleConfig,
        mode: Mode,
        human_seat: int | None,
        match_seed: int,
    ) -> None:
        if mode == "play" and human_seat is None:
            raise SessionError("play mode requires a human_seat")
        self.session_id = session_id
        self.rules = rules
        self.mode: Mode = mode
        self.human_seat = human_seat
        self.match_seed = match_seed
        self.bot_rng = random.Random(match_seed ^ 0x5BAD5EED)
        self.match = Match(rules=rules, dealer=0)
        self.events: list[str] = []
        self.deal: Deal
        self.deal_seed: int
        self.bid_log: list[BidLogEntry] = []
        self.action_log: list[dict[str, Any]] = []
        self._redeal_attempt = 0
        self._start_new_deal()

    # ---------------------------------------------------------------- deal lifecycle

    def _deal_seed(self, deal_number: int, attempt: int) -> int:
        # Deterministic per-(deal, redeal-attempt) seed — reconstructible for replay without
        # storing the hands themselves. `attempt` matters: a 4-pass abort must reshuffle, not
        # repeat the same hands, or a deterministic bot policy that passes on this exact deal
        # live-locks forever (same seed -> same hands -> same passes -> abort -> repeat).
        return (self.match_seed * 1_000_003 + deal_number * 97 + attempt) & 0x7FFF_FFFF

    def _start_new_deal(self) -> None:
        deal_number = self.match.deal_number
        self.deal_seed = self._deal_seed(deal_number, self._redeal_attempt)
        hands = deal_hands(random.Random(self.deal_seed), self.rules)
        self.deal = Deal(self.rules, dealer=self.match.dealer, hands=hands, deal_id=deal_number)
        self.bid_log = []
        self.action_log = []
        self.events.append(f"Deal {deal_number}: dealer is seat {self.match.dealer}")

    def next_deal(self) -> None:
        if self.deal.phase == Phase.TERMINAL:
            self._redeal_attempt = 0
        elif self.deal.phase == Phase.ABORTED:
            self._redeal_attempt += 1
        else:
            raise SessionError("current deal is not finished")
        self._start_new_deal()

    # ---------------------------------------------------------------- turn logic

    def is_bot_turn(self) -> bool:
        if self.deal.phase not in (Phase.AUCTION, Phase.PLAY):
            return False
        if self.mode == "watch":
            return True
        return self.deal.to_act != self.human_seat

    def step_bot(self) -> None:
        if not self.is_bot_turn():
            raise SessionError("it is not a bot's turn")
        seat = self.deal.to_act
        action = bot_action(self.deal, seat, self.bot_rng)
        self._apply(seat, action)

    def apply_human_action(self, seat: int, action: Action) -> None:
        if self.mode == "play" and seat != self.human_seat:
            raise NotYourTurnError(f"seat {seat} is not the human seat ({self.human_seat})")
        if self.deal.to_act != seat:
            raise NotYourTurnError(f"it is seat {self.deal.to_act}'s turn, not {seat}'s")
        self._apply(seat, action)

    def _apply(self, seat: int, action: Action) -> None:
        entry = _bid_log_entry(self.deal.deal_id, seat, action)
        if entry is not None:
            self.bid_log.append(entry)
        self.action_log.append(_serialize_action(action))

        self.deal.step(action)
        self.deal.check_invariants()

        if self.deal.phase == Phase.ABORTED:
            self.events.append(f"Deal {self.deal.deal_id}: auction aborted (4 passes) — redealing")
        elif self.deal.phase == Phase.TERMINAL:
            assert self.deal.result is not None
            r = self.deal.result
            self.match.apply_deal_result(r)
            outcome = "MADE" if r.made else "FAILED"
            capot_tag = " Capot" if r.contract.capot else ""
            self.events.append(
                f"Deal {self.deal.deal_id} complete: "
                f"{r.contract.contract_type}-{r.contract.level}{capot_tag} "
                f"by team {r.attackers_team} — {outcome}. "
                f"Scores +{r.score_attackers}/+{r.score_defenders}. "
                f"Match: {self.match.scores[0]}-{self.match.scores[1]}"
            )
            if self.match.finished:
                self.events.append(f"Match over: team {self.match.winner} wins")

    def autoplay_bots(self, max_steps: int = 200) -> int:
        """Advance through consecutive bot turns. In `watch` mode this can run the whole
        deal; in `play` mode it stops the moment it's the human's turn (or the deal ends)."""
        steps = 0
        while self.is_bot_turn() and steps < max_steps:
            self.step_bot()
            steps += 1
            if self.deal.phase == Phase.ABORTED and self.mode == "watch":
                self.next_deal()  # a redeal has nothing for a spectator to see; skip through
        return steps

    # ---------------------------------------------------------------- replay export

    def replay_log(self) -> dict[str, Any]:
        return {
            "rules_hash": self.rules.rules_hash,
            "match_seed": self.match_seed,
            "deal_number": self.deal.deal_id,
            "dealer": self.deal.dealer,
            "deal_seed": self.deal_seed,
            "actions": list(self.action_log),
        }


class ReplaySession:
    """Reconstructs one deal from a `GameSession.replay_log()` blob and steps through the
    recorded actions one at a time — no bots, no human, no Match. Full visibility always:
    reviewing a finished deal isn't "playing" it, so there is nothing to hide.
    """

    def __init__(self, session_id: str, rules: RuleConfig, log: dict[str, Any]) -> None:
        if log.get("rules_hash") != rules.rules_hash:
            raise SessionError(
                f"replay log was recorded under rules_hash={log.get('rules_hash')!r}, "
                f"current rules_hash={rules.rules_hash!r} — rules have drifted, replay would lie"
            )
        self.session_id = session_id
        self.rules = rules
        self.dealer = int(log["dealer"])
        self.deal_number = int(log["deal_number"])
        self.deal_seed = int(log["deal_seed"])
        self.raw_actions: list[dict[str, Any]] = list(log["actions"])
        self.index = 0
        self.deal: Deal = self._fresh_deal()

    def _fresh_deal(self) -> Deal:
        hands = deal_hands(random.Random(self.deal_seed), self.rules)
        return Deal(self.rules, dealer=self.dealer, hands=hands, deal_id=self.deal_number)

    def step_forward(self) -> bool:
        from bazarblot.ui.views import decode_action

        if self.index >= len(self.raw_actions):
            return False
        action = decode_action(self.raw_actions[self.index])
        self.deal.step(action)
        self.index += 1
        return True

    def goto(self, index: int) -> None:
        """Jump to a specific point in the log. `Deal` has no undo, so going backward means
        rebuilding from scratch and replaying — cheap enough for a 32-ply deal."""
        if not (0 <= index <= len(self.raw_actions)):
            raise SessionError(f"index {index} out of range (0..{len(self.raw_actions)})")
        self.deal = self._fresh_deal()
        self.index = 0
        while self.index < index:
            self.step_forward()


class SessionStore:
    """Process-global, thread-safe (uvicorn's default worker is single-process but may use
    a thread pool for sync endpoints) map of session id -> session object."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._games: dict[str, GameSession] = {}
        self._replays: dict[str, ReplaySession] = {}

    def new_game(self, mode: Mode, human_seat: int | None, match_seed: int | None) -> GameSession:
        session_id = uuid.uuid4().hex
        seed = match_seed if match_seed is not None else random.SystemRandom().randrange(2**31)
        session = GameSession(session_id, load_default(), mode, human_seat, seed)
        with self._lock:
            self._games[session_id] = session
        return session

    def get_game(self, session_id: str) -> GameSession:
        with self._lock:
            session = self._games.get(session_id)
        if session is None:
            raise SessionError(f"no such game session: {session_id}")
        return session

    def new_replay(self, log: dict[str, Any]) -> ReplaySession:
        session_id = uuid.uuid4().hex
        session = ReplaySession(session_id, load_default(), log)
        with self._lock:
            self._replays[session_id] = session
        return session

    def get_replay(self, session_id: str) -> ReplaySession:
        with self._lock:
            session = self._replays.get(session_id)
        if session is None:
            raise SessionError(f"no such replay session: {session_id}")
        return session


STORE = SessionStore()
