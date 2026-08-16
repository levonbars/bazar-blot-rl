"""FastAPI app. Run with: `uvicorn bazarblot.ui.app:app --reload` (or `python -m bazarblot.ui.app`).

Three families of endpoint, matching the three modes:
- `/api/watch/*`   — full-state view, bots control all four seats.
- `/api/play/*`    — seat-scoped view, bots control every seat but the human's.
- `/api/replay/*`  — full-state view over a recorded action log, no bots, no live decisions.

`watch`/`play` responses are built exclusively by `views.full_state_view` /
`views.player_view` respectively — see `views.py` for why those are two functions, not one.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from bazarblot.core.deal import DealError, DealFinishedError, IllegalActionError, Phase
from bazarblot.ui.session import STORE, NotYourTurnError, SessionError
from bazarblot.ui.views import (
    ActionDecodeError,
    FullStateView,
    PlayerView,
    decode_action,
    full_state_view,
    player_view,
)

app = FastAPI(title="Bazar Blot — local UI")

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(str(STATIC_DIR / "index.html"))


def _http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, NotYourTurnError):
        return HTTPException(status_code=403, detail=str(exc))
    if isinstance(exc, (IllegalActionError, ActionDecodeError, DealFinishedError)):
        return HTTPException(status_code=400, detail=str(exc))
    if isinstance(exc, (SessionError, DealError)):
        return HTTPException(status_code=404, detail=str(exc))
    return HTTPException(status_code=500, detail=str(exc))


def _to_jsonable(obj: Any) -> Any:
    """Dataclasses -> plain dicts, recursively, for the response body."""
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _to_jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, tuple | list):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    return obj


# ---------------------------------------------------------------- request bodies


class NewGameRequest(BaseModel):
    mode: str  # "watch" | "play"
    human_seat: int | None = None
    match_seed: int | None = None


class ActionRequest(BaseModel):
    seat: int
    action: dict[str, Any]


class NewReplayRequest(BaseModel):
    log: dict[str, Any]


class GotoRequest(BaseModel):
    index: int


# ---------------------------------------------------------------- watch/play (shared) endpoints


@app.post("/api/game")
def new_game(req: NewGameRequest) -> dict[str, Any]:
    if req.mode not in ("watch", "play"):
        raise HTTPException(status_code=400, detail="mode must be 'watch' or 'play'")
    if req.mode == "play" and req.human_seat is None:
        raise HTTPException(status_code=400, detail="play mode requires human_seat")
    try:
        session = STORE.new_game(req.mode, req.human_seat, req.match_seed)  # type: ignore[arg-type]
        if session.mode == "play":
            # The opener isn't necessarily the human — advance any bot turns that come
            # before the human's first move, same as after every subsequent action.
            session.autoplay_bots()
    except SessionError as exc:
        raise _http_error(exc) from exc
    return _game_state(session.session_id)


@app.get("/api/game/{session_id}")
def get_game(session_id: str) -> dict[str, Any]:
    return _game_state(session_id)


def _game_state(session_id: str) -> dict[str, Any]:
    try:
        session = STORE.get_game(session_id)
    except SessionError as exc:
        raise _http_error(exc) from exc

    match_score = (session.match.scores[0], session.match.scores[1])
    view: FullStateView | PlayerView
    if session.mode == "watch":
        view = full_state_view(session.deal, match_score, session.rules.match.target)
    else:
        assert session.human_seat is not None
        view = player_view(
            session.deal, session.human_seat, match_score, session.rules.match.target
        )

    return {
        "session_id": session.session_id,
        "mode": session.mode,
        "human_seat": session.human_seat,
        "match_scores": list(match_score),
        "match_target": session.rules.match.target,
        "match_finished": session.match.finished,
        "match_winner": session.match.winner,
        "bid_log": [_to_jsonable(e) for e in session.bid_log],
        "events": session.events[-30:],
        "is_bot_turn": session.is_bot_turn(),
        "state": _to_jsonable(view),
    }


@app.post("/api/game/{session_id}/action")
def post_action(session_id: str, req: ActionRequest) -> dict[str, Any]:
    try:
        session = STORE.get_game(session_id)
        action = decode_action(req.action)
        session.apply_human_action(req.seat, action)
        if session.mode == "play":
            session.autoplay_bots()
    except (SessionError, ActionDecodeError, IllegalActionError, DealError) as exc:
        raise _http_error(exc) from exc
    return _game_state(session_id)


@app.post("/api/game/{session_id}/bot_step")
def bot_step(session_id: str) -> dict[str, Any]:
    """Single-step one bot action. `watch` mode's step-through control."""
    try:
        session = STORE.get_game(session_id)
        session.step_bot()
    except (SessionError, DealError) as exc:
        raise _http_error(exc) from exc
    return _game_state(session_id)


@app.post("/api/game/{session_id}/autoplay")
def autoplay(session_id: str) -> dict[str, Any]:
    """Run bot turns to completion (or until it's the human's turn). `watch` mode's autoplay."""
    try:
        session = STORE.get_game(session_id)
        session.autoplay_bots()
    except (SessionError, DealError) as exc:
        raise _http_error(exc) from exc
    return _game_state(session_id)


@app.post("/api/game/{session_id}/next_deal")
def next_deal(session_id: str) -> dict[str, Any]:
    try:
        session = STORE.get_game(session_id)
        session.next_deal()
        if session.mode == "play":
            session.autoplay_bots()
    except (SessionError, DealError) as exc:
        raise _http_error(exc) from exc
    return _game_state(session_id)


@app.get("/api/game/{session_id}/replay_log")
def get_replay_log(session_id: str) -> dict[str, Any]:
    try:
        session = STORE.get_game(session_id)
    except SessionError as exc:
        raise _http_error(exc) from exc
    if session.deal.phase != Phase.TERMINAL:
        raise HTTPException(status_code=400, detail="current deal is not finished yet")
    return session.replay_log()


# ---------------------------------------------------------------- replay endpoints


@app.post("/api/replay")
def new_replay(req: NewReplayRequest) -> dict[str, Any]:
    try:
        session = STORE.new_replay(req.log)
    except SessionError as exc:
        raise _http_error(exc) from exc
    return _replay_state(session.session_id)


def _replay_state(session_id: str) -> dict[str, Any]:
    try:
        session = STORE.get_replay(session_id)
    except SessionError as exc:
        raise _http_error(exc) from exc
    view = full_state_view(session.deal, (0, 0), session.rules.match.target)
    return {
        "session_id": session.session_id,
        "index": session.index,
        "total_actions": len(session.raw_actions),
        "state": _to_jsonable(view),
    }


@app.post("/api/replay/{session_id}/step")
def replay_step(session_id: str) -> dict[str, Any]:
    try:
        session = STORE.get_replay(session_id)
        session.step_forward()
    except (SessionError, DealError) as exc:
        raise _http_error(exc) from exc
    return _replay_state(session_id)


@app.post("/api/replay/{session_id}/goto")
def replay_goto(session_id: str, req: GotoRequest) -> dict[str, Any]:
    try:
        session = STORE.get_replay(session_id)
        session.goto(req.index)
    except (SessionError, DealError) as exc:
        raise _http_error(exc) from exc
    return _replay_state(session_id)


def main() -> None:
    import uvicorn

    uvicorn.run("bazarblot.ui.app:app", host="127.0.0.1", port=8420, reload=True)


if __name__ == "__main__":
    main()
