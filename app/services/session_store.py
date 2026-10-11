"""In-process chat session store, with per-owner access control.

NOTE: this is a module-level dict, which only works correctly for a
single-worker deployment. Running with multiple workers/processes (e.g.
`uvicorn --workers N` or multiple pods) means each worker has its own copy
and a session started on one worker won't be visible on another. A real
production deployment needs a shared store instead -- Redis or a DB table.
Everything is lost on restart.

Ownership: every session stores the sha256 of its owner's cookie token
(app/api/deps.py). Sessions are created ONLY by create_session(owner) with a
server-generated id; every lookup takes the owner and returns None unless it
matches (constant-time), so an unknown id and someone else's id are
indistinguishable to the caller. Nothing here creates a session implicitly.

Caps: max_sessions_per_owner and max_sessions_total evict the
least-recently-updated session. TTL expiry and rate limiting come in Step 3.

Requests run in a threadpool, so mutations and the snapshot iteration hold
_lock (an RLock, so helpers can call each other).
"""

from __future__ import annotations

import hmac
import secrets
import threading
from datetime import datetime, timezone

from app.core.config import settings

MAX_HISTORY = 20

_sessions: dict[str, dict] = {}
_lock = threading.RLock()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _owns(session: dict, owner: str) -> bool:
    return hmac.compare_digest(session["owner"], owner)


def _evict_oldest(candidates: list[tuple[str, dict]]) -> None:
    sid, _ = min(candidates, key=lambda item: item[1]["updated_at"])
    _sessions.pop(sid, None)


def create_session(owner: str) -> tuple[str, dict]:
    """A new empty session owned by `owner`, evicting the oldest if over a cap."""
    with _lock:
        while True:
            mine = [(sid, s) for sid, s in _sessions.items() if _owns(s, owner)]
            if len(mine) < settings.max_sessions_per_owner:
                break
            _evict_oldest(mine)
        while len(_sessions) >= settings.max_sessions_total:
            _evict_oldest(list(_sessions.items()))
        session_id = secrets.token_urlsafe(32)
        while session_id in _sessions:  # astronomically unlikely; never overwrite
            session_id = secrets.token_urlsafe(32)
        now = _now()
        session = {
            "owner": owner,
            "mode": None,
            "persona": None,
            "history": [],
            "created_at": now,
            "updated_at": now,
        }
        _sessions[session_id] = session
        return session_id, session


def get_owned_session(session_id: str, owner: str) -> dict | None:
    """The live session dict (callers mutate it in place), or None if the id is
    unknown OR belongs to someone else."""
    with _lock:
        session = _sessions.get(session_id)
    if session is None or not _owns(session, owner):
        return None
    return session


def update_session(session_id: str, **fields) -> dict | None:
    """Merge fields into an existing session. Never creates one: if it vanished
    mid-request (concurrent DELETE or eviction) this does nothing."""
    with _lock:
        session = _sessions.get(session_id)
        if session is None:
            return None
        history = fields.pop("history", None)
        session.update(fields)
        if history is not None:
            session["history"] = history[-MAX_HISTORY:]
        # Every request path ends with an update_session() call, so touching the
        # timestamp here keeps updated_at accurate for all modes -- including
        # roleplay, which appends to history directly rather than via
        # append_history().
        session["updated_at"] = _now()
        return session


def append_history(session_id: str, role: str, content: str) -> dict | None:
    with _lock:
        session = _sessions.get(session_id)
        if session is None:
            return None
        session["history"].append({"role": role, "content": content})
        session["history"] = session["history"][-MAX_HISTORY:]
        session["updated_at"] = _now()
        return session


def _derive_title(history: list[dict]) -> str:
    """First user turn, trimmed to a sidebar-friendly length."""
    for entry in history:
        if entry.get("role") == "user" and entry.get("content", "").strip():
            text = " ".join(entry["content"].split())
            return text[:60] + "…" if len(text) > 60 else text
    return "New chat"


def list_sessions(owner: str) -> list[dict]:
    """Newest-activity-first summary of `owner`'s sessions only."""
    with _lock:
        snapshot = [(sid, s) for sid, s in _sessions.items() if _owns(s, owner)]
        summaries = [
            {
                "id": sid,
                "title": _derive_title(s["history"]),
                "message_count": len(s["history"]),
                "created_at": s["created_at"],
                "updated_at": s["updated_at"],
            }
            for sid, s in snapshot
        ]
    summaries.sort(key=lambda item: item["updated_at"], reverse=True)
    return summaries


def get_history(session_id: str, owner: str) -> list[dict] | None:
    """The [{role, content}] turns of an owned session, or None (unknown or not
    yours). Does not create anything."""
    with _lock:
        session = get_owned_session(session_id, owner)
        return None if session is None else list(session["history"])


def delete_session(session_id: str, owner: str) -> bool:
    with _lock:
        session = _sessions.get(session_id)
        if session is None or not _owns(session, owner):
            return False
        del _sessions[session_id]
        return True
