"""In-process chat session store.

NOTE: this is a module-level dict, which only works correctly for a
single-worker deployment. Running with multiple workers/processes (e.g.
`uvicorn --workers N` or multiple pods) means each worker has its own copy
and a session started on one worker won't be visible on another. A real
production deployment needs a shared store instead -- Redis or a DB table.
The same limitation applies to the conversation-list endpoints built on
list_sessions() / get_history() below: they only see sessions this worker
handled, and everything is lost on restart.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

MAX_HISTORY = 20

_sessions: dict[str, dict] = {}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _new_session() -> dict:
    now = _now()
    return {
        "mode": None,
        "persona": None,
        "history": [],
        "created_at": now,
        "updated_at": now,
    }


def get_or_create_session(session_id: str | None) -> tuple[str, dict]:
    if session_id is None:
        session_id = str(uuid.uuid4())
    if session_id not in _sessions:
        _sessions[session_id] = _new_session()
    return session_id, _sessions[session_id]


def update_session(session_id: str, **fields) -> dict:
    _, session = get_or_create_session(session_id)
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


def append_history(session_id: str, role: str, content: str) -> dict:
    _, session = get_or_create_session(session_id)
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


def list_sessions() -> list[dict]:
    """Newest-activity-first summary of every session this worker has seen.

    Iterates _sessions.copy() rather than _sessions directly: a concurrent
    /chat (which inserts) or DELETE /conversations (which pops) runs on
    another threadpool thread, and iterating the live dict while it changes
    size raises RuntimeError. dict.copy() is a single C call that never
    re-enters the interpreter, so it takes a consistent snapshot even under
    that concurrency; the per-row work below then runs against the copy.
    """
    summaries = [
        {
            "id": sid,
            "title": _derive_title(s["history"]),
            "message_count": len(s["history"]),
            "created_at": s["created_at"],
            "updated_at": s["updated_at"],
        }
        for sid, s in _sessions.copy().items()
    ]
    summaries.sort(key=lambda item: item["updated_at"], reverse=True)
    return summaries


def get_history(session_id: str) -> list[dict] | None:
    """The [{role, content}] turns for a session, or None if unknown.

    Does NOT create the session (unlike get_or_create_session) -- an unknown
    id should 404, not spawn an empty conversation.
    """
    session = _sessions.get(session_id)
    return None if session is None else list(session["history"])


def delete_session(session_id: str) -> bool:
    return _sessions.pop(session_id, None) is not None
