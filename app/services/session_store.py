"""In-process chat session store.

NOTE: this is a module-level dict, which only works correctly for a
single-worker deployment. Running with multiple workers/processes (e.g.
`uvicorn --workers N` or multiple pods) means each worker has its own copy
and a session started on one worker won't be visible on another. A real
production deployment needs a shared store instead -- Redis or a DB table.
"""

from __future__ import annotations

import uuid

MAX_HISTORY = 20

_sessions: dict[str, dict] = {}


def _new_session() -> dict:
    return {"mode": None, "persona": None, "history": []}


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
    return session


def append_history(session_id: str, role: str, content: str) -> dict:
    _, session = get_or_create_session(session_id)
    session["history"].append({"role": role, "content": content})
    session["history"] = session["history"][-MAX_HISTORY:]
    return session
