"""Tests for the /conversations endpoints (sidebar list, message history,
delete). These read app.services.session_store's in-process dict -- see that
module's docstring for the single-worker limitation.

The session store is process-global and shared with the other test modules,
so every assertion here filters to the session id it created rather than
assuming the store is empty.
"""

from __future__ import annotations

from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app
from app.schemas.query import Citation

client = TestClient(app)

_CITATIONS = [
    Citation(
        book="Test Book",
        chapter="Chapter 1",
        source_db="chroma",
        content="প্রাসঙ্গিক অংশ।",
        similarity=0.9,
        rerank_score=0.95,
    )
]


def _make_qa_session(message: str) -> str:
    with (
        patch("app.api.router.classify_intent", return_value="QA"),
        patch("app.services.qa_service.retrieve_relevant_docs", return_value=_CITATIONS),
        patch("app.services.qa_service.generate_answer", return_value="উত্তর।"),
    ):
        r = client.post("/chat", json={"message": message})
    assert r.status_code == 200
    return r.json()["session_id"]


def test_conversation_lifecycle_list_messages_delete():
    sid = _make_qa_session("যাকাত কী?")

    # appears in the list, with a derived title and the user+assistant turn count
    listing = client.get("/conversations")
    assert listing.status_code == 200
    row = next((x for x in listing.json() if x["id"] == sid), None)
    assert row is not None
    assert row["title"] == "যাকাত কী?"
    assert row["message_count"] == 2

    # message history is the two turns, in order
    msgs = client.get(f"/conversations/{sid}/messages")
    assert msgs.status_code == 200
    assert [m["role"] for m in msgs.json()] == ["user", "assistant"]
    assert msgs.json()[0]["content"] == "যাকাত কী?"
    assert msgs.json()[1]["content"] == "উত্তর।"

    # delete -> 204, then gone
    assert client.delete(f"/conversations/{sid}").status_code == 204
    assert client.get(f"/conversations/{sid}/messages").status_code == 404
    assert all(x["id"] != sid for x in client.get("/conversations").json())


def test_messages_unknown_session_is_404():
    assert client.get("/conversations/does-not-exist/messages").status_code == 404


def test_delete_unknown_session_is_404():
    assert client.delete("/conversations/does-not-exist").status_code == 404


def test_list_sessions_is_safe_under_concurrent_mutation():
    """list_sessions() must not raise "dictionary changed size during
    iteration" while other threads insert/pop sessions. Reverting the
    _sessions.copy() snapshot in list_sessions makes this fail intermittently.
    """
    import threading
    import uuid

    from app.services import session_store

    stop = threading.Event()
    errors: list[Exception] = []

    def churn():
        while not stop.is_set():
            sid = str(uuid.uuid4())
            session_store.get_or_create_session(sid)
            session_store.delete_session(sid)

    def lister():
        try:
            for _ in range(3000):
                session_store.list_sessions()
        except Exception as e:  # noqa: BLE001 -- the point is to catch RuntimeError
            errors.append(e)
        finally:
            stop.set()

    t_churn = threading.Thread(target=churn, daemon=True)
    t_list = threading.Thread(target=lister, daemon=True)
    t_churn.start()
    t_list.start()
    t_list.join(timeout=15)
    stop.set()
    t_churn.join(timeout=5)

    assert not errors, errors
