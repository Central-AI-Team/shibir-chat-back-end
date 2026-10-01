"""Tests for the /conversations endpoints (sidebar list, message history,
delete). These read the owned database-backed conversation store.

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


def test_store_persists_full_history_across_reopening(isolated_chat_store):
    import uuid
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from app.services import session_store

    owner = isolated_chat_store['user_id']
    sid, _ = session_store.get_or_create_session(None, owner)
    for i in range(15):
        rid = str(uuid.uuid4())
        lease = session_store.begin_turn(sid, owner, rid, f'question {i}')
        session_store.finish_turn(sid, owner, rid, f'answer {i}', [], 'QA', lease=lease)
    assert len(session_store.get_history(sid, owner)) == 30
    fresh_engine = create_engine(isolated_chat_store['engine'].url).execution_options(
        **isolated_chat_store['engine'].get_execution_options())
    original = session_store.SessionLocal
    try:
        session_store.SessionLocal = sessionmaker(bind=fresh_engine)
        history = session_store.get_history(sid, owner)
        assert len(history) == 30
        assert history[0]['content'] == 'question 0'
        assert session_store.get_or_create_session(sid, owner)[1]['history'][-1]['content'] == 'answer 14'
    finally:
        session_store.SessionLocal = original
        fresh_engine.dispose()
