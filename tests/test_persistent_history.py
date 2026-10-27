"""Chat history on Postgres (app/services/session_store.py, roadmap §2).

The store-level tests run on the TEST database through the db_session fixture
(skipped unless the database name contains "test"), with the store's
SESSION_FACTORY bound to the fixture's connection so every write lands in the
rolled-back transaction. /ready and the API-shape tests use the usual SQLite
TestClient setup.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select, update
from sqlalchemy.orm import sessionmaker

from app.auth.principal import user_owner_key
from app.core.config import settings
from app.db.models import Conversation, Message, User
from app.main import app
from app.schemas.query import Citation
from app.services import session_store
from scripts import purge_history

GUEST = "g" * 64
OTHER = "o" * 64


@pytest.fixture
def store(db_session, monkeypatch):
    """session_store on the test database, inside the fixture's transaction.
    Returns a function that re-binds a fresh factory ("restart")."""

    def restart():
        factory = sessionmaker(
            bind=db_session.connection(), join_transaction_mode="create_savepoint", future=True
        )
        monkeypatch.setattr(session_store, "SESSION_FACTORY", factory)
        return factory

    restart()
    return restart


def _make_user(db_session) -> str:
    uid = str(uuid.uuid4())
    at = datetime.now(timezone.utc)
    db_session.add(
        User(
            id=uid, email=f"{uid}@example.com", password_hash="x", role="user",
            is_active=True, failed_login_count=0, created_at=at, password_changed_at=at,
        )
    )
    db_session.flush()
    return uid


def _age(db_session, sid: str, **delta) -> None:
    db_session.execute(
        update(Conversation)
        .where(Conversation.id == sid)
        .values(updated_at=datetime.now(timezone.utc) - timedelta(**delta))
    )
    db_session.flush()


# ---------------------------------------------------------------------------
# store
# ---------------------------------------------------------------------------


def test_history_survives_a_restart(store):
    sid, _ = session_store.create_session(GUEST)
    session_store.update_session(sid, mode="ROLEPLAY", persona="শার্লক")
    session_store.append_history(sid, "user", "প্রশ্ন")
    session_store.append_history(sid, "assistant", "উত্তর", sources=[Citation(
        book="b", chapter="c", source_db="tarun", content="x", similarity=0.5, rerank_score=0.9,
    )])

    store()  # a new process: nothing in memory, only the database

    session = session_store.get_owned_session(sid, GUEST)
    assert session is not None
    assert session["mode"] == "ROLEPLAY" and session["persona"] == "শার্লক"
    assert [t["content"] for t in session["history"]] == ["প্রশ্ন", "উত্তর"]
    assert session["history"][1]["sources"][0]["book"] == "b"
    assert "sources" not in session["history"][0]
    listing = session_store.list_sessions(GUEST)
    assert [(r["id"], r["title"], r["message_count"], r["mode"]) for r in listing] == [
        (sid, "প্রশ্ন", 2, "ROLEPLAY")
    ]


def test_ownership_isolation(store):
    sid, _ = session_store.create_session(GUEST)
    session_store.append_history(sid, "user", "গোপন")

    assert session_store.get_owned_session(sid, OTHER) is None
    assert session_store.get_history(sid, OTHER) is None
    assert session_store.delete_session(sid, OTHER) is False
    assert session_store.list_sessions(OTHER) == []
    # the failed attempts changed nothing for the owner
    assert session_store.get_history(sid, GUEST) == [{"role": "user", "content": "গোপন"}]
    # a deleted conversation is as unknown as a foreign one
    assert session_store.delete_session(sid, GUEST) is True
    assert session_store.get_owned_session(sid, GUEST) is None
    assert session_store.delete_session(sid, GUEST) is False


def test_guest_to_user_migration(store, db_session):
    uid = _make_user(db_session)
    user_key = user_owner_key(uid)
    mine, _ = session_store.create_session(GUEST)
    session_store.append_history(mine, "user", "আমার")
    theirs, _ = session_store.create_session(OTHER)
    deleted, _ = session_store.create_session(GUEST)
    session_store.delete_session(deleted, GUEST)

    # the login route passes its own db session; commit is the caller's
    assert session_store.reassign_owner(GUEST, user_key, user_id=uid, db=db_session) == 1
    db_session.flush()

    assert session_store.list_sessions(GUEST) == []
    assert [r["id"] for r in session_store.list_sessions(user_key)] == [mine]
    assert session_store.get_history(mine, user_key) == [{"role": "user", "content": "আমার"}]
    assert db_session.execute(
        select(Conversation.user_id).where(Conversation.id == mine)
    ).scalar_one() == uid
    # untouched: someone else's, and the guest's deleted one
    assert [r["id"] for r in session_store.list_sessions(OTHER)] == [theirs]
    assert db_session.execute(
        select(Conversation.owner_key).where(Conversation.id == deleted)
    ).scalar_one() == GUEST


def test_read_window_is_twenty_but_everything_is_stored(store, db_session):
    sid, _ = session_store.create_session(GUEST)
    for i in range(25):
        session_store.append_history(sid, "user" if i % 2 == 0 else "assistant", f"t{i}")

    history = session_store.get_history(sid, GUEST)
    assert len(history) == session_store.MAX_HISTORY == 20
    assert [t["content"] for t in history] == [f"t{i}" for i in range(5, 25)]
    assert len(session_store.get_owned_session(sid, GUEST)["history"]) == 20
    assert session_store.list_sessions(GUEST)[0]["message_count"] == 25
    assert db_session.execute(
        select(Message.id).where(Message.conversation_id == sid)
    ).scalars().all().__len__() == 25


def test_per_owner_cap_soft_deletes_the_least_recently_updated(store, db_session, monkeypatch):
    monkeypatch.setattr(settings, "guest_conversation_cap", 2)
    a, _ = session_store.create_session(GUEST)
    b, _ = session_store.create_session(GUEST)
    _age(db_session, a, minutes=1)
    _age(db_session, b, minutes=30)
    o, _ = session_store.create_session(OTHER)

    c, _ = session_store.create_session(GUEST)

    assert {r["id"] for r in session_store.list_sessions(GUEST)} == {a, c}
    assert session_store.get_owned_session(b, GUEST) is None
    assert db_session.execute(
        select(Conversation.deleted_at).where(Conversation.id == b)
    ).scalar_one() is not None  # soft-deleted, still a row
    assert session_store.get_owned_session(o, OTHER) is not None


def test_purge_dry_run_counts_and_apply_deletes(store, db_session, monkeypatch):
    monkeypatch.setattr(settings, "guest_history_retention_days", 30)
    uid = _make_user(db_session)
    user_key = user_owner_key(uid)
    old_guest, _ = session_store.create_session(GUEST)
    session_store.append_history(old_guest, "user", "x")
    session_store.append_history(old_guest, "assistant", "y")
    fresh_guest, _ = session_store.create_session(GUEST)
    old_user_live, _ = session_store.create_session(user_key, user_id=uid)
    old_user_deleted, _ = session_store.create_session(user_key, user_id=uid)
    session_store.delete_session(old_user_deleted, user_key)
    for sid in (old_guest, old_user_live, old_user_deleted):
        _age(db_session, sid, days=40)
    db_session.execute(
        update(Conversation).where(Conversation.id == old_user_deleted)
        .values(deleted_at=datetime.now(timezone.utc) - timedelta(days=40))
    )

    cutoff = purge_history.cutoff_for(settings.guest_history_retention_days)
    before = purge_history.counts(db_session, cutoff)
    assert before["conversations_expired"] == 2 and before["messages_expired"] == 2
    # dry run: counting deletes nothing
    assert purge_history.counts(db_session, cutoff) == before
    assert session_store.get_owned_session(old_guest, GUEST) is not None

    assert purge_history.purge(db_session, cutoff) == 2
    remaining = set(db_session.execute(select(Conversation.id)).scalars().all())
    assert {old_guest, old_user_deleted}.isdisjoint(remaining)
    assert {fresh_guest, old_user_live} <= remaining
    assert db_session.execute(
        select(Message.id).where(Message.conversation_id == old_guest)
    ).first() is None
    assert purge_history.counts(db_session, cutoff)["conversations_expired"] == 0


# ---------------------------------------------------------------------------
# API shape and readiness (SQLite TestClient)
# ---------------------------------------------------------------------------

_CITATION = Citation(book="Test Book", chapter="1", source_db="chroma", content="অংশ", rerank_score=0.9)


def test_list_has_mode_and_assistant_turns_carry_sources():
    client = TestClient(app)
    with (
        patch("app.api.router.classify_intent", return_value="QA"),
        patch("app.services.qa_service.retrieve_relevant_docs", return_value=[_CITATION]),
        patch("app.services.qa_service.generate_answer", return_value="উত্তর।"),
    ):
        sid = client.post("/chat", json={"message": "যাকাত কী?"}).json()["session_id"]
    (row,) = client.get("/conversations").json()
    assert row["id"] == sid and row["mode"] == "QA"
    assert {"title", "message_count", "created_at", "updated_at"} <= set(row)
    user_turn, assistant_turn = client.get(f"/conversations/{sid}/messages").json()
    assert user_turn["sources"] == []
    assert assistant_turn["sources"][0]["book"] == "Test Book"


def _ready(postgres_ok: bool, chroma_ok: bool):
    engine = MagicMock()
    if not postgres_ok:
        engine.connect.side_effect = RuntimeError("db down")
    collection = MagicMock()
    if not chroma_ok:
        collection.count.side_effect = RuntimeError("chroma down")
    with (
        patch("app.db.session.engine", engine),
        patch("app.rag.chroma_client.get_collection", return_value=collection),
    ):
        return TestClient(app).get("/ready")


def test_ready_is_200_when_both_dependencies_answer():
    r = _ready(True, True)
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "checks": {"postgres": "ok", "chroma": "ok"}}


@pytest.mark.parametrize("postgres_ok, chroma_ok", [(False, True), (True, False), (False, False)])
def test_ready_is_503_naming_the_failed_check(postgres_ok, chroma_ok):
    r = _ready(postgres_ok, chroma_ok)
    assert r.status_code == 503
    body = r.json()
    assert body["status"] == "unavailable"
    assert body["checks"]["postgres"] == ("ok" if postgres_ok else "failed")
    assert body["checks"]["chroma"] == ("ok" if chroma_ok else "failed")
    assert "down" not in r.text  # exception text stays in the log


def test_health_is_unchanged():
    assert TestClient(app).get("/health").json() == {"status": "ok"}
