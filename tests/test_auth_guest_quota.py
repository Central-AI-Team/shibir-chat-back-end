"""Guest message quota (app/auth/guest_quota.py and its use in app/api/router.py)."""

from __future__ import annotations

import json
import os
import threading
from contextlib import contextmanager
from datetime import timedelta
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.engine import make_url
from sqlalchemy.orm import sessionmaker

import app.auth as auth
from app.auth import guest_quota, open_db
from app.core.config import settings
from app.db.base import Base
from app.db.models import AuthSession, GuestUsage, User
from app.main import app
from app.rag.gpu_client import GPUServiceError
from app.services import rate_limiter, session_store
from tests.conftest import AUTH_TEST_PASSWORD


@pytest.fixture(autouse=True)
def _limits(monkeypatch, auth_origin, auth_clock):
    monkeypatch.setattr(settings, "guest_message_limit", 5)
    monkeypatch.setattr(settings, "guest_ip_limit_per_day", 15)


def _client(ip: str = "10.0.0.1") -> TestClient:
    return TestClient(app, client=(ip, 50000), headers={"Origin": "http://localhost:5173"})


@contextmanager
def _chitchat(answer: str = "উত্তর", **respond_kwargs):
    with (
        patch("app.api.router.classify_intent", return_value="CHITCHAT"),
        patch("app.api.router.respond", return_value=answer, **respond_kwargs) as mock,
    ):
        yield mock


def _chat(client: TestClient, message: str = "প্রশ্ন", **extra):
    with _chitchat():
        return client.post("/chat", json={"message": message, **extra})


def _stream(client: TestClient, message: str = "প্রশ্ন"):
    with _chitchat():
        return client.post("/chat/stream", json={"message": message})


def _events(response) -> list[tuple[str, dict]]:
    out = []
    for block in response.text.strip().split("\n\n"):
        event, data = block.split("\n", 1)
        out.append((event.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
    return out


def _remaining(client: TestClient) -> int:
    body = client.get("/auth/me").json()
    assert body["authenticated"] is False and body["guest_limit"] == 5
    return body["guest_remaining"]


def _assert_limit_401(r) -> None:
    assert r.status_code == 401
    assert r.headers["content-type"].startswith("application/json")
    body = r.json()
    assert body["code"] == "guest_limit_reached"
    assert body["guest_limit"] == 5 and body["guest_remaining"] == 0
    assert "লগইন" in body["detail"]


# ---------------------------------------------------------------------------
# the limit
# ---------------------------------------------------------------------------


def test_five_responses_then_401_on_chat():
    client = _client()
    for expected in (4, 3, 2, 1, 0):
        r = _chat(client)
        assert r.status_code == 200
        assert r.json()["guest_remaining"] == expected
        assert r.json()["guest_limit"] == 5
    with _chitchat() as mock:
        r = client.post("/chat", json={"message": "প্রশ্ন"})
    _assert_limit_401(r)
    mock.assert_not_called()


def test_five_responses_then_plain_401_before_the_stream():
    client = _client()
    for expected in (4, 3, 2, 1, 0):
        r = _stream(client)
        assert r.status_code == 200
        event, data = _events(r)[-1]
        assert event == "done" and data["guest_remaining"] == expected and data["guest_limit"] == 5
    r = _stream(client)
    _assert_limit_401(r)
    assert "event:" not in r.text


def test_chat_with_stream_flag_shares_the_quota():
    client = _client()
    for _ in range(3):
        assert _chat(client).status_code == 200
    for _ in range(2):
        with _chitchat():
            assert client.post("/chat", json={"message": "প্রশ্ন", "stream": True}).status_code == 200
    _assert_limit_401(_chat(client))


def test_new_chats_and_sessions_do_not_reset_it():
    client = _client()
    sid = _chat(client).json()["session_id"]
    _chat(client, session_id=sid)
    _chat(client)  # a new chat
    _chat(client, session_id="x" * 43)  # unknown id -> new session
    _chat(client, session_id=sid)
    _assert_limit_401(_chat(client))
    assert len(session_store.list_sessions(_owner(client))) == 3


def _owner(client: TestClient) -> str:
    from app.api.deps import COOKIE_NAME, hash_token

    return hash_token(client.cookies[COOKIE_NAME])


def test_clearing_the_cookie_does_not_reset_it(monkeypatch):
    monkeypatch.setattr(settings, "guest_ip_limit_per_day", 7)
    first = _client()
    for _ in range(5):
        assert _chat(first).status_code == 200
    second = _client()  # same IP, no cookie: a fresh owner
    assert _chat(second).status_code == 200
    assert _chat(second).json()["guest_remaining"] == 0
    _assert_limit_401(_chat(second))
    assert _remaining(_client()) == 0  # cookie-less /auth/me sees the IP counter


def test_ip_window_rolls_after_24_hours(monkeypatch, auth_clock):
    monkeypatch.setattr(settings, "guest_ip_limit_per_day", 2)
    assert _chat(_client()).status_code == 200
    assert _chat(_client()).status_code == 200
    _assert_limit_401(_chat(_client()))
    auth_clock["now"] += timedelta(hours=23, minutes=59)
    _assert_limit_401(_chat(_client()))
    auth_clock["now"] += timedelta(minutes=2)
    assert _chat(_client()).status_code == 200


def test_ipv6_is_counted_per_64(monkeypatch):
    monkeypatch.setattr(settings, "guest_ip_limit_per_day", 1)
    assert _chat(_client("2001:db8:1:2::1")).status_code == 200
    _assert_limit_401(_chat(_client("2001:db8:1:2:ffff::9")))
    assert _chat(_client("2001:db8:1:3::1")).status_code == 200


def test_quota_survives_a_restart(tmp_path, monkeypatch):
    url = f"sqlite:///{tmp_path / 'quota.db'}"

    def boot():
        engine = create_engine(url, connect_args={"check_same_thread": False})
        Base.metadata.create_all(engine, tables=[User.__table__, AuthSession.__table__, GuestUsage.__table__])
        monkeypatch.setattr(auth, "SESSION_FACTORY", sessionmaker(bind=engine, future=True))
        return engine

    engine = boot()
    client = _client()
    for _ in range(5):
        assert _chat(client).status_code == 200
    engine.dispose()

    # "restart": every in-process store is empty, the database is not
    rate_limiter.reset()
    monkeypatch.setattr(session_store, "_sessions", {})
    engine = boot()
    restarted = TestClient(app, client=("10.0.0.1", 50000), headers={"Origin": "http://localhost:5173"},
                           cookies=dict(client.cookies))
    _assert_limit_401(_chat(restarted))
    engine.dispose()


def test_zero_limit_means_guests_cannot_chat(monkeypatch):
    monkeypatch.setattr(settings, "guest_message_limit", 0)
    r = _chat(_client())
    assert r.status_code == 401 and r.json()["code"] == "guest_limit_reached"


# ---------------------------------------------------------------------------
# what consumes, what refunds
# ---------------------------------------------------------------------------


def test_invalid_messages_consume_nothing(monkeypatch):
    monkeypatch.setattr(settings, "max_message_chars", 10)
    client = _client()
    assert _chat(client).status_code == 200  # get a cookie; 4 left
    assert _chat(client, message="").status_code == 400
    assert _chat(client, message="   ").status_code == 400
    assert _chat(client, message="x" * 11).status_code == 413
    with _chitchat():
        assert client.post("/chat/stream", json={"message": ""}).status_code == 400
        assert client.post("/chat/stream", json={"message": "x" * 11}).status_code == 413
        assert client.post("/chat", json={"nope": 1}).status_code == 422
    assert _remaining(client) == 4


def test_upstream_failure_is_refunded():
    client = _client()
    assert _chat(client).status_code == 200
    with _chitchat(side_effect=GPUServiceError("down")):
        assert client.post("/chat", json={"message": "প্রশ্ন"}).status_code == 503
    assert _remaining(client) == 4
    with _chitchat(side_effect=GPUServiceError("down")):
        r = client.post("/chat/stream", json={"message": "প্রশ্ন"})
    assert _events(r)[-1][0] == "error"
    assert _remaining(client) == 4


def test_guest_note_is_403_and_refunded():
    client = _client()
    assert _chat(client).status_code == 200
    with patch("app.api.router.classify_intent", return_value="NOTE"), \
            patch("app.api.router.generate_book_notes_from_text") as note:
        r = client.post("/chat", json={"message": "নোট বানাও বই"})
        assert r.status_code == 403
        assert "লগইন" in r.json()["detail"]
        s = client.post("/chat/stream", json={"message": "নোট বানাও বই"})
    note.assert_not_called()
    (event, data), = _events(s)
    assert event == "error" and "লগইন" in data["detail"]
    assert _remaining(client) == 4


def test_success_is_never_refunded():
    client = _client()
    _chat(client)
    _stream(client)
    assert _remaining(client) == 3


def test_me_for_a_guest_without_a_cookie_does_not_mint_one():
    client = _client()
    r = client.get("/auth/me")
    assert r.json() == {"authenticated": False, "guest_limit": 5, "guest_remaining": 5}
    assert "set-cookie" not in r.headers


def test_logged_in_users_have_no_guest_limit(make_user):
    make_user()
    client = _client()
    assert client.post("/auth/login", json={"email": "alice@example.com", "password": AUTH_TEST_PASSWORD}).status_code == 200
    for _ in range(7):
        r = _chat(client)
        assert r.status_code == 200
        assert r.json()["guest_remaining"] is None and r.json()["guest_limit"] is None
    done = _events(_stream(client))[-1][1]
    assert done["guest_remaining"] is None and done["guest_limit"] is None


def test_database_down_fails_closed(monkeypatch):
    broken = create_engine("sqlite:////nonexistent-dir/quota.db")
    monkeypatch.setattr(auth, "SESSION_FACTORY", sessionmaker(bind=broken))
    with _chitchat() as mock:
        r = _client().post("/chat", json={"message": "প্রশ্ন"})
    assert r.status_code == 503
    mock.assert_not_called()


# ---------------------------------------------------------------------------
# concurrency
# ---------------------------------------------------------------------------


def _hammer(n_threads: int, owner_for) -> list:
    barrier = threading.Barrier(n_threads)
    results = [None] * n_threads

    def worker(i):
        barrier.wait()
        results[i] = guest_quota.reserve(owner_for(i), "10.9.9.9")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results


def _file_db(tmp_path, monkeypatch):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'race.db'}", connect_args={"check_same_thread": False, "timeout": 30}
    )
    Base.metadata.create_all(engine, tables=[GuestUsage.__table__])
    monkeypatch.setattr(auth, "SESSION_FACTORY", sessionmaker(bind=engine, future=True))
    return engine


def _counts() -> dict[str, int]:
    with open_db() as db:
        return {row.key: row.count for row in db.execute(select(GuestUsage)).scalars()}


def test_concurrent_requests_never_exceed_the_owner_limit(tmp_path, monkeypatch):
    _file_db(tmp_path, monkeypatch)
    results = _hammer(20, lambda i: "same-owner")
    assert sum(r is not None for r in results) == 5
    assert _counts() == {"owner:same-owner": 5, "ip:10.9.9.9": 5}


def test_concurrent_requests_never_exceed_the_ip_limit(tmp_path, monkeypatch):
    _file_db(tmp_path, monkeypatch)
    monkeypatch.setattr(settings, "guest_ip_limit_per_day", 6)
    results = _hammer(20, lambda i: f"owner-{i}")
    assert sum(r is not None for r in results) == 6
    counts = _counts()
    assert counts["ip:10.9.9.9"] == 6
    # owners whose IP reservation failed were released again
    assert sum(v for k, v in counts.items() if k.startswith("owner:")) == 6


def test_concurrency_on_postgres(monkeypatch):
    """Real row locking; runs only against a Postgres test database."""
    url = os.environ.get("TEST_DATABASE_URL") or settings.database_url
    parsed = make_url(url)
    if "test" not in (parsed.database or "") or not parsed.drivername.startswith("postgresql"):
        pytest.skip("needs TEST_DATABASE_URL pointing at a Postgres database whose name contains 'test'")
    engine = create_engine(url)
    Base.metadata.create_all(engine, tables=[GuestUsage.__table__])
    monkeypatch.setattr(auth, "SESSION_FACTORY", sessionmaker(bind=engine, future=True))
    owner = f"pg-race-{os.getpid()}"
    try:
        results = _hammer(20, lambda i: owner)
        assert sum(r is not None for r in results) == 5
    finally:
        with open_db() as db:
            db.query(GuestUsage).filter(GuestUsage.key.in_([f"owner:{owner}", "ip:10.9.9.9"])).delete()
            db.commit()
        engine.dispose()
