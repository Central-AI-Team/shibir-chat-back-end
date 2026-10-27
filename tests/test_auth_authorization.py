"""Who may do what: NOTE gating, conversation isolation between users and
guests, guest-session migration on login, cookie separation, require_role,
the per-user daily cap."""

from __future__ import annotations

import json
from contextlib import contextmanager
from unittest.mock import patch

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from app.api.deps import COOKIE_NAME, hash_token, require_role
from app.auth import principal as principals
from app.auth import sessions
from app.core.config import settings
from app.main import app
from app.services import rate_limiter, session_store
from tests.conftest import AUTH_TEST_PASSWORD as PASSWORD

SESSION_COOKIE = sessions.COOKIE_NAME_PLAIN
_NOTE = {"book": "বই", "chapters": [{"chapter": "অধ্যায়", "pages_used": 1, "note": "নোট"}]}


@pytest.fixture(autouse=True)
def _setup(auth_origin, auth_clock):
    pass


def _client(ip: str = "10.0.0.1") -> TestClient:
    return TestClient(app, client=(ip, 50000), headers={"Origin": "http://localhost:5173"})


def _logged_in(make_user, email: str, role: str = "user", ip: str = "10.0.0.1") -> TestClient:
    make_user(email=email, role=role)
    client = _client(ip)
    assert client.post("/auth/login", json={"email": email, "password": PASSWORD}).status_code == 200
    return client


@contextmanager
def _intent(intent: str):
    with (
        patch("app.api.router.classify_intent", return_value=intent),
        patch("app.api.router.respond", return_value="উত্তর"),
        patch("app.api.router.generate_book_notes_from_text", return_value=_NOTE) as note,
    ):
        yield note


def _chat(client: TestClient, intent: str = "CHITCHAT", **extra):
    with _intent(intent):
        return client.post("/chat", json={"message": "নোট বানাও বই", **extra})


# ---------------------------------------------------------------------------
# NOTE
# ---------------------------------------------------------------------------


def test_guest_is_blocked_from_note_and_user_is_allowed(make_user):
    guest = _client()
    with _intent("NOTE") as note:
        r = guest.post("/chat", json={"message": "নোট বানাও বই"})
        s = guest.post("/chat/stream", json={"message": "নোট বানাও বই"})
    assert r.status_code == 403
    assert s.status_code == 200 and s.text.startswith("event: error")
    note.assert_not_called()

    user = _logged_in(make_user, "alice@example.com")
    r = _chat(user, "NOTE")
    assert r.status_code == 200 and r.json()["mode"] == "note"
    with _intent("NOTE"):
        s = user.post("/chat/stream", json={"message": "নোট বানাও বই"})
    assert "event: done" in s.text


def test_roleplay_and_suggestion_stay_open_to_guests():
    guest = _client()
    with patch("app.api.router.classify_intent", return_value="SUGGESTION"), \
            patch("app.api.router.give_suggestion", return_value=("পরামর্শ", [])):
        assert guest.post("/chat", json={"message": "পরামর্শ দাও"}).status_code == 200
    from app.services.roleplay_service import RoleplayResult

    canned = RoleplayResult(answer="ভূমিকা", sources=[], suggestions=[], role_id="bondhu",
                            level="beginner", refused=False)
    with patch("app.api.router.classify_intent", return_value="ROLEPLAY"), \
            patch("app.api.router.handle_roleplay", return_value=canned):
        assert guest.post("/chat", json={"message": "রোলপ্লে"}).status_code == 200


# ---------------------------------------------------------------------------
# conversation isolation
# ---------------------------------------------------------------------------


def test_users_and_guests_cannot_see_each_others_conversations(make_user):
    alice = _logged_in(make_user, "alice@example.com")
    bob = _logged_in(make_user, "bob@example.com", ip="10.0.0.2")
    guest = _client("10.0.0.3")
    sid = _chat(alice).json()["session_id"]
    assert alice.get(f"/conversations/{sid}/messages").status_code == 200

    unknown = bob.get(f"/conversations/{'z' * 43}/messages")
    for intruder in (bob, guest):
        r = intruder.get(f"/conversations/{sid}/messages")
        assert r.status_code == 404 and r.json() == unknown.json()
        assert intruder.delete(f"/conversations/{sid}").status_code == 404
        assert intruder.get("/conversations").json() == []
        # continuing it starts a new session instead
        assert _chat(intruder, session_id=sid).json()["session_id"] != sid
    assert alice.get(f"/conversations/{sid}/messages").status_code == 200


def test_a_user_sees_the_same_conversations_from_any_browser(make_user):
    first = _logged_in(make_user, "alice@example.com")
    sid = _chat(first).json()["session_id"]
    second = _client("10.0.0.2")
    second.post("/auth/login", json={"email": "alice@example.com", "password": PASSWORD})
    assert [c["id"] for c in second.get("/conversations").json()] == [sid]


# ---------------------------------------------------------------------------
# guest -> user migration
# ---------------------------------------------------------------------------


def test_login_moves_only_the_presenting_guests_sessions(make_user):
    make_user()
    alice_browser = _client()
    other_guest = _client("10.0.0.2")
    mine = _chat(alice_browser).json()["session_id"]
    theirs = _chat(other_guest).json()["session_id"]

    alice_browser.post("/auth/login", json={"email": "alice@example.com", "password": PASSWORD})
    assert [c["id"] for c in alice_browser.get("/conversations").json()] == [mine]
    assert alice_browser.get(f"/conversations/{theirs}/messages").status_code == 404
    assert [c["id"] for c in other_guest.get("/conversations").json()] == [theirs]
    # the conversation continues under the account
    assert _chat(alice_browser, session_id=mine).json()["session_id"] == mine


def test_register_moves_the_guest_sessions_too():
    browser = _client()
    sid = _chat(browser).json()["session_id"]
    r = browser.post("/auth/register", json={"email": "new@example.com", "password": "a long enough secret"})
    assert r.status_code == 201
    assert [c["id"] for c in browser.get("/conversations").json()] == [sid]


def test_reassign_owner_respects_the_per_owner_cap(monkeypatch):
    monkeypatch.setattr(settings, "guest_conversation_cap", 3)
    user_ids = [session_store.create_session("user")[0] for _ in range(2)]
    guest_ids = [session_store.create_session("guest")[0] for _ in range(2)]
    session_store.create_session("someone-else")
    assert session_store.reassign_owner("guest", "user") == 2
    owned = {row["id"] for row in session_store.list_sessions("user")}
    assert len(owned) == 3 and set(guest_ids) <= owned and user_ids[0] not in owned
    assert session_store.list_sessions("guest") == []
    assert len(session_store.list_sessions("someone-else")) == 1


# ---------------------------------------------------------------------------
# cookie separation
# ---------------------------------------------------------------------------


def test_a_guest_cookie_is_never_a_session_cookie_and_vice_versa(make_user):
    guest = _client()
    _chat(guest)
    guest_token = guest.cookies[COOKIE_NAME]
    as_session = _client()
    as_session.cookies.set(SESSION_COOKIE, guest_token, domain="testserver.local")
    assert as_session.get("/auth/me").json()["authenticated"] is False

    alice = _logged_in(make_user, "alice@example.com", ip="10.0.0.2")
    sid = _chat(alice).json()["session_id"]
    session_token = alice.cookies[SESSION_COOKIE]
    as_owner = _client("10.0.0.3")
    as_owner.cookies.set(COOKIE_NAME, session_token, domain="testserver.local")
    assert as_owner.get("/auth/me").json()["authenticated"] is False
    assert as_owner.get(f"/conversations/{sid}/messages").status_code == 404
    assert as_owner.get("/conversations").json() == []


def test_user_owner_key_differs_from_any_guest_owner():
    user = principals.user("u-1", "user")
    assert user.owner_key == principals.user_owner_key("u-1") != hash_token("u-1")


# ---------------------------------------------------------------------------
# require_role
# ---------------------------------------------------------------------------


def test_require_role_admin(make_user):
    mini = FastAPI()

    @mini.get("/admin-only")
    def admin_only(p=Depends(require_role("admin"))):
        return {"ok": True}

    def client_for(email, role):
        make_user(email=email, role=role)
        main_client = _client()
        main_client.post("/auth/login", json={"email": email, "password": PASSWORD})
        c = TestClient(mini)
        c.cookies.set(SESSION_COOKIE, main_client.cookies[SESSION_COOKIE], domain="testserver.local")
        return c

    assert TestClient(mini).get("/admin-only").status_code == 401
    r = client_for("user@example.com", "user").get("/admin-only")
    assert r.status_code == 403 and r.json() == {"detail": "forbidden"}
    assert client_for("admin@example.com", "admin").get("/admin-only").status_code == 200
    assert require_role("admin") is require_role("admin")


# ---------------------------------------------------------------------------
# per-user daily cap
# ---------------------------------------------------------------------------


def test_user_daily_cap(make_user, monkeypatch):
    monkeypatch.setattr(settings, "user_daily_message_limit", 2)
    monkeypatch.setattr(rate_limiter, "_time", lambda: 86_400 * 20_000 + 3_600)  # 01:00 UTC
    alice = _logged_in(make_user, "alice@example.com")
    assert _chat(alice).status_code == 200
    assert _chat(alice).status_code == 200
    r = _chat(alice)
    assert r.status_code == 429
    assert r.json()["detail"] == rate_limiter.USER_DAILY_LIMIT_DETAIL
    assert r.headers["Retry-After"] == str(23 * 3600)
    bob = _logged_in(make_user, "bob@example.com", ip="10.0.0.2")
    assert _chat(bob).status_code == 200
    with _intent("CHITCHAT"):
        s = alice.post("/chat/stream", json={"message": "প্রশ্ন"})
    assert s.status_code == 429  # plain HTTP, before the stream


def test_guest_principal_dependency_mints_the_owner_cookie_as_before():
    r = _client().get("/conversations")
    assert r.status_code == 200 and COOKIE_NAME in r.headers.get("set-cookie", "")
    assert json.loads(r.text) == []
