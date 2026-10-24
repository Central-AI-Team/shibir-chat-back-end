"""Anonymous per-browser session ownership (Step 2: the session IDOR fix).

"alice" and "bob" are separate TestClients, so each has its own cookie jar and
therefore its own `shibir_owner` identity. No LLM, retriever or network is
used: every chat patches intent + answer generation.
"""

from __future__ import annotations

import json
import re
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from sqlalchemy import select, update

from app.api.deps import COOKIE_NAME, hash_token
from app.core.config import Settings, settings
from app.db.models import Conversation
from app.main import app
from app.services import session_store

ALLOWED_ORIGIN = "https://shibirgpt.potropollob.com"
NOT_FOUND = {"detail": "conversation পাওয়া যায়নি।"}
UNKNOWN_ID = "A" * 43  # well-formed, never issued


@contextmanager
def _qa(answer: str = "উত্তর।"):
    with (
        patch("app.api.router.classify_intent", return_value="QA"),
        patch("app.services.qa_service.retrieve_relevant_docs", return_value=[]),
        patch("app.services.qa_service.generate_answer", return_value=answer),
    ):
        yield


@contextmanager
def _stream_qa():
    with (
        patch("app.api.router.classify_intent", return_value="QA"),
        patch("app.services.qa_service.retrieve_relevant_docs", return_value=[]),
        patch("app.services.qa_service.stream_answer", return_value=iter(["উত্তর।"])),
    ):
        yield


def _chat(client: TestClient, message: str = "যাকাত কী?", **extra):
    with _qa():
        return client.post("/chat", json={"message": message, **extra})


@pytest.fixture
def alice():
    return TestClient(app)


@pytest.fixture
def bob():
    return TestClient(app)


def _set_cookie_headers(response) -> list[str]:
    return [v for k, v in response.headers.multi_items() if k.lower() == "set-cookie"]


def _row(sid: str) -> Conversation | None:
    """The stored row by id alone (any owner, deleted or not): the store's
    public functions can't answer "was anything created under this id?"."""
    with session_store._open() as db:
        return db.execute(select(Conversation).where(Conversation.id == sid)).scalar_one_or_none()


# ---------------------------------------------------------------------------
# cross-owner access
# ---------------------------------------------------------------------------


def test_other_owner_gets_the_same_404_as_an_unknown_id(alice, bob):
    sid = _chat(alice).json()["session_id"]

    unknown_get = bob.get(f"/conversations/{UNKNOWN_ID}/messages")
    for r in (
        bob.get(f"/conversations/{sid}/messages"),
        bob.delete(f"/conversations/{sid}"),
        unknown_get,
        bob.delete(f"/conversations/{UNKNOWN_ID}"),
    ):
        assert r.status_code == 404
        assert r.json() == NOT_FOUND
        assert r.content == unknown_get.content

    # alice's session is untouched by bob's attempts
    msgs = alice.get(f"/conversations/{sid}/messages")
    assert msgs.status_code == 200
    assert [m["content"] for m in msgs.json()] == ["যাকাত কী?", "উত্তর।"]


def _done_session_id(sse_text: str) -> str:
    for block in sse_text.split("\n\n"):
        if block.startswith("event: done"):
            return json.loads(block.split("data: ", 1)[1])["session_id"]
    raise AssertionError(f"no done event in {sse_text!r}")


def test_chat_with_other_owners_session_starts_a_new_one(alice, bob):
    sid = _chat(alice).json()["session_id"]

    r = _chat(bob, "ববের প্রশ্ন", session_id=sid)
    assert r.status_code == 200
    new_sid = r.json()["session_id"]
    assert new_sid != sid
    with _stream_qa():
        rs = bob.post("/chat/stream", json={"message": "প্রশ্ন", "session_id": sid})
    assert rs.status_code == 200
    assert _done_session_id(rs.text) not in (sid, new_sid)

    # bob's turns went to his own sessions; alice's is untouched.
    msgs = alice.get(f"/conversations/{sid}/messages").json()
    assert [m["content"] for m in msgs] == ["যাকাত কী?", "উত্তর।"]
    assert sid not in {row["id"] for row in bob.get("/conversations").json()}


def test_conversation_list_is_per_owner(alice, bob):
    a1 = _chat(alice, "প্রথম").json()["session_id"]
    a2 = _chat(alice, "দ্বিতীয়").json()["session_id"]
    b1 = _chat(bob, "ববের প্রশ্ন").json()["session_id"]

    alice_ids = {row["id"] for row in alice.get("/conversations").json()}
    bob_ids = {row["id"] for row in bob.get("/conversations").json()}
    assert alice_ids == {a1, a2}
    assert bob_ids == {b1}


def test_unknown_session_id_starts_a_new_session_under_a_fresh_id(alice):
    # e.g. an id the front-end kept across a server restart (in-memory store).
    r = _chat(alice, session_id=UNKNOWN_ID)
    assert r.status_code == 200
    new_sid = r.json()["session_id"]
    assert new_sid != UNKNOWN_ID
    with _stream_qa():
        rs = alice.post("/chat/stream", json={"message": "প্রশ্ন", "session_id": UNKNOWN_ID})
    assert rs.status_code == 200
    assert _done_session_id(rs.text) not in (UNKNOWN_ID, new_sid)
    assert _row(UNKNOWN_ID) is None

    # the new id works for the next turn
    assert _chat(alice, "আবার", session_id=new_sid).json()["session_id"] == new_sid


@pytest.mark.parametrize(
    "bad", ["short", "x" * 44, "a" * 42 + "!", "../" + "a" * 40, "a" * 65,
            "3f1c9a52-7b1e-4d2a-9c3e-0a1b2c3d4e5f"],  # last: a uuid4 id from main
)
def test_malformed_body_session_id_starts_a_new_session(alice, bad):
    r = _chat(alice, session_id=bad)
    assert r.status_code == 200
    assert r.json()["session_id"] != bad
    assert _row(bad) is None


@pytest.mark.parametrize("bad", ["does-not-exist", "a" * 42 + "$", "a" * 44])
def test_malformed_path_id_is_the_same_404(alice, bad):
    for r in (alice.get(f"/conversations/{bad}/messages"), alice.delete(f"/conversations/{bad}")):
        assert r.status_code == 404
        assert r.json() == NOT_FOUND


def test_roleplay_persona_and_history_never_reach_another_owner(alice, bob):
    with (
        patch("app.api.router.classify_intent", return_value="ROLEPLAY"),
        patch("app.core.llm.get_client") as get_client,
    ):
        fake = MagicMock()
        get_client.return_value = fake

        def completion(text):
            msg = MagicMock(content=text)
            return MagicMock(choices=[MagicMock(message=msg)])

        fake.chat.completions.create.side_effect = [
            completion("গোপন চরিত্র"), completion("অ্যালিসের উত্তর"),
        ]
        sid = alice.post("/chat", json={"message": "অ্যালিসের গোপন বার্তা"}).json()["session_id"]

        # bob sending alice's id gets a fresh roleplay of his own, with none
        # of alice's persona or history.
        fake.chat.completions.create.reset_mock()
        fake.chat.completions.create.side_effect = [
            completion("ববের চরিত্র"), completion("ববের উত্তর"),
        ]
        r = bob.post("/chat", json={"message": "ববের বার্তা", "session_id": sid})
        assert r.status_code == 200
        assert r.json()["session_id"] != sid

    sent = str(fake.chat.completions.create.call_args_list)
    assert "অ্যালিসের" not in sent and "গোপন" not in sent


# ---------------------------------------------------------------------------
# owner cookie
# ---------------------------------------------------------------------------


def _cookie_attrs(header: str) -> set[str]:
    return {part.strip().split("=")[0].lower() for part in header.split(";")[1:]}


@pytest.mark.parametrize("route", ["/chat", "/chat/stream", "/chat?stream"])
def test_first_request_sets_cookie_second_does_not(route, monkeypatch):
    monkeypatch.setattr(settings, "cookie_secure", True)
    client = TestClient(app, base_url="https://testserver")  # Secure cookies need https
    path = "/chat/stream" if route == "/chat/stream" else "/chat"
    body = {"message": "প্রশ্ন", "stream": route == "/chat?stream"}

    with _qa(), _stream_qa():
        first = client.post(path, json=body)
    assert first.status_code == 200
    headers = _set_cookie_headers(first)
    assert len(headers) == 1 and headers[0].startswith(f"{COOKIE_NAME}=")
    attrs = _cookie_attrs(headers[0])
    assert {"httponly", "secure", "path", "max-age", "samesite"} <= attrs
    assert "samesite=lax" in headers[0].lower()
    assert "max-age=2592000" in headers[0].lower()

    with _qa(), _stream_qa():
        second = client.post(path, json=body)
    assert second.status_code == 200
    assert _set_cookie_headers(second) == []


def test_secure_flag_off_when_disabled():
    r = _chat(TestClient(app))
    (header,) = _set_cookie_headers(r)
    assert "secure" not in _cookie_attrs(header)


def test_get_conversations_sets_cookie_for_a_new_browser():
    r = TestClient(app).get("/conversations")
    assert r.status_code == 200 and r.json() == []
    assert len(_set_cookie_headers(r)) == 1


@pytest.mark.parametrize("forged", ["x", "a" * 42 + "!", "a" * 44, "", "%00" * 20])
def test_invalid_cookie_is_replaced_not_trusted(forged, alice):
    sid = _chat(alice).json()["session_id"]
    alice_cookie = alice.cookies.get(COOKIE_NAME)

    mallory = TestClient(app)
    mallory.cookies.set(COOKIE_NAME, forged)
    r = _chat(mallory)
    (header,) = _set_cookie_headers(r)
    new_token = header.split(";")[0].split("=", 1)[1]
    assert re.fullmatch(r"[A-Za-z0-9_-]{43}", new_token)
    assert new_token not in (forged, alice_cookie)
    assert sid not in {row["id"] for row in mallory.get("/conversations").json()}


def test_store_holds_only_the_hash_of_the_token(alice):
    sid = _chat(alice).json()["session_id"]
    token = alice.cookies.get(COOKIE_NAME)
    stored = _row(sid).owner_key
    assert stored == hash_token(token)
    assert token not in repr(session_store.get_owned_session(sid, stored))


# ---------------------------------------------------------------------------
# session store
# ---------------------------------------------------------------------------


def test_session_ids_are_43_char_urlsafe_and_unique():
    owner = "u" * 64
    ids = set()
    try:
        for _ in range(1000):
            sid, _ = session_store.create_session(owner)
            assert re.fullmatch(r"[A-Za-z0-9_-]{43}", sid)
            ids.add(sid)
        assert len(ids) == 1000
    finally:
        for sid in ids:
            session_store.delete_session(sid, owner)


def _age(sid: str, minutes: int) -> None:
    with session_store._open() as db:
        db.execute(
            update(Conversation)
            .where(Conversation.id == sid)
            .values(updated_at=datetime.now(timezone.utc) - timedelta(minutes=minutes))
        )
        db.commit()


def test_per_owner_cap_evicts_that_owners_least_recently_updated(monkeypatch):
    monkeypatch.setattr(settings, "guest_conversation_cap", 3)
    owner, other = "p" * 64, "q" * 64
    s1, _ = session_store.create_session(owner)
    s2, _ = session_store.create_session(owner)
    s3, _ = session_store.create_session(owner)
    o1, _ = session_store.create_session(other)
    _age(s1, 5); _age(s2, 30); _age(s3, 1); _age(o1, 999)  # noqa: E702

    s4, _ = session_store.create_session(owner)

    mine = {row["id"] for row in session_store.list_sessions(owner)}
    assert mine == {s1, s3, s4}  # s2 was the least recently updated
    assert session_store.get_owned_session(o1, other) is not None  # other owners untouched
    for sid in (s1, s3, s4):
        session_store.delete_session(sid, owner)
    session_store.delete_session(o1, other)


def test_user_cap_applies_to_logged_in_owners(monkeypatch):
    # The global cap of the in-memory store is gone (Postgres); users get a
    # larger per-owner cap than guests, selected by create_session(user_id=).
    monkeypatch.setattr(settings, "guest_conversation_cap", 1)
    monkeypatch.setattr(settings, "user_conversation_cap", 2)
    owner, uid = "w" * 64, "00000000-0000-4000-8000-00000000abcd"
    a, _ = session_store.create_session(owner, user_id=uid)
    b, _ = session_store.create_session(owner, user_id=uid)
    _age(a, 5)
    c, _ = session_store.create_session(owner, user_id=uid)
    assert {row["id"] for row in session_store.list_sessions(owner)} == {b, c}
    assert _row(c).user_id == uid


def test_update_and_append_never_create_and_tolerate_a_vanished_session():
    assert session_store.update_session(UNKNOWN_ID, mode="QA") is None
    assert session_store.append_history(UNKNOWN_ID, "user", "x") is None
    assert _row(UNKNOWN_ID) is None


def test_concurrent_delete_mid_request_does_not_crash(alice):
    """The session disappears between resolve and the final update/append."""
    sid = _chat(alice).json()["session_id"]
    owner = hash_token(alice.cookies.get(COOKIE_NAME))

    def answer_then_delete(*_a, **_k):
        session_store.delete_session(sid, owner)
        return "উত্তর।"

    with (
        patch("app.api.router.classify_intent", return_value="QA"),
        patch("app.services.qa_service.retrieve_relevant_docs", return_value=[]),
        patch("app.services.qa_service.generate_answer", side_effect=answer_then_delete),
    ):
        r = alice.post("/chat", json={"message": "আবার", "session_id": sid})
    assert r.status_code == 200
    assert session_store.get_owned_session(sid, owner) is None


# ---------------------------------------------------------------------------
# CSRF (Origin) and CORS settings
# ---------------------------------------------------------------------------


def test_disallowed_origin_is_403_on_post_and_delete(alice):
    sid = _chat(alice).json()["session_id"]
    evil = {"Origin": "https://evil.example"}
    with _qa():
        assert alice.post("/chat", json={"message": "x"}, headers=evil).status_code == 403
    with _stream_qa():
        assert alice.post("/chat/stream", json={"message": "x"}, headers=evil).status_code == 403
    assert alice.delete(f"/conversations/{sid}", headers=evil).status_code == 403
    assert alice.get(f"/conversations/{sid}/messages").status_code == 200  # not deleted


def test_allowed_and_missing_origin_work(alice):
    with _qa():
        assert alice.post("/chat", json={"message": "x"}, headers={"Origin": ALLOWED_ORIGIN}).status_code == 200
        sid = alice.post("/chat", json={"message": "y"}).json()["session_id"]
    assert alice.delete(f"/conversations/{sid}", headers={"Origin": ALLOWED_ORIGIN}).status_code == 204


def test_same_origin_is_allowed(alice):
    # The /docs page posts with Origin = the API's own address; TestClient's
    # host is http://testserver. Same host, different port or scheme is not.
    with _qa():
        assert alice.post("/chat", json={"message": "x"}, headers={"Origin": "http://testserver"}).status_code == 200
        assert alice.post("/chat", json={"message": "x"}, headers={"Origin": "http://testserver:8080"}).status_code == 403
        assert alice.post("/chat", json={"message": "x"}, headers={"Origin": "https://testserver"}).status_code == 403


def test_cors_allows_credentials_for_the_front_end(alice):
    r = alice.options(
        "/chat",
        headers={"Origin": ALLOWED_ORIGIN, "Access-Control-Request-Method": "POST",
                 "Access-Control-Request-Headers": "content-type"},
    )
    assert r.status_code == 200
    assert r.headers["access-control-allow-origin"] == ALLOWED_ORIGIN
    assert r.headers["access-control-allow-credentials"] == "true"


def test_settings_reject_wildcard_cors_origin():
    with pytest.raises(ValueError):
        Settings(openai_api_key="x", cors_allow_origins=["*"])


# ---------------------------------------------------------------------------
# tracing: no raw identifiers leave the process
# ---------------------------------------------------------------------------


def test_trace_gets_owner_prefix_not_client_user_id_or_session_id(alice):
    with patch("app.api.router.tracing.start_request_trace") as start:
        with _qa():
            r = alice.post("/chat", json={"message": "প্রশ্ন", "user_id": "spoofed-user"})
    sid = r.json()["session_id"]
    kwargs = start.call_args.kwargs
    owner = hash_token(alice.cookies.get(COOKIE_NAME))
    assert kwargs["user_id"] == owner[:12]
    assert kwargs["session_id"] != sid and sid not in str(start.call_args)
    assert alice.cookies.get(COOKIE_NAME) not in str(start.call_args)
    assert owner not in str(start.call_args)
