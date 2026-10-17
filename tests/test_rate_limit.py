"""In-process rate limits (app/services/rate_limiter.py, wired in app/api/deps.py
and the NOTE branch of app/api/router.py).

Time is a fake clock (rate_limiter._monotonic / _time), like test_gpu_client.
Client IPs come from TestClient(client=(ip, port)), which is what
request.client.host reports. conftest.py resets every limiter between tests.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.api.deps import COOKIE_NAME
from app.core.config import settings
from app.main import app
from app.services import rate_limiter, session_store

_DAY = 86_400
# 100 s before a UTC midnight.
_NEAR_MIDNIGHT = 20_000 * _DAY - 100


@pytest.fixture
def clock(monkeypatch):
    """Advance with clock["mono"] += s (windows) / clock["time"] += s (UTC day)."""
    state = {"mono": 1_000.0, "time": float(_NEAR_MIDNIGHT - 10_000)}
    monkeypatch.setattr(rate_limiter, "_monotonic", lambda: state["mono"])
    monkeypatch.setattr(rate_limiter, "_time", lambda: state["time"])
    return state


@pytest.fixture
def limits(monkeypatch):
    """Small limits; each test tightens the one it exercises."""
    def _set(ip=100, owner=100, read=100, note=100, budget=0):
        monkeypatch.setattr(settings, "rate_limit_per_ip_per_minute", ip)
        monkeypatch.setattr(settings, "rate_limit_per_owner_per_minute", owner)
        monkeypatch.setattr(settings, "rate_limit_read_per_ip_per_minute", read)
        monkeypatch.setattr(settings, "note_limit_per_owner_per_hour", note)
        monkeypatch.setattr(settings, "daily_llm_request_budget", budget)
    return _set


@contextmanager
def _qa():
    with (
        patch("app.api.router.classify_intent", return_value="QA"),
        patch("app.services.qa_service.retrieve_relevant_docs", return_value=[]),
        patch("app.services.qa_service.generate_answer", return_value="উত্তর।"),
    ):
        yield


@contextmanager
def _note():
    result = {"book": "বই", "chapters": [{"chapter": "অধ্যায়", "pages_used": 1, "note": "নোট।"}]}
    with (
        patch("app.api.router.classify_intent", return_value="NOTE"),
        patch("app.api.router.generate_book_notes_from_text", return_value=result) as mock_note,
    ):
        yield mock_note


def _client(ip: str = "10.0.0.1", token: str | None = None) -> TestClient:
    cookies = {COOKIE_NAME: token} if token else None
    return TestClient(app, client=(ip, 50000), cookies=cookies)


def _chat(client: TestClient, message: str = "যাকাত কী?"):
    with _qa():
        return client.post("/chat", json={"message": message})


def _token(client: TestClient) -> str:
    return client.cookies[COOKIE_NAME]


# ---------------------------------------------------------------------------
# per-IP / per-owner on /chat and /chat/stream
# ---------------------------------------------------------------------------


def test_per_ip_limit_trips_at_n_plus_one(clock, limits):
    limits(ip=3)
    # A fresh cookie per request: rotating owners does not escape the IP limit.
    for _ in range(3):
        assert _chat(_client("10.0.0.1")).status_code == 200
    r = _chat(_client("10.0.0.1"))
    assert r.status_code == 429
    assert r.json()["detail"] == rate_limiter.RATE_LIMIT_DETAIL
    assert int(r.headers["Retry-After"]) >= 1
    # another IP still gets through
    assert _chat(_client("10.0.0.2")).status_code == 200


def test_per_owner_limit_trips_across_ips_and_spares_other_owners(clock, limits):
    limits(owner=2)
    first = _client("10.0.0.1")
    assert _chat(first).status_code == 200
    token = _token(first)
    assert _chat(_client("10.0.0.2", token)).status_code == 200
    assert _chat(_client("10.0.0.3", token)).status_code == 429
    # a different owner, even on an IP the first owner used, is unaffected
    assert _chat(_client("10.0.0.3")).status_code == 200


def test_window_expires(clock, limits):
    limits(ip=1)
    client = _client()
    assert _chat(client).status_code == 200
    clock["mono"] += 10
    r = _chat(client)
    assert r.status_code == 429
    assert r.headers["Retry-After"] == "50"
    clock["mono"] += 50
    assert _chat(client).status_code == 200


def test_stream_429_is_plain_http_before_the_stream(clock, limits):
    limits(ip=1)
    client = _client()
    assert _chat(client).status_code == 200
    with patch("app.api.router.classify_intent") as mock_classify:
        r = client.post("/chat/stream", json={"message": "প্রশ্ন"})
    assert r.status_code == 429
    assert r.headers["content-type"].startswith("application/json")
    assert "Retry-After" in r.headers
    mock_classify.assert_not_called()


def test_rejected_request_creates_no_session(clock, limits):
    limits(ip=1)
    client = _client()
    assert _chat(client).status_code == 200
    before = len(session_store._sessions)
    with patch("app.api.router.classify_intent") as mock_classify:
        assert client.post("/chat", json={"message": "আবার"}).status_code == 429
        assert client.post("/chat/stream", json={"message": "আবার"}).status_code == 429
    assert len(session_store._sessions) == before
    mock_classify.assert_not_called()


def test_read_routes_have_their_own_per_ip_limit(clock, limits):
    limits(ip=1, read=2)
    client = _client()
    sid = _chat(client).json()["session_id"]  # uses up the chat limit (1) ...
    # ... but reads are counted separately
    assert client.get("/conversations").status_code == 200
    assert client.get(f"/conversations/{sid}/messages").status_code == 200
    r = client.delete(f"/conversations/{sid}")
    assert r.status_code == 429
    assert "Retry-After" in r.headers
    assert client.get(f"/conversations/{sid}/messages").status_code == 429
    clock["mono"] += 60
    assert client.delete(f"/conversations/{sid}").status_code == 204


# ---------------------------------------------------------------------------
# NOTE hourly limit
# ---------------------------------------------------------------------------


def test_note_hourly_limit_on_chat(clock, limits, logged_in_user):
    limits(note=2)
    client = _client()
    with _note() as mock_note:
        for _ in range(2):
            assert client.post("/chat", json={"message": "নোট বানাও বই"}).status_code == 200
        r = client.post("/chat", json={"message": "নোট বানাও বই"})
    assert r.status_code == 429
    assert r.json()["detail"] == rate_limiter.NOTE_LIMIT_DETAIL
    assert int(r.headers["Retry-After"]) == 3600
    assert mock_note.call_count == 2

    # other modes are not affected by the NOTE limit
    assert _chat(client).status_code == 200
    clock["mono"] += 3600
    with _note():
        assert client.post("/chat", json={"message": "নোট বানাও বই"}).status_code == 200


def test_note_hourly_limit_on_stream_is_an_sse_error(clock, limits, logged_in_user):
    limits(note=1)
    client = _client()
    with _note() as mock_note:
        assert client.post("/chat/stream", json={"message": "নোট বানাও বই"}).status_code == 200
        r = client.post("/chat/stream", json={"message": "নোট বানাও বই"})
    assert r.status_code == 200  # the stream had started
    events = [block for block in r.text.strip().split("\n\n") if block]
    assert len(events) == 1 and events[0].startswith("event: error")
    assert json.loads(events[0].split("data: ", 1)[1])["detail"] == rate_limiter.NOTE_LIMIT_DETAIL
    assert mock_note.call_count == 1


# ---------------------------------------------------------------------------
# daily budget
# ---------------------------------------------------------------------------


def test_daily_budget_returns_429_until_utc_midnight(clock, limits):
    limits(budget=2)
    clock["time"] = float(_NEAR_MIDNIGHT)
    assert _chat(_client("10.0.0.1")).status_code == 200
    assert _chat(_client("10.0.0.2")).status_code == 200
    r = _chat(_client("10.0.0.3"))  # any IP, any owner: the budget is global
    assert r.status_code == 429
    assert r.json()["detail"] == rate_limiter.DAILY_BUDGET_DETAIL
    assert r.headers["Retry-After"] == "100"

    clock["time"] += 100  # the UTC day rolls over
    assert _chat(_client("10.0.0.3")).status_code == 200


def test_daily_budget_zero_is_unlimited(clock, limits):
    limits(budget=0)
    for i in range(10):
        assert _chat(_client(f"10.0.1.{i}")).status_code == 200


def test_requests_rejected_by_a_rate_limit_do_not_spend_the_budget(clock, limits):
    limits(ip=1, budget=2)
    assert _chat(_client("10.0.0.1")).status_code == 200
    assert _chat(_client("10.0.0.1")).status_code == 429  # IP limit, not counted
    assert _chat(_client("10.0.0.2")).status_code == 200
    assert _chat(_client("10.0.0.3")).json()["detail"] == rate_limiter.DAILY_BUDGET_DETAIL


# ---------------------------------------------------------------------------
# limiter internals: bounded memory
# ---------------------------------------------------------------------------


def test_tracked_key_cap_evicts_least_recently_hit(clock):
    limiter = rate_limiter.SlidingWindowLimiter(60, max_keys=3)
    for key in ("a", "b", "c"):
        assert limiter.hit(key, 1) is None
    assert limiter.hit("a", 1) is not None  # "a" is now the most recent attempt
    assert limiter.hit("d", 1) is None
    assert len(limiter._hits) == 3
    assert "b" not in limiter._hits  # least recently hit, evicted
    assert limiter.hit("b", 1) is None  # an evicted key starts a fresh window


def test_expired_keys_are_pruned(clock):
    limiter = rate_limiter.SlidingWindowLimiter(60, max_keys=100)
    for i in range(50):
        limiter.hit(f"k{i}", 5)
    clock["mono"] += 61
    limiter.hit("fresh", 5)
    assert list(limiter._hits) == ["fresh"]


def test_sliding_window_blocked_checks_without_recording(clock):
    limiter = rate_limiter.SlidingWindowLimiter(60)
    for _ in range(10):  # never counts
        assert limiter.blocked("k", 2) is None
    assert limiter.hit("k", 2) is None
    assert limiter.blocked("k", 2) is None
    assert limiter.hit("k", 2) is None
    assert limiter.blocked("k", 2) == 60
    assert limiter.hit("k", 2) == 60
    clock["mono"] += 61
    assert limiter.blocked("k", 2) is None
    assert limiter.hit("k", 2) is None
