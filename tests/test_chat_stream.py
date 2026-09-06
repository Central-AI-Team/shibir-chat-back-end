"""Tests for POST /chat/stream -- the Server-Sent Events variant of /chat.

Covers the SSE event sequence per mode, plus a regression test for the
ContextVar-loss bug: Starlette pulls the response generator one next() at a
time in a threadpool, each call in its own copied context, so the request
trace set in the first iteration is invisible by the time the generator's
finally runs -- finalize_request_trace() has to be handed the trace object
explicitly or every streamed trace ends up with no output/latency/tags.

Mocks at the service boundary; no embedding model, reranker, or real LLM.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app.core import tracing
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


def _events(raw: str) -> list[tuple[str, str]]:
    """Parse an SSE body into [(event, data), ...]."""
    out = []
    for block in raw.strip().split("\n\n"):
        if not block.strip():
            continue
        event = data = None
        for line in block.splitlines():
            if line.startswith("event: "):
                event = line[len("event: "):]
            elif line.startswith("data: "):
                data = line[len("data: "):]
        out.append((event, data))
    return out


# ---------------------------------------------------------------------------
# SSE event sequence
# ---------------------------------------------------------------------------


def test_stream_qa_emits_sources_then_tokens_then_done():
    with (
        patch("app.api.router.classify_intent", return_value="QA"),
        patch("app.api.router.retrieve_relevant_docs", return_value=_CITATIONS),
        patch("app.api.router.stream_answer", return_value=iter(["উ", "ত্ত", "র।"])),
    ):
        r = client.post("/chat/stream", json={"message": "তৃতীয় অধ্যায়ে কী আছে?"})

    assert r.status_code == 200
    evs = _events(r.text)
    kinds = [e for e, _ in evs]
    assert kinds[0] == "sources"
    assert kinds.count("token") == 3
    assert kinds[-1] == "done"
    assert json.loads(evs[-1][1])["mode"] == "qa"
    assert "".join(json.loads(d)["text"] for e, d in evs if e == "token") == "উত্তর।"


def test_stream_note_emits_single_token_and_done():
    note_result = {
        "book": "Test Book",
        "chapters": [{"chapter": "Chapter 1", "pages_used": 3, "note": "নোট।"}],
    }
    with (
        patch("app.api.router.classify_intent", return_value="NOTE"),
        patch("app.api.router.generate_book_notes_from_text", return_value=note_result),
    ):
        r = client.post("/chat/stream", json={"message": "নোট বানাও পরীক্ষা বই"})

    assert r.status_code == 200
    kinds = [e for e, _ in _events(r.text)]
    assert kinds == ["sources", "token", "done"]


def test_stream_rejects_empty_message():
    r = client.post("/chat/stream", json={"message": "   "})
    assert r.status_code == 400


# ---------------------------------------------------------------------------
# regression: the streamed trace must be finalized despite ContextVar loss
# ---------------------------------------------------------------------------


class _FakeTrace:
    def __init__(self):
        self.id = "tr-stream-test"
        self.updates: list[dict] = []
        self.spans: list[dict] = []

    def update(self, **kw):
        self.updates.append(kw)

    def span(self, **kw):
        self.spans.append(kw)
        return MagicMock()


def test_stream_trace_is_finalized_with_the_answer(monkeypatch):
    ft = _FakeTrace()
    fake_client = MagicMock()
    fake_client.trace.return_value = ft

    monkeypatch.setattr(tracing, "is_enabled", lambda: True)
    monkeypatch.setattr(tracing, "_client", lambda: fake_client)

    with (
        patch("app.api.router.classify_intent", return_value="QA"),
        patch("app.api.router.retrieve_relevant_docs", return_value=_CITATIONS),
        patch("app.api.router.stream_answer", return_value=iter(["হ্যাঁ", "।"])),
    ):
        r = client.post("/chat/stream", json={"message": "প্রশ্ন?"})

    assert r.status_code == 200
    # finalize_request_trace ran even though the ContextVar was gone by the
    # generator's finally -- because gen() passed `trace=` explicitly.
    assert ft.updates, "trace.update() was never called -> trace left unfinalized"
    out = ft.updates[-1]["output"]
    assert out["answer"] == "হ্যাঁ।"
    assert out["mode"] == "qa"
    assert ft.updates[-1]["metadata"]["response_time_ms"] is not None


def test_stream_qa_generation_receives_the_trace_id(monkeypatch):
    ft = _FakeTrace()
    fake_client = MagicMock()
    fake_client.trace.return_value = ft
    monkeypatch.setattr(tracing, "is_enabled", lambda: True)
    monkeypatch.setattr(tracing, "_client", lambda: fake_client)

    seen = {}

    def _spy_stream_answer(message, citations, *, trace_id=None):
        seen["trace_id"] = trace_id
        return iter(["x"])

    with (
        patch("app.api.router.classify_intent", return_value="QA"),
        patch("app.api.router.retrieve_relevant_docs", return_value=_CITATIONS),
        patch("app.api.router.stream_answer", side_effect=_spy_stream_answer),
    ):
        r = client.post("/chat/stream", json={"message": "প্রশ্ন?"})

    assert r.status_code == 200
    assert seen["trace_id"] == "tr-stream-test"


def test_stream_works_when_tracing_disabled(monkeypatch):
    # Default state: no trace_id must reach stream_answer (it would leak into
    # the plain OpenAI call as an unknown kwarg).
    assert tracing.is_enabled() is False
    seen = {}

    def _spy_stream_answer(message, citations, *, trace_id=None):
        seen["trace_id"] = trace_id
        return iter(["x"])

    with (
        patch("app.api.router.classify_intent", return_value="QA"),
        patch("app.api.router.retrieve_relevant_docs", return_value=_CITATIONS),
        patch("app.api.router.stream_answer", side_effect=_spy_stream_answer),
    ):
        r = client.post("/chat/stream", json={"message": "প্রশ্ন?"})

    assert r.status_code == 200
    assert seen["trace_id"] is None
