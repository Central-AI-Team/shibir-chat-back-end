"""Tests for POST /chat/stream -- the Server-Sent Events variant of /chat.

Covers the SSE event sequence per mode, plus a regression test for the
ambient-context-loss bug: Starlette pulls the response generator one next()
at a time in a threadpool, each call getting its own copied context, so the
request's root span context set in the first iteration is invisible by the
time the generator's finally runs -- finalize_request_trace() has to be
handed the trace context explicitly, and the streamed QA generation needs an
explicit trace_id/parent_observation_id, or every streamed trace ends up
unfinalized / with a disconnected generation. See app/core/tracing.py's
module docstring for the full explanation.

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


class _FakeRootObservation:
    """Mimics the LangfuseSpan object start_as_current_observation() yields."""

    def __init__(self):
        self.trace_id = "tr-stream-test"
        self.id = "obs-stream-root"
        self.updates: list[dict] = []

    def update(self, **kw):
        self.updates.append(kw)


class _FakeObservationCM:
    """Mimics the context manager start_as_current_observation()/
    propagate_attributes() return -- tracing.py drives these manually via
    __enter__/__exit__ instead of a `with` block, since the streaming path
    opens the root span in one request-handler call and closes it in
    another (see app/core/tracing.py's module docstring)."""

    def __init__(self, value):
        self._value = value

    def __enter__(self):
        return self._value

    def __exit__(self, *exc_info):
        return False


class _FakeClient:
    def __init__(self):
        self.root = _FakeRootObservation()

    def start_as_current_observation(self, **kw):
        return _FakeObservationCM(self.root)

    def start_observation(self, **kw):
        return MagicMock()


def test_stream_trace_is_finalized_with_the_answer(monkeypatch):
    fake_client = _FakeClient()

    monkeypatch.setattr(tracing, "is_enabled", lambda: True)
    monkeypatch.setattr(tracing, "_client", lambda: fake_client)

    with (
        patch("app.api.router.classify_intent", return_value="QA"),
        patch("app.api.router.retrieve_relevant_docs", return_value=_CITATIONS),
        patch("app.api.router.stream_answer", return_value=iter(["হ্যাঁ", "।"])),
    ):
        r = client.post("/chat/stream", json={"message": "প্রশ্ন?"})

    assert r.status_code == 200
    # finalize_request_trace ran even though ambient context was gone by the
    # generator's finally -- because gen() passed `trace=` explicitly.
    assert fake_client.root.updates, "root.update() was never called -> trace left unfinalized"
    out = fake_client.root.updates[-1]["output"]
    assert out["answer"] == "হ্যাঁ।"
    assert out["mode"] == "qa"
    assert fake_client.root.updates[-1]["metadata"]["response_time_ms"] is not None


def test_stream_qa_generation_receives_the_trace_id(monkeypatch):
    fake_client = _FakeClient()
    monkeypatch.setattr(tracing, "is_enabled", lambda: True)
    monkeypatch.setattr(tracing, "_client", lambda: fake_client)

    seen = {}

    def _spy_stream_answer(message, citations, *, trace_id=None, parent_observation_id=None):
        seen["trace_id"] = trace_id
        seen["parent_observation_id"] = parent_observation_id
        return iter(["x"])

    with (
        patch("app.api.router.classify_intent", return_value="QA"),
        patch("app.api.router.retrieve_relevant_docs", return_value=_CITATIONS),
        patch("app.api.router.stream_answer", side_effect=_spy_stream_answer),
    ):
        r = client.post("/chat/stream", json={"message": "প্রশ্ন?"})

    assert r.status_code == 200
    assert seen["trace_id"] == "tr-stream-test"
    assert seen["parent_observation_id"] == "obs-stream-root"


def test_stream_works_when_tracing_disabled(monkeypatch):
    # Default state: no trace_id/parent_observation_id must reach
    # stream_answer (they would leak into the plain OpenAI call as unknown
    # kwargs). Force-disabled explicitly rather than relying on the repo's
    # .env having no LANGFUSE_* keys (a dev box may have real ones set).
    monkeypatch.setattr(tracing.settings, "langfuse_public_key", "")
    monkeypatch.setattr(tracing.settings, "langfuse_secret_key", "")
    assert tracing.is_enabled() is False
    seen = {}

    def _spy_stream_answer(message, citations, *, trace_id=None, parent_observation_id=None):
        seen["trace_id"] = trace_id
        seen["parent_observation_id"] = parent_observation_id
        return iter(["x"])

    with (
        patch("app.api.router.classify_intent", return_value="QA"),
        patch("app.api.router.retrieve_relevant_docs", return_value=_CITATIONS),
        patch("app.api.router.stream_answer", side_effect=_spy_stream_answer),
    ):
        r = client.post("/chat/stream", json={"message": "প্রশ্ন?"})

    assert r.status_code == 200
    assert seen["trace_id"] is None
    assert seen["parent_observation_id"] is None
