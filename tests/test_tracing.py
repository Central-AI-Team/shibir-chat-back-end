"""Guardrail tests for app/core/tracing.py -- the Langfuse layer.

The non-negotiable for this feature is that observability can never harm the
request path. Two properties are pinned here:

  1. DISABLED = NO-OP. With no Langfuse keys configured (the default, and the
     state in CI / this repo's .env), tracing is completely inert: the langfuse
     SDK is never imported, get_client() hands back the plain OpenAI client,
     every tracing helper is a silent no-op, and complete() adds no extra
     kwargs to the OpenAI call -- i.e. /chat behaves byte-identically to a
     build without this module.

  2. A BROKEN BACKEND CANNOT BREAK /chat. With tracing switched on but the
     Langfuse client raising on every call (slow / down / misconfigured host),
     POST /chat still returns a normal 200 answer -- the tracing exception is
     swallowed, not propagated.

These mock at the boundaries only; no embedding model, reranker, or real LLM
call runs.
"""

from __future__ import annotations

import os
import subprocess
import sys
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


# ---------------------------------------------------------------------------
# 1. disabled == no-op
# ---------------------------------------------------------------------------


def test_tracing_is_disabled_without_keys():
    # The repo .env carries no LANGFUSE_* keys, so this is the real default.
    assert tracing.is_enabled() is False


def test_all_helpers_are_silent_noops_when_disabled():
    assert tracing.is_enabled() is False  # precondition

    # None of these may raise, and none may return a trace object.
    assert tracing.start_request_trace(
        name="chat", query="প্রশ্ন?", session_id="s1", user_id=None
    ) is None
    assert tracing.current_trace() is None
    assert tracing.current_trace_id() is None
    assert tracing.record_intent("QA", method="regex") is None
    assert tracing.record_rewrite(["a", "b"]) is None
    assert tracing.record_retrieval(_CITATIONS, top_k=5) is None
    assert tracing.record_gate(grounded=True, top_score=0.9, threshold=0.5) is None
    assert tracing.finalize_request_trace(
        answer="উত্তর।", sources=_CITATIONS, mode="qa", response_time_ms=1.0
    ) is None
    assert tracing.score_trace("trace-123", "helpful", 1.0) is None

    # Lifecycle hooks are no-ops too.
    tracing.init()
    tracing.shutdown()
    tracing.clear_request_trace()


def test_get_client_is_the_plain_openai_client_when_disabled():
    from openai import OpenAI

    from app.core.llm import _openai_class

    assert tracing.is_enabled() is False
    assert _openai_class() is OpenAI


def test_complete_adds_no_langfuse_kwargs_to_the_openai_call_when_disabled():
    """complete() must issue the exact same chat.completions.create() call it
    would in a build without tracing -- no name=, metadata=, trace_id=."""
    from app.core import llm

    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = MagicMock()

    with patch.object(llm, "_client_for_provider", return_value=fake_client):
        llm.complete("qa", [{"role": "user", "content": "hi"}])

    _, kwargs = fake_client.chat.completions.create.call_args
    assert set(kwargs) == {"model", "messages"}, (
        f"tracing-only kwargs leaked into the OpenAI call while disabled: {kwargs}"
    )


def test_complete_strips_langfuse_kwargs_when_wrapper_patch_is_absent(monkeypatch):
    """Enabled but langfuse.openai's patch never installed (broken import):
    complete() must NOT pass name/metadata/trace_id to the plain client --
    that would 400 every call. Even an explicitly-forwarded trace_id is
    dropped."""
    from app.core import llm

    monkeypatch.setattr(tracing, "is_enabled", lambda: True)
    monkeypatch.setattr(tracing, "_openai_wrapper_ready", False)
    assert tracing.openai_wrapper_active() is False

    fake_client = MagicMock()
    with patch.object(llm, "_client_for_provider", return_value=fake_client):
        llm.complete("qa", [{"role": "user", "content": "hi"}], trace_id="tr-x")

    _, kwargs = fake_client.chat.completions.create.call_args
    assert set(kwargs) == {"model", "messages"}


def test_complete_passes_langfuse_kwargs_when_wrapper_patch_is_active(monkeypatch):
    from app.core import llm

    monkeypatch.setattr(tracing, "is_enabled", lambda: True)
    monkeypatch.setattr(tracing, "_openai_wrapper_ready", True)
    assert tracing.openai_wrapper_active() is True

    fake_client = MagicMock()
    with patch.object(llm, "_client_for_provider", return_value=fake_client):
        llm.complete("qa", [{"role": "user", "content": "hi"}], trace_id="tr-x")

    _, kwargs = fake_client.chat.completions.create.call_args
    assert kwargs["name"] == "llm:qa"
    assert kwargs["metadata"]["task"] == "qa"
    assert kwargs["trace_id"] == "tr-x"


def test_langfuse_sdk_is_not_imported_when_disabled():
    """Strongest form of the no-op guarantee: importing the whole app with no
    Langfuse keys must not pull the langfuse package into sys.modules."""
    code = (
        "import sys\n"
        "import app.main\n"
        "leaked = sorted(m for m in sys.modules if m == 'langfuse' or m.startswith('langfuse.'))\n"
        "assert not leaked, leaked\n"
        "from app.core.llm import _openai_class\n"
        "import openai\n"
        "assert _openai_class() is openai.OpenAI\n"
        "print('OK')\n"
    )
    env = {**os.environ, "LANGFUSE_PUBLIC_KEY": "", "LANGFUSE_SECRET_KEY": ""}
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


# ---------------------------------------------------------------------------
# 2. a broken tracing backend cannot break /chat
# ---------------------------------------------------------------------------


def test_chat_still_answers_when_the_tracing_backend_errors_on_every_call(monkeypatch):
    """Tracing switched on, but the Langfuse client raises on everything it is
    asked to do (unreachable host / outage). /chat must still return 200."""
    exploding = MagicMock()
    exploding.trace.side_effect = ConnectionError("langfuse unreachable")
    exploding.score.side_effect = ConnectionError("langfuse unreachable")
    exploding.flush.side_effect = ConnectionError("langfuse unreachable")
    exploding.shutdown.side_effect = ConnectionError("langfuse unreachable")

    monkeypatch.setattr(tracing, "is_enabled", lambda: True)
    monkeypatch.setattr(tracing, "_client", lambda: exploding)

    with (
        patch("app.api.router.classify_intent", return_value="QA"),
        patch("app.services.qa_service.retrieve_relevant_docs", return_value=_CITATIONS),
        patch("app.services.qa_service.generate_answer", return_value="উত্তর।"),
    ):
        response = client.post("/chat", json={"message": "প্রশ্ন?"})

    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "qa"
    assert body["answer"] == "উত্তর।"
    assert len(body["sources"]) == 1

    # start_request_trace() swallowed the error, so no trace was ever stashed.
    assert tracing.current_trace() is None


def test_pipeline_span_helpers_swallow_backend_errors(monkeypatch):
    """The per-stage helpers run with a live (but broken) trace object; a
    raising span/update call must be swallowed, not propagated."""
    broken_trace = MagicMock()
    broken_trace.span.side_effect = RuntimeError("langfuse span failed")
    broken_trace.update.side_effect = RuntimeError("langfuse update failed")

    monkeypatch.setattr(tracing, "is_enabled", lambda: True)
    token = tracing._current_trace.set(broken_trace)
    try:
        # None of these may raise.
        tracing.record_intent("QA", method="llm")
        tracing.record_rewrite(["ক", "খ"])
        tracing.record_retrieval(_CITATIONS, top_k=5)
        tracing.record_gate(grounded=False, top_score=0.42, threshold=0.5)
        tracing.finalize_request_trace(
            answer="উত্তর।", sources=_CITATIONS, mode="qa", response_time_ms=1.0
        )
    finally:
        tracing._current_trace.reset(token)


# ---------------------------------------------------------------------------
# retrieve span payload shape (gap #1: dropped candidates must be visible)
# ---------------------------------------------------------------------------


def test_retrieve_span_lists_dropped_candidates_with_their_rerank_scores(monkeypatch):
    captured = {}

    class _Span:
        def end(self):
            pass

    class _Trace:
        def span(self, *, name, output):
            captured["name"] = name
            captured["output"] = output
            return _Span()

    monkeypatch.setattr(tracing, "is_enabled", lambda: True)
    token = tracing._current_trace.set(_Trace())
    try:
        pool = [
            Citation(book="B", chapter="keep-1", source_db="d", content="x",
                     similarity=0.71, rerank_score=0.95),
            Citation(book="B", chapter="keep-2", source_db="d", content="y",
                     similarity=0.68, rerank_score=0.80),
            Citation(book="B", chapter="dropped", source_db="d", content="z",
                     similarity=0.61, rerank_score=0.42),
        ]
        tracing.record_retrieval(pool, top_k=2)
    finally:
        tracing._current_trace.reset(token)

    assert captured["name"] == "retrieve"
    cands = captured["output"]["candidates"]
    assert captured["output"]["n"] == 3
    assert captured["output"]["top_k"] == 2
    assert [c["kept"] for c in cands] == [True, True, False]
    dropped = cands[-1]
    assert dropped["chapter"] == "dropped"
    assert dropped["similarity"] == 0.61
    assert dropped["rerank_score"] == 0.42  # the row a miss investigation needs
