"""Latency work: reasoning effort / token caps, rewrite skip, merged
intent+rewrite call, greeting shortcut, embedding LRU, note concurrency.

All model/network boundaries are mocked."""
from __future__ import annotations

import contextvars
import json
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from app.core import llm
from app.rag import embedder, query_rewriter as qr
from app.services import intent_classifier as ic


def _resp(text: str):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


# ── llm.complete: reasoning effort + token caps ──────────────────────────────

def _create_kwargs(task, monkeypatch, model="gpt-5-mini", **kw):
    monkeypatch.setattr(llm.settings, "openai_model", model)
    client = MagicMock()
    with patch.object(llm, "_client_for_provider", return_value=client):
        llm.complete(task, [{"role": "user", "content": "x"}], **kw)
    return client.chat.completions.create.call_args.kwargs


def test_gpt5_gets_minimal_reasoning_and_task_cap(monkeypatch):
    kwargs = _create_kwargs("qa", monkeypatch)
    assert kwargs["reasoning_effort"] == "minimal"
    assert kwargs["max_completion_tokens"] == llm.settings.max_tokens_qa


def test_explicit_token_budget_beats_the_cap(monkeypatch):
    assert _create_kwargs("qa", monkeypatch, token_budget=77)["max_completion_tokens"] == 77


def test_reasoning_effort_is_not_sent_to_non_gpt5_models(monkeypatch):
    kwargs = _create_kwargs("qa", monkeypatch, model="openai/gpt-oss-20b")
    assert "reasoning_effort" not in kwargs


def test_empty_reasoning_effort_sends_nothing(monkeypatch):
    monkeypatch.setattr(llm.settings, "openai_reasoning_effort", "")
    assert "reasoning_effort" not in _create_kwargs("qa", monkeypatch)


# ── rewrite skip ─────────────────────────────────────────────────────────────

def test_bengali_query_skips_the_rewrite_llm():
    qr.expand_query.cache_clear()
    with patch.object(qr, "complete") as complete:
        assert qr.expand_query("নামাজের গুরুত্ব কী?") == ("নামাজের গুরুত্ব কী?",)
    complete.assert_not_called()


def test_banglish_query_still_calls_the_llm():
    qr.expand_query.cache_clear()
    with patch.object(qr, "complete", return_value=_resp("নামাজের রাকাত কত")) as complete:
        assert qr.expand_query("namaz koto rakat") == ("নামাজের রাকাত কত",)
    complete.assert_called_once()


def test_primed_rewrite_is_used_without_an_llm_call():
    qr.expand_query.cache_clear()
    qr.prime_rewrite("zakat koto taka", "যাকাত কত টাকা")
    with patch.object(qr, "complete") as complete:
        assert qr.expand_query("zakat koto taka") == ("যাকাত কত টাকা",)
    complete.assert_not_called()


# ── intent: regex, greetings, merged call ────────────────────────────────────

@pytest.mark.parametrize("text", [
    "assalamualikum", "Assalamu alaikum", "আসসালামু আলাইকুম", "salam", "hi", "hello", "slm",
])
def test_greetings_never_reach_the_llm(text):
    with patch.object(ic, "complete") as complete:
        assert ic.classify_intent(text, False) == "QA"
    complete.assert_not_called()


@pytest.mark.parametrize("text,expected", [
    ("যাকাতের নোট দাও", "NOTE"),
    ("write notes on zakat", "NOTE"),
    ("এই অধ্যায় সংক্ষেপে লিখে দাও", "NOTE"),
    ("একজন আলেমের ভূমিকায় থেকে কথা বলো", "ROLEPLAY"),
    ("ঈমান বাড়াতে কোন বই পড়া উচিত?", "SUGGESTION"),
])
def test_clear_patterns_route_by_regex(text, expected):
    with patch.object(ic, "complete") as complete:
        assert ic.classify_intent(text, False) == expected
    complete.assert_not_called()


def test_plain_bengali_question_is_not_hijacked_by_regex():
    # Falls to the LLM (Bengali -> classification only, no merged call).
    with patch.object(ic, "complete", return_value=_resp("QA")) as complete:
        assert ic.classify_intent("যাকাতের নিসাব কত?", False) == "QA"
    assert complete.call_count == 1


def test_banglish_uses_one_merged_call_and_primes_the_rewrite():
    qr.expand_query.cache_clear()
    reply = json.dumps({"intent": "QA", "rewritten_query": "নামাজ কত রাকাত"}, ensure_ascii=False)
    with patch.object(ic, "complete", return_value=_resp(reply)) as complete:
        assert ic.classify_intent("namaz koto rakat", False) == "QA"
    assert complete.call_count == 1
    with patch.object(qr, "complete") as rewrite:
        assert qr.expand_query("namaz koto rakat") == ("নামাজ কত রাকাত",)
    rewrite.assert_not_called()


def test_merged_call_with_invalid_json_falls_back_to_two_calls():
    qr.expand_query.cache_clear()
    with patch.object(ic, "complete", side_effect=[_resp("{bad"), _resp("QA")]) as complete:
        assert ic.classify_intent("rojar niyom ki ki", False) == "QA"
    assert complete.call_count == 2


def test_merged_call_rejects_an_unknown_intent():
    reply = json.dumps({"intent": "WEATHER", "rewritten_query": "x"})
    with patch.object(ic, "complete", side_effect=[_resp(reply), _resp("QA")]) as complete:
        assert ic.classify_intent("rojar niyom ki ki hoy", False) == "QA"
    assert complete.call_count == 2


# ── embedding LRU ────────────────────────────────────────────────────────────

def test_query_embeddings_are_cached(monkeypatch):
    monkeypatch.setattr(embedder.settings, "embed_cache_size", 2)
    embedder._query_cache.clear()
    calls = []

    def fake_embed(texts, *a, **k):
        calls.append(list(texts))
        return [[float(len(t))] for t in texts]

    monkeypatch.setattr(embedder, "embed_texts", fake_embed)
    assert embedder.embed_queries(["a", "bb"]) == [[1.0], [2.0]]
    assert embedder.embed_queries(["bb", "a"]) == [[2.0], [1.0]]
    assert calls == [["a", "bb"]]            # second call fully served from cache
    embedder.embed_queries(["ccc"])           # evicts the least recently used
    assert len(embedder._query_cache) == 2
    embedder._query_cache.clear()


def test_embed_cache_can_be_disabled(monkeypatch):
    monkeypatch.setattr(embedder.settings, "embed_cache_size", 0)
    with patch.object(embedder, "embed_texts", return_value=[[1.0]]) as e:
        embedder.embed_queries(["a"])
        embedder.embed_queries(["a"])
    assert e.call_count == 2


# ── prompt context limits ────────────────────────────────────────────────────

def test_context_is_limited_by_count_and_length(monkeypatch):
    from app.rag.generator import format_context
    from app.schemas.query import Citation

    monkeypatch.setattr("app.rag.generator.settings.context_top_k", 2)
    monkeypatch.setattr("app.rag.generator.settings.context_max_chars", 5)
    cites = [Citation(book="b", chapter="c", source_db="x", content="0123456789",
                      similarity=0.9, rerank_score=0.9) for _ in range(4)]
    out = format_context(cites)
    assert "[১]" in out and "[২]" in out and "[৩]" not in out
    assert "01234" in out and "012345" not in out


# ── note map concurrency ─────────────────────────────────────────────────────

def test_note_map_calls_run_concurrently_and_keep_order(monkeypatch):
    from app.services import note_service as ns

    monkeypatch.setattr(ns.settings, "note_map_concurrency", 4)
    active, peak, lock = 0, 0, threading.Lock()

    marker = contextvars.ContextVar("marker", default="unset")
    marker.set("request")
    seen = []

    def fake_llm(prompt, max_tokens=2000):
        nonlocal active, peak
        seen.append(marker.get())
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.05)
        with lock:
            active -= 1
        return prompt[-3:] if "অংশ:" in prompt else "NOTE"

    chapter = SimpleNamespace(id=1, name="ch", book=SimpleNamespace(name="bk"))
    pages = [SimpleNamespace(content="x" * 9000) for _ in range(4)]
    session = MagicMock()
    q = session.query.return_value
    q.options.return_value.filter.return_value.one_or_none.return_value = chapter
    q.filter.return_value.filter.return_value.order_by.return_value.all.return_value = pages
    monkeypatch.setattr(ns, "SessionLocal", lambda: session)
    monkeypatch.setattr(ns, "_llm", fake_llm)
    monkeypatch.setattr(ns, "_group", lambda texts, budget=12000: ["g1xxx", "g2xxx", "g3xxx", "g4xxx"])

    result = ns.generate_chapter_note(1)
    assert result["note"] == "NOTE"
    assert peak > 1, "map calls were not concurrent"
    assert set(seen) == {"request"}, "request context did not reach the worker threads"


def test_intent_and_rewrite_use_their_own_reasoning_effort(monkeypatch):
    assert _create_kwargs("intent", monkeypatch)["reasoning_effort"] == "low"
    assert _create_kwargs("rewrite", monkeypatch)["reasoning_effort"] == "low"
    assert _create_kwargs("qa", monkeypatch)["reasoning_effort"] == "minimal"
    monkeypatch.setattr(llm.settings, "openai_reasoning_effort_by_task", {})
    assert _create_kwargs("intent", monkeypatch)["reasoning_effort"] == "minimal"
