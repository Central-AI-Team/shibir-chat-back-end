"""generator.citation_labels_valid: every [n] within 1..n_sources."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from app.core import tracing
from app.rag import generator
from app.schemas.query import Citation

valid = generator.citation_labels_valid


@pytest.mark.parametrize("answer, n, expected", [
    ("যাকাত ফরজ [১]। নিসাব [২][৩]।", 3, True),          # Bengali digits
    ("Zakat is obligatory [1], nisab [2].", 2, True),   # ASCII digits
    ("মিশ্র [১] এবং [2]।", 2, True),
    ("উত্তর [৪]।", 3, False),                            # out of range
    ("উত্তর [0]।", 3, False),
    ("উত্তর [১০]।", 9, False),
    ("উত্তর [ ১ ]।", 1, True),                           # spaces inside the brackets
    ("কোনো উৎস নেই।", 0, True),                          # no sources, no labels
    ("তবুও [১] উদ্ধৃত।", 0, False),                       # no sources but a label
    ("", 0, True),
    ("[১][১][১]", 1, True),
])
def test_citation_labels_valid(answer, n, expected):
    assert valid(answer, n) is expected


def _cit(i):
    return Citation(book=f"b{i}", chapter="c", source_db="tarun", content="x", rerank_score=0.9)


def _resp(text):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


def test_production_path_logs_a_warning_and_leaves_the_answer_unchanged(caplog):
    with patch("app.rag.generator.complete", return_value=_resp("উত্তর [৩]।")), caplog.at_level(logging.WARNING):
        out = generator.generate_answer("প্রশ্ন", [_cit(1), _cit(2)])
    assert out == "উত্তর [৩]।"
    assert any("outside 1..2" in r.message for r in caplog.records)
    caplog.clear()
    with patch("app.rag.generator.complete", return_value=_resp("উত্তর [২]।")), caplog.at_level(logging.WARNING):
        generator.generate_answer("প্রশ্ন", [_cit(1), _cit(2)])
    assert not any("outside" in r.message for r in caplog.records)


def test_eval_path_with_explicit_model_is_not_checked():
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: _resp("[৯]"))))
    with patch("app.rag.generator.tracing.record_citation_validity") as rec:
        assert generator.generate_answer("প্রশ্ন", [], model="x", client=client) == "[৯]"
    rec.assert_not_called()


def test_citation_valid_reaches_the_root_span_metadata(monkeypatch):
    updates = []
    root = SimpleNamespace(update=lambda **kw: updates.append(kw))
    monkeypatch.setattr(tracing, "_current_ctx", tracing.contextvars.ContextVar("t", default=None))
    tracing._current_ctx.set({"root": root})
    tracing.record_citation_validity(False)
    tracing.finalize_request_trace(answer="a", sources=[], mode="qa", response_time_ms=1.0)
    assert updates[-1]["metadata"]["citation_valid"] is False


def test_no_trace_is_a_no_op():
    tracing.record_citation_validity(True)  # no active request: nothing to attach to, no error
