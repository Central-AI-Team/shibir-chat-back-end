"""QA orchestration: the ONE implementation behind POST /chat and /chat/stream.

    message --preface--> (greeting, question)
            --condense-> standalone question (+ previous turn's cited pages)
            --retrieve-> citations, each with a rerank score
            --gate-----> grounding = citations that clear the bar, individually
            --generate-> answer (generate_answer) or deltas (stream_answer)

Both routes call _prepare() for everything up to the gate, so they cannot
drift apart (they did once: the router carried its own copy of the gate and
the small-talk split). They differ only in the last step: answer_question()
calls generate_answer(), which runs the stray-script check (CLAUDE.md §5);
qa_stream() calls stream_answer(), which cannot (tokens are already sent).

Gate: decided in code, by score, per citation. Every citation whose
rerank_score >= settings.min_rerank_score is grounding and a source; the
LLM is called either way, with an empty list when nothing clears the bar, and
the prompt then has it say the books do not cover the question. Sources and
answer can therefore never contradict (the old bug: "not in the database"
next to a populated source list). Before 2026-10 only citations[0] was
checked and all TOP_K went to the prompt and `sources`.

Small talk: chitchat_service.preface() splits the message BEFORE retrieval --
pure small talk gets a matching reply with no retrieval and no QA LLM call;
"hi, <question>" gets a short greeting in front and retrieval, gate and
generation run on the question alone.
"""

from __future__ import annotations

import itertools
import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import NamedTuple

from app.core import timing, tracing
from app.core.config import settings
from app.rag.generator import generate_answer, stream_answer
from app.rag.retriever import retrieve_relevant_docs
from app.schemas.query import Citation, QueryResponse
from app.services.chitchat_service import classify_chitchat, preface
from app.services.intent_classifier import condense

logger = logging.getLogger(__name__)


def _is_conversational(query: str) -> bool:
    """True for pure small talk (no question attached).

    Kept importable for existing callers; the matching itself lives in
    chitchat_service.classify_chitchat().
    """
    hit = classify_chitchat(query)
    return hit is not None and not hit[1]


class QAStream(NamedTuple):
    """qa_stream()'s result: the grounding sources (what the `sources` SSE
    event carries), the answer deltas, and the standalone question the turn
    was retrieved with (stored on the user turn by the router)."""

    sources: list[Citation]
    deltas: Iterator[str]
    standalone_query: str


@dataclass
class _Prepared:
    greeting: str | None  # reply to the small-talk part, if any
    question: str  # "" for pure small talk
    standalone: str  # what retrieval ran on
    citations: list[Citation]  # everything the reranker returned
    grounding: list[Citation]  # the citations that cleared the gate

    @property
    def grounded(self) -> bool:
        return bool(self.grounding)


def previous_row_keys(history: list[dict] | None) -> list[str]:
    """The pages the previous assistant turn cited (stored sources'
    row_key), for retriever extra_row_keys. [] when the last turn was not an
    assistant turn, cited nothing, or predates the row_key field."""
    if not history or history[-1].get("role") != "assistant":
        return []
    keys = [s.get("row_key") for s in history[-1].get("sources") or []]
    return list(dict.fromkeys(k for k in keys if k))


def gate(citations: list[Citation]) -> list[Citation]:
    """The citations that individually clear settings.min_rerank_score, in
    rerank order. Grounded iff non-empty."""
    return [
        c for c in citations
        if c.rerank_score is not None and c.rerank_score >= settings.min_rerank_score
    ]


def _prepare(message: str, history: list[dict] | None) -> _Prepared:
    greeting, question = preface(message)
    if not question:  # pure small talk: `greeting` is the whole reply
        return _Prepared(greeting, "", message, [], [])

    # The standalone question from the intent call (primed by message), else
    # the question itself; see intent_classifier.condense().
    standalone = condense(message, history, default=question)
    prev_keys = previous_row_keys(history)
    # The `retrieve-context` trace span (full candidate pool + rerank scores)
    # is emitted inside retrieve_stages(); nothing to record here.
    # The keyword is passed only when there is something to carry over, so a
    # first turn calls the retriever exactly as before.
    citations = retrieve_relevant_docs(
        standalone, **({"extra_row_keys": prev_keys} if prev_keys else {})
    )
    grounding = gate(citations)
    best = max((c.rerank_score or 0.0 for c in citations), default=None)
    tracing.record_gate(
        grounded=bool(grounding), top_score=best, threshold=settings.min_rerank_score
    )
    if grounding:
        logger.info(
            "grounded query=%r n_sources=%d/%d top_rerank=%.3f books=%s",
            message, len(grounding), len(citations), best, [c.book for c in grounding],
        )
    else:
        logger.info("no_match query=%r best_rerank=%s", message, best)
    return _Prepared(greeting, question, standalone, citations, grounding)


def _prompt_kwargs(
    prep: _Prepared, message: str, history: list[dict] | None, style_rules: str | None
) -> dict:
    """history / original_message / style_rules for the generator. Each is
    passed only when it carries something, so a first turn with no greeting
    and no condensation calls the generator exactly as before. The generator
    gets the condensed standalone as the question and the user's actual
    message (with the small-talk prefix removed) as original_message."""
    kwargs: dict = {}
    recent = [t for t in (history or []) if t.get("role") in ("user", "assistant")]
    if recent:
        kwargs["history"] = recent[-settings.qa_history_turns:]
    if prep.standalone != prep.question:
        kwargs["original_message"] = prep.question
    if style_rules:
        kwargs["style_rules"] = style_rules
    return kwargs


def answer_question(
    message: str, history: list[dict] | None = None, *, style_rules: str | None = None
) -> QueryResponse:
    """Non-streaming QA. `history` is the session's recent turns
    ([{role, content, sources?}]); `style_rules` is reserved for a later
    caller and is appended to the system prompt when given. The returned
    QueryResponse.query is the standalone question retrieval ran on."""
    start = time.perf_counter()
    prep = _prepare(message, history)
    if not prep.question:
        response_time_ms = round((time.perf_counter() - start) * 1000, 2)
        logger.info("conversational_shortcut query=%r in %.2fms", message, response_time_ms)
        return QueryResponse(
            query=message, answer=prep.greeting or "", sources=[], response_time_ms=response_time_ms
        )

    # The model answers the standalone (condensed) question; the user's own
    # wording travels as original_message when the two differ.
    answer = generate_answer(
        prep.standalone, prep.grounding, **_prompt_kwargs(prep, message, history, style_rules)
    )
    if prep.greeting:
        answer = f"{prep.greeting}\n\n{answer}"

    response_time_ms = round((time.perf_counter() - start) * 1000, 2)
    return QueryResponse(
        query=prep.standalone, answer=answer, sources=prep.grounding, response_time_ms=response_time_ms
    )


def qa_stream(
    message: str,
    history: list[dict] | None = None,
    *,
    style_rules: str | None = None,
    trace_id: str | None = None,
    parent_observation_id: str | None = None,
) -> QAStream:
    """Streaming QA: the same preparation as answer_question(), then
    stream_answer() deltas. trace_id/parent_observation_id are threaded
    through to stream_answer() so the streamed generation nests under the
    request's root span: the streaming path cannot rely on ambient trace
    context (see app/core/tracing.py's module docstring)."""
    rec = timing.current()
    prep = _prepare(message, history)
    if not prep.question:
        return QAStream([], iter([prep.greeting or ""]), message)

    deltas = stream_answer(
        prep.standalone, prep.grounding,
        trace_id=trace_id, parent_observation_id=parent_observation_id,
        **({"timing_rec": rec} if rec else {}),
        **_prompt_kwargs(prep, message, history, style_rules),
    )
    if prep.greeting:
        # Same text as the non-streaming "<greeting>\n\n<answer>".
        deltas = itertools.chain([f"{prep.greeting}\n\n"], deltas)
    return QAStream(prep.grounding, deltas, prep.standalone)
