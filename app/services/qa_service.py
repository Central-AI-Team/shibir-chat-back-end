"""Orchestration.

Fixes the exact bug you described -- "direct bole je database e nai, but niche
source diye dey". The old code was:

    citations = retrieve_relevant_docs(query)
    answer = generate_answer(query, citations)
    return QueryResponse(query=query, answer=answer, sources=citations)

Sources were attached unconditionally, with no idea whether the answer used
them. Now retrieval either clears the relevance bar (answer + its sources) or
it does not (answer, grounded in nothing, + empty sources). The two can no
longer contradict.

CHANGED: below-the-bar queries used to short-circuit to a canned refusal
without ever calling the LLM. They now still call generate_answer(), just
with an empty citation list -- generator.py's prompt handles that case by
plainly saying the books don't cover it, instead of refusing outright.
Sources stay empty either way: nothing here grounded the answer, so nothing
is cited.

SMALL TALK: greetings / thanks / farewells used to reach the rerank gate
above, fail it (a "hi" has no book to rank against), and get a stiff "not
found in the books" refusal; they were then given a random canned reply that
ignored what was said (a salam could get a goodbye). chitchat_service.preface()
now splits the message *before* retrieval, using the same code as the
streaming path in app/api/router.py:
  - pure small talk gets a reply that matches it, with no retrieval and no
    book-QA LLM call;
  - "hi, <question>" gets a short greeting in front, and retrieval, the gate
    and generation run on the question alone. Sources come only from that
    RAG part.
"""

from __future__ import annotations

import logging
import time

from app.core import tracing
from app.core.config import settings
from app.rag.generator import generate_answer
from app.rag.retriever import retrieve_relevant_docs
from app.schemas.query import QueryResponse
from app.services.chitchat_service import classify_chitchat, preface

logger = logging.getLogger(__name__)

def _is_conversational(query: str) -> bool:
    """True for pure small talk (no question attached).

    Kept importable for existing callers; the matching itself lives in
    chitchat_service.classify_chitchat().
    """
    hit = classify_chitchat(query)
    return hit is not None and not hit[1]


def answer_question(query: str) -> QueryResponse:
    start = time.perf_counter()

    greeting, question = preface(query)
    if not question:  # pure small talk: `greeting` is the whole reply
        response_time_ms = round((time.perf_counter() - start) * 1000, 2)
        logger.info("conversational_shortcut query=%r in %.2fms", query, response_time_ms)
        return QueryResponse(
            query=query, answer=greeting, sources=[], response_time_ms=response_time_ms
        )

    # The `retrieve` trace span (full candidate pool + rerank scores) is
    # emitted inside retrieve_stages(); nothing to record here.
    citations = retrieve_relevant_docs(question)

    # The reranker score is the honest relevance signal. If even the best
    # candidate is below the bar, the corpus does not cover this question --
    # so do not hand Gemini a pile of noise as "sources". It still generates
    # an answer (generator.py's prompt has it say plainly that the books
    # don't cover this), just with no citations to attach.
    relevant = bool(citations) and citations[0].rerank_score >= settings.min_rerank_score
    tracing.record_gate(
        grounded=relevant,
        top_score=citations[0].rerank_score if citations else None,
        threshold=settings.min_rerank_score,
    )
    grounding = citations if relevant else []

    answer = generate_answer(question, grounding)
    if greeting:
        answer = f"{greeting}\n\n{answer}"

    response_time_ms = round((time.perf_counter() - start) * 1000, 2)
    if relevant:
        logger.info(
            "answered query=%r n_sources=%d top_rerank=%.3f books=%s in %.2fms",
            query, len(citations), citations[0].rerank_score,
            [c.book for c in citations], response_time_ms,
        )
    else:
        best = citations[0].rerank_score if citations else None
        logger.info(
            "no_match query=%r best_rerank=%s in %.2fms", query, best, response_time_ms
        )
    return QueryResponse(
        query=query, answer=answer, sources=grounding, response_time_ms=response_time_ms
    )