"""Orchestration.

Fixes the exact bug you described -- "direct bole je database e nai, but niche
source diye dey". The old code was:

    citations = retrieve_relevant_docs(query)
    answer = generate_answer(query, citations)
    return QueryResponse(query=query, answer=answer, sources=citations)

Sources were attached unconditionally, with no idea whether the answer used
them. Now retrieval either clears the relevance bar (answer + its sources) or
it does not (refusal + empty sources). The two can no longer contradict.
"""

from __future__ import annotations

import logging

from app.core.config import settings
from app.rag.generator import generate_answer
from app.rag.retriever import retrieve_relevant_docs
from app.schemas.query import QueryResponse

logger = logging.getLogger(__name__)

_NO_MATCH = (
    "এই প্রশ্নের উত্তর সংশ্লিষ্ট বইগুলোতে খুঁজে পাওয়া যায়নি। "
    "প্রশ্নটি অন্যভাবে বা আরেকটু নির্দিষ্ট করে জিজ্ঞাসা করে দেখতে পারেন।"
)


def answer_question(query: str) -> QueryResponse:
    citations = retrieve_relevant_docs(query)

    # The reranker score is the honest relevance signal. If even the best
    # candidate is below the bar, the corpus does not cover this question --
    # so do not send Gemini a pile of noise and do not show sources.
    if not citations or citations[0].rerank_score < settings.min_rerank_score:
        best = citations[0].rerank_score if citations else None
        logger.info("no_match query=%r best_rerank=%s", query, best)
        return QueryResponse(query=query, answer=_NO_MATCH, sources=[])

    answer = generate_answer(query, citations)

    logger.info(
        "answered query=%r n_sources=%d top_rerank=%.3f books=%s",
        query, len(citations), citations[0].rerank_score,
        [c.book for c in citations],
    )
    return QueryResponse(query=query, answer=answer, sources=citations)