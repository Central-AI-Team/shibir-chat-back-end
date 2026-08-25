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

CONVERSATIONAL SHORTCUT: greetings / thanks / farewells used to reach the
rerank gate above, fail it (a "hi" has no book to rank against), and get
whatever generator.py did with an empty citation list -- previously a stiff
"not found in the books" refusal. _is_conversational() catches this class of
query with a cheap, deterministic keyword check *before* retrieval, so
small talk gets a short friendly reply without ever calling the retriever or
the LLM.
"""

from __future__ import annotations

import logging
import random
import re
import time

from app.core.config import settings
from app.rag.generator import generate_answer
from app.rag.retriever import retrieve_relevant_docs
from app.schemas.query import QueryResponse

logger = logging.getLogger(__name__)

# Cheap, deterministic small-talk detection -- no LLM/retriever call. Each
# pattern is anchored at the start of the (stripped, lowercased) query, since
# these phrases are how conversational turns actually open. Bengali and the
# common Banglish (Bengali typed in Latin letters) spellings are both covered.
_GREETING_RE = re.compile(
    r"^(?:আসসালামু\s*আলাইকুম|ওয়ালাইকুম\s*আসসালাম|সালাম|হ্যালো|হাই+|হেই|হ্যাই|নমস্কার|"
    r"assalamu\s*alaikum|salam|hello+|hi+|hey+|namaskar)\b"
)
_WELLBEING_RE = re.compile(
    r"^(?:কেমন\s+আছ(?:েন|ো|িস)|কী\s+খবর|কি\s+খবর|"
    r"kemon\s+ach(?:en|o|is)|ki\s*khobor)\b"
)
_THANKS_RE = re.compile(
    r"^(?:ধন্যবাদ|থ্যাংক(?:স|িউ)?|থ্যাঙ্ক(?:স|\s*ইউ)?|শুকরিয়া|"
    r"dhonnobad|dhonyobad|dhannobad|thanks?|thank\s*you|shukriya)\b"
)
_FAREWELL_RE = re.compile(
    r"^(?:বিদায়|আল্লাহ্?\s*হাফেজ|খোদা\s*হাফেজ|ভালো\s+থাকবেন|টাটা|বাই|"
    r"bidae|allah\s*hafez|khoda\s*hafez|tata|bye)\b"
)
_CONVERSATIONAL_PATTERNS = (_GREETING_RE, _WELLBEING_RE, _THANKS_RE, _FAREWELL_RE)

_CONVERSATIONAL_REPLIES = (
    "আসসালামু আলাইকুম! আমি ভালো আছি, আপনাকে ধন্যবাদ। বই সম্পর্কিত কোনো প্রশ্ন থাকলে জিজ্ঞাসা করতে পারেন।",
    "জি, আলহামদুলিল্লাহ ভালো আছি। আপনার জন্য কী সাহায্য করতে পারি?",
    "আপনাকেও ধন্যবাদ! বইয়ের কোনো বিষয়ে জানতে চাইলে বলুন।",
    "আল্লাহ হাফেজ! প্রয়োজন হলে আবার প্রশ্ন নিয়ে আসবেন।",
)


def _is_conversational(query: str) -> bool:
    normalized = query.strip().lower().strip(" .!?,।-")
    # A real book question can still open with a greeting word ("হ্যালো,
    # তৃতীয় অধ্যায়ে কী লেখা আছে?"), so only short pleasantries are treated
    # as small talk -- anything longer than a handful of words falls through
    # to normal retrieval instead.
    if len(normalized.split()) > 6:
        return False
    return any(p.search(normalized) for p in _CONVERSATIONAL_PATTERNS)


def answer_question(query: str) -> QueryResponse:
    start = time.perf_counter()

    if _is_conversational(query):
        answer = random.choice(_CONVERSATIONAL_REPLIES)
        response_time_ms = round((time.perf_counter() - start) * 1000, 2)
        logger.info("conversational_shortcut query=%r in %.2fms", query, response_time_ms)
        return QueryResponse(
            query=query, answer=answer, sources=[], response_time_ms=response_time_ms
        )

    citations = retrieve_relevant_docs(query)

    # The reranker score is the honest relevance signal. If even the best
    # candidate is below the bar, the corpus does not cover this question --
    # so do not hand Gemini a pile of noise as "sources". It still generates
    # an answer (generator.py's prompt has it say plainly that the books
    # don't cover this), just with no citations to attach.
    relevant = bool(citations) and citations[0].rerank_score >= settings.min_rerank_score
    grounding = citations if relevant else []

    answer = generate_answer(query, grounding)

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