"""Shared QA/suggestion grounding; saved book excerpts are rechecked, never trusted by score alone."""
import logging

from app.core import tracing
from app.core.config import settings
from app.rag.reranker import rerank

logger = logging.getLogger(__name__)


def select_grounding(citations, search_query, context=None):
    grounding = []
    top_score = citations[0].rerank_score if citations else None
    if top_score is not None and top_score >= settings.min_rerank_score:
        grounding = citations
    elif context and context.followup and context.prior_sources:
        # Same-conversation, completed, server-owned book excerpts only. The
        # previous answer and its historical rerank score cannot establish
        # relevance to the new question or authorize cross-chat evidence.
        sources = context.prior_sources
        ranked = rerank(search_query, [source.content for source in sources], top_n=len(sources))
        top_score = ranked[0][1] if ranked else None
        grounding = [sources[index].model_copy(update={'rerank_score': score})
                     for index, score in ranked if score >= settings.min_rerank_score][:settings.top_k]
        if grounding:
            logger.info('Follow-up grounding recovered %d stored book excerpts', len(grounding))
    tracing.record_gate(grounded=bool(grounding), top_score=top_score, threshold=settings.min_rerank_score)
    return grounding
