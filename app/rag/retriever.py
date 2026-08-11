"""Retrieval.

CHANGES vs the original:
  1. top_k 3 -> fetch 25 candidates, rerank, keep 5.
  2. Searches every expanded query form (Banglish -> Bengali) and merges.
  3. Asks Chroma for "distances" and converts them to a similarity score.
     The old code discarded distances entirely, which is why there was no way
     to tell "found nothing relevant" from "found something".
  4. Returns Citations carrying a score, so qa_service can decide whether the
     corpus actually covers the question.
"""

from __future__ import annotations

from app.core.config import settings
from app.rag.chroma_client import get_collection
from app.rag.embedder import embed_texts
from app.rag.query_rewriter import expand_query
from app.rag.reranker import rerank
from app.schemas.query import Citation


def retrieve_relevant_docs(
    query: str,
    top_k: int | None = None,
    fetch_k: int | None = None,
) -> list[Citation]:
    top_k = top_k or settings.top_k
    fetch_k = fetch_k or settings.fetch_k

    queries = expand_query(query) or (query,)
    embeddings = embed_texts(list(queries))

    result = get_collection().query(
        query_embeddings=embeddings,
        n_results=fetch_k,
        include=["documents", "metadatas", "distances"],
    )

    # Merge results from all query variants, keeping the best score per chunk.
    pool: dict[str, tuple[str, dict, float]] = {}
    for docs, metas, dists in zip(
        result.get("documents", []),
        result.get("metadatas", []),
        result.get("distances", []),
    ):
        for doc, meta, dist in zip(docs, metas, dists):
            # cosine space: distance in [0, 2]; similarity = 1 - distance.
            score = 1.0 - float(dist)
            key = f"{meta.get('page_id')}:{meta.get('chunk_index', 0)}"
            if key not in pool or score > pool[key][2]:
                pool[key] = (doc, meta, score)

    candidates = sorted(pool.values(), key=lambda x: x[2], reverse=True)
    if not candidates:
        return []

    # Drop obvious noise before paying for the cross-encoder.
    candidates = [c for c in candidates if c[2] >= settings.min_similarity][:fetch_k]
    if not candidates:
        return []

    search_query = queries[0]
    ranked = rerank(search_query, [c[0] for c in candidates], top_n=top_k)

    citations: list[Citation] = []
    for idx, rerank_score in ranked:
        doc, meta, sim = candidates[idx]
        citations.append(
            Citation(
                book=meta.get("book", "Unknown"),
                chapter=meta.get("chapter", "Unknown"),
                source_db=meta.get("source_db", "unknown"),
                content=doc,
                similarity=round(sim, 4),
                rerank_score=round(rerank_score, 4),
            )
        )
    return citations