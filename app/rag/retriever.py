"""Retrieval.

CHANGES vs the original:
  1. top_k 3 -> fetch 25 candidates, rerank, keep 5.
  2. Searches every expanded query form (Banglish -> Bengali) and merges.
  3. Asks Chroma for "distances" and converts them to a similarity score.
     The old code discarded distances entirely, which is why there was no way
     to tell "found nothing relevant" from "found something".
  4. Returns Citations carrying a score, so qa_service can decide whether the
     corpus actually covers the question.
  5. The pipeline is split into retrieve_stages(), which hands back BOTH the
     pre-rerank candidate pool and the post-rerank final list.
     retrieve_relevant_docs() is unchanged from a caller's point of view -- it
     is now a thin wrapper that keeps only the final list and maps it to
     Citations. The extra stage exists because a Citation cannot tell you
     whether a missing document was never retrieved (embedder / chunking /
     query-rewrite problem) or was retrieved and then dropped by the reranker
     (reranker problem). scripts/eval_retrieval.py scores both stages.
  6. Both functions take an optional collection_name, defaulting to None
     (production, get_collection()). Added for scripts/eval_chunking.py, which
     points retrieval at a temporary chunk_eval_* collection to score a
     candidate chunking config without touching production. Every existing
     caller is unaffected -- they just never pass it.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.core.config import settings
from app.rag.chroma_client import get_collection, get_named_collection
from app.rag.chunker import normalize
from app.rag.embedder import embed_texts
from app.rag.query_rewriter import expand_query
from app.rag.reranker import rerank
from app.schemas.query import Citation


@dataclass(frozen=True)
class RetrievedChunk:
    """One chunk plus the metadata a Citation deliberately hides.

    row_key ("page_8219" / "article_1234", see ingest.py) is the stable
    page/article-level identifier -- the unit relevance judgements are made
    against, since a human can label "this page answers the question" but not
    "chunk 3 of this page".
    """

    row_key: str
    chunk_index: int
    book: str
    chapter: str
    source_db: str
    content: str
    similarity: float
    rerank_score: float | None = None


def retrieve_stages(
    query: str,
    top_k: int | None = None,
    fetch_k: int | None = None,
    rerank_top_n: int | None = None,
    use_rewrite: bool = True,
    collection_name: str | None = None,
) -> tuple[list[RetrievedChunk], list[RetrievedChunk]]:
    """Run retrieval and return (candidates, final).

    candidates -- the pool that is actually handed to the cross-encoder:
        post vector-search, post min_similarity filter, truncated to fetch_k,
        sorted by similarity. Anything not in here was never seen by the
        reranker.
    final -- the reranked list, best first, truncated to rerank_top_n
        (default top_k, i.e. exactly what the user gets).

    use_rewrite=False skips the Banglish/English -> Bengali LLM rewrite and
    searches the raw (NFC-normalized) query only, so the rewriter's
    contribution can be measured.

    collection_name=None (default) queries the production collection. Pass a
    name to query a different one instead -- see module docstring, point 6.
    """
    top_k = top_k or settings.top_k
    fetch_k = fetch_k or settings.fetch_k
    rerank_top_n = rerank_top_n or top_k

    if use_rewrite:
        queries = expand_query(query) or (query,)
    else:
        queries = (normalize(query),) if normalize(query) else (query,)
    embeddings = embed_texts(list(queries))

    collection = get_collection() if collection_name is None else get_named_collection(collection_name)
    result = collection.query(
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
            # row_key, not page_id: page ids and article ids are independent
            # sequences, so keying on page_id alone collides across the two.
            key = f"{_row_key(meta)}:{meta.get('chunk_index', 0)}"
            if key not in pool or score > pool[key][2]:
                pool[key] = (doc, meta, score)

    candidates = sorted(pool.values(), key=lambda x: x[2], reverse=True)
    if not candidates:
        return [], []

    # Drop obvious noise before paying for the cross-encoder.
    candidates = [c for c in candidates if c[2] >= settings.min_similarity][:fetch_k]
    if not candidates:
        return [], []

    search_query = queries[0]
    ranked = rerank(search_query, [c[0] for c in candidates], top_n=rerank_top_n)

    stage_a = [_chunk(doc, meta, sim) for doc, meta, sim in candidates]
    stage_b = [
        _chunk(*candidates[idx], rerank_score=rerank_score)
        for idx, rerank_score in ranked
    ]
    return stage_a, stage_b


def retrieve_relevant_docs(
    query: str,
    top_k: int | None = None,
    fetch_k: int | None = None,
    collection_name: str | None = None,
) -> list[Citation]:
    _, final = retrieve_stages(
        query, top_k=top_k, fetch_k=fetch_k, collection_name=collection_name
    )
    return [
        Citation(
            book=c.book,
            chapter=c.chapter,
            source_db=c.source_db,
            content=c.content,
            similarity=round(c.similarity, 4),
            rerank_score=round(c.rerank_score, 4) if c.rerank_score is not None else None,
        )
        for c in final
    ]


def _row_key(meta: dict) -> str:
    """Page/article-level id. Falls back for chunks written before row_key."""
    return str(meta.get("row_key") or f"page_{meta.get('page_id')}")


def _chunk(
    doc: str, meta: dict, similarity: float, rerank_score: float | None = None
) -> RetrievedChunk:
    return RetrievedChunk(
        row_key=_row_key(meta),
        chunk_index=int(meta.get("chunk_index", 0)),
        book=meta.get("book", "Unknown"),
        chapter=meta.get("chapter", "Unknown"),
        source_db=meta.get("source_db", "unknown"),
        content=doc,
        similarity=similarity,
        rerank_score=rerank_score,
    )
