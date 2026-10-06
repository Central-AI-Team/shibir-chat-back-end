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
  7. retrieve_stages() takes an optional max_variants, defaulting to None
     (expand_query's own default, settings.max_variants). Added for
     scripts/eval_query_expansion.py, which sweeps variant counts against the
     real pipeline. Every existing caller is unaffected -- they just never
     pass it.
  8. retrieve_stages() emits the Langfuse `rewrite` and `retrieve` spans (via
     app.core.tracing -- a no-op unless a live request trace is active). This
     is the one place that still holds the FULL pre-truncation candidate pool
     and every candidate's rerank score, so the `retrieve` span can show the
     candidates the reranker dropped, not just the five the user gets.
  9. retrieve_stages() takes an optional `hybrid` flag (None -> settings
     .hybrid_enabled, default off). When on, retrieval is handled by
     _retrieve_hybrid(): a BM25 leg (app/rag/lexical.py) next to the dense
     leg, fused with Reciprocal Rank Fusion, reranked as usual, then capped at
     settings.max_chunks_per_page per page. When off, the dense path below is
     the original code, untouched.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from sqlalchemy import func

from app.core import timing, tracing
from app.core.config import settings
from app.db.models import Book
from app.db.session import SessionLocal
from app.rag import lexical
from app.rag.chroma_client import get_collection, get_named_collection
from app.rag.chunker import normalize
from app.rag.embedder import embed_queries
from app.rag.query_rewriter import expand_query
from app.rag.reranker import rerank
from app.schemas.query import Citation

logger = logging.getLogger(__name__)


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
    book_id: int | None = None


def retrieve_stages(
    query: str,
    top_k: int | None = None,
    fetch_k: int | None = None,
    rerank_top_n: int | None = None,
    use_rewrite: bool = True,
    collection_name: str | None = None,
    max_variants: int | None = None,
    hybrid: bool | None = None,
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

    max_variants=None (default) leaves expand_query's own default
    (settings.max_variants) alone -- no existing caller needs to pass this.
    It exists so scripts/eval_query_expansion.py can sweep variant counts
    against the real pipeline without a second retrieval code path.

    hybrid=None (default) follows settings.hybrid_enabled; True/False forces
    the hybrid / dense-only path. In hybrid mode `candidates` is the RRF-fused
    pool (truncated to fetch_k) and `final` is the reranked list after the
    per-page cap -- see _retrieve_hybrid().
    """
    top_k = top_k or settings.top_k
    fetch_k = fetch_k or settings.fetch_k
    rerank_top_n = rerank_top_n or top_k

    if use_rewrite:
        with timing.timed("query_rewrite"):
            queries = expand_query(query, max_variants=max_variants) or (query,)
    else:
        timing.mark("query_rewrite", "disabled")
        queries = (normalize(query),) if normalize(query) else (query,)
    tracing.record_rewrite(queries)
    if settings.hybrid_enabled if hybrid is None else hybrid:
        return _retrieve_hybrid(
            query, queries, fetch_k, rerank_top_n, collection_name
        )
    with timing.timed("query_embedding"):
        embeddings = embed_queries(list(queries))

    collection = get_collection() if collection_name is None else get_named_collection(collection_name)
    with timing.timed("vector_search"):
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
        tracing.record_retrieval([], top_k=rerank_top_n)
        return [], []

    # Drop obvious noise before paying for the cross-encoder.
    candidates = [c for c in candidates if c[2] >= settings.min_similarity][:fetch_k]
    if not candidates:
        tracing.record_retrieval([], top_k=rerank_top_n)
        return [], []

    search_query = queries[0]
    # Rerank the WHOLE candidate pool, not just the survivors. CrossEncoder
    # .predict() already scores every (query, doc) pair -- top_n is only a
    # slice -- so asking for all of them costs nothing extra and lets the
    # `retrieve` trace span carry the rerank score of a candidate that was
    # dropped ("retrieved at similarity 0.61 but reranked to 0.42"), which is
    # the row a missed-answer investigation actually needs. The user-facing
    # list (stage_b) is still cut to rerank_top_n right below.
    with timing.timed("rerank"):
        ranked = rerank(search_query, [c[0] for c in candidates], top_n=len(candidates))

    reranked = [
        _chunk(*candidates[idx], rerank_score=rerank_score)
        for idx, rerank_score in ranked
    ]
    tracing.record_retrieval(reranked, top_k=rerank_top_n)

    stage_a = [_chunk(doc, meta, sim) for doc, meta, sim in candidates]
    stage_b = reranked[:rerank_top_n]
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


def _retrieve_hybrid(
    query: str,
    queries: tuple[str, ...],
    fetch_k: int,
    rerank_top_n: int,
    collection_name: str | None,
) -> tuple[list[RetrievedChunk], list[RetrievedChunk]]:
    """Dense + BM25 retrieval fused with Reciprocal Rank Fusion.

    The query set is the rewritten query plus the original normalized one:
    the rewrite fixes Banglish, but it can also drop or respell a rare term
    that a lexical match on the user's own words would still find. One ranked
    list is built per (leg, query), all lists are fused (score = sum of
    1 / (rrf_k + rank)), and the pool is cut to fetch_k.

    The reranker then scores the whole fused pool against the REWRITTEN query,
    exactly as the dense path does, and the per-page cap is applied to its
    output so one long page cannot fill every slot.
    """
    collection = get_collection() if collection_name is None else get_named_collection(collection_name)
    search_query = queries[0]
    query_set = hybrid_query_set(search_query, query)

    # --- dense leg -------------------------------------------------------
    with timing.timed("query_embedding"):
        embeddings = embed_queries(query_set)
    with timing.timed("vector_search"):
        result = collection.query(
            query_embeddings=embeddings,
            n_results=fetch_k,
            include=["documents", "metadatas", "distances"],
        )

    # Every dense hit, with its best cosine similarity across the queries.
    seen: dict[str, tuple[str, dict, float]] = {}
    ranked_lists: list[list[str]] = []
    for docs, metas, dists in zip(
        result.get("documents", []),
        result.get("metadatas", []),
        result.get("distances", []),
    ):
        ranking: list[str] = []
        for doc, meta, dist in zip(docs, metas, dists):
            similarity = 1.0 - float(dist)  # cosine space, see retrieve_stages
            key = lexical.chunk_key(meta)
            if key not in seen or similarity > seen[key][2]:
                seen[key] = (doc, meta, similarity)
            # min_similarity gates the DENSE leg only: a weak vector hit earns
            # no dense rank, but BM25 can still bring the chunk in below.
            if similarity >= settings.min_similarity:
                ranking.append(key)
        ranked_lists.append(ranking)

    # --- lexical leg -----------------------------------------------------
    with timing.timed("bm25_search"):
        for q in query_set:
            try:
                hits = lexical.search(q, settings.bm25_fetch_k, collection_name)
            except Exception:
                # Degrade to dense-only rather than fail the request; the
                # warning keeps an eval from silently passing off dense results
                # as hybrid ones.
                logger.warning("BM25 leg failed, continuing dense-only", exc_info=True)
                hits = []
            ranked_lists.append([key for key, _ in hits])

    # --- fuse ------------------------------------------------------------
    order = rrf_fuse(ranked_lists)[:fetch_k]

    missing = [key for key in order if key not in seen]
    if missing:
        seen.update(_fetch_chunks(collection, missing, embeddings))
    candidates = [seen[key] for key in order if key in seen]
    if not candidates:
        tracing.record_retrieval([], top_k=rerank_top_n)
        return [], []

    # --- rerank, then cap per page ---------------------------------------
    with timing.timed("rerank"):
        ranked = rerank(search_query, [c[0] for c in candidates], top_n=len(candidates))
    reranked = [
        _chunk(*candidates[idx], rerank_score=rerank_score)
        for idx, rerank_score in ranked
    ]
    tracing.record_retrieval(reranked, top_k=rerank_top_n)

    stage_a = [_chunk(doc, meta, sim) for doc, meta, sim in candidates]
    return stage_a, _cap_per_page(reranked, rerank_top_n, settings.max_chunks_per_page)


def hybrid_query_set(search_query: str, original: str) -> list[str]:
    """The rewritten query plus the user's own normalized words, deduplicated."""
    return list(dict.fromkeys([search_query, normalize(original) or original]))


def rrf_fuse(ranked_lists: list[list[str]]) -> list[str]:
    """Reciprocal Rank Fusion: keys best first by sum(1 / (rrf_k + rank)).

    Ranks start at 1. Ties keep first-seen order, so with the dense lists
    passed first a dense hit wins a tie against a BM25 one.
    """
    fused: dict[str, float] = {}
    for ranking in ranked_lists:
        for rank, key in enumerate(ranking, start=1):
            fused[key] = fused.get(key, 0.0) + 1.0 / (settings.rrf_k + rank)
    return sorted(fused, key=fused.__getitem__, reverse=True)


def _fetch_chunks(
    collection, keys: list[str], query_embeddings: list[list[float]]
) -> dict[str, tuple[str, dict, float]]:
    """Text, metadata and similarity for BM25-only hits, read from Chroma.

    Chroma ids are f"{row_key}_c{chunk_index}" (ingest.py) while the lexical
    leg's keys are f"{row_key}:{chunk_index}". Similarity is the best cosine
    against the already-computed query embeddings, using the vector Chroma
    stores for the chunk, so the embedder is not called again. A chunk whose
    vector cannot be read gets 0.0; a key Chroma does not know is dropped.
    """
    ids = []
    for key in keys:
        row_key, _, chunk_index = key.rpartition(":")
        ids.append(f"{row_key}_c{chunk_index}")
    got = collection.get(ids=ids, include=["documents", "metadatas", "embeddings"])

    vectors = got.get("embeddings")
    queries = np.asarray(query_embeddings, dtype=float)
    out: dict[str, tuple[str, dict, float]] = {}
    for i, (doc, meta) in enumerate(zip(got.get("documents") or [], got.get("metadatas") or [])):
        similarity = 0.0
        try:
            vec = np.asarray(vectors[i], dtype=float)
            norms = np.linalg.norm(queries, axis=1) * np.linalg.norm(vec)
            similarity = float(np.max(queries @ vec / np.where(norms == 0, 1.0, norms)))
        except Exception:
            pass
        out[lexical.chunk_key(meta)] = (doc, meta, similarity)
    return out


def _cap_per_page(
    chunks: list[RetrievedChunk], limit: int, per_page: int
) -> list[RetrievedChunk]:
    """First `limit` chunks, best first, with at most `per_page` per row_key.

    per_page <= 0 disables the cap. Input order (the rerank order) is kept.
    """
    out: list[RetrievedChunk] = []
    counts: dict[str, int] = {}
    for chunk in chunks:
        if per_page > 0 and counts.get(chunk.row_key, 0) >= per_page:
            continue
        counts[chunk.row_key] = counts.get(chunk.row_key, 0) + 1
        out.append(chunk)
        if len(out) >= limit:
            break
    return out


def _row_key(meta: dict) -> str:
    """Page/article-level id. Falls back for chunks written before row_key."""
    return str(meta.get("row_key") or f"page_{meta.get('page_id')}")


# Two different books can share the exact same `books.name` (e.g. book_id 173
# and 210 are both titled "কর্মপদ্ধতি" -- confirmed genuinely different books,
# just an unlucky title collision upstream). They already live under distinct
# book_ids, so retrieval itself never mixes their content -- this is a
# citation-clarity problem only: "সূত্র: কর্মপদ্ধতি" doesn't tell the reader
# which one. Cached for the life of the process -- the set of colliding
# titles changes only when books are added/renamed, which doesn't happen at
# request time.
_AMBIGUOUS_BOOK_NAMES: set[str] | None = None


def _ambiguous_book_names() -> set[str]:
    global _AMBIGUOUS_BOOK_NAMES
    if _AMBIGUOUS_BOOK_NAMES is None:
        with SessionLocal() as s:
            rows = (
                s.query(Book.name)
                .group_by(Book.name)
                .having(func.count(func.distinct(Book.id)) > 1)
                .all()
            )
        _AMBIGUOUS_BOOK_NAMES = {name for (name,) in rows}
    return _AMBIGUOUS_BOOK_NAMES


def _display_book(name: str, category: str | None) -> str:
    """Append a category disambiguator when `name` collides across book_ids.

    Falls back to the bare name if this chunk's metadata predates the
    book_id/category fields (old Chroma vectors from before ingest.py started
    writing them) -- category is None there, so disambiguation is skipped
    rather than appending "(None)".
    """
    if category and name in _ambiguous_book_names():
        return f"{name} ({category})"
    return name


def _chunk(
    doc: str, meta: dict, similarity: float, rerank_score: float | None = None
) -> RetrievedChunk:
    book_id = meta.get("book_id")
    return RetrievedChunk(
        row_key=_row_key(meta),
        chunk_index=int(meta.get("chunk_index", 0)),
        book=_display_book(meta.get("book", "Unknown"), meta.get("category")),
        chapter=meta.get("chapter", "Unknown"),
        source_db=meta.get("source_db", "unknown"),
        content=doc,
        similarity=similarity,
        rerank_score=rerank_score,
        book_id=int(book_id) if book_id is not None else None,
    )
