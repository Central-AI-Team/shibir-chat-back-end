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
  9. NEW: a TTL cache in front of embed + Chroma search + merge/filter +
     rerank (everything between the rewrite and the LLM call), keyed on the
     TRANSLATED query -- expand_query()'s own cache already saved the LLM
     call for a repeat question; this saves the embed, the Chroma round
     trip and the cross-encoder pass too. See _retrieval_cache below.
  10. NEW: hybrid retrieval. Alongside the dense (bge-m3 + Chroma) search,
     app.rag.bm25_index.search() runs a BM25 keyword search over the same
     chunks. BM25 catches exact names and numbers -- Quran sura names, verse
     numbers, rare technical terms -- that a bi-encoder routinely under-ranks
     (see bm25_index.py's module docstring for corpus examples). The two
     ranked lists are combined with Reciprocal Rank Fusion (_rrf_fuse below)
     and the union is truncated back to fetch_k *before* reranking, so this
     changes WHICH candidates reach the cross-encoder, never how many --
     reranking cost is unchanged. The cross-encoder and min_rerank_score gate
     are themselves untouched: a candidate BM25 adds still has to clear the
     same relevance bar as anything dense search finds, so a coincidental
     keyword match cannot by itself make the model answer from an irrelevant
     excerpt. settings.use_bm25 (default True) and the use_bm25= parameter
     below disable it instantly if an eval run ever shows it hurts.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

from cachetools import TTLCache
from sqlalchemy import func

from app.core import tracing
from app.core.config import settings
from app.db.models import Book
from app.db.session import SessionLocal
from app.rag import bm25_index
from app.rag.chroma_client import get_collection, get_named_collection
from app.rag.chunker import normalize
from app.rag.embedder import embed_texts
from app.rag.query_rewriter import expand_query
from app.rag.reranker import rerank
from app.schemas.query import Citation

# Reciprocal Rank Fusion constant. 60 is the standard value from the RRF
# literature (Cormack et al.) and is not tuned per corpus -- it only controls
# how quickly a source's contribution decays with rank, not which items
# qualify at all (that is min_similarity on the dense side and BM25's own
# zero-score cutoff on the lexical side).
_RRF_K = 60


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


# ══════════════════════════════════════════════════════════════════════════
# Retrieval cache -- skips embed + Chroma search + merge/filter + rerank for
# a repeated question.
#
# Keyed on the TRANSLATED query tuple (expand_query()'s own output, already
# lru_cached in query_rewriter.py), not the raw query: "namaz koto rakat" and
# "নামায কত রাকাত" both translate to the same canonical Bengali string, so
# they land on the same entry here even though their raw text differs. This
# is why the key is built AFTER expand_query() runs, not before.
#
# What is cached is the full, UNSLICED post-rerank list (every candidate the
# cross-encoder scored) plus the pre-rerank candidate pool -- the same pair
# retrieve_stages() itself returns for rerank_top_n = len(candidates). A
# caller-specific rerank_top_n (only eval scripts pass a non-default one) is
# then just a slice of that cached list, so two callers asking for different
# top_n out of the same translated query still share one cache entry instead
# of one each. fetch_k changes the candidate pool itself, so it is part of
# the key; collection_name is part of the key so eval scripts pointing at a
# temporary chunk_eval_* collection (see scripts/eval_chunking.py) never
# share an entry with production or with each other.
#
# TTL-bounded rather than indefinite (unlike expand_query's process-lifetime
# lru_cache): app/rag/ingest.py runs as a separate process/cron job with no
# signal back to a running server (CLAUDE.md §9), so nothing tells this
# in-memory cache when a page's content changed underneath it. The TTL is
# the same trade-off session_store.py already makes for session state --
# bounded staleness instead of perfect invalidation. Either maxsize=0 or
# ttl=0 makes cachetools raise, so a 0 in settings disables caching by
# routing around TTLCache entirely -- see _cache() below.
_retrieval_cache: TTLCache | None = None
_retrieval_cache_lock = threading.Lock()


def _cache() -> TTLCache | None:
    """Build (once) or return the process-wide retrieval cache, honoring
    live settings so tests can monkeypatch maxsize/ttl. None means disabled."""
    global _retrieval_cache
    if settings.retrieval_cache_maxsize <= 0 or settings.retrieval_cache_ttl_seconds <= 0:
        return None
    if _retrieval_cache is None:
        with _retrieval_cache_lock:
            if _retrieval_cache is None:
                _retrieval_cache = TTLCache(
                    maxsize=settings.retrieval_cache_maxsize,
                    ttl=settings.retrieval_cache_ttl_seconds,
                )
    return _retrieval_cache


def clear_retrieval_cache() -> None:
    """Drop every cached entry.

    Used by scripts/eval_chunking.py right after it rebuilds a chunk_eval_*
    collection from scratch, so a config name reused within one script run
    can never serve a cached result scored against the PREVIOUS content of
    that same collection name. Also handy in tests.
    """
    cache = _cache()
    if cache is not None:
        with _retrieval_cache_lock:
            cache.clear()


def _cache_key(
    queries: tuple[str, ...],
    fetch_k: int,
    collection_name: str | None,
    use_bm25: bool,
) -> tuple:
    return (queries, fetch_k, collection_name, use_bm25)


def _rrf_fuse(
    dense_sorted: list[tuple[str, str, dict, float]],
    bm25_sorted: list[tuple[str, str, dict, float]],
    fetch_k: int,
) -> list[tuple[str, dict, float]]:
    """Merge two independently-ranked lists into one, by RANK not by raw
    score -- dense similarity (cosine, [0, 1]-ish) and BM25 score (unbounded
    term-frequency) live on different, incomparable scales, so summing the
    raw numbers would let whichever score happens to be larger dominate for
    no principled reason. Reciprocal Rank Fusion sidesteps that: each source
    contributes 1/(_RRF_K + rank) for a chunk it returned, 0 if it didn't,
    and the two contributions are summed. A chunk both sources agree on
    outranks one only a single source found, without ever comparing a
    cosine number to a BM25 number directly.

    Returns (doc, meta, similarity) tuples -- the exact shape the rest of
    retrieve_stages() already expects from the old dense-only `candidates`
    list, so nothing downstream (rerank, _chunk, tracing) needs to change.
    A chunk BM25 found but dense search did not carries similarity=0.0: it
    has no cosine score to report, and similarity is diagnostic only here --
    the reranker's own score, not this one, decides relevance.
    """
    rrf_scores: dict[str, float] = {}
    payload: dict[str, tuple[str, dict, float]] = {}
    for rank, (key, doc, meta, sim) in enumerate(dense_sorted, start=1):
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (_RRF_K + rank)
        payload[key] = (doc, meta, sim)
    for rank, (key, doc, meta, _bm25_score) in enumerate(bm25_sorted, start=1):
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (_RRF_K + rank)
        payload.setdefault(key, (doc, meta, 0.0))

    ordered = sorted(rrf_scores, key=lambda k: rrf_scores[k], reverse=True)[:fetch_k]
    return [payload[key] for key in ordered]


def retrieve_stages(
    query: str,
    top_k: int | None = None,
    fetch_k: int | None = None,
    rerank_top_n: int | None = None,
    use_rewrite: bool = True,
    collection_name: str | None = None,
    max_variants: int | None = None,
    use_bm25: bool | None = None,
) -> tuple[list[RetrievedChunk], list[RetrievedChunk]]:
    """Run retrieval and return (candidates, final).

    candidates -- the pool that is actually handed to the cross-encoder:
        dense vector-search results (post min_similarity filter) fused with
        BM25 keyword-search results (§"hybrid retrieval" in the module
        docstring), truncated to fetch_k. Anything not in here was never
        seen by the reranker.
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

    use_bm25=None (default) follows settings.use_bm25. Pass False to isolate
    the dense-only pipeline -- e.g. scripts/eval_retrieval.py --no-bm25 --
    the same way use_rewrite=False isolates the rewriter's contribution.
    """
    top_k = top_k or settings.top_k
    fetch_k = fetch_k or settings.fetch_k
    rerank_top_n = rerank_top_n or top_k
    use_bm25 = settings.use_bm25 if use_bm25 is None else use_bm25

    if use_rewrite:
        queries = expand_query(query, max_variants=max_variants) or (query,)
    else:
        queries = (normalize(query),) if normalize(query) else (query,)
    tracing.record_rewrite(queries)

    cache = _cache()
    cache_key = _cache_key(queries, fetch_k, collection_name, use_bm25)
    if cache is not None:
        with _retrieval_cache_lock:
            cached = cache.get(cache_key)
        if cached is not None:
            stage_a, reranked = cached
            tracing.record_retrieval(reranked, top_k=rerank_top_n, cached=True)
            return stage_a, reranked[:rerank_top_n]

    def _store(stage_a: list[RetrievedChunk], reranked: list[RetrievedChunk]) -> None:
        if cache is not None:
            with _retrieval_cache_lock:
                cache[cache_key] = (stage_a, reranked)

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

    # Drop obvious noise before paying for the cross-encoder. This is the
    # dense side's own ranking, independent of BM25 -- fed into _rrf_fuse
    # below keyed by rank, not by this raw similarity number.
    dense_sorted = [
        (key, doc, meta, sim)
        for key, (doc, meta, sim) in pool.items()
        if sim >= settings.min_similarity
    ]
    dense_sorted.sort(key=lambda c: c[3], reverse=True)

    search_query = queries[0]

    bm25_sorted: list[tuple[str, str, dict, float]] = []
    if use_bm25:
        # Same merge-keep-best-per-key pattern as the dense pool above, in
        # case a future caller passes multiple query variants -- today
        # expand_query() always returns exactly one, so this is normally a
        # single bm25_index.search() call.
        bm25_pool: dict[str, tuple[str, dict, float]] = {}
        for q in queries:
            for key, doc, meta, score in bm25_index.search(
                q, settings.bm25_fetch_k, collection_name=collection_name
            ):
                if key not in bm25_pool or score > bm25_pool[key][2]:
                    bm25_pool[key] = (doc, meta, score)
        bm25_sorted = [(key, doc, meta, score) for key, (doc, meta, score) in bm25_pool.items()]
        bm25_sorted.sort(key=lambda c: c[3], reverse=True)

    if not dense_sorted and not bm25_sorted:
        tracing.record_retrieval([], top_k=rerank_top_n)
        _store([], [])
        return [], []

    candidates = _rrf_fuse(dense_sorted, bm25_sorted, fetch_k)
    # Rerank the WHOLE candidate pool, not just the survivors. CrossEncoder
    # .predict() already scores every (query, doc) pair -- top_n is only a
    # slice -- so asking for all of them costs nothing extra and lets the
    # `retrieve` trace span carry the rerank score of a candidate that was
    # dropped ("retrieved at similarity 0.61 but reranked to 0.42"), which is
    # the row a missed-answer investigation actually needs. The user-facing
    # list (stage_b) is still cut to rerank_top_n right below.
    ranked = rerank(search_query, [c[0] for c in candidates], top_n=len(candidates))

    reranked = [
        _chunk(*candidates[idx], rerank_score=rerank_score)
        for idx, rerank_score in ranked
    ]
    tracing.record_retrieval(reranked, top_k=rerank_top_n)

    stage_a = [_chunk(doc, meta, sim) for doc, meta, sim in candidates]
    # Cache the FULL reranked list (unsliced) so a caller-specific
    # rerank_top_n is just a slice of one shared entry (see the cache's
    # module comment above).
    _store(stage_a, reranked)
    return stage_a, reranked[:rerank_top_n]


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
