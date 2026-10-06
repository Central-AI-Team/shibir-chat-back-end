"""Hybrid retrieval: BM25 tokenizer, RRF fusion, per-page cap, retriever wiring.

No network and no real models: the embedder, the reranker and the query
rewriter are faked, and Chroma is replaced by a small in-memory collection that
implements just the calls retriever.py / lexical.py make (count, get, query).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from app.core.config import settings
from app.rag import lexical, retriever
from app.rag.retriever import RetrievedChunk, retrieve_stages, rrf_fuse

QUERY_VEC = [1.0, 0.0]


class FakeCollection:
    """rows: (chroma id, document, metadata, embedding)."""

    def __init__(self, rows):
        self.rows = rows

    def count(self):
        return len(self.rows)

    def get(self, ids=None, limit=None, offset=0, include=None):
        if ids is None:
            rows = self.rows[offset : offset + limit] if limit else self.rows[offset:]
        else:
            by_id = {r[0]: r for r in self.rows}
            rows = [by_id[i] for i in ids if i in by_id]
        out = {
            "ids": [r[0] for r in rows],
            "documents": [r[1] for r in rows],
            "metadatas": [r[2] for r in rows],
        }
        if "embeddings" in (include or []):
            out["embeddings"] = np.array([r[3] for r in rows])
        return out

    def query(self, query_embeddings, n_results, include=None):
        docs, metas, dists = [], [], []
        for q in query_embeddings:
            q = np.asarray(q, dtype=float)
            scored = sorted(
                (
                    (1.0 - float(np.dot(q, v) / (np.linalg.norm(q) * np.linalg.norm(v))), d, m)
                    for _, d, m, v in self.rows
                ),
                key=lambda t: t[0],
            )[:n_results]
            dists.append([s[0] for s in scored])
            docs.append([s[1] for s in scored])
            metas.append([s[2] for s in scored])
        return {"documents": docs, "metadatas": metas, "distances": dists}


def _row(row_key: str, chunk: int, doc: str, vec):
    meta = {
        "row_key": row_key,
        "chunk_index": chunk,
        "book": "বই",
        "chapter": "অধ্যায়",
        "source_db": "tarun",
    }
    return (f"{row_key}_c{chunk}", doc, meta, list(vec))


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    # The BM25 index is saved under chroma_persist_dir: never let a test touch
    # the real <chroma_db>/bm25 cache.
    monkeypatch.setattr(settings, "chroma_persist_dir", str(tmp_path / "chroma"))
    lexical.reset()
    monkeypatch.setattr(settings, "hybrid_enabled", False)
    monkeypatch.setattr(settings, "min_similarity", 0.25)
    monkeypatch.setattr(settings, "max_chunks_per_page", 2)
    yield
    lexical.reset()


@pytest.fixture
def wire(monkeypatch):
    """Point retrieval at a FakeCollection with fake models. Returns a spy."""

    spy = SimpleNamespace(embedded=[], reranked_docs=[])

    def install(rows, scores=None, rewrite=None):
        collection = FakeCollection(rows)
        monkeypatch.setattr(retriever, "get_collection", lambda: collection)
        monkeypatch.setattr(lexical, "get_collection", lambda: collection)
        monkeypatch.setattr(
            retriever,
            "expand_query",
            lambda q, max_variants=None: (rewrite or q,),
        )

        def fake_embed(queries):
            spy.embedded.append(list(queries))
            return [QUERY_VEC for _ in queries]

        def fake_rerank(query, docs, top_n):
            spy.reranked_docs.append(list(docs))
            score = (scores or {}).get
            ranked = sorted(
                ((i, score(d, 0.1)) for i, d in enumerate(docs)),
                key=lambda t: t[1],
                reverse=True,
            )
            return ranked[:top_n]

        monkeypatch.setattr(retriever, "embed_queries", fake_embed)
        monkeypatch.setattr(retriever, "rerank", fake_rerank)
        return collection

    spy.install = install
    return spy


# --------------------------------------------------------------------------
# tokenizer
# --------------------------------------------------------------------------


def test_tokenizer_suffix_variants_share_ngrams():
    a, b, c = (set(lexical.tokenize(w)) for w in ("নামাজ", "নামাজে", "নামাজের"))
    # The whole word is always a token...
    assert "নামাজ" in a and "নামাজে" in b and "নামাজের" in c
    # ...and the stem's n-grams appear in every inflected form, so a query for
    # one form matches chunks that only contain another.
    stem_ngrams = {"নাম", "ামা", "মাজ", "নামা", "ামাজ"}
    assert stem_ngrams <= a & b & c
    assert lexical.tokenize("নামাজ").count("নামাজ") == 1


def test_tokenizer_handles_short_words_latin_and_arabic():
    assert lexical.tokenize("যে") == ["যে"]  # shorter than ngram_min: kept whole
    assert lexical.tokenize("Salah") == ["salah", "sal", "ala", "lah", "sala", "alah"]
    # Harakat are combining marks: the word must not be shredded.
    assert "بِسْمِ" in lexical.tokenize("بِسْمِ")
    # Danda and punctuation separate words.
    assert lexical.tokenize("রোজা।নামাজ,") == lexical.tokenize("রোজা নামাজ")


# --------------------------------------------------------------------------
# fusion and cap
# --------------------------------------------------------------------------


def test_rrf_fuse_orders_by_summed_reciprocal_rank(monkeypatch):
    monkeypatch.setattr(settings, "rrf_k", 60)
    order = rrf_fuse([["a", "b"], ["b", "c"], ["b"]])
    assert order == ["b", "a", "c"]  # b is in all three lists


def _chunk(row_key, idx, score):
    return RetrievedChunk(row_key, idx, "b", "c", "s", f"{row_key}-{idx}", 0.5, score)


def test_per_page_cap_keeps_two_per_page_in_rerank_order():
    ranked = [
        _chunk("page_1", 0, 0.9),
        _chunk("page_1", 1, 0.8),
        _chunk("page_1", 2, 0.7),
        _chunk("page_2", 0, 0.6),
        _chunk("page_3", 0, 0.5),
    ]
    out = retriever._cap_per_page(ranked, limit=4, per_page=2)
    assert [(c.row_key, c.chunk_index) for c in out] == [
        ("page_1", 0), ("page_1", 1), ("page_2", 0), ("page_3", 0),
    ]
    assert len(retriever._cap_per_page(ranked, limit=5, per_page=0)) == 5  # cap off


# --------------------------------------------------------------------------
# retriever wiring
# --------------------------------------------------------------------------

DENSE_ROWS = [
    _row("page_1", 0, "নামাজের গুরুত্ব", [1.0, 0.0]),   # similarity 1.0
    _row("page_2", 0, "রোজার নিয়ম", [0.8, 0.6]),        # 0.8
    _row("page_3", 0, "যাকাতের হিসাব", [0.6, 0.8]),      # 0.6
    _row("page_4", 0, "হজের পদ্ধতি", [0.0, 1.0]),        # 0.0 -> below min_similarity
]


def test_dense_only_path_unchanged_when_hybrid_off(wire, monkeypatch):
    wire.install(DENSE_ROWS, scores={"রোজার নিয়ম": 0.9, "নামাজের গুরুত্ব": 0.4})
    monkeypatch.setattr(
        lexical, "search", lambda *a, **k: pytest.fail("BM25 must not run when hybrid is off")
    )

    stage_a, stage_b = retrieve_stages("প্রশ্ন", top_k=2, fetch_k=10)

    # min_similarity drops page_4; the rest are ordered by similarity.
    assert [c.row_key for c in stage_a] == ["page_1", "page_2", "page_3"]
    assert [round(c.similarity, 2) for c in stage_a] == [1.0, 0.8, 0.6]
    # Final is the reranked list cut to top_k, with no per-page cap involved.
    assert [c.row_key for c in stage_b] == ["page_2", "page_1"]
    assert wire.embedded == [["প্রশ্ন"]]


def test_hybrid_flag_overrides_settings_both_ways(wire, monkeypatch):
    wire.install(DENSE_ROWS)
    calls = []
    real = lexical.search
    monkeypatch.setattr(lexical, "search", lambda *a, **k: calls.append(1) or real(*a, **k))

    monkeypatch.setattr(settings, "hybrid_enabled", True)
    retrieve_stages("প্রশ্ন", hybrid=False)
    assert calls == []
    monkeypatch.setattr(settings, "hybrid_enabled", False)
    retrieve_stages("প্রশ্ন", hybrid=True)
    assert calls


def test_bm25_only_hit_survives_into_candidates(wire):
    rows = DENSE_ROWS + [
        _row("page_9", 0, "তাহাজ্জুদ নামাজের ফজিলত", [0.0, 1.0]),  # dense similarity 0.0
    ]
    wire.install(rows, scores={"তাহাজ্জুদ নামাজের ফজিলত": 0.95})

    stage_a, stage_b = retrieve_stages("তাহাজ্জুদ", top_k=3, fetch_k=10, hybrid=True)

    keys = [c.row_key for c in stage_a]
    assert "page_9" in keys, "BM25 hit must not be dropped by min_similarity"
    assert "page_4" not in keys, "a weak dense hit with no lexical match stays out"
    bm25_only = next(c for c in stage_a if c.row_key == "page_9")
    assert bm25_only.similarity == pytest.approx(0.0, abs=1e-6)  # from the stored vector
    assert stage_b[0].row_key == "page_9"  # and the reranker saw it
    assert wire.embedded == [["তাহাজ্জুদ"]]  # the embedder was not called for it


def test_hybrid_embeds_rewritten_and_original_queries(wire):
    wire.install(DENSE_ROWS, rewrite="নামাজ")
    retrieve_stages("namaz", hybrid=True)
    assert wire.embedded == [["নামাজ", "namaz"]]
    # Same text on both sides collapses to one query.
    wire.embedded.clear()
    retrieve_stages("নামাজ", hybrid=True)
    assert wire.embedded == [["নামাজ"]]


def test_hybrid_applies_per_page_cap_after_rerank(wire):
    rows = [
        _row("page_1", 0, "নামাজ ক", [1.0, 0.0]),
        _row("page_1", 1, "নামাজ খ", [1.0, 0.0]),
        _row("page_1", 2, "নামাজ গ", [1.0, 0.0]),
        _row("page_2", 0, "নামাজ ঘ", [0.9, 0.4]),
    ]
    wire.install(
        rows,
        scores={"নামাজ ক": 0.9, "নামাজ খ": 0.8, "নামাজ গ": 0.7, "নামাজ ঘ": 0.6},
    )

    _, final = retrieve_stages("নামাজ", top_k=3, fetch_k=10, hybrid=True)

    assert [(c.row_key, c.chunk_index) for c in final] == [
        ("page_1", 0), ("page_1", 1), ("page_2", 0),
    ]
    assert [c.rerank_score for c in final] == sorted(
        (c.rerank_score for c in final), reverse=True
    )


# --------------------------------------------------------------------------
# empty / odd input
# --------------------------------------------------------------------------


def test_empty_collection_is_safe(wire):
    wire.install([])
    assert lexical.search("নামাজ", 5) == []
    assert retrieve_stages("নামাজ", hybrid=True) == ([], [])


@pytest.mark.parametrize("query", ["", "   ", "।।।", "!!!", "‌"])
def test_odd_queries_return_nothing_without_raising(wire, query):
    wire.install(DENSE_ROWS)
    assert lexical.search(query, 5) == []


# --------------------------------------------------------------------------
# persistence
# --------------------------------------------------------------------------


@pytest.fixture
def build_spy(monkeypatch):
    """Counts real index builds, so a test can tell 'loaded from disk' from 'rebuilt'."""
    calls = []
    real = lexical._build

    def spy(collection, count):
        calls.append(count)
        return real(collection, count)

    monkeypatch.setattr(lexical, "_build", spy)
    return calls


def test_index_is_saved_and_reused_without_rebuilding(wire, build_spy):
    wire.install(DENSE_ROWS)
    first = lexical.search("নামাজ", 5)
    assert build_spy == [4]
    assert (lexical._cache_dir(None) / "meta.json").exists()

    lexical.reset()  # a new process: empty memory, cache on disk
    assert lexical.search("নামাজ", 5) == first
    assert build_spy == [4], "second start must load the saved index, not rebuild"


def test_changed_chunk_ids_invalidate_the_saved_index(wire, build_spy):
    wire.install(DENSE_ROWS)
    lexical.search("নামাজ", 5)

    # Same chunk count, different page: the count alone would not notice.
    swapped = [*DENSE_ROWS[:3], _row("page_77", 0, "হজের পদ্ধতি", [0.0, 1.0])]
    wire.install(swapped)
    lexical.reset()
    keys = [k for k, _ in lexical.search("হজের", 5)]

    assert build_spy == [4, 4]
    assert keys[0] == "page_77:0"  # the rebuilt index sees the swapped-in page


def test_changed_ngram_settings_invalidate_the_saved_index(wire, build_spy, monkeypatch):
    wire.install(DENSE_ROWS)
    lexical.search("নামাজ", 5)
    monkeypatch.setattr(settings, "ngram_max", 4)
    lexical.reset()
    lexical.search("নামাজ", 5)
    assert build_spy == [4, 4]


def test_corrupt_cache_falls_back_to_a_rebuild(wire, build_spy):
    wire.install(DENSE_ROWS)
    expected = lexical.search("নামাজ", 5)
    (lexical._cache_dir(None) / "keys.json").write_text("{not json", encoding="utf-8")

    lexical.reset()
    assert lexical.search("নামাজ", 5) == expected
    assert build_spy == [4, 4]
    lexical.reset()
    lexical.search("নামাজ", 5)
    assert build_spy == [4, 4], "the rebuild must have repaired the cache"


def test_unwritable_cache_still_serves_from_memory(wire, monkeypatch):
    wire.install(DENSE_ROWS)

    def boom(*args, **kwargs):
        raise OSError("read-only filesystem")

    monkeypatch.setattr(lexical, "_save", boom)
    assert [k for k, _ in lexical.search("নামাজ", 5)] == ["page_1:0"]


def test_rebuild_discards_the_saved_index(wire, build_spy):
    wire.install(DENSE_ROWS)
    lexical.search("নামাজ", 5)
    assert lexical.rebuild() == 4
    assert build_spy == [4, 4]


def test_index_survives_collection_growth_via_count_check(wire, build_spy):
    wire.install(DENSE_ROWS)
    lexical.search("নামাজ", 5)
    wire.install([*DENSE_ROWS, _row("page_5", 0, "তাহাজ্জুদ নামাজ", [0.5, 0.5])])
    # No reset(): a live process must notice the collection changed size.
    assert "page_5:0" in [k for k, _ in lexical.search("তাহাজ্জুদ", 5)]
    assert build_spy == [4, 5]
