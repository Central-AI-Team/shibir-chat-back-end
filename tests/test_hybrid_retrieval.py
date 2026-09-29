"""app/rag/retriever.py -- hybrid retrieval (dense + BM25 fused with
Reciprocal Rank Fusion). No network, no real models: expand_query, the
Chroma collection, bm25_index.search, and rerank() are all mocked, exactly
like tests/test_retrieval_cache.py. Each test controls what dense search and
BM25 each "find" independently and asserts what reaches the reranker.
"""

from __future__ import annotations

import unittest.mock as mock

import pytest

from app.core.config import settings
from app.rag import retriever


@pytest.fixture(autouse=True)
def _reset_cache(monkeypatch):
    monkeypatch.setattr(settings, "retrieval_cache_maxsize", 2048)
    monkeypatch.setattr(settings, "retrieval_cache_ttl_seconds", 1800)
    monkeypatch.setattr(settings, "use_bm25", True)
    monkeypatch.setattr(settings, "bm25_fetch_k", 25)
    retriever._retrieval_cache = None
    yield
    retriever._retrieval_cache = None


def _chroma_result(hits: list[tuple[str, dict, float]]):
    """hits: list of (doc, meta, distance)."""
    docs = [h[0] for h in hits]
    metas = [h[1] for h in hits]
    dists = [h[2] for h in hits]
    return {"documents": [docs], "metadatas": [metas], "distances": [dists]}


@pytest.fixture
def mocks(monkeypatch):
    collection = mock.MagicMock()
    collection.query.return_value = _chroma_result([])
    m = {
        "expand_query": mock.MagicMock(return_value=("অনুবাদিত প্রশ্ন",)),
        "embed_texts": mock.MagicMock(return_value=[[0.0] * 1024]),
        "get_collection": mock.MagicMock(return_value=collection),
        "bm25_search": mock.MagicMock(return_value=[]),
        "rerank": mock.MagicMock(side_effect=lambda q, docs, top_n: [(i, 1.0) for i in range(len(docs))]),
        "collection": collection,
    }
    monkeypatch.setattr(retriever, "expand_query", m["expand_query"])
    monkeypatch.setattr(retriever, "embed_texts", m["embed_texts"])
    monkeypatch.setattr(retriever, "get_collection", m["get_collection"])
    monkeypatch.setattr(retriever.bm25_index, "search", m["bm25_search"])
    monkeypatch.setattr(retriever, "rerank", m["rerank"])
    return m


class TestHybridFusion:
    def test_bm25_only_hit_reaches_the_reranker(self, mocks):
        """A chunk dense search never returns at all (0 hits) but BM25 finds
        by exact keyword match must still reach the cross-encoder -- this is
        the whole point of hybrid search: recovering an exact-term match a
        bi-encoder missed entirely."""
        mocks["collection"].query.return_value = _chroma_result([])  # dense finds nothing
        mocks["bm25_search"].return_value = [
            ("page_9:0", "সূরা আল মুমিনুন সম্পর্কে", {"row_key": "page_9", "chunk_index": 0,
                                                      "book": "B", "chapter": "C"}, 8.5),
        ]

        _, final = retriever.retrieve_stages("সূরা আল মুমিনুন")

        assert [c.row_key for c in final] == ["page_9"]

    def test_dense_only_hit_still_works_with_bm25_enabled(self, mocks):
        """BM25 finding nothing must not suppress a dense-only hit."""
        mocks["collection"].query.return_value = _chroma_result([
            ("dense content", {"row_key": "page_1", "chunk_index": 0, "book": "B", "chapter": "C"}, 0.1),
        ])
        mocks["bm25_search"].return_value = []

        _, final = retriever.retrieve_stages("কোনো প্রশ্ন")

        assert [c.row_key for c in final] == ["page_1"]

    def test_chunk_found_by_both_sources_is_not_duplicated(self, mocks):
        meta = {"row_key": "page_5", "chunk_index": 0, "book": "B", "chapter": "C"}
        mocks["collection"].query.return_value = _chroma_result([("shared content", meta, 0.1)])
        mocks["bm25_search"].return_value = [("page_5:0", "shared content", meta, 5.0)]

        candidates, final = retriever.retrieve_stages("প্রশ্ন")

        assert len(candidates) == 1
        assert len(final) == 1

    def test_agreement_between_sources_outranks_a_single_source_hit(self, mocks):
        """RRF: a chunk both dense and BM25 rank #1 should outrank a chunk
        only one of the two sources found at all."""
        meta_both = {"row_key": "page_both", "chunk_index": 0, "book": "B", "chapter": "C"}
        meta_dense_only = {"row_key": "page_dense", "chunk_index": 0, "book": "B", "chapter": "C"}
        mocks["collection"].query.return_value = _chroma_result([
            ("both", meta_both, 0.05),
            ("dense only", meta_dense_only, 0.06),
        ])
        mocks["bm25_search"].return_value = [("page_both:0", "both", meta_both, 9.0)]

        candidates, _ = retriever.retrieve_stages("প্রশ্ন")

        assert candidates[0].row_key == "page_both"

    def test_use_bm25_false_ignores_bm25_entirely(self, mocks):
        mocks["collection"].query.return_value = _chroma_result([])
        mocks["bm25_search"].return_value = [
            ("page_9:0", "keyword hit", {"row_key": "page_9", "chunk_index": 0,
                                          "book": "B", "chapter": "C"}, 8.5),
        ]

        candidates, final = retriever.retrieve_stages("প্রশ্ন", use_bm25=False)

        mocks["bm25_search"].assert_not_called()
        assert candidates == [] and final == []

    def test_settings_use_bm25_false_disables_by_default(self, mocks, monkeypatch):
        monkeypatch.setattr(settings, "use_bm25", False)
        mocks["bm25_search"].return_value = [
            ("page_9:0", "keyword hit", {"row_key": "page_9", "chunk_index": 0,
                                          "book": "B", "chapter": "C"}, 8.5),
        ]

        retriever.retrieve_stages("প্রশ্ন")

        mocks["bm25_search"].assert_not_called()

    def test_reranker_still_gates_a_bm25_only_false_positive(self, mocks):
        """A BM25 hit that is really just noise (shares a common word) must
        still be dropped by the SAME reranker + gate qa_service relies on --
        hybrid search only widens the candidate pool, it never bypasses the
        relevance check. This is the hallucination-safety guarantee: adding
        BM25 candidates cannot make a low-relevance chunk look "found"."""
        mocks["collection"].query.return_value = _chroma_result([])
        mocks["bm25_search"].return_value = [
            ("page_noise:0", "irrelevant noise", {"row_key": "page_noise", "chunk_index": 0,
                                                    "book": "B", "chapter": "C"}, 1.0),
        ]
        mocks["rerank"].side_effect = None
        mocks["rerank"].return_value = [(0, 0.01)]  # cross-encoder says: not relevant

        _, final = retriever.retrieve_stages("প্রশ্ন")

        # retrieve_stages() itself does not apply the gate (qa_service does),
        # but the low rerank_score is what the gate keys on -- confirm it
        # survives the fusion untouched.
        assert final[0].rerank_score == 0.01

    def test_bm25_pool_capped_by_bm25_fetch_k(self, mocks, monkeypatch):
        monkeypatch.setattr(settings, "bm25_fetch_k", 3)
        retriever.retrieve_stages("প্রশ্ন")

        assert mocks["bm25_search"].call_args.args[1] == 3

    def test_candidate_pool_never_exceeds_fetch_k(self, mocks):
        dense_hits = [
            (f"dense doc {i}", {"row_key": f"page_d{i}", "chunk_index": 0, "book": "B", "chapter": "C"}, 0.1)
            for i in range(20)
        ]
        mocks["collection"].query.return_value = _chroma_result(dense_hits)
        mocks["bm25_search"].return_value = [
            (f"page_b{i}:0", f"bm25 doc {i}", {"row_key": f"page_b{i}", "chunk_index": 0,
                                                "book": "B", "chapter": "C"}, 5.0)
            for i in range(20)
        ]

        candidates, _ = retriever.retrieve_stages("প্রশ্ন", fetch_k=10)

        assert len(candidates) <= 10

    def test_use_bm25_is_part_of_the_cache_key(self, mocks):
        mocks["bm25_search"].return_value = [
            ("page_9:0", "keyword hit", {"row_key": "page_9", "chunk_index": 0,
                                          "book": "B", "chapter": "C"}, 8.5),
        ]

        retriever.retrieve_stages("প্রশ্ন", use_bm25=True)
        retriever.retrieve_stages("প্রশ্ন", use_bm25=False)

        assert mocks["embed_texts"].call_count == 2

    def test_bm25_only_result_gets_zero_similarity_not_none(self, mocks):
        """similarity is diagnostic-only for a BM25-only find (no cosine
        score exists), not a crash -- Citation.similarity must still be a
        plain float."""
        mocks["collection"].query.return_value = _chroma_result([])
        mocks["bm25_search"].return_value = [
            ("page_9:0", "keyword hit", {"row_key": "page_9", "chunk_index": 0,
                                          "book": "B", "chapter": "C"}, 8.5),
        ]

        _, final = retriever.retrieve_stages("প্রশ্ন")

        assert final[0].similarity == 0.0
