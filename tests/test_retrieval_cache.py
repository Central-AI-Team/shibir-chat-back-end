"""app/rag/retriever.py -- the retrieval cache (embed + Chroma search +
merge/filter + rerank), keyed on the TRANSLATED query.

No network, no real models: expand_query, the Chroma collection, and
rerank() are all mocked. Each test controls exactly what a "repeat question"
looks like and asserts whether the expensive calls happened again.
"""

from __future__ import annotations

import unittest.mock as mock

import pytest

from app.core.config import settings
from app.rag import retriever


# ══════════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════════

@pytest.fixture(autouse=True)
def _reset_cache(monkeypatch):
    """Every test starts with a fresh, default-sized/TTL'd cache.

    The cache object is built lazily and stored in a module global, so a
    stale one (or one built with a previous test's monkeypatched settings)
    must not leak between tests.

    use_bm25=False: these tests predate hybrid retrieval and exercise the
    dense-only caching path specifically (see tests/test_hybrid_retrieval.py
    for BM25 fusion itself). Without this, bm25_index.search() would run for
    real here -- unmocked, against whatever the local Chroma collection
    happens to contain -- since it is not one of the calls `mocks` patches.
    """
    monkeypatch.setattr(settings, "retrieval_cache_maxsize", 2048)
    monkeypatch.setattr(settings, "retrieval_cache_ttl_seconds", 1800)
    monkeypatch.setattr(settings, "use_bm25", False)
    retriever._retrieval_cache = None
    yield
    retriever._retrieval_cache = None


def _chroma_result(doc: str, meta: dict, distance: float):
    """One-variant, one-hit Chroma query() response."""
    return {
        "documents": [[doc]],
        "metadatas": [[meta]],
        "distances": [[distance]],
    }


@pytest.fixture
def mocks(monkeypatch):
    """Patch everything retrieve_stages() calls between rewrite and rerank.

    expand_query is a plain Mock (not the real lru_cache'd one) so each test
    controls the "translated query" independently of the raw input text --
    that mapping is exactly what the retrieval cache keys on.
    """
    collection = mock.MagicMock()
    collection.query.return_value = _chroma_result(
        "some excerpt", {"book": "Book", "chapter": "Ch1", "row_key": "page_1", "chunk_index": 0}, 0.1
    )
    m = {
        "expand_query": mock.MagicMock(return_value=("অনুবাদিত প্রশ্ন",)),
        "embed_texts": mock.MagicMock(return_value=[[0.0] * 1024]),
        "get_collection": mock.MagicMock(return_value=collection),
        "rerank": mock.MagicMock(return_value=[(0, 0.9)]),
        "collection": collection,
    }
    monkeypatch.setattr(retriever, "expand_query", m["expand_query"])
    monkeypatch.setattr(retriever, "embed_texts", m["embed_texts"])
    monkeypatch.setattr(retriever, "get_collection", m["get_collection"])
    monkeypatch.setattr(retriever, "rerank", m["rerank"])
    return m


# ══════════════════════════════════════════════════════════════════════════
# Cache hits / misses
# ══════════════════════════════════════════════════════════════════════════

class TestRetrievalCache:
    def test_repeat_translated_query_skips_embed_search_and_rerank(self, mocks):
        """Two raw queries that translate to the SAME Bengali string (the
        Banglish-vs-Bengali-script case the feature targets) must hit
        embed/search/rerank only once."""
        retriever.retrieve_stages("namaj koto rakat")
        retriever.retrieve_stages("নামাজ কত রাকাত")

        mocks["embed_texts"].assert_called_once()
        mocks["collection"].query.assert_called_once()
        mocks["rerank"].assert_called_once()

    def test_second_call_returns_equivalent_result(self, mocks):
        stage_a1, stage_b1 = retriever.retrieve_stages("q1")
        stage_a2, stage_b2 = retriever.retrieve_stages("q2 (different raw text, same translation)")

        assert [c.row_key for c in stage_b1] == [c.row_key for c in stage_b2]
        assert stage_b1[0].rerank_score == stage_b2[0].rerank_score

    def test_different_translated_query_is_not_cached_together(self, mocks):
        retriever.retrieve_stages("first question")
        mocks["expand_query"].return_value = ("সম্পূর্ণ ভিন্ন প্রশ্ন",)
        retriever.retrieve_stages("second question")

        assert mocks["embed_texts"].call_count == 2
        assert mocks["collection"].query.call_count == 2
        assert mocks["rerank"].call_count == 2

    def test_different_collection_name_bypasses_cache(self, mocks):
        retriever.retrieve_stages("same question")
        retriever.retrieve_stages("same question", collection_name="chunk_eval_900_150")

        assert mocks["embed_texts"].call_count == 2

    def test_different_fetch_k_bypasses_cache(self, mocks):
        retriever.retrieve_stages("same question", fetch_k=10)
        retriever.retrieve_stages("same question", fetch_k=25)

        assert mocks["embed_texts"].call_count == 2

    def test_different_rerank_top_n_shares_one_cache_entry(self, mocks):
        """rerank_top_n only slices the cached (already-ranked) list -- it is
        not part of the cache key, so two callers asking for different
        amounts of the SAME translated query still share one entry."""
        retriever.retrieve_stages("same question", rerank_top_n=1)
        retriever.retrieve_stages("same question", rerank_top_n=5)

        mocks["rerank"].assert_called_once()

    def test_empty_result_is_cached_too(self, mocks):
        """An off-topic question (nothing survives min_similarity) must not
        re-run embed/search on a repeat, either."""
        mocks["collection"].query.return_value = _chroma_result(
            "irrelevant", {"book": "Book", "chapter": "Ch1", "row_key": "page_1", "chunk_index": 0}, 1.9
        )

        stage_a1, stage_b1 = retriever.retrieve_stages("off topic")
        stage_a2, stage_b2 = retriever.retrieve_stages("off topic, same translation")

        assert stage_a1 == stage_a2 == []
        assert stage_b1 == stage_b2 == []
        mocks["embed_texts"].assert_called_once()

    def test_clear_retrieval_cache(self, mocks):
        retriever.retrieve_stages("q")
        retriever.clear_retrieval_cache()
        retriever.retrieve_stages("q")

        assert mocks["embed_texts"].call_count == 2

    def test_cache_disabled_when_maxsize_zero(self, monkeypatch, mocks):
        monkeypatch.setattr(settings, "retrieval_cache_maxsize", 0)
        retriever._retrieval_cache = None

        retriever.retrieve_stages("q")
        retriever.retrieve_stages("q")

        assert mocks["embed_texts"].call_count == 2

    def test_cache_disabled_when_ttl_zero(self, monkeypatch, mocks):
        monkeypatch.setattr(settings, "retrieval_cache_ttl_seconds", 0)
        retriever._retrieval_cache = None

        retriever.retrieve_stages("q")
        retriever.retrieve_stages("q")

        assert mocks["embed_texts"].call_count == 2

    def test_use_rewrite_false_bypasses_expand_query_but_still_caches(self, mocks):
        """use_rewrite=False searches the raw (normalized) query directly --
        still cacheable, just keyed on that raw text instead of a
        translation."""
        retriever.retrieve_stages("রাকাত", use_rewrite=False)
        retriever.retrieve_stages("রাকাত", use_rewrite=False)

        mocks["expand_query"].assert_not_called()
        mocks["embed_texts"].assert_called_once()
