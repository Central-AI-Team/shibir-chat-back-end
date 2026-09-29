"""app/rag/bm25_index.py -- tokenization, scoring, staleness, and per-
collection isolation. No network, no real Chroma: get_collection() /
get_named_collection() are mocked to return a fake in-memory `.get()`
response, and rank_bm25 itself (pure Python, no weights to download) runs
for real -- it is cheap and deterministic, unlike the reranker/embedder.
"""

from __future__ import annotations

import unittest.mock as mock

import pytest

from app.core.config import settings
from app.rag import bm25_index


@pytest.fixture(autouse=True)
def _reset_indexes(monkeypatch):
    monkeypatch.setattr(settings, "bm25_rebuild_interval_seconds", 1800)
    bm25_index._indexes.clear()
    bm25_index._built_at.clear()
    yield
    bm25_index._indexes.clear()
    bm25_index._built_at.clear()


def _fake_collection(docs: list[str], metas: list[dict]):
    collection = mock.MagicMock()
    collection.get.return_value = {"documents": docs, "metadatas": metas}
    return collection


def _meta(row_key: str, chunk_index: int = 0) -> dict:
    return {"row_key": row_key, "chunk_index": chunk_index, "book": "B", "chapter": "C"}


# BM25's idf formula goes negative (and rank_bm25 floors it to a small
# epsilon) for a term that appears in most/all of a tiny corpus -- correct
# behaviour (a term every document shares carries no discriminating power),
# but it makes a 1-2 document test corpus an unrealistic, degenerate case.
# These filler sentences pad every corpus below out to a size where a query
# term appearing in exactly one document gets a normal, clearly positive
# score, the way it would against the real ~12k-chunk production corpus.
_FILLER_DOCS = [
    "যাকাতের হিসাব ও নিয়মাবলী বিস্তারিত আলোচনা",
    "হজ্জ পালনের ধাপসমূহ ও প্রস্তুতি সম্পর্কিত বিবরণ",
    "ঈমানের মৌলিক বিষয়াবলী ও তাওহীদের ব্যাখ্যা",
    "চরিত্র গঠনের মৌলিক উপাদান নিয়ে আলোচনা",
    "সংগঠন পরিচালনার পদ্ধতি ও কর্মসূচি বিবরণ",
    "ইসলামী রাষ্ট্র ব্যবস্থার রূপরেখা ও মূলনীতি",
]


class TestTokenize:
    def test_splits_bengali_arabic_and_latin_the_same_way(self):
        assert bm25_index._tokenize("সূরা আল মুমিনুন verse 121") == [
            "সূরা", "আল", "মুমিনুন", "verse", "121",
        ]

    def test_lowercases_latin_only(self):
        assert bm25_index._tokenize("Namaz নামাজ") == ["namaz", "নামাজ"]

    def test_empty_string(self):
        assert bm25_index._tokenize("") == []


class TestSearch:
    def test_exact_term_ranks_above_a_paraphrase(self, monkeypatch):
        """The whole point of BM25: a rare exact term (a specific sura name)
        should win over a longer, unrelated document that happens to share
        only common words."""
        docs = [
            "সূরা আল মুমিনুন সম্পর্কে আলোচনা",
            "নামাজের নিয়ম ও গুরুত্ব সম্পর্কে বিস্তারিত আলোচনা এখানে দেওয়া হলো",
            *_FILLER_DOCS,
        ]
        metas = [_meta("page_1"), _meta("page_2")] + [_meta(f"filler_{i}") for i in range(len(_FILLER_DOCS))]
        collection = _fake_collection(docs, metas)
        monkeypatch.setattr(bm25_index, "get_collection", lambda: collection)

        results = bm25_index.search("সূরা আল মুমিনুন", top_n=10)

        assert results[0][0] == "page_1:0"

    def test_no_shared_term_returns_nothing(self, monkeypatch):
        collection = _fake_collection(
            docs=["সালাতের গুরুত্ব"], metas=[_meta("page_1")]
        )
        monkeypatch.setattr(bm25_index, "get_collection", lambda: collection)

        assert bm25_index.search("সম্পূর্ণ ভিন্ন বিষয়বস্তু", top_n=10) == []

    def test_empty_query_returns_nothing(self, monkeypatch):
        collection = _fake_collection(docs=["কিছু একটা"], metas=[_meta("page_1")])
        monkeypatch.setattr(bm25_index, "get_collection", lambda: collection)

        assert bm25_index.search("!!!", top_n=10) == []

    def test_empty_collection_returns_nothing(self, monkeypatch):
        collection = _fake_collection(docs=[], metas=[])
        monkeypatch.setattr(bm25_index, "get_collection", lambda: collection)

        assert bm25_index.search("যেকোনো প্রশ্ন", top_n=10) == []

    def test_top_n_caps_results(self, monkeypatch):
        # BM25 matches whole tokens, no stemming: the query word must appear
        # verbatim ("নামাজ", not the inflected "নামাজের") for this to score.
        docs = ["নামাজ পড়ার নিয়ম আলোচনা"] * 5 + _FILLER_DOCS
        metas = [_meta(f"page_{i}") for i in range(5)] + [
            _meta(f"filler_{i}") for i in range(len(_FILLER_DOCS))
        ]
        collection = _fake_collection(docs, metas)
        monkeypatch.setattr(bm25_index, "get_collection", lambda: collection)

        assert len(bm25_index.search("নামাজ", top_n=2)) == 2

    def test_result_shape(self, monkeypatch):
        docs = ["নামাজ পড়ার নিয়ম", *_FILLER_DOCS]
        metas = [_meta("page_7", 2)] + [_meta(f"filler_{i}") for i in range(len(_FILLER_DOCS))]
        collection = _fake_collection(docs, metas)
        monkeypatch.setattr(bm25_index, "get_collection", lambda: collection)

        [(key, doc, meta, score)] = bm25_index.search("নামাজ", top_n=5)
        assert key == "page_7:2"
        assert doc == "নামাজ পড়ার নিয়ম"
        assert meta["row_key"] == "page_7"
        assert score > 0


class TestCollectionScoping:
    def test_named_collection_uses_get_named_collection(self, monkeypatch):
        prod = _fake_collection(
            ["প্রোডাকশন কনটেন্ট", *_FILLER_DOCS],
            [_meta("page_1")] + [_meta(f"filler_{i}") for i in range(len(_FILLER_DOCS))],
        )
        temp = _fake_collection(
            ["টেম্প কনটেন্ট", *_FILLER_DOCS],
            [_meta("page_2")] + [_meta(f"filler_{i}") for i in range(len(_FILLER_DOCS))],
        )
        monkeypatch.setattr(bm25_index, "get_collection", lambda: prod)
        monkeypatch.setattr(
            bm25_index, "get_named_collection", lambda name: temp if name == "chunk_eval_x" else prod
        )

        results = bm25_index.search("টেম্প", top_n=5, collection_name="chunk_eval_x")

        assert results and results[0][0] == "page_2:0"

    def test_production_and_named_indexes_are_independent(self, monkeypatch):
        prod = _fake_collection(["ক খ গ"], [_meta("page_prod")])
        temp = _fake_collection(["ক খ গ"], [_meta("page_temp")])
        monkeypatch.setattr(bm25_index, "get_collection", lambda: prod)
        monkeypatch.setattr(bm25_index, "get_named_collection", lambda name: temp)

        bm25_index.search("ক", top_n=5)
        bm25_index.search("ক", top_n=5, collection_name="chunk_eval_x")

        assert set(bm25_index._indexes.keys()) == {None, "chunk_eval_x"}

    def test_reset_index_forces_rebuild(self, monkeypatch):
        filler_metas = [_meta(f"filler_{i}") for i in range(len(_FILLER_DOCS))]
        first = _fake_collection(["পুরাতন বিষয়বস্তু", *_FILLER_DOCS], [_meta("page_old")] + filler_metas)
        second = _fake_collection(["নতুন বিষয়বস্তু", *_FILLER_DOCS], [_meta("page_new")] + filler_metas)
        calls = {"n": 0}

        def get_collection():
            calls["n"] += 1
            return first if calls["n"] == 1 else second

        monkeypatch.setattr(bm25_index, "get_collection", get_collection)

        bm25_index.search("পুরাতন", top_n=5)  # builds from `first`
        bm25_index.reset_index()
        results = bm25_index.search("নতুন", top_n=5)  # must rebuild from `second`

        assert results and results[0][0] == "page_new:0"


class TestRebuildTTL:
    def test_index_is_not_rebuilt_within_ttl(self, monkeypatch):
        collection = _fake_collection(["ক খ গ"], [_meta("page_1")])
        get_calls = mock.MagicMock(side_effect=lambda: collection)
        monkeypatch.setattr(bm25_index, "get_collection", get_calls)

        bm25_index.search("ক", top_n=5)
        bm25_index.search("ক", top_n=5)

        get_calls.assert_called_once()

    def test_index_rebuilds_after_ttl_expires(self, monkeypatch):
        collection = _fake_collection(["ক খ গ"], [_meta("page_1")])
        get_calls = mock.MagicMock(side_effect=lambda: collection)
        monkeypatch.setattr(bm25_index, "get_collection", get_calls)
        monkeypatch.setattr(settings, "bm25_rebuild_interval_seconds", 0)

        bm25_index.search("ক", top_n=5)
        bm25_index.search("ক", top_n=5)

        assert get_calls.call_count == 2
