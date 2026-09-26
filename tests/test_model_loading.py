"""Concurrent first requests must load each model exactly once.

The API runs handlers in a threadpool, and lru_cache does not lock -- without
the module locks in embedder.py / reranker.py, N concurrent first calls each
built their own ~2.2 GB model copy. These tests swap in a slow fake model
class (no real model is loaded) and hammer the public functions from several
threads at once.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from app.core.config import settings
from app.rag import embedder, reranker


class _SlowFake:
    """Counts constructions; the sleep widens the race window."""

    instances = 0
    _count_lock = threading.Lock()

    def __init__(self, *args, **kwargs):
        time.sleep(0.2)
        with _SlowFake._count_lock:
            _SlowFake.instances += 1

    def encode(self, texts, **kwargs):
        return np.ones((len(texts), 4))

    def predict(self, pairs):
        return [0.5] * len(pairs)


@pytest.fixture(autouse=True)
def _fresh_models(monkeypatch):
    import sentence_transformers

    _SlowFake.instances = 0
    # Local-model path only -- force it even if .env points at the GPU service.
    monkeypatch.setattr(settings, "gpu_service_url", "")
    # embedder/reranker import these lazily inside _model(), so patch the
    # source module's attributes.
    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", _SlowFake)
    monkeypatch.setattr(sentence_transformers, "CrossEncoder", _SlowFake)
    embedder._model.cache_clear()
    reranker._model.cache_clear()
    yield
    embedder._model.cache_clear()
    reranker._model.cache_clear()


def _run_concurrently(fn, n=6):
    barrier = threading.Barrier(n)

    def call(_):
        barrier.wait()
        return fn()

    with ThreadPoolExecutor(max_workers=n) as pool:
        return list(pool.map(call, range(n)))


def test_embedder_loads_once_under_concurrency():
    results = _run_concurrently(lambda: embedder.embed_texts(["ক", "খ"]))
    assert _SlowFake.instances == 1
    assert all(len(r) == 2 for r in results)


def test_reranker_loads_once_under_concurrency():
    results = _run_concurrently(lambda: reranker.rerank("q", ["a", "b", "c"], top_n=2))
    assert _SlowFake.instances == 1
    assert all(len(r) == 2 for r in results)
