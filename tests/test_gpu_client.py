"""GPU-service mode for embedder / reranker, against an httpx.MockTransport.

No network, no model: every request goes to a handler function here, which
records what was sent and returns a canned response.
"""

from __future__ import annotations

import json
import sys

import httpx
import pytest

from app.core.config import settings
from app.rag import embedder, gpu_client, reranker
from app.rag.gpu_client import GPUServiceError

DIM = 1024


@pytest.fixture
def gpu(monkeypatch):
    """Turn GPU mode on and route the shared client through a handler.

    Usage: gpu(handler) -> list of recorded (path, json_body, headers).
    """
    monkeypatch.setattr(settings, "gpu_service_url", "https://gpu.test")
    monkeypatch.setattr(settings, "gpu_api_key", "secret-key")
    monkeypatch.setattr(settings, "gpu_max_retries", 2)
    monkeypatch.setattr(settings, "embedding_model_name", "BAAI/bge-m3")
    monkeypatch.setattr(gpu_client, "_sleep", lambda _s: None)
    calls: list[tuple[str, dict, httpx.Headers]] = []

    def install(handler):
        def recording(request: httpx.Request) -> httpx.Response:
            calls.append((request.url.path, json.loads(request.content), request.headers))
            return handler(request)

        gpu_client.reset(httpx.MockTransport(recording))
        return calls

    yield install
    gpu_client.reset()


def _embed_ok(request: httpx.Request) -> httpx.Response:
    n = len(json.loads(request.content)["texts"])
    return httpx.Response(
        200, json={"model": "BAAI/bge-m3", "dim": DIM, "vectors": [[0.1] * DIM] * n}
    )


# ---------------------------------------------------------------------------
# reranker
# ---------------------------------------------------------------------------


def test_rerank_uses_prob_not_score_and_sends_max_length(gpu):
    calls = gpu(lambda r: httpx.Response(200, json={
        "model": "BAAI/bge-reranker-v2-m3",
        "results": [
            {"index": 1, "score": 4.2, "prob": 0.985},
            {"index": 0, "score": -3.0, "prob": 0.047},
        ],
    }))

    ranked = reranker.rerank("q", ["a", "b", "c"], top_n=2)

    assert ranked == [(1, 0.985), (0, 0.047)]
    path, body, headers = calls[0]
    assert path == "/rerank"
    assert body["max_length"] == 1024
    assert body["top_k"] == 2
    assert headers["X-API-Key"] == "secret-key"


def test_rerank_splits_into_200_doc_calls_and_merges(gpu):
    def handler(request):
        docs = json.loads(request.content)["documents"]
        # Score each doc by the number embedded in its text, so the global
        # best-first order is predictable across slices.
        results = [{"index": i, "score": 0.0, "prob": int(d) / 1000} for i, d in enumerate(docs)]
        results.sort(key=lambda r: r["prob"], reverse=True)
        return httpx.Response(200, json={"model": "m", "results": results})

    calls = gpu(handler)
    docs = [str(i) for i in range(450)]

    ranked = reranker.rerank("q", docs, top_n=5)

    assert [len(body["documents"]) for _, body, _ in calls] == [200, 200, 50]
    # Indices are mapped back to the ORIGINAL list, not the slice.
    assert [i for i, _ in ranked] == [449, 448, 447, 446, 445]
    assert ranked[0][1] == pytest.approx(0.449)


# ---------------------------------------------------------------------------
# embedder
# ---------------------------------------------------------------------------


def test_embed_splits_into_256_text_calls(gpu):
    calls = gpu(_embed_ok)

    vectors = embedder.embed_texts([f"t{i}" for i in range(600)])

    assert len(vectors) == 600
    assert [len(body["texts"]) for _, body, _ in calls] == [256, 256, 88]
    assert all(body["normalize"] is True and body["batch_size"] == 64 for _, body, _ in calls)


def test_embed_dim_mismatch_raises(gpu):
    gpu(lambda r: httpx.Response(200, json={"model": "BAAI/bge-m3", "dim": 768, "vectors": [[0.1] * 768]}))
    with pytest.raises(GPUServiceError, match="dim"):
        embedder.embed_texts(["x"])


def test_embed_model_mismatch_raises(gpu):
    gpu(lambda r: httpx.Response(200, json={"model": "other/model", "dim": DIM, "vectors": [[0.1] * DIM]}))
    with pytest.raises(GPUServiceError, match="model"):
        embedder.embed_texts(["x"])


def test_gpu_mode_does_not_import_torch(gpu):
    if "torch" in sys.modules:
        pytest.skip("torch already imported by an earlier test in this process")
    gpu(_embed_ok)
    embedder.embed_query("x")
    assert "torch" not in sys.modules
    assert "sentence_transformers" not in sys.modules


# ---------------------------------------------------------------------------
# retry policy
# ---------------------------------------------------------------------------


def test_retries_on_503_then_succeeds(gpu):
    responses = iter([httpx.Response(503), httpx.Response(503)])
    calls = gpu(lambda r: next(responses, None) or _embed_ok(r))

    assert len(embedder.embed_texts(["x"])) == 1
    assert len(calls) == 3


def test_retries_on_429(gpu):
    responses = iter([httpx.Response(429)])
    calls = gpu(lambda r: next(responses, None) or _embed_ok(r))

    embedder.embed_texts(["x"])
    assert len(calls) == 2


def test_gives_up_after_max_retries(gpu):
    calls = gpu(lambda r: httpx.Response(503))
    with pytest.raises(GPUServiceError) as exc:
        embedder.embed_texts(["x"])
    assert exc.value.status_code == 503
    assert len(calls) == 1 + settings.gpu_max_retries


def test_does_not_retry_on_401(gpu):
    calls = gpu(lambda r: httpx.Response(401))
    with pytest.raises(GPUServiceError) as exc:
        embedder.embed_texts(["x"])
    assert exc.value.status_code == 401
    assert len(calls) == 1


def test_retries_on_connection_error(gpu):
    state = {"n": 0}

    def handler(request):
        state["n"] += 1
        if state["n"] == 1:
            raise httpx.ConnectError("boom", request=request)
        return _embed_ok(request)

    gpu(handler)
    embedder.embed_texts(["x"])
    assert state["n"] == 2


def test_cold_timeout_first_then_normal(gpu, monkeypatch):
    monkeypatch.setattr(settings, "gpu_timeout_seconds", 30)
    monkeypatch.setattr(settings, "gpu_cold_timeout_seconds", 120)
    seen: list[float] = []

    def handler(request):
        seen.append(request.extensions["timeout"]["read"])
        return _embed_ok(request)

    gpu(handler)
    embedder.embed_texts(["a"])
    embedder.embed_texts(["b"])
    assert seen == [120, 30]
