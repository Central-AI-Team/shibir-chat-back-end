"""GPU-service mode for embedder / reranker, against an httpx.MockTransport.

No network, no model: every request goes to a handler function here, which
records what was sent and returns a canned response.
"""

from __future__ import annotations

import json
import logging
import sys

import httpx
import pytest

from app.core import tracing
from app.core.config import settings
from app.rag import embedder, gpu_client, reranker
from app.rag.gpu_client import GPUServiceError

DIM = 1024
RERANKER = "BAAI/bge-reranker-v2-m3"


@pytest.fixture
def gpu(monkeypatch):
    """Turn GPU mode on and route the shared client through a handler.

    Usage: gpu(handler) -> list of recorded (path, json_body, headers).
    """
    monkeypatch.setattr(settings, "gpu_service_url", "https://gpu.test")
    monkeypatch.setattr(settings, "gpu_api_key", "secret-key")
    monkeypatch.setattr(settings, "gpu_max_retries", 2)
    monkeypatch.setattr(settings, "embedding_model_name", "BAAI/bge-m3")
    monkeypatch.setattr(settings, "reranker_model_name", RERANKER)
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
        return httpx.Response(200, json={"model": RERANKER, "results": results})

    calls = gpu(handler)
    docs = [str(i) for i in range(450)]

    ranked = reranker.rerank("q", docs, top_n=5)

    assert [len(body["documents"]) for _, body, _ in calls] == [200, 200, 50]
    # Indices are mapped back to the ORIGINAL list, not the slice.
    assert [i for i, _ in ranked] == [449, 448, 447, 446, 445]
    assert ranked[0][1] == pytest.approx(0.449)


def test_rerank_model_mismatch_raises(gpu):
    gpu(lambda r: httpx.Response(200, json={
        "model": "other/reranker",
        "results": [{"index": 0, "score": 1.0, "prob": 0.7}],
    }))
    with pytest.raises(GPUServiceError, match="model"):
        reranker.rerank("q", ["a"], top_n=1)


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


def test_non_retryable_status_is_logged_without_secrets(gpu, caplog):
    gpu(lambda r: httpx.Response(401))
    with caplog.at_level(logging.ERROR, logger=gpu_client.__name__):
        with pytest.raises(GPUServiceError):
            embedder.embed_texts(["private user text"])

    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 1
    assert "401" in errors[0].getMessage()
    assert "/embed" in errors[0].getMessage()
    assert "secret-key" not in caplog.text
    assert "private user text" not in caplog.text


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


@pytest.mark.parametrize("error", [httpx.ReadError, httpx.WriteError])
def test_retries_when_connection_drops_mid_request(gpu, error):
    state = {"n": 0}

    def handler(request):
        state["n"] += 1
        if state["n"] == 1:
            raise error("dropped", request=request)
        return _embed_ok(request)

    gpu(handler)
    embedder.embed_texts(["x"])
    assert state["n"] == 2


def test_does_not_retry_read_timeout(gpu):
    def handler(request):
        raise httpx.ReadTimeout("slow", request=request)

    calls = gpu(handler)
    with pytest.raises(GPUServiceError, match="ReadTimeout"):
        embedder.embed_texts(["x"])
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# warm window
# ---------------------------------------------------------------------------


@pytest.fixture
def clock(monkeypatch):
    """A controllable gpu_client._monotonic; advance with clock["now"] += s."""
    state = {"now": 1000.0}
    monkeypatch.setattr(gpu_client, "_monotonic", lambda: state["now"])
    return state


def _record_timeouts(gpu, monkeypatch) -> list[float]:
    monkeypatch.setattr(settings, "gpu_timeout_seconds", 30)
    monkeypatch.setattr(settings, "gpu_cold_timeout_seconds", 120)
    monkeypatch.setattr(settings, "gpu_warm_window_seconds", 240)
    seen: list[float] = []

    def handler(request):
        seen.append(request.extensions["timeout"]["read"])
        return _embed_ok(request)

    gpu(handler)
    return seen


def test_cold_timeout_again_after_warm_window_expires(gpu, monkeypatch, clock):
    seen = _record_timeouts(gpu, monkeypatch)

    embedder.embed_texts(["a"])  # first call: cold
    clock["now"] += 10
    embedder.embed_texts(["b"])  # recently succeeded: warm
    clock["now"] += 241          # idle past the window: Modal may have scaled down
    embedder.embed_texts(["c"])

    assert seen == [120, 30, 120]


def test_warm_window_measured_from_last_success(gpu, monkeypatch, clock):
    seen = _record_timeouts(gpu, monkeypatch)

    embedder.embed_texts(["a"])
    for _ in range(3):  # steady traffic keeps it warm well past 240s in total
        clock["now"] += 200
        embedder.embed_texts(["b"])

    assert seen == [120, 30, 30, 30]


def test_reset_clears_warm_state(gpu, monkeypatch, clock):
    _record_timeouts(gpu, monkeypatch)
    embedder.embed_texts(["a"])
    assert gpu_client._last_ok is not None

    gpu_client.reset()

    assert gpu_client._last_ok is None


# ---------------------------------------------------------------------------
# tracing
# ---------------------------------------------------------------------------


@pytest.fixture
def spans(monkeypatch):
    """Capture gpu_client's span calls instead of talking to Langfuse."""
    recorded: list[dict] = []

    def start(name):
        span = {"name": name}
        recorded.append(span)
        return span

    def end(span, *, metadata, error=False):
        span.update(metadata=metadata, error=error)

    monkeypatch.setattr(tracing, "start_gpu_span", start)
    monkeypatch.setattr(tracing, "end_gpu_span", end)
    return recorded


def test_span_per_call_with_counts_and_no_texts(gpu, spans):
    gpu(lambda r: _embed_ok(r) if r.url.path == "/embed" else httpx.Response(
        200, json={"model": RERANKER, "results": [{"index": 0, "score": 1.0, "prob": 0.7}]}
    ))

    embedder.embed_texts(["secret text one", "secret text two"])
    reranker.rerank("secret query", ["secret doc"], top_n=1)

    assert [s["name"] for s in spans] == ["gpu:embed", "gpu:rerank"]
    embed = spans[0]
    assert embed["error"] is False
    assert set(embed["metadata"]) == {"n_items", "latency_ms", "cold", "attempts", "status"}
    assert embed["metadata"]["n_items"] == 2
    assert embed["metadata"]["cold"] is True
    assert embed["metadata"]["attempts"] == 1
    assert embed["metadata"]["status"] == 200
    assert spans[1]["metadata"]["n_items"] == 1
    assert spans[1]["metadata"]["cold"] is False
    assert "secret" not in repr(spans)


def test_span_records_failure(gpu, spans):
    gpu(lambda r: httpx.Response(503))
    with pytest.raises(GPUServiceError):
        embedder.embed_texts(["x"])

    (span,) = spans
    assert span["error"] is True
    assert span["metadata"]["attempts"] == 1 + settings.gpu_max_retries
    assert span["metadata"]["status"] == 503


def test_gpu_span_helpers_are_noops_when_tracing_disabled(monkeypatch):
    monkeypatch.setattr(settings, "langfuse_public_key", "")
    assert tracing.start_gpu_span("gpu:embed") is None
    tracing.end_gpu_span(None, metadata={})  # must not raise
