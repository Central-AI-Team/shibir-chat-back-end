"""HTTP client for the external GPU service (shibir-chat-gpu-service on Modal).

Used by embedder.py and reranker.py when settings.gpu_service_url is set.
The service exposes:
  POST /embed   {texts, normalize, batch_size}      -> {model, dim, vectors}
  POST /rerank  {query, documents, top_k, max_length} -> {model, results}
both authenticated with an X-API-Key header.

One shared httpx.Client per process, so connections (and TLS sessions) are
pooled across requests instead of re-handshaking on every embed/rerank call.

Timeouts: Modal scales the service to zero after 300s idle, so a call can
land on a cold container that still has to boot and load both models. A call
gets settings.gpu_cold_timeout_seconds unless the last successful call was
less than settings.gpu_warm_window_seconds ago (kept below Modal's scaledown
window); otherwise settings.gpu_timeout_seconds. "Warm once, warm forever"
was wrong: after an idle stretch the short timeout hit the cold start, read-
timed-out, and the user got a 503.

Retries (exponential backoff) happen ONLY on connection-level failures
(including a connection dropped mid-request -- /embed and /rerank are
idempotent, so resending is safe), 5xx and 429 -- those are transient. Any
other 4xx (bad key, bad payload) will fail identically on retry, so it is
logged at ERROR and raised immediately. A read timeout is not
retried either: the service may still be working on the request, and
retrying a slow call just multiplies the user's wait.

Each call is traced as a Langfuse span "gpu:embed" / "gpu:rerank" (no-op when
tracing is off) carrying counts and timings only.

Never log or trace request bodies (they are user queries / book text) or the
API key.
"""

from __future__ import annotations

import logging
import threading
import time

import httpx

from app.core import tracing
from app.core.config import settings

logger = logging.getLogger(__name__)

# Transient, safe-to-retry transport failures: the request never reached the
# service, or the connection was dropped (e.g. a container being replaced).
# ReadTimeout is deliberately absent -- see the module docstring.
_RETRYABLE_TRANSPORT_ERRORS = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.RemoteProtocolError,
    httpx.ReadError,
    httpx.WriteError,
)

_BACKOFF_BASE_SECONDS = 0.5

_lock = threading.Lock()
_client: httpx.Client | None = None
# time.monotonic() of the last successful response; None = none yet.
_last_ok: float | None = None

# Indirection so tests can skip real sleeps and move the clock.
_sleep = time.sleep
_monotonic = time.monotonic


class GPUServiceError(RuntimeError):
    """The GPU service could not produce a usable response."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


def is_enabled() -> bool:
    return bool(settings.gpu_service_url)


def _get_client() -> httpx.Client:
    global _client
    with _lock:
        if _client is None:
            _client = httpx.Client(base_url=settings.gpu_service_url.rstrip("/"))
        return _client


def reset(transport: httpx.BaseTransport | None = None) -> None:
    """Drop the shared client (and warm state). Tests pass a MockTransport."""
    global _client, _last_ok
    with _lock:
        if _client is not None:
            _client.close()
        _client = (
            httpx.Client(base_url=settings.gpu_service_url.rstrip("/"), transport=transport)
            if transport is not None
            else None
        )
        _last_ok = None


def _retryable_status(code: int) -> bool:
    return code == 429 or code >= 500


def _is_cold() -> bool:
    return _last_ok is None or _monotonic() - _last_ok > settings.gpu_warm_window_seconds


def post(path: str, json: dict) -> dict:
    """POST `json` to the GPU service and return the decoded JSON response.

    Raises GPUServiceError after the final failed attempt.
    """
    span = tracing.start_gpu_span(f"gpu:{path.strip('/')}")
    started = _monotonic()
    cold = _is_cold()
    # Filled in by _post() as it goes, so the span is accurate on failure too.
    outcome: dict = {"attempts": 0, "status": None}
    error = True
    try:
        data = _post(path, json, outcome)
        error = False
        return data
    finally:
        tracing.end_gpu_span(
            span,
            error=error,
            metadata={
                "n_items": len(json.get("texts") or json.get("documents") or ()),
                "latency_ms": round((_monotonic() - started) * 1000, 1),
                "cold": cold,
                "attempts": outcome["attempts"],
                "status": outcome["status"],
            },
        )


def _post(path: str, json: dict, outcome: dict) -> dict:
    global _last_ok
    client = _get_client()
    headers = {"X-API-Key": settings.gpu_api_key}
    attempts = 1 + max(0, settings.gpu_max_retries)
    last_error: GPUServiceError | None = None

    for attempt in range(attempts):
        outcome["attempts"] = attempt + 1
        timeout = settings.gpu_cold_timeout_seconds if _is_cold() else settings.gpu_timeout_seconds
        try:
            resp = client.post(path, json=json, headers=headers, timeout=timeout)
        except _RETRYABLE_TRANSPORT_ERRORS as e:
            outcome["status"] = type(e).__name__
            last_error = GPUServiceError(f"GPU service {path}: {type(e).__name__}")
        except httpx.HTTPError as e:
            outcome["status"] = type(e).__name__
            raise GPUServiceError(f"GPU service {path}: {type(e).__name__}") from e
        else:
            outcome["status"] = resp.status_code
            if resp.status_code < 400:
                try:
                    data = resp.json()
                except ValueError as e:
                    raise GPUServiceError(
                        f"GPU service {path}: invalid JSON response", resp.status_code
                    ) from e
                _last_ok = _monotonic()
                return data
            if not _retryable_status(resp.status_code):
                # The router turns this into a bare 503, so without this line
                # a bad key or payload would leave no trace in the logs.
                error = GPUServiceError(
                    f"GPU service {path}: HTTP {resp.status_code}", resp.status_code
                )
                logger.error("%s -- not retryable, giving up", error)
                raise error
            last_error = GPUServiceError(
                f"GPU service {path}: HTTP {resp.status_code}", resp.status_code
            )

        if attempt < attempts - 1:
            delay = _BACKOFF_BASE_SECONDS * (2 ** attempt)
            logger.warning(
                "%s (attempt %d/%d), retrying in %.1fs",
                last_error, attempt + 1, attempts, delay,
            )
            _sleep(delay)

    assert last_error is not None
    logger.error("%s -- giving up after %d attempt(s)", last_error, attempts)
    raise last_error
