"""HTTP client for the external GPU service (shibir-chat-gpu-service on Modal).

Used by embedder.py and reranker.py when settings.gpu_service_url is set.
The service exposes:
  POST /embed   {texts, normalize, batch_size}      -> {model, dim, vectors}
  POST /rerank  {query, documents, top_k, max_length} -> {model, results}
both authenticated with an X-API-Key header.

One shared httpx.Client per process, so connections (and TLS sessions) are
pooled across requests instead of re-handshaking on every embed/rerank call.

Timeouts: Modal scales the service to zero, so the first call in a process can
land on a cold container that still has to boot and load both models. Until
the first call succeeds we use settings.gpu_cold_timeout_seconds; after that,
settings.gpu_timeout_seconds.

Retries (exponential backoff) happen ONLY on connection-level failures, 5xx
and 429 -- those are transient. Any other 4xx (bad key, bad payload) will
fail identically on retry, so it raises immediately. A read timeout is not
retried either: the service may still be working on the request, and
retrying a slow call just multiplies the user's wait.

Never log request bodies (they are user queries / book text) or the API key.
"""

from __future__ import annotations

import logging
import threading
import time

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)

# Transient, safe-to-retry transport failures: the request never reached the
# service, or the connection was dropped (e.g. a container being replaced).
_RETRYABLE_TRANSPORT_ERRORS = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.RemoteProtocolError,
)

_BACKOFF_BASE_SECONDS = 0.5

_lock = threading.Lock()
_client: httpx.Client | None = None
_warm = False

# Indirection so tests can skip real sleeps.
_sleep = time.sleep


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
    global _client, _warm
    with _lock:
        if _client is not None:
            _client.close()
        _client = (
            httpx.Client(base_url=settings.gpu_service_url.rstrip("/"), transport=transport)
            if transport is not None
            else None
        )
        _warm = False


def _retryable_status(code: int) -> bool:
    return code == 429 or code >= 500


def post(path: str, json: dict) -> dict:
    """POST `json` to the GPU service and return the decoded JSON response.

    Raises GPUServiceError after the final failed attempt.
    """
    global _warm
    client = _get_client()
    headers = {"X-API-Key": settings.gpu_api_key}
    attempts = 1 + max(0, settings.gpu_max_retries)
    last_error: GPUServiceError | None = None

    for attempt in range(attempts):
        timeout = settings.gpu_timeout_seconds if _warm else settings.gpu_cold_timeout_seconds
        try:
            resp = client.post(path, json=json, headers=headers, timeout=timeout)
        except _RETRYABLE_TRANSPORT_ERRORS as e:
            last_error = GPUServiceError(f"GPU service {path}: {type(e).__name__}")
        except httpx.HTTPError as e:
            raise GPUServiceError(f"GPU service {path}: {type(e).__name__}") from e
        else:
            if resp.status_code < 400:
                try:
                    data = resp.json()
                except ValueError as e:
                    raise GPUServiceError(
                        f"GPU service {path}: invalid JSON response", resp.status_code
                    ) from e
                _warm = True
                return data
            if not _retryable_status(resp.status_code):
                raise GPUServiceError(
                    f"GPU service {path}: HTTP {resp.status_code}", resp.status_code
                )
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
