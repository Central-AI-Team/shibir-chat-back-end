"""Langfuse tracing -- the ONLY module in the app that imports the langfuse SDK.

Pinned to ``langfuse==2.60.10`` (the v2 "manual client" API: ``Langfuse()`` ->
``trace.span()`` / ``trace.generation()``, plus the ``langfuse.openai`` drop-in
wrapped client). Langfuse v3/v4 moved to an OpenTelemetry / ``@observe`` model
with a different surface -- do NOT bump the pin without rewriting this file and
the wiring in ``app/core/llm.py``.

DESIGN RULE: observability must never harm the request path.
  * OPT-IN. With ``settings.langfuse_public_key`` / ``langfuse_secret_key``
    empty, every function here is a no-op, the langfuse SDK is never imported,
    ``app/core/llm.get_client()`` returns the plain OpenAI client, and ``/chat``
    behaves byte-identically to a build without this module.
  * Fire-and-forget. The v2 client batches events on a background thread and
    flushes on its own timer; nothing here calls ``flush()`` on the request
    path. ``shutdown()`` (wired into the app lifespan) does the final flush.
  * Swallow everything. Every public helper is wrapped so any tracing error is
    logged at WARNING and never propagates into ``/chat``.

WHAT A TRACE STORES when enabled (all on your own infra if self-hosted):
  trace        -- raw user query, session id, optional user id, the final
                  answer, the cited sources, which mode ran, total latency
  spans        -- ``intent`` (+ how it was classified), ``rewrite`` (the query
                  variants), ``retrieve`` (EVERY reranked candidate -- the ones
                  kept AND the ones dropped -- each with a text preview + cosine
                  similarity + rerank score), ``gate`` (grounded vs refused, top
                  rerank score, threshold)
  generations  -- every LLM call routed through ``app/core/llm.py``: model,
                  messages, completion, token usage, cost, latency, task label
Set ``LANGFUSE_CAPTURE_IO=false`` to redact every text field (query, excerpts,
prompts, completions, answer) while keeping structure, scores, usage and cost.

API keys are never placed in trace/span metadata by this module.
"""

from __future__ import annotations

import contextvars
import functools
import logging
from typing import Any, Callable, Optional, TypeVar

from app.core.config import settings

logger = logging.getLogger(__name__)

# How much of each retrieved chunk's text to keep in the trace preview.
_PREVIEW_CHARS = 240

# The active request trace. Made visible to app/core/llm.py (so every LLM call
# links to it) and to the pipeline-stage helpers below WITHOUT threading a
# trace argument through any service signature. Set in app/api/router.py at the
# /chat boundary; read everywhere else. ContextVars propagate into
# starlette.run_in_threadpool workers (verified on this Starlette/anyio), so the
# threadpool-dispatched pipeline stages see the same trace object.
_current_trace: contextvars.ContextVar[Optional[Any]] = contextvars.ContextVar(
    "langfuse_current_trace", default=None
)

F = TypeVar("F", bound=Callable[..., Any])


def is_enabled() -> bool:
    """True only when tracing is switched on AND both keys are configured.

    No keys -> False -> the whole module is inert.
    """
    return bool(
        settings.langfuse_enabled
        and settings.langfuse_public_key
        and settings.langfuse_secret_key
    )


def _safe(fn: F) -> F:
    """Wrap a helper so a disabled state is a cheap no-op and any tracing
    failure is logged, never raised into the request path."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if not is_enabled():
            return None
        try:
            return fn(*args, **kwargs)
        except Exception:  # never let observability break a request
            logger.warning("langfuse: %s failed (ignored)", fn.__name__, exc_info=True)
            return None

    return wrapper  # type: ignore[return-value]


def _mask(*, data: Any) -> Any:
    """Client-level mask -- langfuse applies this to every input/output/metadata
    field when LANGFUSE_CAPTURE_IO=false."""
    return "<redacted>"


@functools.lru_cache(maxsize=1)
def _client() -> Optional[Any]:
    """The one shared v2 client. This is also the exact instance the
    ``langfuse.openai`` integration reuses (it fetches the same
    ``LangfuseSingleton``), so traces, generations and scores all flow through
    a single queue / flush thread."""
    if not is_enabled():
        return None
    try:
        from langfuse.openai import LangfuseSingleton

        return LangfuseSingleton().get(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
            release=settings.langfuse_release or None,
            mask=None if settings.langfuse_capture_io else _mask,
            # Keep a failing/unreachable Langfuse from stretching a graceful
            # shutdown's final flush (default is 20s per attempt).
            timeout=10,
            sdk_integration="shibir-chat",
        )
    except Exception:
        logger.warning("langfuse: client init failed; tracing disabled", exc_info=True)
        return None


def init() -> None:
    """Pay the one-time cost of importing ``langfuse.openai`` (~heavy: a few
    seconds cold) and building the client at APP STARTUP, so the first /chat
    request never eats it. Wired into the FastAPI lifespan. No-op when
    disabled; never raises."""
    if not is_enabled():
        return
    try:
        configure_openai_wrapper()
        if _client() is not None:
            logger.info("langfuse: tracing enabled (host=%s)", settings.langfuse_host)
    except Exception:
        logger.warning("langfuse: init failed; tracing effectively disabled", exc_info=True)


# Set True only once ``import langfuse.openai`` has actually run and patched
# ``openai``'s chat-completions method (the import is what installs the wrapper,
# process-wide). app/core/llm.py's complete() reads this via
# openai_wrapper_active(): the langfuse-only kwargs (name/metadata/trace_id) are
# safe to pass ONLY when the patch is in place -- a plain client 400s on them.
# So if langfuse is enabled but the import somehow fails, tracing stays off for
# the OpenAI calls instead of breaking every request.
_openai_wrapper_ready = False


def openai_wrapper_active() -> bool:
    """True iff ``langfuse.openai``'s global patch is installed, so passing its
    kwargs to ``chat.completions.create`` is safe."""
    return _openai_wrapper_ready and is_enabled()


def configure_openai_wrapper() -> None:
    """Point ``langfuse.openai``'s drop-in client at our config. Called by
    ``app/core/llm.py`` immediately before it constructs the wrapped OpenAI
    client. No-op when disabled."""
    global _openai_wrapper_ready
    if not is_enabled():
        return
    try:
        import openai

        import langfuse.openai  # noqa: F401  (import injects openai.langfuse_* attrs + patches create)

        openai.langfuse_public_key = settings.langfuse_public_key
        openai.langfuse_secret_key = settings.langfuse_secret_key
        openai.langfuse_host = settings.langfuse_host
        openai.langfuse_enabled = True
        openai.langfuse_mask = None if settings.langfuse_capture_io else _mask
        # Build the shared singleton NOW so the wrapper's own initialize()
        # returns this instance instead of building a second client.
        _client()
        _openai_wrapper_ready = True
    except Exception:
        logger.warning("langfuse: openai wrapper config failed (ignored)", exc_info=True)


# ---------------------------------------------------------------------------
# request-level trace
# ---------------------------------------------------------------------------


@_safe
def start_request_trace(
    *, name: str, query: str, session_id: str, user_id: str | None = None
) -> Optional[Any]:
    """Open one trace for a /chat request and stash it in the ContextVar so
    every nested LLM call and pipeline stage groups under it. Returns the
    trace (or None when disabled) -- callers ignore it and use the helpers
    below."""
    client = _client()
    if client is None:
        return None
    trace = client.trace(
        name=name,
        input={"message": query},
        session_id=session_id,
        user_id=user_id,
        release=settings.langfuse_release or None,
    )
    _current_trace.set(trace)
    return trace


def current_trace() -> Optional[Any]:
    return _current_trace.get()


def current_trace_id() -> Optional[str]:
    """The active trace id, for app/core/llm.py to link generations to. Cheap;
    safe to call on every LLM call."""
    return trace_id_of(_current_trace.get())


def trace_id_of(trace: Optional[Any]) -> Optional[str]:
    """The id of a trace object (or None). For callers that hold the trace
    explicitly rather than via the ContextVar -- see finalize_request_trace's
    ``trace`` argument."""
    return getattr(trace, "id", None) if trace is not None else None


@_safe
def finalize_request_trace(
    *,
    trace: Optional[Any] = None,
    answer: str,
    sources: list | None,
    mode: str,
    response_time_ms: float | None,
    error: bool = False,
) -> None:
    """Attach the request outcome to the trace. Called from a ``finally`` at
    the /chat boundary, so it also runs on the error paths.

    ``trace`` may be passed explicitly by a caller whose control flow loses
    the ContextVar before the ``finally`` runs. The streaming endpoint is
    exactly that case: Starlette pulls its response generator one ``next()``
    at a time via ``anyio.to_thread.run_sync``, each call in its own copied
    context, so the trace ``set`` in the first iteration is already gone by
    the time the generator's ``finally`` executes. Non-streaming callers omit
    this and the ContextVar is used as before.
    """
    trace = trace if trace is not None else _current_trace.get()
    if trace is None:
        return
    trace.update(
        output={"answer": answer, "sources": _sources_view(sources), "mode": mode},
        metadata={"response_time_ms": response_time_ms, "error": error},
        tags=[f"mode:{mode}"] + (["error"] if error else []),
    )


def clear_request_trace() -> None:
    """Drop the ContextVar reference. Matters for the streaming path, whose
    generator runs on a pooled worker thread that outlives the request."""
    _current_trace.set(None)


# ---------------------------------------------------------------------------
# pipeline stages -- one thin span each, all no-op when there is no trace
# ---------------------------------------------------------------------------


@_safe
def record_intent(intent: str, method: str) -> None:
    """method: "regex" | "llm" | "session" (an ongoing roleplay thread)."""
    trace = _current_trace.get()
    if trace is None:
        return
    trace.span(
        name="intent", output={"intent": intent, "classified_by": method}
    ).end()


@_safe
def record_rewrite(variants) -> None:
    trace = _current_trace.get()
    if trace is None:
        return
    variants = list(variants)
    trace.span(
        name="rewrite",
        output={"variants": variants, "n_variants": len(variants)},
    ).end()


@_safe
def record_retrieval(chunks, *, top_k: int | None = None, name: str = "retrieve") -> None:
    """One span covering the whole retrieve+rerank step.

    ``chunks`` is the ENTIRE candidate pool the cross-encoder scored, best
    (rerank) first, each carrying its cosine ``similarity`` AND its
    ``rerank_score``. ``top_k`` is how many of them the user actually gets, so
    every row is tagged ``kept``.

    This is the key data for diagnosing a miss: a page that was retrieved but
    lost is right here with ``kept: false`` and both scores visible -- e.g.
    ``similarity 0.61, rerank_score 0.42`` pins the loss on the reranker, not
    on retrieval or the gate. Logging only the survivors (the old behaviour)
    hid exactly the row you needed to see.
    """
    trace = _current_trace.get()
    if trace is None:
        return
    chunks = list(chunks)
    trace.span(
        name=name,
        output={
            "n": len(chunks),
            "top_k": top_k,
            "candidates": [
                {
                    "rank": i,
                    "kept": top_k is None or i < top_k,
                    "book": getattr(c, "book", None),
                    "chapter": getattr(c, "chapter", None),
                    "source_db": getattr(c, "source_db", None),
                    "similarity": getattr(c, "similarity", None),
                    "rerank_score": getattr(c, "rerank_score", None),
                    "preview": (getattr(c, "content", "") or "")[:_PREVIEW_CHARS],
                }
                for i, c in enumerate(chunks)
            ],
        },
    ).end()


@_safe
def record_gate(*, grounded: bool, top_score, threshold) -> None:
    trace = _current_trace.get()
    if trace is None:
        return
    trace.span(
        name="gate",
        output={
            "grounded": grounded,
            "outcome": "grounded" if grounded else "refused",
            "top_rerank_score": top_score,
            "threshold": threshold,
        },
    ).end()


# ---------------------------------------------------------------------------
# scoring hook (Part C) -- connect quality signals to logged traces
# ---------------------------------------------------------------------------


@_safe
def score_trace(
    trace_id: str,
    name: str,
    value,
    *,
    comment: str | None = None,
    data_type: str | None = None,
) -> None:
    """Attach a score to a trace by id.

    Use for:
      * a future frontend thumbs-up / thumbs-down (POST the trace id the /chat
        response could carry, then call this),
      * a batch job that scores logged real queries,
      * the offline eval scripts (eval_responses / eval_ragas) emitting their
        per-question scores against traces of logged queries -- see the README
        section "Connecting evals to traces".

    ``value`` is a float for NUMERIC/BOOLEAN scores or a string for CATEGORICAL.
    ``data_type`` is one of "NUMERIC" | "CATEGORICAL" | "BOOLEAN" (or None to
    let Langfuse infer).
    """
    client = _client()
    if client is None:
        return
    client.score(
        trace_id=trace_id,
        name=name,
        value=value,
        comment=comment,
        data_type=data_type,
    )


# ---------------------------------------------------------------------------
# lifecycle
# ---------------------------------------------------------------------------


def shutdown() -> None:
    """Flush and stop the background thread. Wired into the FastAPI lifespan so
    a graceful shutdown does not drop the last batch of events."""
    if not is_enabled():
        return
    try:
        client = _client()
        if client is not None:
            client.flush()
            client.shutdown()
    except Exception:
        logger.warning("langfuse: shutdown flush failed (ignored)", exc_info=True)


def _sources_view(sources) -> list:
    out = []
    for c in sources or []:
        out.append(
            {
                "book": getattr(c, "book", None),
                "chapter": getattr(c, "chapter", None),
                "rerank_score": getattr(c, "rerank_score", None),
            }
        )
    return out
