"""Langfuse tracing -- the ONLY module in the app that imports the langfuse SDK.

Pinned to ``langfuse==4.15.4`` (the OpenTelemetry-based "observations-first"
API: ``Langfuse().start_observation()`` / ``start_as_current_observation()``,
no more ``trace()``/``span()``/``generation()``, plus the ``langfuse.openai``
drop-in wrapped client). Migrated from the v2 "manual client" API on
2026-09-19: Langfuse Cloud moved this project's org onto its v4 backend,
which sunsets the v2 SDK's legacy ingestion path entirely on 2026-11-16 --
this was not optional modernization, the old pin was about to silently stop
sending data at all. Do NOT downgrade without re-reading that deprecation
notice (fetch https://langfuse.com/docs/observability/sdk/upgrade-path first
-- the guidance moves).

WHY THIS FILE HOLDS EXPLICIT OBJECT REFERENCES, NOT AMBIENT CONTEXT:
  v4's "recommended" pattern is a single ``with start_as_current_observation()``
  block whose children nest via ambient OpenTelemetry context (a contextvar
  under the hood). That works fine within one thread/task -- verified live
  that anyio's ``run_in_threadpool`` copies the calling contextvars.Context
  into its worker thread, so ambient nesting survives run_in_threadpool calls
  (used throughout this pipeline to keep blocking retrieval/LLM work off the
  event loop). It does NOT survive Starlette's SSE streaming path: `gen()` in
  router.py's ``chat_stream()`` is a plain sync generator that Starlette pulls
  one ``next()`` at a time, each call getting its OWN fresh context copy from
  the ASGI response sender -- not from the previous iteration. Whatever this
  module attaches to ambient context during iteration 1 (the request's root
  span) is gone by iteration 2. So: the root span/trace id/root observation
  id are held as an explicit dict (``_current_ctx`` / the ``trace`` object
  callers pass around), never re-derived from ambient context alone, and
  ``app/core/llm.py`` always threads ``trace_id``/``parent_observation_id``
  explicitly into the one call that runs post-first-yield
  (``app/rag/generator.py``'s ``stream_answer()``). This mirrors exactly how
  the old v2 code worked around the same Starlette behavior with its ContextVar
  -- see git history on this file for that version if the shape here is
  confusing.

DESIGN RULE: observability must never harm the request path.
  * OPT-IN. With ``settings.langfuse_public_key`` / ``langfuse_secret_key``
    empty, every function here is a no-op, the langfuse SDK is never imported,
    ``app/core/llm.get_client()`` returns the plain OpenAI client, and ``/chat``
    behaves byte-identically to a build without this module.
  * Fire-and-forget. The client batches events on a background thread and
    flushes on its own timer; nothing here calls ``flush()`` on the request
    path. ``shutdown()`` (wired into the app lifespan) does the final flush.
  * Swallow everything. Every public helper is wrapped so any tracing error is
    logged at WARNING and never propagates into ``/chat``.

WHAT A TRACE STORES when enabled (all on your own infra if self-hosted):
  root span    -- raw user query, session id, optional user id, the final
                  answer, the cited sources, which mode ran, total latency
  spans        -- ``classify-intent`` (+ how it was classified), ``expand-
                  query`` (the query variants), ``retrieve-context`` (a
                  ``retriever``-typed observation covering EVERY reranked
                  candidate -- kept AND dropped -- each with a text preview +
                  cosine similarity + rerank score), ``check-relevance-gate``
                  (a ``guardrail``-typed observation: grounded vs refused, top
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

# The active request's trace context: {"root": LangfuseSpan, "root_cm": the
# start_as_current_observation() context manager (manually entered/exited --
# see module docstring), "attr_cm": the propagate_attributes() context
# manager for session_id/user_id, "trace_id": str, "root_id": str}. Made
# visible to app/core/llm.py (so every LLM call links to it) and to the
# pipeline-stage helpers below WITHOUT threading it through every service
# signature. Set in app/api/router.py at the /chat boundary; read everywhere
# else.
_current_ctx: contextvars.ContextVar[Optional[dict]] = contextvars.ContextVar(
    "langfuse_current_ctx", default=None
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
    """The one shared client. Constructing it registers it as THE Langfuse
    singleton (keyed by public_key) -- ``langfuse.openai``'s wrapped calls use
    ``get_client()`` internally with no key, which finds this instance as long
    as it was constructed before the first traced call. That's why ``init()``
    builds this eagerly at app startup instead of waiting for the first
    request."""
    if not is_enabled():
        return None
    try:
        from langfuse import Langfuse

        return Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
            release=settings.langfuse_release or None,
            environment=settings.langfuse_environment,
            mask=None if settings.langfuse_capture_io else _mask,
            # Keep a failing/unreachable Langfuse from stretching a graceful
            # shutdown's final flush (default is 20s per attempt).
            timeout=10,
        )
    except Exception:
        logger.warning("langfuse: client init failed; tracing disabled", exc_info=True)
        return None


def init() -> None:
    """Pay the one-time cost of constructing the client and importing
    ``langfuse.openai`` (~heavy: a few seconds cold) at APP STARTUP, so the
    first /chat request never eats it. Wired into the FastAPI lifespan. No-op
    when disabled; never raises."""
    if not is_enabled():
        return
    try:
        configure_openai_wrapper()
        if _client() is not None:
            logger.info(
                "langfuse: tracing enabled (host=%s, environment=%s)",
                settings.langfuse_host,
                settings.langfuse_environment,
            )
    except Exception:
        logger.warning("langfuse: init failed; tracing effectively disabled", exc_info=True)


# Set True only once ``import langfuse.openai`` has actually run -- that
# import is what monkeypatches the real openai.OpenAI/AsyncOpenAI classes'
# chat-completions methods, process-wide (unlike v2, there is no separate
# per-attribute configure step). app/core/llm.py's complete() reads this via
# openai_wrapper_active(): the langfuse-only kwargs (name/metadata/trace_id/
# parent_observation_id/...) are safe to pass ONLY when the patch is in place
# -- a plain client 400s on them. So if langfuse is enabled but the import
# somehow fails, tracing stays off for the OpenAI calls instead of breaking
# every request.
_openai_wrapper_ready = False


def openai_wrapper_active() -> bool:
    """True iff ``langfuse.openai``'s global patch is installed, so passing its
    kwargs to ``chat.completions.create`` is safe."""
    return _openai_wrapper_ready and is_enabled()


def configure_openai_wrapper() -> None:
    """Construct the shared client (so it's registered as the singleton
    ``langfuse.openai``'s wrapped methods will find via ``get_client()``),
    then import ``langfuse.openai`` to install its patch. Called by
    ``app/core/llm.py`` immediately before it constructs the wrapped OpenAI
    client, and eagerly by ``init()``. No-op when disabled."""
    global _openai_wrapper_ready
    if not is_enabled():
        return
    try:
        _client()  # must exist before the wrapper's get_client() calls resolve to it
        import langfuse.openai  # noqa: F401  (import patches OpenAI's create methods)

        _openai_wrapper_ready = True
    except Exception:
        logger.warning("langfuse: openai wrapper config failed (ignored)", exc_info=True)


# ---------------------------------------------------------------------------
# request-level trace
# ---------------------------------------------------------------------------


@_safe
def start_request_trace(
    *, name: str, query: str, session_id: str, user_id: str | None = None
) -> Optional[dict]:
    """Open the root span for a /chat request and stash its context in the
    ContextVar so every nested LLM call and pipeline stage groups under it.
    Returns the context dict (or None when disabled) -- non-streaming callers
    ignore it and use the helpers below; the streaming endpoint holds onto it
    explicitly and passes it back into finalize_request_trace() (see the
    module docstring for why)."""
    client = _client()
    if client is None:
        return None
    from langfuse import propagate_attributes

    root_cm = client.start_as_current_observation(
        as_type="span", name=name, input={"message": query}
    )
    root = root_cm.__enter__()
    attr_cm = propagate_attributes(session_id=session_id, user_id=user_id)
    attr_cm.__enter__()
    ctx = {
        "root": root,
        "root_cm": root_cm,
        "attr_cm": attr_cm,
        "trace_id": root.trace_id,
        "root_id": root.id,
    }
    _current_ctx.set(ctx)
    return ctx


def current_trace() -> Optional[dict]:
    return _current_ctx.get()


def current_trace_id() -> Optional[str]:
    """The active trace id, for app/core/llm.py to link generations to. Cheap;
    safe to call on every LLM call."""
    ctx = _current_ctx.get()
    return ctx["trace_id"] if ctx else None


def current_parent_observation_id() -> Optional[str]:
    """The active root span's own observation id, so LLM-call generations
    nest under it instead of attaching flat at the trace level."""
    ctx = _current_ctx.get()
    return ctx["root_id"] if ctx else None


def trace_id_of(trace: Optional[dict]) -> Optional[str]:
    """The trace id of an explicitly-held context dict. For callers that hold
    the context explicitly rather than via the ContextVar -- see
    finalize_request_trace's ``trace`` argument."""
    return trace["trace_id"] if trace else None


def parent_observation_id_of(trace: Optional[dict]) -> Optional[str]:
    """The root span's observation id of an explicitly-held context dict."""
    return trace["root_id"] if trace else None


@_safe
def finalize_request_trace(
    *,
    trace: Optional[dict] = None,
    answer: str,
    sources: list | None,
    mode: str,
    response_time_ms: float | None,
    error: bool = False,
) -> None:
    """Attach the request outcome to the root span and close it. Called from
    a ``finally`` at the /chat boundary, so it also runs on the error paths.

    ``trace`` may be passed explicitly by a caller whose control flow loses
    the ContextVar before the ``finally`` runs. The streaming endpoint is
    exactly that case -- see the module docstring. Non-streaming callers omit
    this and the ContextVar is used as before.

    ``mode`` goes into metadata, not a tag: it's only known once intent
    classification finishes, and Langfuse's own guidance is that tags are for
    dimensions known at observation-creation time -- metadata is the
    documented escape hatch for exactly this case. ``error`` sets the
    observation ``level`` instead of an ad hoc tag, which is what Langfuse's
    UI natively filters/colors by.
    """
    ctx = trace if trace is not None else _current_ctx.get()
    if ctx is None:
        return
    root = ctx["root"]
    root.update(
        output={"answer": answer, "sources": _sources_view(sources), "mode": mode},
        metadata={"response_time_ms": response_time_ms, "mode": mode},
        level="ERROR" if error else "DEFAULT",
    )
    for key in ("attr_cm", "root_cm"):
        cm = ctx.get(key)
        if cm is None:
            continue
        try:
            cm.__exit__(None, None, None)
        except Exception:
            logger.warning("langfuse: closing %s failed (ignored)", key, exc_info=True)


def clear_request_trace() -> None:
    """Drop the ContextVar reference. Matters for the streaming path, whose
    generator runs on a pooled worker thread that outlives the request."""
    _current_ctx.set(None)


# ---------------------------------------------------------------------------
# pipeline stages -- one thin child observation each, all no-op when there is
# no active trace. Each nests under the request's root span via AMBIENT
# OpenTelemetry context (verified live: this survives anyio's
# run_in_threadpool, and every call site here runs before the streaming
# path's first `yield` -- see the module docstring for the boundary that
# does NOT survive and how app/core/llm.py handles it instead).
# ---------------------------------------------------------------------------


@_safe
def record_intent(intent: str, method: str) -> None:
    """method: "regex" | "llm" | "session" (an ongoing roleplay thread)."""
    if _current_ctx.get() is None:
        return
    client = _client()
    if client is None:
        return
    client.start_observation(
        as_type="span",
        name="classify-intent",
        output={"intent": intent, "classified_by": method},
    ).end()


@_safe
def record_rewrite(variants) -> None:
    if _current_ctx.get() is None:
        return
    client = _client()
    if client is None:
        return
    variants = list(variants)
    client.start_observation(
        as_type="span",
        name="expand-query",
        output={"variants": variants, "n_variants": len(variants)},
    ).end()


@_safe
def start_gpu_span(name: str) -> Optional[Any]:
    """Open a span around one app/rag/gpu_client.post() call ("gpu:embed" /
    "gpu:rerank"). Returns the span, or None when there is no active trace
    (tracing disabled, or outside a request, e.g. app.rag.ingest)."""
    if _current_ctx.get() is None:
        return None
    client = _client()
    if client is None:
        return None
    return client.start_observation(as_type="span", name=name)


@_safe
def end_gpu_span(span: Optional[Any], *, metadata: dict, error: bool = False) -> None:
    """Close a start_gpu_span() span. ``metadata`` is counts and timings only
    -- never the texts / documents sent to the service."""
    if span is None:
        return
    span.update(metadata=metadata, level="ERROR" if error else "DEFAULT")
    span.end()


@_safe
def record_retrieval(chunks, *, top_k: int | None = None, name: str = "retrieve-context") -> None:
    """One ``retriever``-typed observation covering the whole retrieve+rerank
    step.

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
    if _current_ctx.get() is None:
        return
    client = _client()
    if client is None:
        return
    chunks = list(chunks)
    client.start_observation(
        as_type="retriever",
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
    """A ``guardrail``-typed observation: this is the pass/refuse decision
    that keeps an ungrounded answer from being cited (see CLAUDE.md's
    "Prompt contract")."""
    if _current_ctx.get() is None:
        return
    client = _client()
    if client is None:
        return
    client.start_observation(
        as_type="guardrail",
        name="check-relevance-gate",
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
    client.create_score(
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
