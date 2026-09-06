"""Shared chat-completions client + per-task model routing.

Callers use get_client() / get_model() instead of constructing an OpenAI
client directly, so the api_key/model live in one place.

PER-TASK ROUTING: get_model(task) and complete(task, messages, **overrides)
are how a call site asks for "whichever model this task is routed to" instead
of hardcoding the single default. See app/core/config.settings.model_by_task
for the routing table itself (which task -> which model string) and its
docstring for how/when to change it -- this file only holds the MECHANISM
(the lookup + per-model param translation), never model choices.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache

from openai import OpenAI

from app.core import tracing
from app.core.config import settings

logger = logging.getLogger(__name__)


def _openai_class():
    """Return the OpenAI client class to instantiate.

    With tracing enabled this is langfuse.openai's drop-in wrapped class --
    every call made through it is auto-recorded as a Langfuse `generation`
    (model, messages, completion, token usage, cost, latency), and links to
    the active request trace via the `trace_id` kwarg complete() passes.

    With tracing disabled (no keys) this is the plain `openai.OpenAI` -- the
    returned client is byte-identical to today's, and no langfuse code runs.
    """
    if tracing.is_enabled():
        try:
            tracing.configure_openai_wrapper()
            from langfuse.openai import OpenAI as _WrappedOpenAI

            return _WrappedOpenAI
        except Exception:  # never let tracing break client construction
            logger.warning(
                "langfuse: wrapped OpenAI client unavailable; using plain client",
                exc_info=True,
            )
    return OpenAI


@lru_cache(maxsize=1)
def get_client() -> OpenAI:
    return _openai_class()(api_key=settings.openai_api_key)


def get_model(task: str | None = None) -> str:
    """task=None (default) -- today's exact behavior; every pre-existing bare
    get_model() call site (if any remain) is unaffected. A task name looks it
    up in settings.model_by_task, falling back to the single default model
    for any task not (yet) assigned there -- so an unconfigured or misspelled
    task name never breaks a caller, it just isn't routed anywhere special
    yet.
    """
    if task is None:
        return settings.openai_model
    return settings.model_by_task.get(task, settings.openai_model)


@lru_cache(maxsize=8)
def get_client_for(base_url: str | None = None, api_key: str | None = None) -> OpenAI:
    """Client pointed at an arbitrary OpenAI-compatible endpoint (e.g. Groq).

    Purely additive -- get_client() above is untouched. Groq's API is
    OpenAI-compatible, so the same `openai` client works against it with just
    a different base_url/api_key; this is how complete() below (and
    scripts/eval_generation_ab.py) reach a non-OpenAI model without a second
    HTTP library.

    base_url=None, api_key=None returns the same cached client as
    get_client() (same cache key, both None) -- so a caller that always
    passes explicit values for other models never accidentally diverges
    from production's client for the baseline model.
    """
    if base_url is None and api_key is None:
        return get_client()
    return _openai_class()(base_url=base_url, api_key=api_key)


# ---------------------------------------------------------------------------
# Per-model call-parameter adapter.
#
# Different models take different knobs for the same concept: gpt-5-mini and
# gpt-5 are reasoning models (fixed temperature -- passing any other value is
# a 400 -- and max_completion_tokens, not max_tokens, for a token cap; see
# app/rag/generator.py's long-standing comment on this). Groq's models take
# max_tokens regardless of family -- but NOTE the openai/gpt-oss-* models
# hosted there are ALSO reasoning models under the hood (confirmed live:
# completion_tokens_details.reasoning_tokens is nonzero even for a one-line
# Bengali greeting) -- they just take that budget through Groq's max_tokens
# param, not a separate name. A call site should never need to know any of
# this -- it calls complete(task, messages, token_budget=N) and gets the
# right param name for whatever model that task is currently routed to.
#
# CONFIRMED LIVE (Sept 2026) against this project's GROQ_API_KEY:
# llama-3.3-70b-versatile and llama-3.1-8b-instant both 404 ("does not exist
# or you do not have access to it") despite being listed as self-serve on
# Groq's own docs page -- docs and dashboards can lag actual entitlement;
# trust a live call over documentation before relying on a model. Do not
# re-add either without reconfirming access.
#
# Only `provider` (which client/base_url) and `token_param` (the literal
# kwarg name for a token cap) are adapter FACTS. They are NOT here to impose
# a token cap that didn't exist before -- complete() only sets token_param
# when a caller explicitly passes token_budget, so a call site that
# currently passes no cap at all keeps not passing one, for any model.
#
# Add an entry here whenever a genuinely new model enters
# settings.model_by_task and it isn't one of the OpenAI reasoning models or
# llama-3.3-70b-versatile already covered. This table is the reason
# scripts/eval_generation_ab.py doesn't duplicate provider/token-param facts
# for the same models -- it imports MODEL_ADAPTERS from here.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelAdapter:
    provider: str  # "openai" | "groq"
    token_param: str  # kwarg name a token_budget= becomes for this model


MODEL_ADAPTERS: dict[str, ModelAdapter] = {
    "gpt-5-mini": ModelAdapter(provider="openai", token_param="max_completion_tokens"),
    "gpt-5": ModelAdapter(provider="openai", token_param="max_completion_tokens"),
    "gpt-5-nano": ModelAdapter(provider="openai", token_param="max_completion_tokens"),
    "openai/gpt-oss-20b": ModelAdapter(provider="groq", token_param="max_tokens"),
    "openai/gpt-oss-120b": ModelAdapter(provider="groq", token_param="max_tokens"),
    # Not a model_by_task candidate today -- scripts/eval_generation_ab.py's
    # judge model. Listed here anyway so _provider_for()-style lookups (that
    # script derives provider from THIS table, not a private copy) never
    # silently default a Groq-hosted model to the wrong provider.
    "qwen/qwen3.6-27b": ModelAdapter(provider="groq", token_param="max_tokens"),
}
# Any model not listed above (e.g. a typo, or a new one not yet added here)
# is treated as an OpenAI reasoning model -- today's only real production
# family, and the safest assumption (uses the existing default client).
_DEFAULT_ADAPTER = ModelAdapter(provider="openai", token_param="max_completion_tokens")


def _client_for_provider(provider: str) -> OpenAI:
    if provider == "openai":
        return get_client()
    if provider == "groq":
        if not settings.groq_api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set (.env) -- required because a task "
                "in settings.model_by_task is routed to a Groq model."
            )
        return get_client_for(base_url=settings.groq_base_url, api_key=settings.groq_api_key)
    raise ValueError(f"unknown provider: {provider!r}")


def complete(task: str, messages: list[dict], *, token_budget: int | None = None, **overrides):
    """Chat-completion for `task`'s routed model, with that model's own
    param quirks (provider/client, token-cap kwarg name) applied
    automatically -- the caller never needs to know them.

    token_budget, if given, becomes whichever literal kwarg (max_tokens vs
    max_completion_tokens) the resolved model actually needs -- pass this
    instead of hardcoding a param name, so a call site's carefully-chosen
    budget (e.g. note_service's map/reduce sizes) survives that task later
    being routed to a different model family. Any other override (e.g.
    temperature, response_format) passes straight through.
    """
    model = get_model(task)
    adapter = MODEL_ADAPTERS.get(model, _DEFAULT_ADAPTER)
    client = _client_for_provider(adapter.provider)
    params = dict(overrides)
    if token_budget is not None:
        params.setdefault(adapter.token_param, token_budget)

    # langfuse.openai's global patch consumes these and strips them before the
    # real OpenAI call (so per-task cost/latency is filterable and the
    # generation nests under the current /chat trace). A plain, unpatched
    # client 400s on them. Guard on the patch actually being installed --
    # NOT merely on tracing being enabled -- so an enabled-but-broken langfuse
    # (import failed) can never turn every LLM call into a 400. When the patch
    # is absent, strip any that a caller passed explicitly (e.g. stream_answer
    # forwarding a trace_id).
    _LANGFUSE_ONLY = ("name", "metadata", "trace_id", "session_id", "user_id", "tags")
    if tracing.openai_wrapper_active():
        params.setdefault("name", f"llm:{task}")
        params.setdefault("metadata", {"task": task, "model": model})
        trace_id = params.get("trace_id") or tracing.current_trace_id()
        if trace_id:
            params["trace_id"] = trace_id
    else:
        for key in _LANGFUSE_ONLY:
            params.pop(key, None)
    return client.chat.completions.create(model=model, messages=messages, **params)
