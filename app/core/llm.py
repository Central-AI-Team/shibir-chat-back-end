"""Shared chat-completions client.

Gemini (via its OpenAI-compat endpoint) and OpenAI both speak the same
chat-completions API, so one `openai.OpenAI` client covers either -- only
api_key/base_url/model differ. Switch providers with LLM_PROVIDER in .env;
callers just use get_client() / get_model() instead of hardcoding either.
"""

from __future__ import annotations

from functools import lru_cache

from openai import OpenAI

from app.core.config import settings


@lru_cache(maxsize=1)
def get_client() -> OpenAI:
    if settings.llm_provider == "openai":
        return OpenAI(api_key=settings.openai_api_key)
    return OpenAI(api_key=settings.gemini_api_key, base_url=settings.gemini_base_url)


def get_model() -> str:
    if settings.llm_provider == "openai":
        return settings.openai_model
    return settings.gemini_model
