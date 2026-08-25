"""Shared chat-completions client.

Callers use get_client() / get_model() instead of constructing an OpenAI
client directly, so the api_key/model live in one place.
"""

from __future__ import annotations

from functools import lru_cache

from openai import OpenAI

from app.core.config import settings


@lru_cache(maxsize=1)
def get_client() -> OpenAI:
    return OpenAI(api_key=settings.openai_api_key)


def get_model() -> str:
    return settings.openai_model
