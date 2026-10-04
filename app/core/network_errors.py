"""Detect "the network/internet is down or unreachable" failures.

Lives in tracked code (not in the local-only timing module) because the /chat
router uses it to return a clean 503 instead of a 500 stack trace, whether or
not timing is enabled.
"""

from __future__ import annotations

import socket

import httpx
import openai

NETWORK_ERRORS: tuple[type[BaseException], ...] = (
    openai.APIConnectionError,
    openai.APITimeoutError,
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.ReadTimeout,
    socket.gaierror,
    ConnectionError,
    TimeoutError,
)

NETWORK_UNAVAILABLE_DETAIL = (
    "Network/internet problem: could not reach the AI service. "
    "Please check the connection and try again."
)


def is_network_error(exc: BaseException | None) -> bool:
    """True if `exc` or anything in its __cause__/__context__ chain is a network error.

    The chain walk matters: gpu_client wraps httpx failures in GPUServiceError
    and the router wraps everything in HTTPException, so the transport error is
    rarely the outermost exception.
    """
    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        if isinstance(exc, NETWORK_ERRORS):
            return True
        seen.add(id(exc))
        exc = exc.__cause__ or exc.__context__
    return False
