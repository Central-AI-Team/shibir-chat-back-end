"""Request dependencies: anonymous owner identity and the Origin (CSRF) check.

There are no accounts. The server issues each browser a random token in the
`shibir_owner` cookie; chat sessions belong to the hash of that token and are
invisible to everyone else. Only sha256(token) is ever stored or compared, so a
dump of the in-memory store holds nothing that can be replayed as a cookie.

This is deliberately plain dependencies, not @app.middleware("http"): tracing
keeps ContextVars across the streaming path and BaseHTTPMiddleware can break
that. The catch: a route that returns its OWN Response (StreamingResponse, the
204 on DELETE) drops headers set on the injected `response`, so those routes
call apply_owner_cookie() on what they return.
"""

from __future__ import annotations

import hashlib
import re
import secrets

from fastapi import HTTPException, Request, Response

from app.core.config import settings

COOKIE_NAME = "shibir_owner"
COOKIE_MAX_AGE = 30 * 24 * 3600  # 30 days
_TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]{43}$")  # token_urlsafe(32)
_NEW_TOKEN_ATTR = "new_owner_token"


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _set_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        COOKIE_NAME,
        token,
        max_age=COOKIE_MAX_AGE,
        path="/",
        httponly=True,
        samesite="lax",
        secure=settings.cookie_secure,
    )


def get_owner(request: Request, response: Response) -> str:
    """The caller's owner hash. Mints a token (and sets the cookie) when the
    incoming cookie is missing or not in the exact format we issue."""
    token = request.cookies.get(COOKIE_NAME)
    if token is not None and _TOKEN_RE.fullmatch(token):
        return hash_token(token)
    token = secrets.token_urlsafe(32)
    setattr(request.state, _NEW_TOKEN_ATTR, token)
    _set_cookie(response, token)
    return hash_token(token)


def apply_owner_cookie(request: Request, response: Response) -> Response:
    """Set the cookie on a Response the route built itself, if get_owner minted
    one for this request. No-op otherwise (an existing cookie is not reissued)."""
    token = getattr(request.state, _NEW_TOKEN_ATTR, None)
    if token is not None:
        _set_cookie(response, token)
    return response


def check_origin(request: Request) -> None:
    """CSRF guard for state-changing routes: a browser always sends Origin on a
    cross-site POST/DELETE, so reject any that is not on the allow-list.
    Requests with no Origin (curl, server-to-server) pass; they only ever get
    their own anonymous identity.

    Same-origin requests (Origin == this request's own scheme://host, e.g. the
    /docs "Try it out" page) also pass: they are not cross-site, so not CSRF.
    Host is client-supplied, but a page that rebinds its own domain to this
    server only gets that domain's cookies, i.e. a fresh anonymous identity."""
    origin = request.headers.get("origin")
    if origin is None or origin in settings.cors_allow_origins:
        return
    if origin == f"{request.url.scheme}://{request.url.netloc}":
        return
    raise HTTPException(status_code=403, detail="origin not allowed")


def trace_ids(owner: str, session_id: str) -> tuple[str, str]:
    """(user_id, session_id) safe to send to Langfuse: a prefix of the owner
    hash and a hash of the session id, never the real session id or owner."""
    return owner[:12], hashlib.sha256(session_id.encode()).hexdigest()[:16]
