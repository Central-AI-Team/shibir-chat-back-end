"""Request dependencies: caller identity, access levels and the Origin (CSRF) check.

Callers are guests or logged-in users (app/auth/, docs/auth.md). A valid
login-session cookie makes the caller a user whose chat sessions belong to
principal.user_owner_key(user_id); everyone else is a guest, identified as below.

Guests: the server issues each browser a random token in the
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
from functools import lru_cache

from fastapi import Depends, HTTPException, Request, Response
from sqlalchemy.exc import SQLAlchemyError

from app.auth import audit, open_db
from app.auth import principal as principals
from app.auth import sessions as auth_sessions
from app.auth.principal import Principal
from app.core.config import settings
from app.services import rate_limiter

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


# ---------------------------------------------------------------------------
# Accounts (app/auth/). A valid login-session cookie wins; otherwise the guest
# owner-cookie flow above runs exactly as before. Rejections are generic: the
# reason a session was refused goes to the audit log only.
# ---------------------------------------------------------------------------

SERVICE_UNAVAILABLE_DETAIL = "সার্ভিস সাময়িকভাবে অনুপলব্ধ। কিছুক্ষণ পর আবার চেষ্টা করুন।"
AUTH_REQUIRED_DETAIL = "authentication required"
FORBIDDEN_DETAIL = "forbidden"
AUTH_SESSION_ATTR = "auth_session_id"


def public_route() -> None:
    """Marker dependency for routes whose policy level is 'public'
    (app/auth/policy.py); it checks nothing."""


def peek_owner(request: Request) -> str | None:
    """The guest owner hash from a well-formed owner cookie, without minting one."""
    token = request.cookies.get(COOKIE_NAME)
    if token is not None and _TOKEN_RE.fullmatch(token):
        return hash_token(token)
    return None


def current_user(request: Request) -> Principal | None:
    """The logged-in user's principal, or None. Never mints a guest cookie.
    A database failure is a 503 (fail closed), never a silent guest fallback."""
    token = auth_sessions.request_token(request)
    if token is None:
        if auth_sessions.has_session_cookie(request):
            audit.event("session_rejected", ip=_client_ip(request), reason="malformed")
        return None
    try:
        with open_db() as db:
            session, user = auth_sessions.lookup(db, token)
            if session is None:
                audit.event("session_rejected", ip=_client_ip(request), reason=user)
                return None
            setattr(request.state, AUTH_SESSION_ATTR, session.id)
            return principals.user(user.id, user.role)
    except SQLAlchemyError as e:
        raise HTTPException(status_code=503, detail=SERVICE_UNAVAILABLE_DETAIL) from e


def get_principal(
    request: Request, response: Response, user: Principal | None = Depends(current_user)
) -> Principal:
    """Policy level 'guest_ok': the user if logged in, else the guest owner."""
    if user is not None:
        return user
    return principals.guest(get_owner(request, response))


def require_user(request: Request, user: Principal | None = Depends(current_user)) -> Principal:
    """Policy level 'user': 401 for anyone not logged in."""
    if user is None:
        raise HTTPException(status_code=401, detail=AUTH_REQUIRED_DETAIL)
    return user


@lru_cache(maxsize=None)
def require_role(role: str):
    """Policy level for `role` (e.g. "admin"): 401 if not logged in, 403 if
    logged in without the role. Cached so each role is one stable dependency."""

    def dependency(request: Request, user: Principal = Depends(require_user)) -> Principal:
        if user.role != role:
            audit.event("forbidden_access", ip=_client_ip(request), user_id=user.user_id, path=request.url.path)
            raise HTTPException(status_code=403, detail=FORBIDDEN_DETAIL)
        return user

    dependency.required_role = role
    return dependency


def check_session_origin(request: Request) -> None:
    """On top of check_origin, for auth routes: a POST/DELETE that carries a
    login-session cookie must also carry Origin (browsers always send it on
    those), so a cookie-bearing request from a non-browser context that strips
    Origin cannot change account state."""
    if (
        request.method in ("POST", "DELETE")
        and auth_sessions.has_session_cookie(request)
        and "origin" not in request.headers
    ):
        raise HTTPException(status_code=403, detail="origin required")


def require_json(request: Request) -> None:
    """Auth request bodies are JSON only (no form posts)."""
    content_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if content_type != "application/json":
        raise HTTPException(status_code=415, detail="content type must be application/json")


def too_many_requests(detail: str, retry_after: int) -> HTTPException:
    return HTTPException(status_code=429, detail=detail, headers={"Retry-After": str(retry_after)})


def _client_ip(request: Request) -> str:
    # uvicorn --proxy-headers already rewrote this from X-Forwarded-For, trusting
    # it only from FORWARDED_ALLOW_IPS; never read that header here.
    return request.client.host if request.client else "unknown"


def limit_chat(request: Request, principal: Principal = Depends(get_principal)) -> None:
    """POST /chat, /chat/stream: per-IP, then per-owner, then (users) the
    per-user daily cap, then the global daily budget, so a request rejected by
    a limit is not counted against the budget. Runs before the route body, so
    a rejected request creates no session and reaches no LLM. get_principal is
    cached per request: the route sees this principal."""
    retry = rate_limiter.chat_per_ip.hit(_client_ip(request), settings.rate_limit_per_ip_per_minute)
    if retry is None:
        retry = rate_limiter.chat_per_owner.hit(principal.owner_key, settings.rate_limit_per_owner_per_minute)
    if retry is not None:
        raise too_many_requests(rate_limiter.RATE_LIMIT_DETAIL, retry)
    if principal.is_user:
        retry = rate_limiter.user_daily.hit(principal.user_id, settings.user_daily_message_limit)
        if retry is not None:
            raise too_many_requests(rate_limiter.USER_DAILY_LIMIT_DETAIL, retry)
    retry = rate_limiter.daily_budget.hit(settings.daily_llm_request_budget)
    if retry is not None:
        raise too_many_requests(rate_limiter.DAILY_BUDGET_DETAIL, retry)


def limit_reads(request: Request) -> None:
    """GET /conversations, GET /conversations/{id}/messages, DELETE: per IP."""
    retry = rate_limiter.read_per_ip.hit(_client_ip(request), settings.rate_limit_read_per_ip_per_minute)
    if retry is not None:
        raise too_many_requests(rate_limiter.RATE_LIMIT_DETAIL, retry)


def trace_ids(owner: str, session_id: str) -> tuple[str, str]:
    """(user_id, session_id) safe to send to Langfuse: a prefix of the owner
    hash and a hash of the session id, never the real session id or owner."""
    return owner[:12], hashlib.sha256(session_id.encode()).hexdigest()[:16]
