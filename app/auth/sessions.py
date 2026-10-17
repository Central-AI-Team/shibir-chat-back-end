"""Login sessions: opaque random tokens in a cookie, sha256(token) in Postgres.

- A token is secrets.token_urlsafe(32); only its sha256 hex is stored, so a
  database dump holds nothing that can be replayed as a cookie.
- Every login/register issues a NEW token (session-fixation defence); an
  incoming token is never reused or upgraded.
- Idle expiry slides (written at most once per SLIDE_EVERY); the absolute
  expiry is fixed at creation and never extended.
- At most MAX_SESSIONS_PER_USER live sessions per user; the oldest beyond that
  are revoked.

Cookie: "__Host-shibir_session" when settings.cookie_secure (the __Host- prefix
makes the browser require Secure, Path=/ and no Domain), else "shibir_session"
for plain-http local development. HttpOnly, SameSite=Lax, Path=/, no Domain.
It is separate from the guest `shibir_owner` cookie, and the two token spaces
never meet: a guest token is never stored in auth_sessions, so it can never
match as a login token, and a login token in the owner cookie only yields an
anonymous guest identity.
"""

from __future__ import annotations

import hashlib
import re
import secrets
from datetime import datetime, timedelta

from fastapi import Request, Response
from sqlalchemy import delete, select, update
from sqlalchemy.orm import Session

from app.auth import aware, now
from app.auth.audit import ip_prefix
from app.core.config import settings
from app.db.models import AuthSession, User

COOKIE_NAME_SECURE = "__Host-shibir_session"
COOKIE_NAME_PLAIN = "shibir_session"
MAX_SESSIONS_PER_USER = 10
SLIDE_EVERY = timedelta(minutes=5)
PURGE_BATCH = 100
_TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]{43}$")  # token_urlsafe(32)


def cookie_name() -> str:
    return COOKIE_NAME_SECURE if settings.cookie_secure else COOKIE_NAME_PLAIN


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def request_token(request: Request) -> str | None:
    """The session token the request carries, if it has the exact shape we issue."""
    token = request.cookies.get(cookie_name())
    if token is not None and _TOKEN_RE.fullmatch(token):
        return token
    return None


def has_session_cookie(request: Request) -> bool:
    return cookie_name() in request.cookies


def set_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        cookie_name(),
        token,
        max_age=settings.auth_session_absolute_days * 86400,
        path="/",
        httponly=True,
        samesite="lax",
        secure=settings.cookie_secure,
    )


def clear_cookie(response: Response) -> None:
    response.delete_cookie(
        cookie_name(), path="/", httponly=True, samesite="lax", secure=settings.cookie_secure
    )


def _purge_expired(db: Session, at: datetime) -> None:
    """Delete a bounded batch of expired rows (no scheduler; runs on login)."""
    ids = select(AuthSession.id).where(
        (AuthSession.absolute_expires_at < at) | (AuthSession.idle_expires_at < at)
    ).limit(PURGE_BATCH)
    db.execute(delete(AuthSession).where(AuthSession.id.in_(ids.scalar_subquery())))


def create(db: Session, user: User, request: Request) -> str:
    """A fresh session for `user`; returns the token for the cookie. The caller
    commits. Purges expired rows and enforces MAX_SESSIONS_PER_USER."""
    at = now()
    _purge_expired(db, at)
    token = secrets.token_urlsafe(32)
    absolute = at + timedelta(days=settings.auth_session_absolute_days)
    db.add(
        AuthSession(
            user_id=user.id,
            token_hash=hash_token(token),
            created_at=at,
            last_seen_at=at,
            idle_expires_at=min(at + timedelta(days=settings.auth_session_idle_days), absolute),
            absolute_expires_at=absolute,
            user_agent=(request.headers.get("user-agent") or "")[:200] or None,
            ip_prefix=ip_prefix(request.client.host if request.client else None),
        )
    )
    db.flush()
    stale = (
        db.execute(
            select(AuthSession.id)
            .where(AuthSession.user_id == user.id, AuthSession.revoked_at.is_(None))
            .order_by(AuthSession.created_at.desc(), AuthSession.id.desc())
            .offset(MAX_SESSIONS_PER_USER)
        )
        .scalars()
        .all()
    )
    if stale:
        db.execute(update(AuthSession).where(AuthSession.id.in_(stale)).values(revoked_at=at))
    return token


def lookup(db: Session, token: str) -> tuple[AuthSession, User] | tuple[None, str]:
    """(session, user) for a live session, sliding its idle expiry (commits
    when it writes). Otherwise (None, reason) -- the reason is for the audit
    log only and must never reach the client."""
    row = db.execute(
        select(AuthSession, User)
        .join(User, User.id == AuthSession.user_id)
        .where(AuthSession.token_hash == hash_token(token))
    ).first()
    if row is None:
        return None, "unknown"
    session, user = row
    at = now()
    if session.revoked_at is not None:
        return None, "revoked"
    if aware(session.absolute_expires_at) <= at:
        return None, "absolute_expired"
    if aware(session.idle_expires_at) <= at:
        return None, "idle_expired"
    if not user.is_active:
        return None, "inactive_user"
    if at - aware(session.last_seen_at) >= SLIDE_EVERY:
        session.last_seen_at = at
        session.idle_expires_at = min(
            at + timedelta(days=settings.auth_session_idle_days), aware(session.absolute_expires_at)
        )
        db.commit()
    return session, user


def revoke_token(db: Session, token: str) -> None:
    """Revoke the session this token names, if any (e.g. the one a browser
    carried into a new login). The caller commits."""
    db.execute(
        update(AuthSession)
        .where(AuthSession.token_hash == hash_token(token), AuthSession.revoked_at.is_(None))
        .values(revoked_at=now())
    )


def revoke(db: Session, session_id: int) -> None:
    db.execute(
        update(AuthSession)
        .where(AuthSession.id == session_id, AuthSession.revoked_at.is_(None))
        .values(revoked_at=now())
    )


def revoke_all(db: Session, user_id: str) -> None:
    db.execute(
        update(AuthSession)
        .where(AuthSession.user_id == user_id, AuthSession.revoked_at.is_(None))
        .values(revoked_at=now())
    )
