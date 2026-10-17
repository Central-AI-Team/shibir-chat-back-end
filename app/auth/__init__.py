"""Accounts, login sessions, authorization and the guest quota.

See docs/auth.md for the front-end contract. Modules:
  passwords    argon2id hashing and the password policy
  sessions     login-session tokens, cookies, expiry and revocation
  principal    who the caller is (guest or user) and their owner_key
  policy       the access level every route declares (deny by default)
  guest_quota  the Postgres-backed guest message counters
  audit        one structured log line per security event

Database access goes through open_db(), not app.db.session directly, so tests
can point SESSION_FACTORY at SQLite. Every caller uses it as a context manager
(`with open_db() as db:`) and commits explicitly.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy.orm import Session

# Tests set this to a sessionmaker bound to SQLite; None = the app's Postgres.
SESSION_FACTORY = None


def open_db() -> Session:
    if SESSION_FACTORY is not None:
        return SESSION_FACTORY()
    from app.db.session import SessionLocal

    return SessionLocal()


def now() -> datetime:
    """Aware UTC now. Patched by tests to move the clock."""
    return datetime.now(timezone.utc)


def aware(value: datetime | None) -> datetime | None:
    """SQLite returns naive datetimes; everything stored here is UTC."""
    if value is not None and value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value
