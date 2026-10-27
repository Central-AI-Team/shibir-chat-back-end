"""Shared fixtures.

db_session: a SQLAlchemy session on the TEST database, inside a transaction
that is rolled back after the test -- the loaders' own flushes/commits land in
a savepoint and never persist.

Which database: TEST_DATABASE_URL, else DATABASE_URL. Either way its name must
contain "test" (CI's is shibir_test), or the test is skipped: locally
DATABASE_URL is the real corpus, and loader tests write to pages/articles.
"""


from __future__ import annotations

import os

# Tests must not depend on a developer's local .env. Environment variables
# beat .env in pydantic-settings, and this must run before app.main is imported
# because CORSMiddleware reads the list once, at import time.
os.environ["CORS_ALLOW_ORIGINS"] = (
    '["http://localhost:5173","http://127.0.0.1:5173","https://shibirgpt.potropollob.com"]'
)


from functools import lru_cache

import pytest
from sqlalchemy import create_engine
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session

from app.core.config import settings


@lru_cache(maxsize=1)
def _engine(url: str):
    from app.db import models  # noqa: F401  (registers the tables on Base)
    from app.db.base import Base

    engine = create_engine(url, future=True)
    Base.metadata.create_all(engine)  # no-op for tables that already exist
    return engine


@pytest.fixture
def db_session():
    url = os.environ.get("TEST_DATABASE_URL") or settings.database_url
    name = make_url(url).database or ""
    if "test" not in name:
        pytest.skip(
            f"database {name!r} is not a test database; set TEST_DATABASE_URL "
            "to one whose name contains 'test'"
        )
    connection = _engine(url).connect()
    transaction = connection.begin()
    session = Session(bind=connection, join_transaction_mode="create_savepoint")
    try:
        yield session
    finally:
        session.close()
        transaction.rollback()
        connection.close()


@pytest.fixture(autouse=True)
def _plain_http_cookies(monkeypatch):
    """TestClient talks http://testserver, and httpx (like a browser) never
    sends a Secure cookie over http -- so with the production default every
    request would mint a fresh owner and lose its own sessions. Tests that
    check the Secure flag turn it back on and use an https base_url."""
    monkeypatch.setattr(settings, "cookie_secure", False)


@pytest.fixture(autouse=True)
def _timing_off(monkeypatch):
    """A developer's ENABLE_TIMING=1 must not change what the router passes to
    mocked functions (the local timing recorder is exercised on its own)."""
    monkeypatch.setattr(settings, "enable_timing", False)


@pytest.fixture(autouse=True)
def _reset_rate_limits():
    """Every TestClient request comes from the same IP ("testclient"), so the
    in-process limiters must start empty in each test or the suite trips them."""
    from app.services import rate_limiter

    rate_limiter.reset()
    yield
    rate_limiter.reset()


@pytest.fixture(autouse=True)
def auth_db(monkeypatch, tmp_path):
    """Accounts, the guest quota AND the chat session store run on a fresh
    SQLite database (users / auth_sessions / guest_usage / conversations /
    messages; the content tables use JSONB), with a cheap argon2 hasher for
    speed. Guest limits are raised so tests that predate the quota are not
    capped; tests/test_auth_*.py set them back.

    A file under tmp_path, not ":memory:" on a StaticPool: that shares ONE
    sqlite3 connection between threads, and tests that hit the store from
    several threads (test_conversations' churn test) got "cannot rollback -
    no transaction is active" and, at teardown, a segfault in sqlite3. With a
    file every thread checks out its own connection; `timeout` is SQLite's
    busy wait so writers queue instead of failing with "database is locked"."""
    from argon2 import PasswordHasher
    from sqlalchemy.orm import sessionmaker

    import app.auth as auth
    from app.auth import passwords
    from app.db.base import Base
    from app.db.models import AuthSession, Conversation, GuestUsage, Message, User
    from app.services import session_store

    engine = create_engine(
        f"sqlite:///{tmp_path / 'app.db'}",
        connect_args={"check_same_thread": False, "timeout": 30},
    )
    Base.metadata.create_all(
        engine,
        tables=[
            User.__table__, AuthSession.__table__, GuestUsage.__table__,
            Conversation.__table__, Message.__table__,
        ],
    )
    factory = sessionmaker(bind=engine, future=True)
    monkeypatch.setattr(auth, "SESSION_FACTORY", factory)
    monkeypatch.setattr(session_store, "SESSION_FACTORY", factory)
    monkeypatch.setattr(passwords, "_hasher", PasswordHasher(time_cost=1, memory_cost=8, parallelism=1))
    passwords._dummy_hash.cache_clear()
    monkeypatch.setattr(settings, "guest_message_limit", 10_000)
    monkeypatch.setattr(settings, "guest_ip_limit_per_day", 10_000)
    yield engine
    passwords._dummy_hash.cache_clear()
    engine.dispose()


@pytest.fixture
def logged_in_user(auth_db, monkeypatch):
    """Every request in the test is made by one logged-in user (role "user"),
    for tests of user-only behaviour (notes) that are not about login. A real
    user and session exist in the test database; only the cookie lookup is
    patched, so this works whichever `app` object a test module imported
    (tests/test_main_config.py reloads app.main)."""
    from types import SimpleNamespace

    from app.auth import now, open_db, principal, sessions
    from app.db.models import User

    at = now()
    user = User(
        id="00000000-0000-4000-8000-000000000001",
        email="tester@example.com",
        password_hash="unused",
        role="user",
        is_active=True,
        failed_login_count=0,
        created_at=at,
        password_changed_at=at,
    )
    with open_db() as db:
        db.add(user)
        db.flush()
        token = sessions.create(db, user, SimpleNamespace(headers={}, client=None))
        db.commit()
    monkeypatch.setattr(sessions, "request_token", lambda request: token)
    return principal.user("00000000-0000-4000-8000-000000000001", "user")


AUTH_TEST_ORIGIN = "http://localhost:5173"
AUTH_TEST_PASSWORD = "correct horse battery"


@pytest.fixture
def auth_origin(monkeypatch):
    """The allowed front-end origin for auth tests, independent of a local
    .env that overrides CORS_ALLOW_ORIGINS."""
    monkeypatch.setattr(settings, "cors_allow_origins", [AUTH_TEST_ORIGIN])
    return AUTH_TEST_ORIGIN


@pytest.fixture
def auth_clock(monkeypatch):
    """Controls app.auth time: advance with clock["now"] += timedelta(...).
    The modules bind `now` at import, so each is patched."""
    from datetime import datetime, timezone

    import app.auth as auth
    from app.api import auth_routes
    from app.auth import audit, guest_quota, sessions

    state = {"now": datetime(2026, 10, 12, 12, 0, tzinfo=timezone.utc)}
    for module in (auth, audit, guest_quota, sessions, auth_routes):
        monkeypatch.setattr(module, "now", lambda: state["now"])
    return state


@pytest.fixture
def make_user(auth_db):
    """make_user(email=..., password=..., role=..., is_active=...) -> user id."""
    import uuid

    from app.auth import now, open_db, passwords
    from app.db.models import User

    def _make(email="alice@example.com", password=AUTH_TEST_PASSWORD, role="user", is_active=True) -> str:
        user_id = str(uuid.uuid4())
        at = now()
        with open_db() as db:
            db.add(
                User(
                    id=user_id,
                    email=email,
                    password_hash=passwords.hash_password(password),
                    role=role,
                    is_active=is_active,
                    failed_login_count=0,
                    created_at=at,
                    password_changed_at=at,
                )
            )
            db.commit()
        return user_id

    return _make
