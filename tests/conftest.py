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
