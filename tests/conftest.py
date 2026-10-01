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


@pytest.fixture(autouse=True)
def isolated_chat_store(request, monkeypatch, tmp_path):
    """All chat tests use isolated durable storage, never the real corpus or LLM."""
    from sqlalchemy.orm import sessionmaker
    from app.db.chat_models import ChatBase
    from app.services import context, identity, session_store
    from app.api import router
    from app.core import tracing

    # Use a file DB so tests can reopen it to verify persistence.
    url = os.environ.get('CHAT_TEST_DATABASE_URL')
    schema = None
    if url:
        import uuid
        from sqlalchemy.schema import CreateSchema
        schema = 'chat_test_' + uuid.uuid4().hex
        test_engine = create_engine(url)
        with test_engine.begin() as connection:
            connection.execute(CreateSchema(schema))
        test_engine = test_engine.execution_options(schema_translate_map={None: schema})
    else:
        test_engine = create_engine(f"sqlite:///{tmp_path / 'chat.sqlite'}", connect_args={"check_same_thread": False})
    ChatBase.metadata.create_all(test_engine)
    factory = sessionmaker(bind=test_engine, expire_on_commit=False)
    for module in (context, identity, session_store):
        monkeypatch.setattr(module, 'SessionLocal', factory)
    monkeypatch.setattr(router, 'refresh_memory', lambda *args: None)
    monkeypatch.setattr(tracing.settings, 'langfuse_enabled', False)
    client = getattr(request.module, 'client', None)
    credentials = identity.create_identity()
    if client is not None:
        client.headers['X-Chat-Identity'] = credentials['token']
    yield {'engine': test_engine, 'factory': factory, **credentials}
    if client is not None:
        client.headers.pop('X-Chat-Identity', None)
    if schema:
        from sqlalchemy.schema import DropSchema
        with test_engine.begin() as connection:
            connection.execute(DropSchema(schema, cascade=True))
    test_engine.dispose()
