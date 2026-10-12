"""docs/sql/auth_tables.sql must create exactly the columns the models use
(nothing creates these tables at startup, so drift would only show in prod)."""

from __future__ import annotations

import re
from pathlib import Path

from app.db.models import AuthSession, GuestUsage, User

SQL = Path(__file__).resolve().parent.parent / "docs" / "sql" / "auth_tables.sql"


def _sql_columns() -> dict[str, set[str]]:
    text = SQL.read_text()
    tables = {}
    for name, body in re.findall(r"CREATE TABLE IF NOT EXISTS (\w+) \((.*?)\n\);", text, re.S):
        tables[name] = {
            line.split()[0] for line in body.strip().splitlines()
            if line.strip() and not line.strip().startswith("CONSTRAINT")
        }
    return tables


def test_sql_matches_the_models():
    sql = _sql_columns()
    for model in (User, AuthSession, GuestUsage):
        assert sql[model.__tablename__] == {c.name for c in model.__table__.columns}, model.__tablename__


def test_sql_is_idempotent():
    text = SQL.read_text()
    assert text.count("CREATE TABLE IF NOT EXISTS") == 3
    assert "CREATE TABLE " not in text.replace("CREATE TABLE IF NOT EXISTS", "")
    assert all("IF NOT EXISTS" in line for line in text.splitlines() if line.startswith("CREATE INDEX"))


def test_startup_creates_missing_tables(monkeypatch):
    import app.auth as auth
    from sqlalchemy import create_engine, inspect
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool

    from app.main import create_auth_tables

    engine = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    monkeypatch.setattr(auth, "SESSION_FACTORY", sessionmaker(bind=engine))
    create_auth_tables()
    create_auth_tables()  # idempotent
    assert {"users", "auth_sessions", "guest_usage"} <= set(inspect(engine).get_table_names())
    # only the auth tables: the content tables are not created here
    assert "pages" not in inspect(engine).get_table_names()


def test_startup_survives_a_database_that_is_down(monkeypatch, caplog):
    import app.auth as auth
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from app.main import create_auth_tables

    monkeypatch.setattr(auth, "SESSION_FACTORY", sessionmaker(bind=create_engine("sqlite:////nonexistent-dir/x.db")))
    create_auth_tables()  # must not raise
    assert "could not create the auth tables" in caplog.text
