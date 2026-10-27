"""Alembic environment.

The database URL comes from app.core.config.settings (DATABASE_URL in .env),
never from alembic.ini, so the same .env drives the app and its migrations.
target_metadata is the app's declarative Base; autogenerate therefore sees
every model, including the content and auth tables that were created outside
Alembic. Review every generated revision and keep only the ops you intended.

Usage:
    uv run alembic upgrade head                     # apply (scripts/deploy.sh does this)
    uv run alembic revision --autogenerate -m "..."  # new revision; then edit it
"""

from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool

from alembic import context
from app.core.config import settings
from app.db import models  # noqa: F401  (registers every table on Base)
from app.db.base import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

config.set_main_option("sqlalchemy.url", settings.database_url.replace("%", "%%"))

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Emit SQL to stdout instead of connecting (`alembic upgrade head --sql`)."""
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
