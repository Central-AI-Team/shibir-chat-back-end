"""Idempotent additive migration; never modifies the corpus tables.

Usage: uv run python -m scripts.create_chat_tables
Startup also runs this check, protected by a PostgreSQL advisory lock.
"""
from app.db.chat_models import initialize_chat_schema
from app.db.session import engine

if __name__ == '__main__':
    initialize_chat_schema(engine)
    print('Chat identity, conversation, message and memory tables are ready.')
