"""Durable chat data, separate metadata so corpus tables are never migrated here."""
from datetime import datetime, timezone

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, JSON, String, Text, UniqueConstraint, text
from sqlalchemy.orm import declarative_base

ChatBase = declarative_base()


def now():
    return datetime.now(timezone.utc)


class ChatIdentity(ChatBase):
    __tablename__ = 'chat_identities'
    id = Column(String(36), primary_key=True)
    token_hash = Column(String(64), nullable=False, unique=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=now)


class Conversation(ChatBase):
    __tablename__ = 'chat_conversations'
    id = Column(String(36), primary_key=True)
    owner_id = Column(String(36), ForeignKey('chat_identities.id'), nullable=False, index=True)
    title = Column(String(80), nullable=False, default='New chat')
    mode = Column(String(24), nullable=True)
    persona = Column(Text, nullable=True)
    summary = Column(Text, nullable=False, default='')
    summarized_through = Column(Integer, nullable=False, default=0)
    next_sequence = Column(Integer, nullable=False, default=1)
    active_request = Column(String(36), nullable=True)
    active_since = Column(DateTime(timezone=True), nullable=True)
    memory_enabled = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=now)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=now)


class ChatMessage(ChatBase):
    __tablename__ = 'chat_messages'
    __table_args__ = (
        UniqueConstraint('conversation_id', 'sequence'),
        UniqueConstraint('conversation_id', 'request_id', 'role'),
    )
    id = Column(String(36), primary_key=True)
    conversation_id = Column(String(36), ForeignKey('chat_conversations.id', ondelete='CASCADE'), nullable=False, index=True)
    sequence = Column(Integer, nullable=False)
    request_id = Column(String(36), nullable=False)
    role = Column(String(12), nullable=False)
    content = Column(Text, nullable=False, default='')
    status = Column(String(16), nullable=False, default='complete')
    sources = Column(JSON, nullable=False, default=list)
    resources = Column(JSON, nullable=False, default=dict, server_default=text("'{}'"))
    options = Column(JSON, nullable=False, default=dict, server_default=text("'{}'"))
    mode = Column(String(24), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=now)


class UserMemory(ChatBase):
    __tablename__ = 'chat_user_memories'
    __table_args__ = (UniqueConstraint('owner_id', 'key'),)
    id = Column(String(36), primary_key=True)
    owner_id = Column(String(36), ForeignKey('chat_identities.id'), nullable=False, index=True)
    key = Column(String(100), nullable=False)
    content = Column(Text, nullable=False)
    source_conversation_id = Column(String(36), ForeignKey('chat_conversations.id', ondelete='CASCADE'), nullable=False)
    source_message_id = Column(String(36), ForeignKey('chat_messages.id', ondelete='CASCADE'), nullable=False)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=now)


def initialize_chat_schema(engine):
    """Add chat tables only; serialize concurrent PostgreSQL worker startup."""
    from sqlalchemy import inspect
    with engine.begin() as connection:
        if engine.dialect.name == 'postgresql':
            connection.execute(text('SELECT pg_advisory_xact_lock(6401001)'))
        ChatBase.metadata.create_all(connection)
        # create_all does not add columns to databases created by older versions.
        # The PostgreSQL advisory lock also serializes this additive upgrade.
        schema = connection.get_execution_options().get('schema_translate_map', {}).get(None)
        quote = connection.dialect.identifier_preparer.quote
        table = (quote(schema) + '.' if schema else '') + quote('chat_messages')
        columns = {column['name'] for column in inspect(connection).get_columns('chat_messages', schema=schema)}
        for name in ('resources', 'options'):
            if name not in columns:
                connection.exec_driver_sql(f"ALTER TABLE {table} ADD COLUMN {name} JSON NOT NULL DEFAULT '{{}}'")
        if 'resources' not in columns:
            # SQL JSON builders preserve nested source metadata and citation order.
            if engine.dialect.name == 'postgresql':
                connection.exec_driver_sql(f"UPDATE {table} SET resources = json_build_object("
                    "'version', 1, 'sources', COALESCE(sources, '[]'::json), "
                    "'web_results', '[]'::json, 'verification', NULL)")
            elif engine.dialect.name == 'sqlite':
                connection.exec_driver_sql(f"UPDATE {table} SET resources = json_object("
                    "'version', 1, 'sources', json(COALESCE(sources, '[]')), "
                    "'web_results', json('[]'), 'verification', NULL)")
            else:
                raise RuntimeError('Chat resource migration supports PostgreSQL and SQLite.')
