"""conversations and messages

Chat history moves from the in-process dict in app/services/session_store.py
to Postgres (docs/roadmap-q4-2026.md §2). Creates only the two new tables.
Autogenerate also proposed dropping the legacy chat_* tables that exist in some
databases but are not in app/db/models.py; those ops were removed by hand --
this revision never touches a table it did not create.

Revision ID: 309969df1eaf
Revises:
Create Date: 2026-10-24 20:08:23.213801

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '309969df1eaf'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create conversations and messages."""
    op.create_table('conversations',
    sa.Column('id', sa.String(length=43), nullable=False),
    sa.Column('owner_key', sa.Text(), nullable=False),
    sa.Column('user_id', sa.String(length=36), nullable=True),
    sa.Column('title', sa.Text(), nullable=True),
    sa.Column('mode', sa.Text(), server_default='QA', nullable=False),
    sa.Column('persona', sa.Text(), nullable=True),
    sa.Column('role_id', sa.Text(), nullable=True),
    sa.Column('level', sa.Text(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('deleted_at', sa.DateTime(timezone=True), nullable=True),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index('conversations_owner_recent', 'conversations', ['owner_key', sa.literal_column('updated_at DESC')], unique=False, postgresql_where=sa.text('deleted_at IS NULL'), sqlite_where=sa.text('deleted_at IS NULL'))
    op.create_table('messages',
    sa.Column('id', sa.BigInteger().with_variant(sa.Integer(), 'sqlite'), autoincrement=True, nullable=False),
    sa.Column('conversation_id', sa.String(length=43), nullable=False),
    sa.Column('role', sa.Text(), nullable=False),
    sa.Column('content', sa.Text(), nullable=False),
    sa.Column('mode', sa.Text(), nullable=True),
    sa.Column('standalone_query', sa.Text(), nullable=True),
    sa.Column('sources', sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), 'postgresql'), nullable=True),
    sa.Column('rerank_top', sa.Float(), nullable=True),
    sa.Column('grounded', sa.Boolean(), nullable=True),
    sa.Column('response_time_ms', sa.Float(), nullable=True),
    sa.Column('trace_id', sa.Text(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint("role IN ('user', 'assistant')", name='ck_messages_role'),
    sa.ForeignKeyConstraint(['conversation_id'], ['conversations.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index('messages_conversation', 'messages', ['conversation_id', 'id'], unique=False)


def downgrade() -> None:
    """Drop messages and conversations (all chat history is lost)."""
    op.drop_index('messages_conversation', table_name='messages')
    op.drop_table('messages')
    op.drop_index('conversations_owner_recent', table_name='conversations', postgresql_where=sa.text('deleted_at IS NULL'), sqlite_where=sa.text('deleted_at IS NULL'))
    op.drop_table('conversations')
