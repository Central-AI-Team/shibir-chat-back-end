import enum

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Column,
    Date,
    DateTime,
    Enum as SAEnum,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship

from app.db.base import Base


class ContentStatus(str, enum.Enum):
    """Publication state. Only 'published' rows should ever be embedded."""

    draft = "draft"
    published = "published"


class Category(Base):
    __tablename__ = "categories"

    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    language = Column(String(10), nullable=False)  # "bn" / "en" / "ar"

    books = relationship("Book", back_populates="category")


class Book(Base):
    __tablename__ = "books"

    id = Column(Integer, primary_key=True)
    category_id = Column(Integer, ForeignKey("categories.id"), nullable=True)
    name = Column(String(255), nullable=False)
    language = Column(String(10), nullable=False)
    author = Column(String(255), nullable=True)

    __table_args__ = (Index("ix_books_category_id", "category_id"),)

    category = relationship("Category", back_populates="books")
    chapters = relationship("Chapter", back_populates="book")


class Chapter(Base):
    __tablename__ = "chapters"

    id = Column(Integer, primary_key=True)
    book_id = Column(Integer, ForeignKey("books.id"), nullable=False)
    name = Column(String(255), nullable=False)
    position = Column(Integer, nullable=True)  # ordering within the book

    __table_args__ = (Index("ix_chapters_book_id", "book_id"),)

    book = relationship("Book", back_populates="chapters")
    pages = relationship("Page", back_populates="chapter")


class Page(Base):
    __tablename__ = "pages"

    id = Column(Integer, primary_key=True)
    book_id = Column(Integer, ForeignKey("books.id"), nullable=False)
    chapter_id = Column(Integer, ForeignKey("chapters.id"), nullable=True)

    content = Column(Text, nullable=False)
    language = Column(String(10), nullable=False)
    status = Column(SAEnum(ContentStatus), nullable=False, default=ContentStatus.published)

    # RAG cleanup flag. Set by scripts/clean_corpus.py for pages a prior
    # analysis flagged as Arabic-placeholder junk, cross-contaminated
    # duplicates, or orphaned. app/rag/ingest.py._due_pages() skips these, so
    # they are never (re-)embedded. Reversible: flip back to false + re-ingest.
    # Added by the ADD COLUMN IF NOT EXISTS migration in clean_corpus.py --
    # that migration MUST be run before any ingest after this model change.
    excluded_from_rag = Column(
        Boolean, nullable=False, server_default="false", default=False
    )
    exclusion_reason = Column(Text, nullable=True)

    # Provenance only — which legacy SQLite file + row this came from.
    # Not used by the app after migration; kept so re-running the migration
    # script is safe (it can tell what's already been copied over).
    source_db = Column(String(50), nullable=True)
    source_page_id = Column(Integer, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    embedded_at = Column(DateTime(timezone=True), nullable=True)

    # Keep in sync with INDEXES in scripts/clean_database.py (which adds them to
    # an existing database; create_all only builds them for new tables).
    __table_args__ = (
        Index("ix_pages_book_id", "book_id"),
        Index("ix_pages_chapter_id", "chapter_id"),
        Index("ix_pages_status_embedded", "status", "embedded_at"),
        Index("uq_pages_source", "source_db", "source_page_id", unique=True),
    )

    book = relationship("Book")
    chapter = relationship("Chapter", back_populates="pages")


class Article(Base):
    __tablename__ = "articles"

    id = Column(Integer, primary_key=True)
    title = Column(String(500), nullable=False)
    content = Column(Text, nullable=False)
    language = Column(String(10), nullable=False)
    status = Column(SAEnum(ContentStatus), nullable=False, default=ContentStatus.draft)
    author = Column(String(255), nullable=True)

    # Provenance for non-web-article sources (PDF/Word docs, external DBs,
    # CSV/spreadsheets, ...). source_ref holds a URL, file path, or
    # "external_table::id" reference; source_metadata holds whatever extra
    # fields are specific to that source (e.g. PDF page number).
    source_type = Column(String(30), nullable=False, default="web_article")
    source_ref = Column(String(500), nullable=True)
    source_metadata = Column(JSONB, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    embedded_at = Column(DateTime(timezone=True), nullable=True)
    published_at = Column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        Index("ix_articles_status_source", "status", "source_type"),
        Index("uq_articles_source", "source_type", "source_ref", unique=True),
    )


class Shaheed(Base):
    """One martyr record (name, date, place, organisational rank).

    A separate table, not `pages`: the data is structured, not prose. ingest.py
    renders each row to a short labelled text block (`render_shaheed`) and
    embeds that, so the chatbot can answer "who was ...", "who was martyred in
    ...". Column names match the original dump so it loads unchanged.
    """

    __tablename__ = "shaheed"

    id = Column(Integer, primary_key=True)
    serial_no = Column(Integer, nullable=False, unique=True)  # provenance + upsert key
    name = Column(Text, nullable=False)
    shahadat_date = Column(Date, nullable=True)
    shahadat_date_text = Column(Text, nullable=False)  # as written, Bengali digits
    shahadat_place = Column(Text, nullable=False)
    organizational_rank = Column(Text, nullable=False)

    status = Column(SAEnum(ContentStatus), nullable=False, default=ContentStatus.draft)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    embedded_at = Column(DateTime(timezone=True), nullable=True)


# ---------------------------------------------------------------------------
# Accounts (app/auth/). Portable types only: the auth tests run on SQLite.
# Created at startup by app.main.create_auth_tables() (create_all, checkfirst),
# or manually with docs/sql/auth_tables.sql; keep the two in sync --
# tests/test_auth_schema.py compares them.
# ---------------------------------------------------------------------------


class User(Base):
    __tablename__ = "users"

    id = Column(String(36), primary_key=True)  # uuid4 string
    email = Column(String(254), nullable=False, unique=True)  # stripped + lowercased
    password_hash = Column(String(255), nullable=False)  # argon2id, never logged
    role = Column(String(10), nullable=False, default="user", server_default="user")  # user | admin
    is_active = Column(Boolean, nullable=False, default=True, server_default="true")
    failed_login_count = Column(Integer, nullable=False, default=0, server_default="0")
    locked_until = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False)
    last_login_at = Column(DateTime(timezone=True), nullable=True)
    password_changed_at = Column(DateTime(timezone=True), nullable=False)

    __table_args__ = (CheckConstraint("role IN ('user', 'admin')", name="ck_users_role"),)


class AuthSession(Base):
    """One login. Only sha256(token) is stored; the token lives in the cookie."""

    __tablename__ = "auth_sessions"

    id = Column(Integer, primary_key=True)
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    token_hash = Column(String(64), nullable=False, unique=True)
    created_at = Column(DateTime(timezone=True), nullable=False)
    last_seen_at = Column(DateTime(timezone=True), nullable=False)
    idle_expires_at = Column(DateTime(timezone=True), nullable=False)
    absolute_expires_at = Column(DateTime(timezone=True), nullable=False)
    revoked_at = Column(DateTime(timezone=True), nullable=True)
    user_agent = Column(String(200), nullable=True)
    ip_prefix = Column(String(64), nullable=True)  # IPv4 /24 or IPv6 /48, never the full IP

    __table_args__ = (Index("ix_auth_sessions_user_id", "user_id"),)


class GuestUsage(Base):
    """Guest response counters (app/auth/guest_quota.py). `key` is
    "owner:<owner hash>" (lifetime) or "ip:<IPv4 address | IPv6 /64>" (24 h
    window from window_start). Holds an IP address: rows idle 30 days are purged."""

    __tablename__ = "guest_usage"

    key = Column(String(255), primary_key=True)
    count = Column(Integer, nullable=False, default=0, server_default="0")
    first_seen = Column(DateTime(timezone=True), nullable=False)
    last_seen = Column(DateTime(timezone=True), nullable=False)
    window_start = Column(DateTime(timezone=True), nullable=False)

    __table_args__ = (Index("ix_guest_usage_last_seen", "last_seen"),)


# ---------------------------------------------------------------------------
# Chat history (app/services/session_store.py). Schema: docs/roadmap-q4-2026.md
# §2. Created by Alembic (migrations/), never by create_all in production.
# Portable types where the roadmap's SQL is Postgres-only (JSONB, BIGSERIAL),
# because the API tests run the store on in-memory SQLite like the auth tables.
# ---------------------------------------------------------------------------


class Conversation(Base):
    """One chat session. `id` is the 43-char token the server issues; ownership
    is `owner_key` (guest cookie hash or principal.user_owner_key). A row is
    soft-deleted (deleted_at) by DELETE /conversations/{id} and by the
    per-owner cap; scripts/purge_history.py hard-deletes later."""

    __tablename__ = "conversations"

    id = Column(String(43), primary_key=True)
    owner_key = Column(Text, nullable=False)
    # NULL for guests; set when the guest logs in / registers (auth_routes) or
    # when a logged-in user starts a chat. users.id is a uuid string, not the
    # INTEGER the roadmap sketch says.
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=True)
    title = Column(Text, nullable=True)  # first user turn, trimmed; see session_store._derive_title
    mode = Column(Text, nullable=False, default="QA", server_default="QA")
    persona = Column(Text, nullable=True)  # roleplay only
    # Reserved for the roleplay rebuild (roadmap §4); unused until then.
    role_id = Column(Text, nullable=True)
    level = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    deleted_at = Column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        # The sidebar query: an owner's live conversations, newest activity first.
        Index(
            "conversations_owner_recent",
            "owner_key",
            text("updated_at DESC"),
            postgresql_where=text("deleted_at IS NULL"),
            sqlite_where=text("deleted_at IS NULL"),
        ),
    )


class Message(Base):
    """One turn. Every turn is stored; session_store.MAX_HISTORY is only the
    read window. mode / standalone_query / rerank_top / grounded / trace_id are
    the evaluation columns of roadmap §3, stored from day one (NULL until the
    code that produces them lands)."""

    __tablename__ = "messages"

    id = Column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True)
    conversation_id = Column(
        String(43), ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False
    )
    role = Column(Text, nullable=False)
    content = Column(Text, nullable=False)
    mode = Column(Text, nullable=True)  # intent that produced this assistant turn
    standalone_query = Column(Text, nullable=True)  # roadmap §1; on the user turn
    sources = Column(JSON().with_variant(JSONB, "postgresql"), nullable=True)  # Citation dicts
    rerank_top = Column(Float, nullable=True)  # gate score; NULL when not QA/SUGGESTION
    grounded = Column(Boolean, nullable=True)  # gate verdict
    response_time_ms = Column(Float, nullable=True)
    trace_id = Column(Text, nullable=True)  # Langfuse link
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    __table_args__ = (
        CheckConstraint("role IN ('user', 'assistant')", name="ck_messages_role"),
        Index("messages_conversation", "conversation_id", "id"),
    )
