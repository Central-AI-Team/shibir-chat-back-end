import enum

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Enum as SAEnum,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
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

    category = relationship("Category", back_populates="books")
    chapters = relationship("Chapter", back_populates="book")


class Chapter(Base):
    __tablename__ = "chapters"

    id = Column(Integer, primary_key=True)
    book_id = Column(Integer, ForeignKey("books.id"), nullable=False)
    name = Column(String(255), nullable=False)
    position = Column(Integer, nullable=True)  # ordering within the book

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
