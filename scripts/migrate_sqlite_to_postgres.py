"""
One-time migration: reads Tarun_Associate.db and Nobin_Associate.db (SQLite)
and writes their content into the new PostgreSQL schema defined in app/db/models.py.

Usage:
    python -m scripts.migrate_sqlite_to_postgres

Safe to re-run: each page is tagged with (source_db, source_page_id) on the way
in, and already-migrated pages are skipped on subsequent runs. Re-running after
a partial failure will only insert what's missing.

Note: this script currently calls Base.metadata.create_all(engine) to create
tables if they don't exist yet. Once the Alembic migration (next step) is set
up, remove that call and run `alembic upgrade head` instead — Alembic should
own schema creation from that point on, not this script.
"""
import sqlite3
from dataclasses import dataclass

from bs4 import BeautifulSoup

from app.core.config import settings
from app.db.base import Base
from app.db.models import Book, Category, Chapter, ContentStatus, Page
from app.db.session import SessionLocal, engine


@dataclass(frozen=True)
class SourceDB:
    key: str
    path: str
    language: str


SOURCES = [
    SourceDB(key="tarun", path=settings.tarun_db_path, language="bn"),
    SourceDB(key="nobin", path=settings.nobin_db_path, language="bn"),
]


def clean_html(html_content: str) -> str:
    return BeautifulSoup(html_content, "html.parser").get_text(separator="\n").strip()


def migrate_source(source: SourceDB, session) -> int:
    conn = sqlite3.connect(source.path)
    cur = conn.cursor()

    # --- categories: old sqlite id -> new Postgres Category row ---
    cur.execute("SELECT id, name FROM categories")
    category_map = {}
    for old_id, name in cur.fetchall():
        category = Category(name=name or "Unknown", language=source.language)
        session.add(category)
        session.flush()  # assigns category.id without committing yet
        category_map[old_id] = category

    # --- books ---
    cur.execute("SELECT id, name, category FROM books")
    book_map = {}
    for old_id, name, old_category_id in cur.fetchall():
        book = Book(
            name=name or "Unknown",
            language=source.language,
            category=category_map.get(old_category_id),
        )
        session.add(book)
        session.flush()
        book_map[old_id] = book

    # --- chapters ---
    cur.execute("SELECT id, name, book FROM chapters")
    chapter_map = {}
    for old_id, name, old_book_id in cur.fetchall():
        book = book_map.get(old_book_id)
        if book is None:
            continue
        chapter = Chapter(name=name or "Unknown", book=book)
        session.add(chapter)
        session.flush()
        chapter_map[old_id] = chapter

    # --- pages, skipping any already migrated in a previous run ---
    already_migrated = {
        row[0]
        for row in session.query(Page.source_page_id).filter(Page.source_db == source.key)
    }

    cur.execute("SELECT id, book, chapter, content FROM pages WHERE content IS NOT NULL")
    count = 0
    skipped = 0
    for old_id, old_book_id, old_chapter_id, content in cur.fetchall():
        if old_id in already_migrated:
            continue

        book = book_map.get(old_book_id)
        if book is None:
            skipped += 1
            continue

        clean_text = clean_html(content)
        if not clean_text:
            continue

        page = Page(
            book=book,
            chapter=chapter_map.get(old_chapter_id),
            content=clean_text,
            language=source.language,
            status=ContentStatus.published,
            source_db=source.key,
            source_page_id=old_id,
        )
        session.add(page)
        count += 1

        if count % 200 == 0:
            session.commit()
            print(f"[{source.key}] migrated {count} pages so far...")

    session.commit()
    conn.close()
    print(
        f"[{source.key}] migration complete: {count} new pages inserted, "
        f"{skipped} skipped (no matching book)."
    )
    return count


def main() -> None:
    # Temporary — see module docstring. Alembic takes over schema management
    # once the migration in the next step is in place.
    Base.metadata.create_all(engine)

    session = SessionLocal()
    try:
        total = sum(migrate_source(source, session) for source in SOURCES)
        print(f"All sources migrated. Total new pages: {total}")
    finally:
        session.close()


if __name__ == "__main__":
    main()
