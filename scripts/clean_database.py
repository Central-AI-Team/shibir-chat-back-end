"""Clean the content tables and add the indexes the loaders and ingest rely on.

Usage:
    uv run python -m scripts.clean_database                 # DRY RUN (default): report only
    uv run python -m scripts.clean_database --apply         # backup, clean, add indexes
    uv run python -m scripts.clean_database --apply --no-backup   # not recommended

What it does, in one transaction:
  1. deletes chapters that have no pages;
  2. deletes books that have no pages and no remaining chapters;
  3. merges categories with the same (name, language) into one, keeping the one
     whose books hold the most published pages (ties: lowest id), repoints the
     books, and deletes categories left with no books;
  4. creates the indexes below (IF NOT EXISTS) -- the two unique ones only when
     no duplicate keys exist.

Never touched: pages, articles, chat_* tables, reference tables. Nothing is
deleted that has a page, and the script aborts (rolls back) if the page count,
the published-page count or the article count changes.

Not done on purpose: duplicate book titles are only listed (which copy is
canonical is a content decision), and legacy draft rows are left alone.

--apply first writes a custom-format pg_dump to data/backups/ and stops if that
fails. Restore with pg_restore. Run this BEFORE scripts.export_domains so the
export is built from the cleaned data.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.engine import make_url

from app.core.config import settings

BACKUP_DIR = Path("data/backups")

# (name, table, columns, unique?) -- keep in sync with the Index(...) entries in app/db/models.py
INDEXES = [
    ("ix_books_category_id", "books", "category_id", False),
    ("ix_chapters_book_id", "chapters", "book_id", False),
    ("ix_pages_book_id", "pages", "book_id", False),
    ("ix_pages_chapter_id", "pages", "chapter_id", False),
    ("ix_pages_status_embedded", "pages", "status, embedded_at", False),
    ("ix_articles_status_source", "articles", "status, source_type", False),
    ("uq_pages_source", "pages", "source_db, source_page_id", True),
    ("uq_articles_source", "articles", "source_type, source_ref", True),
]

EMPTY_CHAPTERS = "not exists (select 1 from pages p where p.chapter_id = chapters.id)"
EMPTY_BOOKS = (
    "not exists (select 1 from pages p where p.book_id = books.id) "
    "and not exists (select 1 from chapters c where c.book_id = books.id)"
)


def _one(s, sql: str, **params):
    return s.execute(text(sql), params).scalar()


def _counts(s) -> dict[str, int]:
    return {
        "pages": _one(s, "select count(*) from pages"),
        "published_pages": _one(s, "select count(*) from pages where status = 'published'"),
        "articles": _one(s, "select count(*) from articles"),
        "books": _one(s, "select count(*) from books"),
        "chapters": _one(s, "select count(*) from chapters"),
        "categories": _one(s, "select count(*) from categories"),
    }


def category_merges(s) -> dict[int, int]:
    """{duplicate_category_id: canonical_category_id}."""
    rows = s.execute(text("""
        select c.id, c.name, c.language,
               coalesce((select count(*) from pages p join books b on b.id = p.book_id
                         where b.category_id = c.id and p.status = 'published'), 0) as live
        from categories c order by c.name, c.language, live desc, c.id
    """)).all()
    groups: dict[tuple, list[int]] = {}
    for cid, name, lang, _live in rows:  # already ordered best-first within a group
        groups.setdefault((name, lang), []).append(cid)
    return {dup: ids[0] for ids in groups.values() for dup in ids[1:]}


def duplicate_titles(s) -> list[tuple]:
    return s.execute(text("""
        select b.name, array_agg(b.id order by b.id)
        from books b
        where exists (select 1 from pages p where p.book_id = b.id and p.status = 'published')
        group by b.name having count(*) > 1 order by b.name
    """)).all()


def clean(s, *, apply_indexes: bool = True) -> dict[str, int]:
    """Run the cleanup inside the caller's transaction. Returns what was removed."""
    done: dict[str, int] = {}
    done["chapters_deleted"] = s.execute(text(f"delete from chapters where {EMPTY_CHAPTERS}")).rowcount
    done["books_deleted"] = s.execute(text(f"delete from books where {EMPTY_BOOKS}")).rowcount

    merges = category_merges(s)
    for dup, canon in merges.items():
        s.execute(text("update books set category_id = :c where category_id = :d"), {"c": canon, "d": dup})
    done["categories_merged"] = len(merges)
    done["categories_deleted"] = s.execute(text(
        "delete from categories where not exists (select 1 from books b where b.category_id = categories.id)"
    )).rowcount

    if apply_indexes:
        done["indexes_created"] = create_indexes(s)
    return done


def create_indexes(s) -> int:
    created = 0
    for name, table, cols, unique in INDEXES:
        exists = _one(s, "select count(*) from pg_indexes where schemaname='public' and indexname=:n", n=name)
        if exists:
            continue
        if unique and _one(
            s, f"select count(*) from (select 1 from {table} where {cols.split(',')[-1].strip()} is not null "
               f"group by {cols} having count(*) > 1) x"
        ):
            print(f"  skip {name}: duplicate keys exist in {table}")
            continue
        s.execute(text(f"create {'unique ' if unique else ''}index {name} on {table} ({cols})"))
        created += 1
    return created


def backup() -> Path:
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    path = BACKUP_DIR / f"shibir_chat_{datetime.now():%Y%m%d_%H%M%S}.dump"
    dsn = make_url(settings.database_url).set(drivername="postgresql").render_as_string(hide_password=False)
    subprocess.run(["pg_dump", "-Fc", "-f", str(path), dsn], check=True)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--apply", action="store_true", help="make the changes (default: dry run)")
    parser.add_argument("--no-backup", action="store_true")
    parser.add_argument("--no-indexes", action="store_true")
    args = parser.parse_args(argv)

    from app.db.session import SessionLocal

    s = SessionLocal()
    try:
        before = _counts(s)
        print("before:", before)
        print(f"empty chapters: {_one(s, f'select count(*) from chapters where {EMPTY_CHAPTERS}')}")
        print(f"duplicate categories to merge: {len(category_merges(s))}")
        titles = duplicate_titles(s)
        print(f"duplicate book titles (NOT changed, decide by hand): {len(titles)}")
        for name, ids in titles:
            print(f"   {name}  book ids {ids}")

        if not args.apply:
            # Run the real thing and roll back, so the dry run reports exact numbers.
            done = clean(s, apply_indexes=not args.no_indexes)
            after = _counts(s)
            s.rollback()
            print("\nDRY RUN would do:", done)
            print("after (not saved):", after)
            print("Re-run with --apply to make these changes.")
            return 0

        if not args.no_backup:
            path = backup()
            print(f"backup written: {path} ({path.stat().st_size / 1e6:.1f} MB)")

        done = clean(s, apply_indexes=not args.no_indexes)
        after = _counts(s)
        for key in ("pages", "published_pages", "articles"):
            if before[key] != after[key]:
                s.rollback()
                print(f"ABORTED, rolled back: {key} changed {before[key]} -> {after[key]}", file=sys.stderr)
                return 1
        s.commit()
        print("done:", done)
        print("after:", after)
    finally:
        s.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
