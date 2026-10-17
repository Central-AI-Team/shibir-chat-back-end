"""Load the 6 whole-book text blobs from data/raw/books/books.sql into Postgres.


Usage:
    uv run python -m scripts.load_books [--sql data/raw/books/books.sql] [--publish] [--dry-run]

Then embed with `uv run python -m app.rag.ingest` (the only code that writes
to Chroma). See "Adding a data source" in CLAUDE.md.

The dump holds one row per book with the whole text in a single column (plus
author / translator / publisher / edition / price). Only its COPY data is read
(app/loaders/pgdump.py); its own `books` table is never created, so it cannot
collide with ours.

Shape in Postgres: Category "সংযোজিত বই" -> Book (name = title, author) ->
ONE Chapter (the text has no reliable chapter markers: it is hard-wrapped
plain text, with the contents page inline) -> Pages of up to ~PAGE_CHARS
characters, cut on paragraph boundaries, all with source_db="books_sql",
source_page_id = book_id*PAGE_ID_STRIDE + page number, upserted on that pair.
Translator / publisher / edition / price are parsed but not stored (no RAG use);
they stay in the raw dump.

Cleaning: the source is wrapped at ~50 characters per line, so lines are
re-joined into paragraphs (a line clearly shorter than the book's usual width
ends a paragraph). Scanner/publisher junk is then dropped paragraph by
paragraph (never whole pages: page 1 mixes it with real preface text) --
watermarks ("www.icsbook.info"), bare page numbers, Latin-only credits, and
short address / phone / printing-credit lines. The raw dump is untouched, so
this is reversible by re-loading.

Status: new pages are draft unless --publish; existing pages keep their status
unless --publish. --dry-run does everything in a transaction and rolls it back.
Books whose title already exists in the library are listed in the report so a
duplicate can be spotted BEFORE publishing.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy import or_

from app.loaders.sources import iter_copy_rows
from app.rag.chunker import normalize

SOURCE_DB = "books_sql"
CATEGORY_NAME = "সংযোজিত বই"
LANGUAGE = "bn"
DEFAULT_SQL = Path("data/raw/books/books.sql")
PAGE_CHARS = 2000        # target page size; the chunker splits to ~900 anyway
PAGE_ID_STRIDE = 10_000  # source_page_id = source_book_id * stride + page number
_TERMINALS = "।?!॥"

# Short paragraphs containing one of these are publisher front matter.
_JUNK_MARKERS = ("ফোন:", "ফ্যাক্স:", "শিরিশদাস", "বাংলাবাজার", "আঃ প্রঃ", "প্রকাশনায়",
                 "মুদ্রণে", "বিনিময়:", "নির্ধারিত মূল্য", "আধুনিক প্রেস")
_JUNK_MAX_LEN = 80
_LETTER_RE = re.compile(r"[\u0980-\u09FF\u0600-\u06FF]")  # Bengali or Arabic-script letters
_DIGITS_ONLY_RE = re.compile(r"^[\s\d\u09E6-\u09EF\u06F0-\u06F9\u0660-\u0669\u0966-\u096F.:\-–—()]*$")
_WATERMARK_RE = re.compile(r"\bwww\.[\w.-]+\.\w+\b")  # "www.icsbook.info" scanner stamp
_CREDIT_RE = re.compile(r"Published by|Sponsored by")  # English colophon, also mixed with Urdu
_TAIL_MERGE_CHARS = 300  # a shorter last page is appended to the previous one


@dataclass
class BookPage:
    number: int
    content: str


@dataclass
class BookRecord:
    source_id: int
    source_file: str
    title: str
    author: str
    translator: str | None
    publisher: str | None
    edition_note: str | None
    price_note: str | None
    pages: list[BookPage] = field(default_factory=list)
    dropped: int = 0  # junk paragraphs removed


def reflow(text: str) -> list[str]:
    """Hard-wrapped plain text -> paragraphs.

    Blank lines always end a paragraph. Otherwise a line ends one when it is
    clearly shorter than the book's usual wrapped width (the 90th-percentile
    line length), which is how a wrapped paragraph's last line, a heading or a
    one-line item looks. Everything else is joined to the next line with a space.
    """
    lines = [ln.strip() for ln in text.replace("\r\n", "\n").split("\n")]
    lengths = sorted(len(ln) for ln in lines if ln)
    if not lengths:
        return []
    width = lengths[int(len(lengths) * 0.9)]
    short = max(12, int(width * 0.6))

    paragraphs: list[str] = []
    current: list[str] = []
    for ln in lines:
        if not ln:
            if current:
                paragraphs.append(" ".join(current))
                current = []
            continue
        current.append(ln)
        if len(ln) < short or (ln[-1] in _TERMINALS and len(ln) < width * 0.8):
            paragraphs.append(" ".join(current))
            current = []
    if current:
        paragraphs.append(" ".join(current))
    return [re.sub(r"[ \t]{2,}", " ", p) for p in paragraphs if p.strip()]


def paginate(paragraphs: list[str], page_chars: int = PAGE_CHARS) -> list[str]:
    """Greedy-pack paragraphs into pages of about `page_chars`; a paragraph is
    never split here (the chunker splits long pages later)."""
    pages: list[str] = []
    current: list[str] = []
    size = 0
    for para in paragraphs:
        if current and size + len(para) > page_chars:
            pages.append("\n\n".join(current))
            current, size = [], 0
        current.append(para)
        size += len(para) + 2
    if current:
        pages.append("\n\n".join(current))
    if len(pages) > 1 and len(pages[-1]) < _TAIL_MERGE_CHARS:
        pages[-2] += "\n\n" + pages.pop()
    return pages


def is_junk(paragraph: str) -> bool:
    """Scanner/publisher noise: watermark, page number, Latin-only credit line,
    or a short address / phone / printing-credit line."""
    p = paragraph.strip()
    if not p or _DIGITS_ONLY_RE.match(p) or _CREDIT_RE.search(p):
        return True
    if not _LETTER_RE.search(p) and len(p) <= 250:
        return True  # Latin-only: "www.icsbook.info", "Published by Adhunik ..."
    return len(p) <= _JUNK_MAX_LEN and any(m in p for m in _JUNK_MARKERS)


def read_books(sql_path: Path, page_chars: int = PAGE_CHARS) -> list[BookRecord]:
    rows = list(iter_copy_rows(sql_path, "books"))
    books: list[BookRecord] = []
    for row in rows:
        source_id = int(row["id"])
        record = BookRecord(
            source_id=source_id,
            source_file=row["source_file"],
            title=normalize(row["title"]),
            author=normalize(row["author"]),
            translator=normalize(row["translator"]) if row["translator"] else None,
            publisher=normalize(row["publisher"]) if row["publisher"] else None,
            edition_note=normalize(row["edition_note"]) if row["edition_note"] else None,
            price_note=normalize(row["price_note"]) if row["price_note"] else None,
        )
        paragraphs = reflow(_WATERMARK_RE.sub("", normalize(row["content"])))
        kept = [p for p in paragraphs if not is_junk(p)]
        record.dropped = len(paragraphs) - len(kept)
        for n, page in enumerate(paginate(kept, page_chars), 1):
            record.pages.append(BookPage(n, page))
        books.append(record)
    return books


@dataclass
class UpsertStats:
    books_created: int = 0
    pages_inserted: int = 0
    pages_updated: int = 0
    pages_unchanged: int = 0
    possible_duplicates: list[str] = field(default_factory=list)


def upsert_books(session, books: list[BookRecord], *, publish: bool) -> UpsertStats:
    """Upsert category/book/chapter/pages. Flushes; the caller commits or rolls back."""
    from app.db.models import Book, Category, Chapter, ContentStatus, Page

    stats = UpsertStats()
    category = (
        session.query(Category).filter_by(name=CATEGORY_NAME, language=LANGUAGE)
        .order_by(Category.id).first()
    )
    if category is None:
        category = Category(name=CATEGORY_NAME, language=LANGUAGE)
        session.add(category)
        session.flush()

    new_status = ContentStatus.published if publish else ContentStatus.draft
    existing_pages = {
        p.source_page_id: p for p in session.query(Page).filter_by(source_db=SOURCE_DB)
    }

    for rec in books:
        # A book is identified by its already-loaded pages (source_page_id
        # encodes the source book id), so a re-run updates instead of duplicating.
        known = next((p.book_id for sid, p in existing_pages.items()
                      if sid // PAGE_ID_STRIDE == rec.source_id), None)
        book = session.get(Book, known) if known else None
        if book is None:
            # NULL category counts as "elsewhere" (SQL `!=` would skip NULLs).
            others = (
                session.query(Book).filter(Book.name == rec.title)
                .filter(or_(Book.category_id.is_(None), Book.category_id != category.id)).all()
            )
            if others:
                stats.possible_duplicates.append(
                    f"{rec.title!r} already exists as book id(s) {[b.id for b in others]}"
                )
            book = Book(name=rec.title, language=LANGUAGE, author=rec.author,
                        category_id=category.id)
            session.add(book)
            session.flush()
            stats.books_created += 1
        for attr, value in (("name", rec.title), ("author", rec.author),
                            ("category_id", category.id)):
            if getattr(book, attr) != value:
                setattr(book, attr, value)

        chapter = (
            session.query(Chapter).filter_by(book_id=book.id, position=1)
            .order_by(Chapter.id).first()
        )
        if chapter is None:
            chapter = Chapter(book_id=book.id, name=rec.title, position=1)
            session.add(chapter)
            session.flush()

        for page in rec.pages:
            sid = rec.source_id * PAGE_ID_STRIDE + page.number
            row = existing_pages.get(sid)
            if row is None:
                session.add(Page(
                    book_id=book.id, chapter_id=chapter.id, content=page.content,
                    language=LANGUAGE, status=new_status,
                    source_db=SOURCE_DB, source_page_id=sid,
                ))
                stats.pages_inserted += 1
                continue
            changed = False
            # Assign only on a real change so updated_at (and so ingest's
            # "needs re-embedding" check) stays quiet on an identical re-run.
            for attr, value in (("content", page.content), ("book_id", book.id),
                                ("chapter_id", chapter.id)):
                if getattr(row, attr) != value:
                    setattr(row, attr, value)
                    changed = True
            if publish and row.status != ContentStatus.published:
                row.status = ContentStatus.published
                changed = True
            if changed:
                stats.pages_updated += 1
            else:
                stats.pages_unchanged += 1
    session.flush()
    return stats


def _print_report(books: list[BookRecord]) -> None:
    for b in books:
        chars = sum(len(p.content) for p in b.pages)
        print(f"[{b.source_id}] {b.title}  — {b.author}  pages: {len(b.pages)} "
              f"junk paragraphs dropped: {b.dropped}  chars: {chars:,}")
    print(f"total pages: {sum(len(b.pages) for b in books)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sql", type=Path, default=DEFAULT_SQL)
    parser.add_argument("--publish", action="store_true", help="publish pages")
    parser.add_argument("--dry-run", action="store_true", help="roll back instead of committing")
    args = parser.parse_args(argv)

    if not args.sql.exists():
        print(f"not found: {args.sql}", file=sys.stderr)
        return 1
    books = read_books(args.sql)
    _print_report(books)

    from app.db.session import SessionLocal

    session = SessionLocal()
    try:
        s = upsert_books(session, books, publish=args.publish)
        print(f"\nbooks created: {s.books_created}   pages inserted: {s.pages_inserted}   "
              f"updated: {s.pages_updated}   unchanged: {s.pages_unchanged}")
        for d in s.possible_duplicates:
            print(f"POSSIBLE DUPLICATE: {d} -- check before --publish")
        if args.dry_run:
            session.rollback()
            print("DRY RUN -- rolled back, nothing written.")
        else:
            session.commit()
            print("committed. Next: uv run python -m app.rag.ingest")
    finally:
        session.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())