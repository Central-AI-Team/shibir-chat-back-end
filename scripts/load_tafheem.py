"""Load তাফহীমুল কুরআন from data/tafheemul_quran.db into Postgres.

Usage:
    uv run python -m scripts.load_tafheem [--db data/tafheemul_quran.db] [--publish] [--dry-run]

Then embed with `uv run python -m app.rag.ingest` (the only code that writes
to Chroma). See "Adding a data source" in CLAUDE.md.

Shape in Postgres: Category "কুরআন ও তাফসীর" -> Book "তাফহীমুল কুরআন" -> one
Chapter per sura (position = sura number) -> Pages:
  * source_page_id = sura*1000        the sura introduction (vumika_sura)
  * source_page_id = sura*1000+ayah   "সূরা <name> <sura>:<ayah>" + the clean
                                      Bengali translation + that ayah's
                                      footnotes as "টীকা <n>: ..."
all with source_db="tafheem", upserted on (source_db, source_page_id).

Source quirks this handles (all verified against the real file):
  * Every id column is stored as TEXT -> everything is CAST to INTEGER.
  * Footnote markers are Bengali digits glued onto the translation
    ("জন্য২", "রব,৩", in 1:1 wrapped in <sup>) and RESTART at ১ in every sura,
    so a footnote is looked up by (sura, number), never by number alone.
  * Sub-notes "২৫(ক)" are stored with a NEGATIVE expl_id (-25).
  * Footnote bodies carry their own "[[টিকা: ২) ... ]]" wrapper, stripped here.
  * Literal backslash-n sequences stand for newlines.
  * Ayah 0 (112 suras) is only the bismillah plus a link to the introduction;
    it is not a page (its id would also collide with the intro page's).
  * (sura 19, expl_id -27) is stored twice.

Only the alquran, expl, vumika_sura and surah_name tables are ever read; the
file also holds app tables (users, sessions, ...) that must not be touched.
arabic_text is never selected.

Status: new pages are draft unless --publish. --publish also publishes
existing pages, and SUPERSEDES the legacy import of the same book
(pages.source_db="tafheemul_quran", several ayahs per page) by setting those
pages to draft -- nothing is deleted. Without --publish, existing pages keep
their status. --dry-run does everything in a transaction and rolls it back.
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from app.loaders.text import html_to_text

TABLES = ("alquran", "expl", "vumika_sura", "surah_name")
SOURCE_DB = "tafheem"
LEGACY_SOURCE_DB = "tafheemul_quran"
CATEGORY_NAME = "কুরআন ও তাফসীর"
BOOK_NAME = "তাফহীমুল কুরআন"
LANGUAGE = "bn"
DEFAULT_DB = Path("data/tafheemul_quran.db")

_BN_DIGITS = str.maketrans("0123456789", "০১২৩৪৫৬৭৮৯")
_ASCII_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")

# Navigation links to the introduction ("ভূমিকা: <a href='#://3'>২</a>"):
# dropped WITH their text -- the number inside is not a footnote marker.
_LINK_RE = re.compile(r"<a\b[^>]*>.*?</a>", re.IGNORECASE | re.DOTALL)
_NAV_LINE_RE = re.compile(r"^\s*\*")  # the "* ... ভূমিকা:" lines left behind
_MARKER_RE = re.compile(r"([০-৯]+)(\s*\(ক\))?")
_FOOTNOTE_PREFIX_RE = re.compile(
    r"^\s*ট[িী]কা\s*[:ঃ]?\s*[০-৯0-9]+\s*\)?\s*(?:\(\s*ক\s*\)\s*\)?)?\s*"
)
_SLASH_RUN_RE = re.compile(r"/{3,}")
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")


def bn(n: int) -> str:
    return str(n).translate(_BN_DIGITS)


def footnote_label(key: int) -> str:
    """expl_id -> printed label: 2 -> "২", -25 -> "২৫(ক)"."""
    return f"{bn(-key)}(ক)" if key < 0 else bn(key)


def _literal_newlines(text: str) -> str:
    return (text or "").replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\r", "\n")


def clean_ayah(raw: str) -> str:
    """Translation text with HTML and navigation removed; markers still in."""
    text = html_to_text(_LINK_RE.sub("", _literal_newlines(raw)))
    return "\n".join(ln for ln in text.split("\n") if not _NAV_LINE_RE.match(ln)).strip()


def clean_footnote(raw: str) -> str:
    text = html_to_text(_literal_newlines(raw))  # keeps cross-reference link text
    text = _SLASH_RUN_RE.sub("", text.replace("[[", "").replace("]]", ""))
    return _FOOTNOTE_PREFIX_RE.sub("", text, count=1).strip()


def clean_intro(raw: str) -> str:
    return html_to_text(_literal_newlines(raw))


@dataclass
class TafheemPage:
    source_page_id: int
    content: str


@dataclass
class Sura:
    number: int
    name: str
    pages: list[TafheemPage] = field(default_factory=list)


@dataclass
class ReadReport:
    n_footnotes: int = 0
    duplicate_footnotes: list[tuple[int, str]] = field(default_factory=list)
    # footnotes whose marker appears in no ayah of their sura
    unreferenced_footnotes: list[tuple[int, str]] = field(default_factory=list)
    # markers (sura, ayah, label) with no footnote row in that sura
    orphan_markers: list[tuple[int, int, str]] = field(default_factory=list)
    skipped_ayah0: int = 0


def _connect(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def _attach_footnotes(
    sura: int, ayah: int, text: str, notes: dict[int, str], report: ReadReport
) -> tuple[str, list[int]]:
    """Strip footnote markers from `text`; return it plus the matched keys in order."""
    keys: list[int] = []

    def repl(m: re.Match) -> str:
        n = int(m.group(1).translate(_ASCII_DIGITS))
        key = -n if m.group(2) else n
        if key in notes:
            if key not in keys:
                keys.append(key)
            return ""
        report.orphan_markers.append((sura, ayah, footnote_label(key)))
        # "২৫(ক)" is unmistakably a marker; a bare unmatched number might be
        # real prose, so it stays.
        return "" if m.group(2) else m.group(0)

    text = _MARKER_RE.sub(repl, text)
    return _MULTI_SPACE_RE.sub(" ", text).strip(), keys


def read_tafheem(path: Path) -> tuple[list[Sura], ReadReport]:
    """Read the four source tables and build one Sura (with its pages) per sura."""
    report = ReadReport()
    con = _connect(path)
    try:
        names = dict(con.execute("SELECT CAST(_id AS INTEGER), sura_name FROM surah_name"))
        intros: dict[int, list[str]] = defaultdict(list)
        for sura, text in con.execute(
            "SELECT CAST(sura_id AS INTEGER), vumika FROM vumika_sura ORDER BY rowid"
        ):
            if cleaned := clean_intro(text):
                intros[sura].append(cleaned)

        notes: dict[int, dict[int, str]] = defaultdict(dict)
        for sura, key, text in con.execute(
            "SELECT CAST(sura_id AS INTEGER), CAST(expl_id AS INTEGER), expels"
            " FROM expl ORDER BY rowid"
        ):
            if key in notes[sura]:
                report.duplicate_footnotes.append((sura, footnote_label(key)))
                continue
            notes[sura][key] = clean_footnote(text)
        report.n_footnotes = sum(len(v) for v in notes.values())

        ayahs = con.execute(
            "SELECT CAST(sura_id AS INTEGER), CAST(ayat_id AS INTEGER), bangla_trans"
            " FROM alquran ORDER BY 1, 2"
        ).fetchall()
    finally:
        con.close()

    suras = {n: Sura(number=n, name=name.strip()) for n, name in sorted(names.items())}
    for n, parts in intros.items():
        if n in suras:
            suras[n].pages.append(TafheemPage(n * 1000, "\n\n".join(parts)))

    referenced: set[tuple[int, int]] = set()
    for sura, ayah, raw in ayahs:
        if ayah == 0:
            report.skipped_ayah0 += 1
            continue
        if sura not in suras:
            continue
        text, keys = _attach_footnotes(sura, ayah, clean_ayah(raw), notes[sura], report)
        referenced.update((sura, k) for k in keys)
        body = f"সূরা {suras[sura].name} {bn(sura)}:{bn(ayah)}\n{text}"
        for k in keys:
            body += f"\n\nটীকা {footnote_label(k)}: {notes[sura][k]}"
        suras[sura].pages.append(TafheemPage(sura * 1000 + ayah, body))

    report.unreferenced_footnotes = sorted(
        (s, footnote_label(k)) for s in notes for k in notes[s] if (s, k) not in referenced
    )
    return list(suras.values()), report


@dataclass
class UpsertStats:
    inserted: int = 0
    updated: int = 0
    unchanged: int = 0
    superseded: int = 0


def upsert_tafheem(session, suras: list[Sura], *, publish: bool) -> UpsertStats:
    """Upsert category/book/chapters/pages. Flushes; the caller commits or rolls back."""
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

    # Prefer a book already in our category; otherwise adopt the oldest book
    # of that name (the legacy import's), so it isn't split in two.
    book = (
        session.query(Book).filter_by(name=BOOK_NAME)
        .order_by((Book.category_id == category.id).desc(), Book.id).first()
    )
    if book is None:
        book = Book(name=BOOK_NAME, language=LANGUAGE, category_id=category.id)
        session.add(book)
        session.flush()
    elif book.category_id != category.id:
        book.category_id = category.id

    chapters: dict[int, Chapter] = {}
    for ch in session.query(Chapter).filter_by(book_id=book.id).order_by(Chapter.id.desc()):
        chapters[ch.position] = ch  # descending id -> the oldest per position wins

    existing = {
        p.source_page_id: p for p in session.query(Page).filter_by(source_db=SOURCE_DB)
    }
    new_status = ContentStatus.published if publish else ContentStatus.draft

    for sura in suras:
        chapter = chapters.get(sura.number)
        if chapter is None:
            chapter = Chapter(book_id=book.id, name=sura.name, position=sura.number)
            session.add(chapter)
            session.flush()
            chapters[sura.number] = chapter
        elif chapter.name != sura.name:
            chapter.name = sura.name

        for tp in sura.pages:
            page = existing.get(tp.source_page_id)
            if page is None:
                session.add(Page(
                    book_id=book.id, chapter_id=chapter.id, content=tp.content,
                    language=LANGUAGE, status=new_status,
                    source_db=SOURCE_DB, source_page_id=tp.source_page_id,
                ))
                stats.inserted += 1
                continue
            changed = False
            # Assigning only on a real change keeps updated_at (onupdate) --
            # and so ingest's "needs re-embedding" check -- quiet on a re-run.
            for attr, value in (
                ("content", tp.content), ("book_id", book.id), ("chapter_id", chapter.id),
            ):
                if getattr(page, attr) != value:
                    setattr(page, attr, value)
                    changed = True
            if publish and page.status != ContentStatus.published:
                page.status = ContentStatus.published
                changed = True
            if changed:
                stats.updated += 1
            else:
                stats.unchanged += 1

    if publish:
        stats.superseded = (
            session.query(Page)
            .filter(Page.source_db == LEGACY_SOURCE_DB, Page.status == ContentStatus.published)
            .update({Page.status: ContentStatus.draft}, synchronize_session=False)
        )
    session.flush()
    return stats


def _print_report(suras: list[Sura], report: ReadReport) -> None:
    n_pages = sum(len(s.pages) for s in suras)
    print(f"suras: {len(suras)}   pages: {n_pages}   footnotes: {report.n_footnotes}")
    print(f"ayah-0 rows skipped (bismillah + intro link): {report.skipped_ayah0}")
    print(f"duplicate footnote rows dropped: {len(report.duplicate_footnotes)} "
          f"{report.duplicate_footnotes[:10]}")
    print(f"footnotes whose marker was not found in any ayah: "
          f"{len(report.unreferenced_footnotes)} {report.unreferenced_footnotes[:20]}")
    print(f"markers with no footnote: {len(report.orphan_markers)} {report.orphan_markers[:20]}")
    samples = [p for s in suras for p in s.pages if p.source_page_id % 1000][:1]
    samples += [p for s in suras for p in s.pages if "টীকা" in p.content][1:2]
    for p in samples:
        print(f"\n--- sample page (source_page_id={p.source_page_id}) ---\n{p.content[:800]}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--publish", action="store_true",
                        help="publish pages (and supersede the legacy import)")
    parser.add_argument("--dry-run", action="store_true", help="roll back instead of committing")
    args = parser.parse_args(argv)

    if not args.db.exists():
        print(f"not found: {args.db}", file=sys.stderr)
        return 1
    suras, report = read_tafheem(args.db)
    _print_report(suras, report)

    from app.db.session import SessionLocal

    session = SessionLocal()
    try:
        stats = upsert_tafheem(session, suras, publish=args.publish)
        print(f"\ninserted: {stats.inserted}   updated: {stats.updated}   "
              f"unchanged: {stats.unchanged}   legacy pages superseded: {stats.superseded}")
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
