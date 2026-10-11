"""Load the central president / secretary-general list (CSV) into the articles table.

Usage:
    uv run python -m scripts.load_leadership [--csv <path>] [--publish] [--dry-run]

Then embed with `uv run python -m app.rag.ingest` (the only code that writes
to Chroma). See "Adding a data source" in CLAUDE.md.

The CSV has a title line, a header line, then one row per term: serial number,
session, start, end, president, secretary general. The whole table becomes ONE
article (source_type "central-leadership", source_ref "shibir-central-president")
with one sentence-paragraph per term, because a bare row like "৩৮,২০১২,..." says
nothing once it is cut into a chunk; the chunk header carries the title and each
paragraph names its role and period. An end of "-" means the term is ongoing.

Upserted on (source_type, source_ref); columns are assigned only when their value
changed, so updated_at (and so re-embedding) moves only for real edits. New rows
are draft unless --publish. --dry-run rolls back.
"""

from __future__ import annotations

import argparse
import csv
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CSV = Path("data/shibir-central-president.csv")
SOURCE_TYPE = "central-leadership"
SOURCE_REF = "shibir-central-president"
TITLE = "ছাত্রশিবিরের কেন্দ্রীয় সভাপতি ও সেক্রেটারি জেনারেলবৃন্দের তালিকা"
ONGOING = "চলমান"


@dataclass(frozen=True)
class Term:
    serial: int
    session: str
    start: str
    end: str
    president: str
    secretary: str


def _clean(cell: str) -> str:
    return unicodedata.normalize("NFC", " ".join(cell.split()))


def read_terms(path: Path) -> tuple[list[Term], int]:
    """(terms, skipped). Rows without a numeric serial number (title, header) are skipped."""
    terms, skipped = [], 0
    with open(path, encoding="utf-8-sig", newline="") as fh:
        for raw in csv.reader(fh):
            cells = [_clean(c) for c in raw] + [""] * 6
            if not cells[0].isdigit():
                skipped += 1
                continue
            if not (cells[4] and cells[5]):
                skipped += 1
                continue
            terms.append(Term(int(cells[0]), *cells[1:6]))
    return terms, skipped


def render(terms: list[Term]) -> str:
    paras = [TITLE]
    for t in terms:
        end = ONGOING if t.end in ("", "-", "–") else t.end
        paras.append(
            f"ক্রম {t.serial}: সেশন {t.session} ({t.start} থেকে {end})। "
            f"কেন্দ্রীয় সভাপতি: {t.president}। সেক্রেটারি জেনারেল: {t.secretary}।"
        )
    return "\n\n".join(paras)


def upsert_article(session, terms: list[Term], *, publish: bool) -> str:
    """Returns "inserted" | "updated" | "unchanged". Flushes; the caller commits."""
    from app.db.models import Article, ContentStatus

    fields = {
        "title": TITLE,
        "content": render(terms),
        "language": "bn",
        "source_metadata": {"terms": len(terms), "file": DEFAULT_CSV.name},
    }
    rec = (
        session.query(Article)
        .filter(Article.source_type == SOURCE_TYPE, Article.source_ref == SOURCE_REF)
        .one_or_none()
    )
    if rec is None:
        rec = Article(
            source_type=SOURCE_TYPE,
            source_ref=SOURCE_REF,
            status=ContentStatus.published if publish else ContentStatus.draft,
            **fields,
        )
        session.add(rec)
        session.flush()
        return "inserted"
    changed = False
    for attr, value in fields.items():
        if getattr(rec, attr) != value:
            setattr(rec, attr, value)
            changed = True
    if publish and rec.status != ContentStatus.published:
        rec.status = ContentStatus.published
        changed = True
    session.flush()
    return "updated" if changed else "unchanged"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="roll back instead of committing")
    args = parser.parse_args(argv)

    if not args.csv.exists():
        print(f"not found: {args.csv}", file=sys.stderr)
        return 1
    terms, skipped = read_terms(args.csv)
    print(f"{len(terms)} term(s) read from {args.csv} ({skipped} non-data line(s) skipped)")
    if not terms:
        return 1
    serials = [t.serial for t in terms]
    gaps = sorted(set(range(1, max(serials) + 1)) - set(serials))
    if gaps or len(serials) != len(set(serials)):
        print(f"WARNING: serial numbers have gaps {gaps} or duplicates", file=sys.stderr)

    from app.db.session import SessionLocal

    session = SessionLocal()
    try:
        result = upsert_article(session, terms, publish=args.publish)
        print(f"article {result}")
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
