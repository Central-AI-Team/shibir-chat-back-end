"""Load the shaheed (martyr) list from a pg_dump into the shaheed table.

Usage:
    uv run python -m scripts.load_shaheed [--sql <path>] [--publish] [--dry-run]

Then embed with `uv run python -m app.rag.ingest` (the only code that writes
to Chroma). See "Adding a data source" in CLAUDE.md.

Upserted on serial_no. New rows are draft unless --publish; --publish also
publishes existing rows. Columns are assigned only when their value changed, so
updated_at (and so re-embedding) moves only for real edits. Creates the table
if it does not exist (there are no migrations). --dry-run rolls back.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from app.loaders.sources import iter_copy_rows

DEFAULT_SQL = Path("data/raw/shaheed/shaheed.sql")

_TEXT_COLUMNS = ("name", "shahadat_date_text", "shahadat_place", "organizational_rank")


@dataclass
class UpsertStats:
    inserted: int = 0
    updated: int = 0
    unchanged: int = 0
    skipped: int = 0


def read_shaheed(path: Path) -> list[dict]:
    """Parsed rows; malformed ones (no serial_no, name or text fields) are dropped by the caller."""
    return list(iter_copy_rows(path, "shaheed"))


def _clean(raw: dict) -> dict | None:
    try:
        serial_no = int(raw["serial_no"])
    except (KeyError, TypeError, ValueError):
        return None
    row = {col: (raw.get(col) or "").strip() for col in _TEXT_COLUMNS}
    if not row["name"]:
        return None
    iso = raw.get("shahadat_date")
    try:
        row["shahadat_date"] = date.fromisoformat(iso) if iso else None
    except ValueError:
        row["shahadat_date"] = None
    row["serial_no"] = serial_no
    return row


def upsert_shaheed(session, rows: list[dict], *, publish: bool) -> UpsertStats:
    """Flushes; the caller commits or rolls back."""
    from app.db.models import ContentStatus, Shaheed

    stats = UpsertStats()
    existing = {s.serial_no: s for s in session.query(Shaheed)}
    new_status = ContentStatus.published if publish else ContentStatus.draft

    for raw in rows:
        row = _clean(raw)
        if row is None:
            stats.skipped += 1
            continue
        rec = existing.get(row["serial_no"])
        if rec is None:
            rec = Shaheed(status=new_status, **row)
            session.add(rec)
            existing[row["serial_no"]] = rec
            stats.inserted += 1
            continue
        changed = False
        for attr, value in row.items():
            if getattr(rec, attr) != value:
                setattr(rec, attr, value)
                changed = True
        if publish and rec.status != ContentStatus.published:
            rec.status = ContentStatus.published
            changed = True
        stats.updated += changed
        stats.unchanged += not changed
    session.flush()
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sql", type=Path, default=DEFAULT_SQL)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="roll back instead of committing")
    args = parser.parse_args(argv)

    if not args.sql.exists():
        print(f"not found: {args.sql}", file=sys.stderr)
        return 1
    rows = read_shaheed(args.sql)
    print(f"{len(rows)} row(s) read from {args.sql}")

    from app.db.base import Base
    from app.db.models import Shaheed
    from app.db.session import SessionLocal, engine

    Base.metadata.create_all(engine, tables=[Shaheed.__table__])  # checkfirst
    session = SessionLocal()
    try:
        stats = upsert_shaheed(session, rows, publish=args.publish)
        print(f"inserted: {stats.inserted}   updated: {stats.updated}   "
              f"unchanged: {stats.unchanged}   skipped: {stats.skipped}")
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
