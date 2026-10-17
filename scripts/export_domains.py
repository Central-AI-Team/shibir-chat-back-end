"""Export the Postgres corpus as one self-contained SQL file per domain.

Usage:
    uv run python -m scripts.export_domains [--out data/export] [--only books tafheem ...]

Writes <out>/<domain>.sql for each domain in DOMAINS. Restore on the server:

    docker compose exec -T db psql -U postgres -d shibir_chat < data/export/books.sql
    docker compose exec app python -m app.rag.ingest        # embeds what was loaded

Design (each point is here because the alternative broke something):
  * Published rows only. Drafts are superseded legacy imports (see CLAUDE.md
    §9); they must not travel.
  * `embedded_at` is exported as NULL, so the server's ingest embeds the rows
    (the local Chroma index is not transferred).
  * Primary keys are preserved and every INSERT is `ON CONFLICT (id) DO UPDATE`,
    so a file can be re-run, files can be restored in any order, and the shared
    parent rows (categories/books/chapters, e.g. Tafheem book 166 also holds
    older Tarun pages) may appear in several files. The update never touches
    `embedded_at`/`created_at`, so unchanged rows are not re-embedded.
  * Sequences are bumped at the end so new rows on the server don't collide.
  * Split by page `source_db` / article `source_type`, not by book: one book can
    mix sources.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path

from psycopg2.extensions import register_adapter
from psycopg2.extras import Json

register_adapter(dict, Json)  # jsonb columns come back as dicts

BATCH = 200


@dataclass(frozen=True)
class Domain:
    name: str
    table: str  # "pages" or "articles"
    column: str  # filter column
    values: tuple[str, ...]
    doc: str


DOMAINS = [
    Domain("books", "pages", "source_db", ("tarun", "nobin", "books_sql"),
           "all book pages (Tarun + Nobin + the 6 added books)"),
    Domain("tafheem", "pages", "source_db", ("tafheem",), "Tafheemul Quran, one page per ayah"),
    Domain("articles", "articles", "source_type", ("cs-post", "pp-article"), "cs-posts + pp-articles"),
    Domain("shaheed", "articles", "source_type", ("shahid_basic_info", "web_article"),
           "martyr biographies and summaries"),
    Domain("leadership", "articles", "source_type", ("central-leadership",),
           "central president / secretary-general list"),
]
# Add a domain (e.g. iconic persons) by appending a Domain once its rows exist.

_PARENTS = {"pages": ["categories", "books", "chapters"], "articles": []}
_NEVER_UPDATE = {"id", "embedded_at", "created_at"}


def _columns(cur, table: str) -> list[str]:
    cur.execute(
        "select column_name from information_schema.columns "
        "where table_schema='public' and table_name=%s order by ordinal_position", (table,))
    return [r[0] for r in cur.fetchall()]


def _insert_sql(cur, table: str, cols: list[str], rows: list[tuple]) -> str:
    update = [c for c in cols if c not in _NEVER_UPDATE]
    values = ",\n".join(
        cur.mogrify("(" + ",".join(["%s"] * len(cols)) + ")", r).decode() for r in rows)
    sets = ", ".join(f"{c}=EXCLUDED.{c}" for c in update)
    return (f"INSERT INTO {table} ({', '.join(cols)}) VALUES\n{values}\n"
            f"ON CONFLICT (id) DO UPDATE SET {sets};\n")


def _dump(cur, out, table: str, where: str, params: tuple, order: str = "id") -> int:
    cols = _columns(cur, table)
    cur.execute(f"select {', '.join(cols)} from {table} where {where} order by {order}", params)
    n = 0
    while True:
        rows = cur.fetchmany(BATCH)
        if not rows:
            return n
        if "embedded_at" in cols:
            i = cols.index("embedded_at")
            rows = [r[:i] + (None,) + r[i + 1:] for r in rows]
        out.write(_insert_sql(cur, table, cols, rows))
        n += len(rows)


def export_domain(conn, domain: Domain, path: Path) -> dict[str, int]:
    cur = conn.cursor()
    counts: dict[str, int] = {}
    where = f"status = 'published' and {domain.column} = any(%s)"
    params = (list(domain.values),)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as out:
        out.write(f"-- domain: {domain.name} ({domain.doc})\n-- idempotent; see scripts/export_domains.py\n")
        out.write("BEGIN;\nSET client_encoding = 'UTF8';\n\n")
        if domain.table == "pages":
            sub = f"select {{col}} from pages where {where}"
            counts["categories"] = _dump(
                cur, out, "categories",
                f"id in (select category_id from books where id in ({sub.format(col='book_id')}))",
                params)
            counts["books"] = _dump(cur, out, "books", f"id in ({sub.format(col='book_id')})", params)
            counts["chapters"] = _dump(
                cur, out, "chapters", f"id in ({sub.format(col='chapter_id')})", params)
        counts[domain.table] = _dump(cur, out, domain.table, where, params)
        for t in _PARENTS[domain.table] + [domain.table]:
            out.write(f"SELECT setval(pg_get_serial_sequence('{t}', 'id'), "
                      f"(SELECT COALESCE(MAX(id), 1) FROM {t}));\n")
        out.write("COMMIT;\n")
    tmp.replace(path)
    return counts


@dataclass(frozen=True)
class DumpDomain:
    """A reference dataset that exists only as a pg_dump file (not in our Postgres)."""

    name: str
    src: str  # path under --dumps-dir
    tables: tuple[str, ...]  # drop order: children first
    doc: str


DUMP_DOMAINS = [
    DumpDomain("syllabus", "Postgres-data/syllabus/syllabus.sql",
               ("syllabus_items", "syllabus_sections", "syllabus_subjects", "syllabus_levels"),
               "syllabus: levels, subjects, sections, items"),
    DumpDomain("site-directory", "shibir-site.postgresql.sql",
               ("websites_website", "departments_department"),
               "departments and websites"),
]
# pg_dump session noise that breaks a restore on an older/other server.
_DROP_LINES = ("\\restrict", "\\unrestrict", "SET transaction_timeout")


def export_dump(src: Path, dst: Path, domain: DumpDomain) -> int:
    """Make a pg_dump re-runnable: DROP ... IF EXISTS first, one transaction, and
    without the lines Postgres 16 rejects (transaction_timeout is PG17+; \\restrict
    is psql-version specific). Safe because these tables belong to this dataset
    alone; the app never writes to them. Returns the number of COPY data rows."""
    lines = src.read_text(encoding="utf-8").splitlines()
    body = [ln for ln in lines if not ln.startswith(_DROP_LINES)]
    rows, in_copy = 0, False
    for ln in body:
        if ln.startswith("COPY "):
            in_copy = True
        elif in_copy and ln == "\\.":
            in_copy = False
        elif in_copy:
            rows += 1
    drops = "".join(f"DROP TABLE IF EXISTS public.{t} CASCADE;\n" for t in domain.tables)
    dst.write_text(
        f"-- domain: {domain.name} ({domain.doc})\n-- re-runnable: drops and recreates only its own tables\n"
        f"BEGIN;\n{drops}\n" + "\n".join(body) + "\nCOMMIT;\n", encoding="utf-8")
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=Path("data/export"))
    parser.add_argument("--dumps-dir", type=Path, default=Path("data/archive-data"),
                        help="where the raw pg_dump files for the dump-only domains live")
    parser.add_argument("--only", nargs="+", choices=[d.name for d in DOMAINS + DUMP_DOMAINS])
    args = parser.parse_args(argv)

    from app.db.session import engine

    args.out.mkdir(parents=True, exist_ok=True)
    conn = engine.raw_connection()
    try:
        for d in DOMAINS:
            if args.only and d.name not in args.only:
                continue
            path = args.out / f"{d.name}.sql"
            counts = export_domain(conn, d, path)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
            rows = ", ".join(f"{k}={v}" for k, v in counts.items())
            print(f"{path}  {path.stat().st_size / 1e6:6.1f} MB  sha256:{digest}  [{rows}]")
    finally:
        conn.close()
    for d in DUMP_DOMAINS:
        if args.only and d.name not in args.only:
            continue
        src, path = args.dumps_dir / d.src, args.out / f"{d.name}.sql"
        if not src.exists():
            print(f"skip {d.name}: not found {src}", file=sys.stderr)
            continue
        rows = export_dump(src, path, d)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
        print(f"{path}  {path.stat().st_size / 1e6:6.1f} MB  sha256:{digest}  [rows={rows}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
