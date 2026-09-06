"""Remove placeholder / duplicate / orphaned pages from the RAG corpus.

A prior analysis produced three CSVs at the repo root listing pages that
pollute retrieval:

  arabic_placeholder_pages.csv     - pages flagged as Arabic-placeholder / junk.
                                     Column `page_id` (= pages.id).
  book203_cross_contamination.csv  - duplicate pairs. `loser_page_id` is the
                                     bad copy to drop; `duplicates_page_id` is
                                     the canonical copy that MUST be kept.
  orphaned_source_pages.csv        - legacy source rows never migrated to
                                     Postgres. Identified by
                                     (source_db, source_page_id).

Those pages are still `status = published` in Postgres AND still embedded in
Chroma, so they keep coming back on every retrieval and get re-embedded on
every `python -m app.rag.ingest`. This script fixes both, safely:

  * Postgres  - REVERSIBLE soft-exclude. Sets pages.excluded_from_rag = true
                (+ a per-page exclusion_reason). No row is ever deleted.
                app/rag/ingest.py._due_pages() now also skips
                excluded_from_rag rows, so they can never be re-embedded.
  * Chroma    - real delete of every chunk of those pages, via the same
                delete-by-row_key path ingest already uses
                (app/rag/ingest._drop_stale). The Chroma index is rebuildable
                from Postgres, so this is the only destructive step and it is
                recoverable by flipping excluded_from_rag back and re-ingesting.

ID -> row_key mapping (confirmed in app/rag/ingest.py): every chunk of a page
is stored with metadata row_key == f"page_{pages.id}" (chunk ids are
f"page_{id}_c{n}"). Deleting where row_key == "page_<id>" removes all chunks
of that page. Articles are unaffected.

Usage:
    python -m scripts.clean_corpus                  # dry-run (default): report only
    python -m scripts.clean_corpus --apply          # perform it (implies --migrate)
    python -m scripts.clean_corpus --migrate        # only add the Postgres columns
    python -m scripts.clean_corpus --verify         # re-run the post-conditions

--apply is idempotent: a second run updates 0 rows and deletes 0 vectors.

Run from a shell with the app's deps + .env (same as `python -m app.rag.ingest`).
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

from sqlalchemy import text

from app.db.session import SessionLocal, engine
from app.db.models import Page
from app.rag.chroma_client import get_collection
from app.rag.ingest import _drop_stale, _due_pages

ROOT = Path(__file__).parent.parent
ARABIC_CSV = ROOT / "arabic_placeholder_pages.csv"
CONTAM_CSV = ROOT / "book203_cross_contamination.csv"
ORPHAN_CSV = ROOT / "orphaned_source_pages.csv"

SAMPLE_IDS = 15  # how many example ids to print in the dry-run report

# CSV -> the reason label written to pages.exclusion_reason.
SOURCE_KEYS = {
    "arabic": "arabic_placeholder_pages.csv",
    "book203": "book203_cross_contamination.csv:loser",
    "orphaned": "orphaned_source_pages.csv",
}

# Migration -- matches the ADD COLUMN IF NOT EXISTS style of
# scripts/add_source_type_to_articles.py (this repo has no Alembic).
MIGRATION = [
    "ALTER TABLE pages ADD COLUMN IF NOT EXISTS excluded_from_rag BOOLEAN "
    "NOT NULL DEFAULT FALSE",
    "ALTER TABLE pages ADD COLUMN IF NOT EXISTS exclusion_reason TEXT",
]

# Detects the "(আরবী****)" / "****" stubs left where an Arabic original was
# lost in migration -- used only for the advisory content breakdown.
_ASTERISK_STUB = re.compile(r"আরব[িী]\s*[*)ঃ:]|\*{4,}|\(আরব[িী]")


# ---------------------------------------------------------------------------
# exclusion set
# ---------------------------------------------------------------------------

def _read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_exclusions(sources: set[str]) -> tuple[dict[int, set[str]], set[int], dict]:
    """Return (page_id -> {reasons}, canonical_keep_ids, stats).

    `sources` selects which CSVs actually contribute to the removal set
    (canonical KEEP ids are always read from book203, regardless). stats
    carries the per-source id sets so the report and the sanity checks can be
    built without re-parsing.
    """
    arabic = {int(r["page_id"]) for r in _read_csv(ARABIC_CSV)}

    contam = _read_csv(CONTAM_CSV)
    losers = {int(r["loser_page_id"]) for r in contam}
    canonical = {int(r["duplicates_page_id"]) for r in contam}

    # orphaned rows are keyed by source, not by pages.id -- resolve against PG.
    orphan_refs = {(r["source_db"], int(r["source_page_id"]))
                   for r in _read_csv(ORPHAN_CSV)}
    orphan_ids: set[int] = set()
    if orphan_refs:
        with SessionLocal() as s:
            q = s.query(Page.id, Page.source_db, Page.source_page_id).filter(
                Page.source_page_id.in_({spid for _, spid in orphan_refs})
            )
            for pid, sdb, spid in q:
                if (sdb, spid) in orphan_refs:
                    orphan_ids.add(pid)

    reasons: dict[int, set[str]] = defaultdict(set)
    if "arabic" in sources:
        for pid in arabic:
            reasons[pid].add(SOURCE_KEYS["arabic"])
    if "book203" in sources:
        for pid in losers:
            reasons[pid].add(SOURCE_KEYS["book203"])
    if "orphaned" in sources:
        for pid in orphan_ids:
            reasons[pid].add(SOURCE_KEYS["orphaned"])

    stats = {
        "arabic": arabic,
        "losers": losers,
        "canonical": canonical,
        "orphan_refs": orphan_refs,
        "orphan_ids": orphan_ids,
    }
    return reasons, canonical, stats


def sanity_check(reasons: dict[int, set[str]], canonical: set[int], stats: dict) -> dict[int, set[str]]:
    """Fail loudly on keep/remove contradictions. Returns a cleaned reasons map.

    Rule: a page that is a canonical `duplicates_page_id` (a KEEP) must never
    be in the removal set, whatever another CSV says. A book203 loser that is
    also a book203 canonical is a self-contradicting CSV -> hard fail. Any
    other keep/remove overlap (e.g. arabic vs canonical, which is what
    actually occurs here) is stripped from the removal set with a loud warning.
    """
    loser_keep = sorted(stats["losers"] & canonical)
    if loser_keep:
        sys.exit(
            f"FATAL: {len(loser_keep)} id(s) are both a book203 loser AND a "
            f"book203 canonical duplicate: {loser_keep}. The contamination CSV "
            "contradicts itself -- refusing to run."
        )

    cleaned = {pid: rs for pid, rs in reasons.items()}
    conflict = sorted(set(cleaned) & canonical)
    if conflict:
        print("!" * 74)
        print(f"WARNING: {len(conflict)} id(s) are in the removal set AND are a "
              "book203 canonical\n  duplicates_page_id (a KEEP):")
        print(f"  {conflict}")
        print("  Reason(s) flagged: "
              f"{sorted({r for pid in conflict for r in cleaned[pid]})}")
        print("  They are being DROPPED from the removal set -- a page another "
              "CSV marks\n  canonical is never deleted. Fix the CSV conflict "
              "upstream if that is wrong.")
        print("!" * 74)
        for pid in conflict:
            cleaned.pop(pid, None)

    return cleaned


# ---------------------------------------------------------------------------
# report (dry-run)
# ---------------------------------------------------------------------------

def _content_breakdown(session, ids: set[int]) -> dict:
    rows = session.query(Page.id, Page.content).filter(Page.id.in_(ids)).all()
    stub = real = 0
    for _pid, content in rows:
        c = content or ""
        if _ASTERISK_STUB.search(c):
            stub += 1
        else:
            real += 1
    return {"with_arabic_stub": stub, "no_defect_detected": real,
            "found_in_pg": len(rows)}


def report(reasons: dict[int, set[str]], canonical: set[int], stats: dict) -> None:
    exclude = set(reasons)
    col = get_collection()

    print("=" * 74)
    print("CORPUS CLEANUP -- DRY RUN (nothing will be changed)")
    print("=" * 74)

    active = sorted({r for rs in reasons.values() for r in rs})
    print("\n-- exclusion set by source (all CSV contents, then what is active) --")
    print(f"  arabic_placeholder_pages.csv page_id      : {len(stats['arabic'])}")
    print(f"  book203_cross_contamination.csv losers    : {len(stats['losers'])}")
    print(f"  orphaned_source_pages.csv (refs / matched): "
          f"{len(stats['orphan_refs'])} / {len(stats['orphan_ids'])}")
    print(f"  reasons active this run                   : {active or '(none)'}")
    print(f"  COMBINED UNIQUE pages to exclude          : {len(exclude)}")
    print(f"  book203 canonical pages to KEEP           : {len(canonical)}")

    has_col = _col_exists("excluded_from_rag")
    with SessionLocal() as s:
        # explicit column list, not the full entity: the new columns may not
        # exist yet on a fresh DB and a `query(Page)` would SELECT them.
        cols = [Page.id, Page.status] + ([Page.excluded_from_rag] if has_col else [])
        rows = s.query(*cols).filter(Page.id.in_(exclude)).all()
        present_ids = {r[0] for r in rows}
        missing = sorted(exclude - present_ids)
        by_status: dict[str, int] = defaultdict(int)
        already = 0
        for r in rows:
            status = r[1].value if hasattr(r[1], "value") else str(r[1])
            by_status[status] += 1
            if has_col and r[2]:
                already += 1
        canon_present = s.query(Page.id).filter(Page.id.in_(canonical)).count()
        breakdown = _content_breakdown(s, present_ids)
    present = present_ids

    print("\n-- Postgres --")
    print(f"  exclude ids found in pages   : {len(present)} / {len(exclude)}")
    if missing:
        print(f"  exclude ids NOT in pages     : {len(missing)} "
              f"(ignored) e.g. {missing[:10]}")
    print(f"  current status of those pages: {dict(by_status)}")
    if _col_exists("excluded_from_rag"):
        print(f"  already excluded_from_rag    : {already}  "
              f"(a second --apply would touch {len(present) - already})")
    else:
        print("  excluded_from_rag column     : does not exist yet "
              "(--apply / --migrate will add it)")
    print(f"  canonical KEEP ids in pages  : {canon_present} / {len(canonical)}")

    print("\n-- content check on the flagged pages (advisory) --")
    print(f"  pages with an '(আরবী****)' stub : {breakdown['with_arabic_stub']}")
    print(f"  pages with no defect detected   : {breakdown['no_defect_detected']}")
    print("  NOTE: an Arabic stub means only the Arabic ORIGINAL was lost -- the")
    print("  Bengali translation + commentary is still there and retrievable.")
    print("  'no defect detected' pages may be ordinary, usable content.")
    print("  Spot-check before --apply.")

    print("\n-- Chroma --")
    print(f"  collection                   : {col.name}")
    print(f"  total vectors (before)       : {col.count()}")
    print(f"  vectors matching exclude ids : {_chroma_match(col, exclude)}")
    print(f"  vectors for canonical KEEPs  : {_chroma_match(col, canonical)}  "
          "(must remain after --apply)")

    print("\n-- sample exclude ids --")
    for pid in sorted(exclude)[:SAMPLE_IDS]:
        print(f"  page_{pid:<6} <- {', '.join(sorted(reasons[pid]))}")
    if len(exclude) > SAMPLE_IDS:
        print(f"  ... (+{len(exclude) - SAMPLE_IDS} more)")

    print("\nNo changes made. Re-run with --apply to perform the cleanup.")


def _chroma_match(col, ids: set[int]) -> int:
    if not ids:
        return 0
    keys = [f"page_{i}" for i in ids]
    got = col.get(where={"row_key": {"$in": keys}}, include=[])
    return len(got["ids"])


# ---------------------------------------------------------------------------
# migration + apply
# ---------------------------------------------------------------------------

_COL_CACHE: dict[str, bool] = {}


def _col_exists(name: str) -> bool:
    if name not in _COL_CACHE:
        with engine.connect() as conn:
            row = conn.execute(text(
                "SELECT 1 FROM information_schema.columns "
                "WHERE table_name = 'pages' AND column_name = :c"
            ), {"c": name}).first()
        _COL_CACHE[name] = row is not None
    return _COL_CACHE[name]


def migrate() -> None:
    with engine.begin() as conn:
        for stmt in MIGRATION:
            conn.execute(text(stmt))
    _COL_CACHE.clear()
    print("Postgres: pages.excluded_from_rag (bool, default false) + "
          "pages.exclusion_reason (text) present.")


def apply(reasons: dict[int, set[str]], canonical: set[int], stats: dict) -> None:
    exclude = set(reasons)
    migrate()

    with SessionLocal() as s:
        present_ids = {pid for (pid,) in
                       s.query(Page.id).filter(Page.id.in_(exclude)).all()}
        to_update = s.query(Page).filter(
            Page.id.in_(present_ids),
            Page.excluded_from_rag.is_(False),
        ).all()

        for p in to_update:
            p.excluded_from_rag = True
            p.exclusion_reason = "; ".join(sorted(reasons[p.id]))
        s.commit()
        print(f"\nPostgres: set excluded_from_rag on {len(to_update)} page(s) "
              f"({len(present_ids) - len(to_update)} already excluded).")

    col = get_collection()
    before = col.count()
    matched = _chroma_match(col, present_ids)
    for pid in sorted(present_ids):
        _drop_stale(col, "page", pid)          # reuse ingest.py's delete path
    after = col.count()
    print(f"Chroma: {matched} vector(s) matched the exclude ids; "
          f"collection {before} -> {after} (removed {before - after}).")

    verify(reasons, canonical, stats)


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------

def verify(reasons: dict[int, set[str]], canonical: set[int], stats: dict) -> None:
    exclude = set(reasons)
    col = get_collection()
    failures: list[str] = []

    left = _chroma_match(col, exclude)
    print(f"\n[verify] excluded row_keys still in Chroma : {left}")
    if left:
        failures.append(f"{left} excluded vector(s) still present in Chroma")

    canon_left = _chroma_match(col, canonical)
    print(f"[verify] canonical KEEP row_keys in Chroma : {canon_left} / "
          f"expected ~{len(canonical)} pages present")
    if canonical and canon_left == 0:
        failures.append("canonical duplicate pages have no vectors -- the WRONG "
                        "side may have been deleted")

    if not _col_exists("excluded_from_rag"):
        failures.append("pages.excluded_from_rag column missing")
    else:
        with SessionLocal() as s:
            leaked = s.query(Page.id).filter(
                Page.id.in_(exclude), Page.excluded_from_rag.is_(False)
            ).count()
            print(f"[verify] exclude pages NOT soft-excluded in PG : {leaked}")
            if leaked:
                failures.append(f"{leaked} exclude page(s) not soft-excluded")

            due_ids = {p.id for p in _due_pages(s)}
            re_embed = sorted(due_ids & exclude)
            print(f"[verify] exclude pages ingest would re-embed  : {len(re_embed)}")
            if re_embed:
                failures.append(f"ingest._due_pages() still returns excluded ids: "
                                f"{re_embed[:20]}")

    if failures:
        print("\nVERIFY FAILED:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("\n[verify] all post-conditions hold.")


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        prog="python -m scripts.clean_corpus",
        description=__doc__.split("\n\n")[0],
    )
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--apply", action="store_true",
                   help="perform the cleanup (soft-exclude in PG + delete "
                        "Chroma vectors); implies --migrate; idempotent")
    g.add_argument("--migrate", action="store_true",
                   help="only add the pages.excluded_from_rag / exclusion_reason "
                        "columns, change nothing else")
    g.add_argument("--verify", action="store_true",
                   help="re-check the post-conditions against the live stores")
    ap.add_argument("--dry-run", action="store_true",
                    help="explicit no-op report (this is also the default)")
    ap.add_argument("--sources", default="arabic,book203,orphaned",
                    help="comma list of CSVs to apply: any of "
                         "arabic,book203,orphaned (default: all three)")
    args = ap.parse_args()

    sources = {s.strip() for s in args.sources.split(",") if s.strip()}
    unknown = sources - set(SOURCE_KEYS)
    if unknown:
        ap.error(f"unknown --sources value(s): {sorted(unknown)}; "
                 f"choose from {sorted(SOURCE_KEYS)}")
    print(f"sources in play: {sorted(sources)}\n")

    reasons, canonical, stats = build_exclusions(sources)
    reasons = sanity_check(reasons, canonical, stats)

    if args.migrate:
        migrate()
    elif args.verify:
        verify(reasons, canonical, stats)
    elif args.apply:
        apply(reasons, canonical, stats)
    else:
        report(reasons, canonical, stats)


if __name__ == "__main__":
    main()
