"""Flag byte-for-byte duplicate `pages.content` rows so they stop being

embedded twice.

Two independent causes produce these duplicates:
  * the same book was migrated from BOTH legacy SQLite DBs
    (source_db='tarun' and source_db='nobin'), so it exists twice.
  * the same source_db has duplicate rows on its own -- a re-run of the
    ingest/migration script without an idempotency check
    (scripts/migrate_sqlite_to_postgres.py).

Grouping is by md5(content.strip()) -- an exact-content match, not a fuzzy
one. Within each duplicate group, ONE row is kept as canonical and every
other row is soft-excluded:

    excluded_from_rag = True
    exclusion_reason  = "duplicate_content:<canonical_page_id>"
    (appended with "; " if the row already carried a different reason --
    e.g. from scripts/clean_corpus.py -- never overwritten)

Canonical choice, in order:
  1. prefer a row that is NOT already excluded_from_rag=True
  2. among those, prefer source_db='tarun' (larger/primary corpus)
  3. among those, prefer the lowest id

Step 2 is a real assumption, not a certainty -- a duplicate group where the
tie survives step 2 (e.g. every surviving copy is already 'tarun', or none
of them are) is printed as AMBIGUOUS for manual review. The id tiebreak
still produces a deterministic answer either way; ambiguity only means "a
human should eyeball this one", it never blocks the run.

Nothing is ever deleted. This is a soft, reversible flag -- see
scripts/clean_corpus.py, which established the same excluded_from_rag /
exclusion_reason convention this script reuses.

A subset of AMBIGUOUS groups turned out, on manual inspection, to be real
cross-book content mislabeling rather than a harmless source_db tie -- the
not-excluded -> tarun -> lowest-id heuristic has no way to know which book a
page's content actually belongs to, and picking wrong would silently exclude
correctly-labeled content while leaving a mislabeled duplicate active.
scripts/dedupe_overrides.csv (content_hash,canonical_page_id) lets a human
override the canonical choice for a specific group once they've checked it.
Any AMBIGUOUS group with no matching override is SKIPPED by --apply (and its
rows counted separately in the dry-run report) rather than resolved by
guessing.

Chroma is handled the same way clean_corpus.py handles it: app/rag/ingest.py
_due_pages() filters `excluded_from_rag = False` BEFORE deciding what needs
(re-)embedding, so a page that becomes excluded here is simply never visited
again by `python -m app.rag.ingest` -- which means a plain ingest re-run will
NOT remove vectors a duplicate page already has sitting in Chroma. --apply
therefore also deletes those vectors directly, via ingest.py's own
_drop_stale() (row_key == "page_<id>"), exactly like clean_corpus.py --apply
does.

Usage:
    python -m scripts.dedupe_pages             # dry-run (default): report only
    python -m scripts.dedupe_pages --apply      # perform it (Postgres + Chroma)
    python -m scripts.dedupe_pages --verify     # re-check post-conditions

--apply is idempotent: a second run touches 0 new rows and deletes 0 vectors.

Run from a shell with the app's deps + .env (same as `python -m app.rag.ingest`).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy.orm import joinedload

from app.db.models import Book, Page
from app.db.session import SessionLocal
from app.rag.chroma_client import get_collection
from app.rag.ingest import _drop_stale

REASON_PREFIX = "duplicate_content"
SAMPLE_NON_AMBIGUOUS = 10
OVERRIDES_PATH = Path(__file__).parent / "dedupe_overrides.csv"


def load_overrides(path: Path = OVERRIDES_PATH) -> dict[str, int]:
    """Manual canonical-id decisions for AMBIGUOUS groups: content_hash ->
    canonical_page_id, from a human-reviewed spreadsheet converted to CSV.

    Missing file or a header-only file just means nothing has been resolved
    yet -- every AMBIGUOUS group stays blocked until it has a row here.
    """
    if not path.exists():
        return {}
    overrides: dict[str, int] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            content_hash = (row.get("content_hash") or "").strip()
            page_id = (row.get("canonical_page_id") or "").strip()
            if content_hash and page_id:
                overrides[content_hash] = int(page_id)
    return overrides


@dataclass
class Group:
    content_hash: str
    rows: list[Page]
    canonical: Page
    ambiguous: bool
    all_pre_excluded: bool
    was_overridden: bool = False
    new_excludes: list[Page] = field(default_factory=list)


# ---------------------------------------------------------------------------
# grouping + canonical choice
# ---------------------------------------------------------------------------

def _content_hash(content: str | None) -> str:
    return hashlib.md5((content or "").strip().encode("utf-8")).hexdigest()


def load_pages(session) -> list[Page]:
    return (
        session.query(Page)
        .options(joinedload(Page.book).joinedload(Book.category))
        .all()
    )


def find_duplicate_groups(pages: list[Page]) -> dict[str, list[Page]]:
    by_hash: dict[str, list[Page]] = defaultdict(list)
    for p in pages:
        by_hash[_content_hash(p.content)].append(p)
    return {h: rows for h, rows in by_hash.items() if len(rows) > 1}


def choose_canonical(
    rows: list[Page],
    content_hash: str,
    overrides: dict[str, int] | None = None,
) -> tuple[Page, bool, bool, bool]:
    """Return (canonical_row, ambiguous, all_pre_excluded, was_overridden).

    A manual override for `content_hash` wins outright -- a human has already
    inspected the actual content and decided which row is correctly labeled,
    so ambiguous is forced False and was_overridden=True. Otherwise falls
    through to the original not-excluded -> tarun -> lowest-id heuristic.

    ambiguous: the source_db preference (step 2) did not narrow the field to
    a single row -- the id tiebreak (step 3) still ran, but a human should
    look at this group before trusting the choice (see dedupe_overrides.csv).
    all_pre_excluded: every row in the group was already excluded_from_rag
    before this script ran (canonical is being chosen among already-excluded
    rows purely so a "duplicate_content:<id>" pointer exists; nothing new
    gets excluded in this group).
    """
    if overrides and content_hash in overrides:
        override_id = overrides[content_hash]
        canonical = next((r for r in rows if r.id == override_id), None)
        if canonical is not None:
            all_pre_excluded = not any(not r.excluded_from_rag for r in rows)
            return canonical, False, all_pre_excluded, True
        # override points at an id not in this group -- don't silently
        # ignore a bad override, surface it and fall through to the heuristic
        print(f"WARNING: override for {content_hash[:12]} names id="
              f"{override_id}, which is not in this group's rows "
              f"({[r.id for r in rows]}). Ignoring override for this group.")

    not_excluded = [r for r in rows if not r.excluded_from_rag]
    all_pre_excluded = not not_excluded
    pool = not_excluded if not_excluded else rows

    tarun = [r for r in pool if r.source_db == "tarun"]
    candidates = tarun if tarun else pool
    ambiguous = len(candidates) > 1

    canonical = min(candidates, key=lambda r: r.id)
    return canonical, ambiguous, all_pre_excluded, False


def build_groups(
    duplicate_groups: dict[str, list[Page]],
    overrides: dict[str, int] | None = None,
) -> list[Group]:
    groups = []
    for h, rows in duplicate_groups.items():
        canonical, ambiguous, all_pre_excluded, was_overridden = choose_canonical(
            rows, h, overrides
        )
        new_excludes = [
            r for r in rows if r.id != canonical.id and not r.excluded_from_rag
        ]
        groups.append(Group(
            content_hash=h, rows=sorted(rows, key=lambda r: r.id),
            canonical=canonical, ambiguous=ambiguous,
            all_pre_excluded=all_pre_excluded, was_overridden=was_overridden,
            new_excludes=new_excludes,
        ))
    return groups


def _split_actionable(groups: list[Group]) -> tuple[list[Group], list[Group]]:
    """(actionable, blocked). blocked = still ambiguous with no override --
    left completely untouched by apply()/verify()."""
    blocked = [g for g in groups if g.ambiguous and not g.was_overridden]
    actionable = [g for g in groups if not (g.ambiguous and not g.was_overridden)]
    return actionable, blocked


def _row_label(r: Page) -> str:
    book = r.book.name if r.book else "Unknown"
    cat = r.book.category.name if r.book and r.book.category else "Unknown"
    excl = f"EXCLUDED[{r.exclusion_reason}]" if r.excluded_from_rag else "active"
    return (f"id={r.id:<6} source_db={r.source_db or 'unknown':<6} "
            f"book={book!r} ({cat}) {excl}")


# ---------------------------------------------------------------------------
# report (dry-run)
# ---------------------------------------------------------------------------

def report(groups: list[Group]) -> None:
    actionable, blocked = _split_actionable(groups)
    overridden = [g for g in actionable if g.was_overridden]
    auto = [g for g in actionable if not g.was_overridden]

    print("=" * 78)
    print("DEDUPE PAGES -- DRY RUN (nothing will be changed)")
    print("=" * 78)

    print(f"\nduplicate-content groups found        : {len(groups)}")
    print(f"  auto-resolved (no ambiguity)        : {len(auto)}")
    print(f"  resolved via manual override        : {len(overridden)}  "
          "(scripts/dedupe_overrides.csv)")
    print(f"  BLOCKED pending manual review       : {len(blocked)}  "
          "<- add these to dedupe_overrides.csv before --apply touches them")

    actionable_rows = sum(len(g.rows) for g in actionable)
    actionable_already_excluded = sum(
        1 for g in actionable for r in g.rows if r.excluded_from_rag
    )
    actionable_newly_excluded = sum(len(g.new_excludes) for g in actionable)
    all_pre_excluded_groups = [g for g in actionable if g.all_pre_excluded]
    blocked_rows = sum(len(g.rows) for g in blocked)

    print(f"\npages in ACTIONABLE groups (all copies) : {actionable_rows}")
    print(f"  already excluded_from_rag=True        : {actionable_already_excluded}")
    print(f"  NOT excluded (would be flagged)       : "
          f"{actionable_rows - actionable_already_excluded}")
    print(f"rows this run would newly exclude       : {actionable_newly_excluded}")
    print(f"groups fully pre-excluded already       : {len(all_pre_excluded_groups)} "
          "(no new exclusions, canonical chosen for pointer only)")
    print(f"\npages in BLOCKED groups (left untouched): {blocked_rows}  "
          "-- excluded_from_rag NOT changed for these")

    print("\n" + "-" * 78)
    print(f"BLOCKED groups -- pending manual review ({len(blocked)}):")
    print("-" * 78)
    if not blocked:
        print("  (none)")
    for g in blocked:
        print(f"\n  group {g.content_hash[:12]}  "
              "-> NOT resolved (source_db tie, no matching override)")
        for r in g.rows:
            print(f"    {_row_label(r)}")

    if overridden:
        print("\n" + "-" * 78)
        print(f"Resolved via manual override ({len(overridden)}):")
        print("-" * 78)
        for g in overridden:
            print(f"\n  group {g.content_hash[:12]}  "
                  f"-> canonical id={g.canonical.id} (manual override)")
            for r in g.rows:
                marker = "  KEEP -> " if r.id == g.canonical.id else "  drop -> "
                print(f"{marker}{_row_label(r)}")

    print("\n" + "-" * 78)
    print(f"Auto-resolved groups -- sample ({min(SAMPLE_NON_AMBIGUOUS, len(auto))} "
          f"of {len(auto)}):")
    print("-" * 78)
    for g in auto[:SAMPLE_NON_AMBIGUOUS]:
        print(f"\n  group {g.content_hash[:12]}  -> canonical id={g.canonical.id}")
        for r in g.rows:
            marker = "  KEEP -> " if r.id == g.canonical.id else "  drop -> "
            print(f"{marker}{_row_label(r)}")

    print("\n" + "-" * 78)
    print("Per-book breakdown (rows this run would newly exclude, actionable only):")
    print("-" * 78)
    by_book: dict[str, int] = defaultdict(int)
    for g in actionable:
        for r in g.new_excludes:
            by_book[r.book.name if r.book else "Unknown"] += 1
    for name, count in sorted(by_book.items(), key=lambda kv: -kv[1]):
        print(f"  {count:>4}  {name}")
    if not by_book:
        print("  (none)")

    print(f"\nSUMMARY: {len(auto)} group(s) auto-resolved, "
          f"{len(overridden)} group(s) resolved via manual override, "
          f"{len(blocked)} group(s) still BLOCKED pending manual review.")
    print("No changes made. Re-run with --apply to perform the exclusion "
          "(BLOCKED groups are skipped until dedupe_overrides.csv resolves them).")


# ---------------------------------------------------------------------------
# apply
# ---------------------------------------------------------------------------

def _chroma_match(col, ids: set[int]) -> int:
    if not ids:
        return 0
    keys = [f"page_{i}" for i in ids]
    got = col.get(where={"row_key": {"$in": keys}}, include=[])
    return len(got["ids"])


def apply(session, groups: list[Group]) -> None:
    actionable, blocked = _split_actionable(groups)

    if blocked:
        print(f"SKIPPING {len(blocked)} group(s) pending manual review "
              "(ambiguous, no override in scripts/dedupe_overrides.csv):")
        for g in blocked:
            print(f"  {g.content_hash[:12]}  ids={[r.id for r in g.rows]}")
        print("Resolve these via the manual-review spreadsheet, add rows to "
              "dedupe_overrides.csv, then re-run.\n")

    updated = 0
    reason_appended = 0
    exclude_ids: set[int] = set()

    for g in actionable:
        reason = f"{REASON_PREFIX}:{g.canonical.id}"
        for r in g.rows:
            if r.id == g.canonical.id:
                continue
            exclude_ids.add(r.id)
            if not r.excluded_from_rag:
                r.excluded_from_rag = True
                r.exclusion_reason = reason
                updated += 1
            elif reason not in (r.exclusion_reason or ""):
                r.exclusion_reason = (
                    f"{r.exclusion_reason}; {reason}" if r.exclusion_reason else reason
                )
                reason_appended += 1
    session.commit()
    print(f"Postgres: newly excluded {updated} page(s); appended the "
          f"duplicate_content reason to {reason_appended} already-excluded "
          "page(s) that lacked it.")

    col = get_collection()
    before = col.count()
    matched = _chroma_match(col, exclude_ids)
    for pid in sorted(exclude_ids):
        _drop_stale(col, "page", pid)
    after = col.count()
    print(f"Chroma: {matched} vector(s) matched the excluded ids; "
          f"collection {before} -> {after} (removed {before - after}).")

    overridden = [g for g in actionable if g.was_overridden]
    auto = [g for g in actionable if not g.was_overridden]
    print(f"\nSUMMARY: {len(auto)} group(s) auto-resolved, "
          f"{len(overridden)} group(s) resolved via manual override, "
          f"{len(blocked)} group(s) still BLOCKED pending manual review.")


def verify(session, groups: list[Group]) -> None:
    actionable, blocked = _split_actionable(groups)
    col = get_collection()
    failures: list[str] = []

    exclude_ids = {r.id for g in actionable for r in g.rows if r.id != g.canonical.id}
    canonical_ids = {g.canonical.id for g in actionable}

    left = _chroma_match(col, exclude_ids)
    print(f"[verify] excluded duplicate row_keys still in Chroma : {left}")
    if left:
        failures.append(f"{left} excluded duplicate vector(s) still present in Chroma")

    canon_left = _chroma_match(col, canonical_ids)
    print(f"[verify] canonical row_keys in Chroma                : {canon_left} "
          f"/ up to {len(canonical_ids)} pages (0 is fine -- means not yet (re-)ingested)")

    not_excluded = session.query(Page.id).filter(
        Page.id.in_(exclude_ids), Page.excluded_from_rag.is_(False)
    ).count()
    print(f"[verify] excluded ids NOT soft-excluded in PG        : {not_excluded}")
    if not_excluded:
        failures.append(f"{not_excluded} duplicate id(s) not soft-excluded in Postgres")

    print(f"[verify] BLOCKED groups skipped (left untouched)     : {len(blocked)}")

    if failures:
        print("\nVERIFY FAILED:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print(f"\n[verify] all post-conditions hold ({len(blocked)} group(s) "
          "intentionally left blocked pending manual review).")


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        prog="python -m scripts.dedupe_pages",
        description=__doc__.split("\n\n")[0],
    )
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--apply", action="store_true",
                   help="perform the exclusion (Postgres soft-exclude + Chroma "
                        "vector delete); idempotent")
    g.add_argument("--verify", action="store_true",
                   help="re-check post-conditions against the live stores")
    ap.add_argument("--dry-run", action="store_true",
                     help="explicit no-op report (this is also the default)")
    args = ap.parse_args()

    overrides = load_overrides()
    session = SessionLocal()
    try:
        pages = load_pages(session)
        duplicate_groups = find_duplicate_groups(pages)
        groups = build_groups(duplicate_groups, overrides)

        if args.verify:
            verify(session, groups)
        elif args.apply:
            apply(session, groups)
        else:
            report(groups)
    finally:
        session.close()


if __name__ == "__main__":
    main()
