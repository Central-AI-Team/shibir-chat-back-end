"""Reproducible, spread-out sample of real published pages for hand-labelling.

Feeds the retrieval eval dataset build (scripts/retrieval_eval_questions.json):
you read the emitted pages, write ONE paraphrased question per page whose
answer is fully contained on that page, and use the page's row_key as the
gold id. See scripts/eval_retrieval.py's _README.

What it does:
  1. Loads the KNOWN-BAD page ids from the three cleanup CSVs in the repo root
     (arabic_placeholder_pages.csv, book203_cross_contamination.csv,
     orphaned_source_pages.csv) and excludes them.
  2. Reads every published Page from Postgres with its book / chapter /
     category, drops excluded ids and pages too short to self-contain an
     answer.
  3. Round-robins across books (ordered to spread over categories and
     early/mid/late chapter positions) so the sample never clusters on one
     book.
  4. Cross-checks each sampled page's row_key against the LIVE Chroma
     collection (settings.chroma_collection_name) so you never label a page
     that retrieval can't return.
  5. Writes the sample (id, row_key, book, chapter, category, position,
     full content) as JSON.

Deterministic: fixed SEED. Re-running produces the same sample.

Usage:
    python -m scripts.sample_corpus_pages [--n 70] [--min-chars 600]
                                          [--out scripts/_corpus_sample.json]

Run from a shell with the app's deps + .env (same as `python -m app.rag.ingest`).
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

from sqlalchemy.orm import joinedload

from app.db.models import Book, ContentStatus, Page
from app.db.session import SessionLocal
from app.rag.chroma_client import get_collection

ROOT = Path(__file__).parent.parent
SEED = 20260904
DEFAULT_N = 70
DEFAULT_MIN_CHARS = 600
DEFAULT_OUT = Path(__file__).parent / "_corpus_sample.json"


def load_bad_page_ids() -> set[int]:
    """Page.id values flagged unusable by the three repo-root cleanup CSVs."""
    bad: set[int] = set()

    arabic = ROOT / "arabic_placeholder_pages.csv"
    with open(arabic, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            bad.add(int(row["page_id"]))

    contam = ROOT / "book203_cross_contamination.csv"
    with open(contam, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            bad.add(int(row["loser_page_id"]))

    # orphaned_source_pages.csv holds legacy source ids that were never
    # migrated to Postgres, so there is no Page.id to exclude -- but guard
    # by (source_db, source_page_id) anyway in case one slipped in.
    orphan_refs: set[tuple[str, int]] = set()
    orphan = ROOT / "orphaned_source_pages.csv"
    with open(orphan, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            orphan_refs.add((row["source_db"], int(row["source_page_id"])))

    load_bad_page_ids.orphan_refs = orphan_refs  # type: ignore[attr-defined]
    return bad


def live_row_keys() -> set[str]:
    """Every row_key present in the collection the eval harness queries."""
    col = get_collection()
    got = col.get(include=["metadatas"])
    return {m["row_key"] for m in got["metadatas"] if m.get("row_key")}


def position_bucket(pos: int | None, n_in_book: int) -> str:
    if pos is None or n_in_book <= 1:
        return "unknown"
    frac = pos / n_in_book
    if frac <= 0.33:
        return "early"
    if frac <= 0.66:
        return "mid"
    return "late"


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m scripts.sample_corpus_pages")
    ap.add_argument("--n", type=int, default=DEFAULT_N,
                    help=f"pages to emit (default {DEFAULT_N})")
    ap.add_argument("--min-chars", type=int, default=DEFAULT_MIN_CHARS,
                    help=f"skip pages shorter than this (default {DEFAULT_MIN_CHARS})")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    bad_ids = load_bad_page_ids()
    orphan_refs = load_bad_page_ids.orphan_refs  # type: ignore[attr-defined]
    keys = live_row_keys()

    session = SessionLocal()
    try:
        pages = (
            session.query(Page)
            .options(
                joinedload(Page.book).joinedload(Book.category),
                joinedload(Page.chapter),
            )
            .filter(Page.status == ContentStatus.published)
            .all()
        )

        per_book_positions: dict[int, list[int]] = defaultdict(list)
        for p in pages:
            if p.chapter and p.chapter.position is not None:
                per_book_positions[p.book_id].append(p.chapter.position)
        book_maxpos = {b: (max(v) if v else 0) for b, v in per_book_positions.items()}

        eligible: list[dict] = []
        for p in pages:
            if p.id in bad_ids:
                continue
            if (p.source_db, p.source_page_id) in orphan_refs:
                continue
            row_key = f"page_{p.id}"
            if row_key not in keys:
                continue
            text = (p.content or "").strip()
            if len(text) < args.min_chars:
                continue
            eligible.append({
                "id": p.id,
                "row_key": row_key,
                "book": p.book.name if p.book else "Unknown",
                "book_id": p.book_id,
                "chapter": p.chapter.name if p.chapter else "Unknown",
                "category": (
                    p.book.category.name if p.book and p.book.category else "Unknown"
                ),
                "position": p.chapter.position if p.chapter else None,
                "position_bucket": position_bucket(
                    p.chapter.position if p.chapter else None,
                    book_maxpos.get(p.book_id, 0) or 1,
                ),
                "char_len": len(text),
                "content": text,
            })
    finally:
        session.close()

    rng = random.Random(SEED)

    # Round-robin across books so the sample spreads. Books are visited in a
    # category-interleaved order (one book from each category in turn) and
    # each book's pages are shuffled, so we also spread across chapter
    # positions without hard-quota-ing them.
    by_book: dict[int, list[dict]] = defaultdict(list)
    for row in eligible:
        by_book[row["book_id"]].append(row)
    for rows in by_book.values():
        rng.shuffle(rows)

    by_cat: dict[str, list[int]] = defaultdict(list)
    for book_id, rows in by_book.items():
        by_cat[rows[0]["category"]].append(book_id)
    for book_ids in by_cat.values():
        rng.shuffle(book_ids)
    cat_order = sorted(by_cat)
    rng.shuffle(cat_order)

    book_queue: list[int] = []
    cursors = {c: 0 for c in cat_order}
    while any(cursors[c] < len(by_cat[c]) for c in cat_order):
        for c in cat_order:
            if cursors[c] < len(by_cat[c]):
                book_queue.append(by_cat[c][cursors[c]])
                cursors[c] += 1

    sample: list[dict] = []
    book_cursors = {b: 0 for b in by_book}
    while len(sample) < args.n and any(
        book_cursors[b] < len(by_book[b]) for b in book_queue
    ):
        for b in book_queue:
            if len(sample) >= args.n:
                break
            if book_cursors[b] < len(by_book[b]):
                sample.append(by_book[b][book_cursors[b]])
                book_cursors[b] += 1

    args.out.write_text(
        json.dumps(sample, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    cats = defaultdict(int)
    books = defaultdict(int)
    buckets = defaultdict(int)
    for r in sample:
        cats[r["category"]] += 1
        books[r["book"]] += 1
        buckets[r["position_bucket"]] += 1
    print(f"wrote {len(sample)} pages to {args.out}")
    print(f"  eligible pool: {len(eligible)} pages / {len(by_book)} books")
    print(f"  excluded bad ids: {len(bad_ids)}")
    print(f"  books in sample: {len(books)}")
    print(f"  by category: {dict(sorted(cats.items(), key=lambda x: -x[1]))}")
    print(f"  by position bucket: {dict(buckets)}")


if __name__ == "__main__":
    main()
