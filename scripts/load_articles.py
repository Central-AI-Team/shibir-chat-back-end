"""Load the pp-articles / cs-posts MySQL dumps in data/ into the articles table.

Usage:
    uv run python -m scripts.load_articles --source pp-articles|cs-posts
                                           [--sql <path>] [--publish] [--dry-run]

Then embed with `uv run python -m app.rag.ingest` (the only code that writes
to Chroma). See "Adding a data source" in CLAUDE.md.

Per row: HTML -> text (app.loaders.text.html_to_text, WordPress [caption]
shortcodes removed first), then drop_repeated_sentences. Skipped: malformed
dump lines, text shorter than 80 chars, is_repetitive() spam, and exact
duplicates (_fingerprint) of an existing published page/article or of an
earlier row in the same file.

Upserted on (source_type, source_ref): source_type "pp-article" / "cs-post",
source_ref = the original row id.

Status: new rows are draft unless --publish. --publish also publishes
existing rows, and SUPERSEDES the legacy import of the same source
(source_type "persxpect_article" / "chhatrasangbad_post", source_ref
"pp_article_<id>" / "cs_post_<id>") by setting the legacy rows for the ids
in this file to draft -- nothing is deleted. Legacy rows are never used as
dedupe targets, or every re-cleaned row identical to its legacy copy would be
skipped as a "duplicate" and then lose its legacy copy too. --dry-run does
everything in a transaction and rolls it back.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from app.loaders.sources import DumpStats, RawDoc, read_cs_posts, read_pp_articles
from app.loaders.text import _fingerprint, drop_repeated_sentences, html_to_text, is_repetitive

MIN_CHARS = 80

# Both sites are Bangladeshi; their dump dates carry no zone.
_SOURCE_TZ = ZoneInfo("Asia/Dhaka")
_DATE_FORMATS = ("%d-%m-%Y %H:%M:%S", "%d-%m-%Y", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d")
_SHORTCODE_RE = re.compile(r"\[/?caption\b[^\]]*\]", re.IGNORECASE)
_BENGALI_RE = re.compile(r"[ঀ-৿]")
_LATIN_RE = re.compile(r"[A-Za-z]")


@dataclass(frozen=True)
class Source:
    source_type: str
    default_sql: Path
    reader: Callable[..., Iterator[RawDoc]]
    author_key: str
    metadata_keys: tuple[str, ...]
    legacy_type: str
    legacy_ref: str  # format string for the legacy source_ref, given the id


SOURCES = {
    "pp-articles": Source(
        source_type="pp-article",
        default_sql=Path("data/pp-articles-modified.sql"),
        reader=read_pp_articles,
        author_key="byline",
        metadata_keys=("video_url",),
        legacy_type="persxpect_article",
        legacy_ref="pp_article_{}",
    ),
    "cs-posts": Source(
        source_type="cs-post",
        default_sql=Path("data/cs-posts-modified.sql"),
        reader=read_cs_posts,
        author_key="writer",
        metadata_keys=("designation",),
        legacy_type="chhatrasangbad_post",
        legacy_ref="cs_post_{}",
    ),
}


def clean(html: str) -> str:
    return drop_repeated_sentences(html_to_text(_SHORTCODE_RE.sub("", html or "")))


def parse_date(raw: str) -> datetime | None:
    """A timezone-aware datetime, or None for empty / invalid ("0000-00-00") values."""
    raw = (raw or "").strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(raw, fmt).replace(tzinfo=_SOURCE_TZ)
        except ValueError:
            continue
    return None


def detect_language(text: str) -> str:
    """"bn" unless Latin letters outnumber Bengali ones (pp-articles are English)."""
    return "en" if len(_LATIN_RE.findall(text)) > len(_BENGALI_RE.findall(text)) else "bn"


@dataclass
class Prepared:
    source_ref: str
    title: str
    content: str
    language: str
    author: str | None
    published_at: datetime | None
    source_metadata: dict


@dataclass
class PrepareStats:
    parsed: int = 0
    skipped: Counter = field(default_factory=Counter)  # malformed/short/repetitive/duplicate


def prepare(
    docs: Iterable[RawDoc], source: Source, existing_fingerprints: set[str], stats: PrepareStats
) -> list[Prepared]:
    """Clean, filter and dedupe raw rows. Pure: no DB access."""
    seen = set(existing_fingerprints)
    out: list[Prepared] = []
    for doc in docs:
        stats.parsed += 1
        text = clean(doc.html)
        if len(text) < MIN_CHARS:
            stats.skipped["short"] += 1
            continue
        if is_repetitive(text):
            stats.skipped["repetitive"] += 1
            continue
        fp = _fingerprint(text)
        if fp in seen:
            stats.skipped["duplicate"] += 1
            continue
        seen.add(fp)

        author = " ".join((doc.meta.get(source.author_key) or "").split())
        metadata = {k: doc.meta.get(k) for k in source.metadata_keys}
        metadata["raw_date"] = doc.meta.get("date")
        out.append(Prepared(
            source_ref=doc.doc_id,
            title=(" ".join(doc.title.split()) or text[:80])[:500],
            content=text,
            language=detect_language(text),
            author=author[:255] or None,
            published_at=parse_date(doc.meta.get("date") or ""),
            source_metadata={k: v for k, v in metadata.items() if v not in (None, "")},
        ))
    return out


def existing_fingerprints(session, source: Source) -> set[str]:
    """Fingerprints of every published page and article, except this source's
    own rows (so a re-run isn't all "duplicates") and its legacy rows (which
    this load supersedes)."""
    from app.db.models import Article, ContentStatus, Page

    fps = {
        _fingerprint(c)
        for (c,) in session.query(Page.content).filter(Page.status == ContentStatus.published)
    }
    fps.update(
        _fingerprint(c)
        for (c,) in session.query(Article.content)
        .filter(Article.status == ContentStatus.published)
        .filter(Article.source_type.notin_([source.source_type, source.legacy_type]))
    )
    return fps


@dataclass
class UpsertStats:
    inserted: int = 0
    updated: int = 0
    unchanged: int = 0
    superseded: int = 0


def upsert_articles(
    session, rows: list[Prepared], source: Source, *, publish: bool, file_ids: Iterable[str]
) -> UpsertStats:
    """Upsert prepared rows. `file_ids` = every id in the file (kept or skipped),
    whose legacy copies --publish supersedes. Flushes; the caller commits."""
    from app.db.models import Article, ContentStatus

    stats = UpsertStats()
    existing = {
        a.source_ref: a
        for a in session.query(Article).filter(Article.source_type == source.source_type)
    }
    new_status = ContentStatus.published if publish else ContentStatus.draft

    for row in rows:
        fields = {
            "title": row.title,
            "content": row.content,
            "language": row.language,
            "author": row.author,
            "published_at": row.published_at,
            "source_metadata": row.source_metadata,
        }
        article = existing.get(row.source_ref)
        if article is None:
            session.add(Article(
                **fields, status=new_status,
                source_type=source.source_type, source_ref=row.source_ref,
            ))
            stats.inserted += 1
            continue
        changed = False
        # Assign only real changes, so updated_at (onupdate) -- and ingest's
        # "needs re-embedding" check -- stay quiet on an identical re-run.
        for attr, value in fields.items():
            if getattr(article, attr) != value:
                setattr(article, attr, value)
                changed = True
        if publish and article.status != ContentStatus.published:
            article.status = ContentStatus.published
            changed = True
        stats.updated += changed
        stats.unchanged += not changed

    if publish:
        legacy_refs = [source.legacy_ref.format(i) for i in file_ids]
        for start in range(0, len(legacy_refs), 1000):
            stats.superseded += (
                session.query(Article)
                .filter(Article.source_type == source.legacy_type)
                .filter(Article.source_ref.in_(legacy_refs[start : start + 1000]))
                .filter(Article.status == ContentStatus.published)
                .update({Article.status: ContentStatus.draft}, synchronize_session=False)
            )
    session.flush()
    return stats


def load(session, source: Source, sql_path: Path, *, publish: bool):
    """Parse + prepare + upsert. Returns (dump_stats, prepare_stats, upsert_stats, rows)."""
    dump_stats = DumpStats()
    docs = list(source.reader(sql_path, dump_stats))
    prep_stats = PrepareStats()
    rows = prepare(docs, source, existing_fingerprints(session, source), prep_stats)
    prep_stats.skipped["malformed"] = dump_stats.skipped
    upsert_stats = upsert_articles(
        session, rows, source, publish=publish, file_ids=[d.doc_id for d in docs]
    )
    return dump_stats, prep_stats, upsert_stats, rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--source", required=True, choices=sorted(SOURCES))
    parser.add_argument("--sql", type=Path, help="dump path (default: the source's file in data/)")
    parser.add_argument("--publish", action="store_true",
                        help="publish rows (and supersede the legacy import)")
    parser.add_argument("--dry-run", action="store_true", help="roll back instead of committing")
    args = parser.parse_args(argv)

    source = SOURCES[args.source]
    sql_path = args.sql or source.default_sql
    if not sql_path.exists():
        print(f"not found: {sql_path}", file=sys.stderr)
        return 1

    from app.db.session import SessionLocal

    session = SessionLocal()
    try:
        _, prep, ups, rows = load(session, source, sql_path, publish=args.publish)
        sk = prep.skipped
        print(f"parsed: {prep.parsed}   skipped: malformed {sk['malformed']} / short "
              f"{sk['short']} / repetitive {sk['repetitive']} / duplicate {sk['duplicate']}")
        print(f"inserted: {ups.inserted}   updated: {ups.updated}   unchanged: "
              f"{ups.unchanged}   legacy {source.legacy_type} superseded: {ups.superseded}")
        for row in rows[:2]:
            print(f"\n--- sample ({source.source_type} {row.source_ref}, {row.language}, "
                  f"{row.published_at}) ---\n{row.title}\n{row.content[:500]}")
        if args.dry_run:
            session.rollback()
            print("\nDRY RUN -- rolled back, nothing written.")
        else:
            session.commit()
            print("\ncommitted. Next: uv run python -m app.rag.ingest")
    finally:
        session.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
