"""Readers for the MySQL dumps in data/ (cs-posts, pp-articles).

Ported (standard library only) from shibir-chat-gpu-service/ingestion/sources.py.
The SQLite book reader is not ported: those books are already in Postgres.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RawDoc:
    source: str  # e.g. "tarun", "cs-posts"
    doc_id: str  # stable id within the source
    title: str
    html: str
    meta: dict = field(default_factory=dict)


# ---------- MySQL dumps ----------

_INSERT_RE = re.compile(r"INSERT INTO `(\w+)` \(([^)]*)\) VALUES\s*", re.IGNORECASE)
_TOKEN_RE = re.compile(
    r"""
    '(?P<str>(?:[^'\\]|\\.|'')*)'   # quoted string with backslash or '' escapes
    | (?P<null>NULL)\b
    | (?P<num>-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)
    | (?P<punct>[(),;])
    | (?P<ws>\s+)
    """,
    re.VERBOSE | re.DOTALL,
)
_ESCAPES = {
    "0": "\0", "b": "\b", "n": "\n", "r": "\r", "t": "\t", "Z": "\x1a",
    "\\": "\\", "'": "'", '"': '"', "%": "\\%", "_": "\\_",
}  # fmt: skip
_ESC_RE = re.compile(r"\\(.)|''", re.DOTALL)


def _unescape(s: str) -> str:
    return _ESC_RE.sub(
        lambda m: "'" if m.group(1) is None else _ESCAPES.get(m.group(1), m.group(1)), s
    )


@dataclass
class DumpStats:
    rows: int = 0
    skipped: int = 0


_ROW_START_RE = re.compile(r"^\(\d+,")


def _parse_tuple(line: str) -> list | None:
    """Parse the leading `(v1, v2, ...)` of a line; None if it is malformed."""
    pos, vals = 1, []
    while True:
        m = _TOKEN_RE.match(line, pos)
        if m is None:
            return None
        pos = m.end()
        kind = m.lastgroup
        if kind == "str":
            vals.append(_unescape(m.group("str")))
        elif kind == "null":
            vals.append(None)
        elif kind == "num":
            vals.append(m.group("num"))
        elif kind == "punct":
            p = m.group("punct")
            if p == ")":
                return vals
            if p != ",":
                return None


def iter_mysql_values(
    text: str, table: str, stats: DumpStats | None = None
) -> Iterator[tuple[list[str], list]]:
    """Yield (declared_columns, values) for each row of `table` in a MySQL dump.

    Assumes mysqldump/phpMyAdmin layout: one row per line after an
    `INSERT INTO ... VALUES` header. Malformed lines inside an INSERT block are
    counted in `stats.skipped` instead of aborting the whole file.
    """
    stats = stats if stats is not None else DumpStats()
    cols: list[str] | None = None
    for line in text.splitlines():
        header = _INSERT_RE.match(line)
        if header:
            is_ours = header.group(1) == table
            cols = [c.strip().strip("`") for c in header.group(2).split(",")] if is_ours else None
            continue
        if cols is None or not line.strip():
            continue
        row_cols = cols
        if line.rstrip().endswith(";"):
            cols = None  # last line of this INSERT statement
        vals = _parse_tuple(line) if _ROW_START_RE.match(line) else None
        if vals is None:
            stats.skipped += 1  # truncated row or continuation garbage
            continue
        stats.rows += 1
        yield row_cols, vals


def read_cs_posts(path: Path, stats: DumpStats | None = None) -> Iterator[RawDoc]:
    text = path.read_text(encoding="utf-8")
    for cols, vals in iter_mysql_values(text, "posts", stats):
        if len(vals) != len(cols):
            if stats is not None:
                stats.skipped += 1
            continue
        r = dict(zip(cols, vals, strict=True))
        yield RawDoc(
            source="cs-posts",
            doc_id=str(r["id"]),
            title=r.get("title") or "",
            html=r.get("post") or "",
            meta={
                "writer": r.get("writer") or "",
                "designation": r.get("designation") or "",
                "date": r.get("news_date") or "",
            },
        )


# pp-articles-modified.sql declares 5 columns, but most rows still carry the
# original 16-column layout. Map those positionally.
_ARTICLE_16 = {"id": 0, "title": 2, "published_date": 5, "description": 8, "byline": 11}


def read_pp_articles(path: Path, stats: DumpStats | None = None) -> Iterator[RawDoc]:
    text = path.read_text(encoding="utf-8")
    for cols, vals in iter_mysql_values(text, "articles", stats):
        if len(vals) == len(cols):
            r = dict(zip(cols, vals, strict=True))
            byline = ""
        elif len(vals) == 16:
            r = {k: vals[i] for k, i in _ARTICLE_16.items()}
            byline = r["byline"] or ""
        else:
            if stats is not None:
                stats.skipped += 1
            continue
        yield RawDoc(
            source="pp-articles",
            doc_id=str(r["id"]),
            title=r.get("title") or "",
            html=r.get("description") or "",
            # video_url: not in the upstream port; scripts/load_articles.py
            # keeps it in source_metadata. Unknown for 16-column rows.
            meta={
                "date": r.get("published_date") or "",
                "byline": byline,
                "video_url": r.get("video_url") or "",
            },
        )
