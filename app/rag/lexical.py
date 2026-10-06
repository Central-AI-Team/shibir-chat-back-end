"""Lexical (BM25) leg of hybrid retrieval.

Indexes the SAME chunks that live in Chroma (same ids, documents and
metadatas), so the dense and lexical legs share chunk identity and
retriever.py can fuse their rankings by key. Chroma stays the single source
of truth; the BM25 index is derived data that can always be rebuilt from it.

Tokenizer. Bengali is suffix-heavy (নামাজ / নামাজে / নামাজের), so matching
whole words alone misses inflected forms. Each word is therefore indexed as
the whole word plus its character n-grams (settings.ngram_min..ngram_max),
which lets those three share n-grams. Words are cut with the `regex` package
on letters, marks and digits: stdlib `\\w` does NOT count Bengali vowel signs
or Arabic harakat (combining marks) as word characters and shreds every
Bengali word into single consonants.

Persistence. Building the index is the expensive part: measured 2026-10-06 on
the production corpus (76,461 chunks, ~88M n-gram tokens) it took 186 s and
peaked at 3.5 GB, and an in-memory-only index paid that on the first hybrid
request of every process and every worker. The index is therefore saved under
<chroma_persist_dir>/bm25/<collection>/ and loaded back with mmap, so a start
costs a few seconds and the arrays are paged in lazily and shared between
workers. Living inside the persist dir means the documented full reindex
(`rm -rf chroma_db`) removes it too. Chroma ignores the extra directory
(checked across a process restart).

A saved index is reused only if its fingerprint matches the collection now:
chunk count, a SHA-256 of every chunk id, the n-gram settings and the bm25s
version. Count alone would miss a re-ingest that swaps pages one for one, and
sqlite's mtime cannot be used because Chroma touches it on read-only opens.
What the fingerprint CANNOT see is a chunk whose text changed under an
unchanged id (a page edited without changing its chunk count). After such an
ingest run `uv run python -m app.rag.lexical`, which rebuilds from scratch.

Concurrency. Within a process the load/build is serialised by a lock (same
reasoning as embedder._lock); across workers an flock on a sibling lock file
lets one worker build while the others wait and then load its result. A cache
that cannot be written or read never fails a request: the index is built in
memory (or rebuilt) and a warning is logged.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tempfile
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import regex

try:  # POSIX only; without it the cross-worker lock is skipped, nothing else changes
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None

from app.core.config import settings
from app.rag.chroma_client import get_collection, get_named_collection
from app.rag.chunker import normalize

logger = logging.getLogger(__name__)

_FORMAT_VERSION = 1
_WORD = regex.compile(r"[\p{L}\p{M}\p{N}]+")
_GET_BATCH = 2000  # collection.get() page size, bounds peak memory while loading
_IDS_BATCH = 5000  # ids only are tiny, so page bigger when fingerprinting


@dataclass(frozen=True)
class _Index:
    bm25: object  # bm25s.BM25
    keys: list[str]  # position -> chunk key, parallel to the indexed documents
    count: int  # collection.count() the index was built from
    ngram_min: int
    ngram_max: int


_lock = threading.Lock()
_indexes: dict[str, _Index] = {}


def chunk_key(meta: dict) -> str:
    """f"{row_key}:{chunk_index}" -- the key retriever.py pools chunks by.

    Mirrors retriever._row_key, including the fallback for chunks written
    before row_key existed (it cannot be imported from there: retriever
    imports this module).
    """
    row_key = str(meta.get("row_key") or f"page_{meta.get('page_id')}")
    return f"{row_key}:{meta.get('chunk_index', 0)}"


def tokenize(
    text: str, ngram_min: int | None = None, ngram_max: int | None = None
) -> list[str]:
    """normalize -> lowercase -> words -> whole word + character n-grams.

    A word shorter than ngram_min is kept whole. An n-gram as long as the word
    would just repeat it, so only strictly shorter n-grams are added.
    """
    lo = settings.ngram_min if ngram_min is None else ngram_min
    hi = settings.ngram_max if ngram_max is None else ngram_max
    tokens: list[str] = []
    for word in _WORD.findall(normalize(text or "").lower()):
        tokens.append(word)
        for n in range(lo, min(hi, len(word) - 1) + 1):
            tokens.extend(word[i : i + n] for i in range(len(word) - n + 1))
    return tokens


def _build(collection, count: int):
    """Tokenize every chunk and index it. Returns (bm25, keys), or None if empty."""
    import bm25s  # lazy: nothing outside hybrid mode pays for the import

    lo, hi = settings.ngram_min, settings.ngram_max
    keys: list[str] = []
    # Token ids, not strings. With 3-5 character n-grams a ~900-character chunk
    # yields ~1,150 tokens, so the production corpus is ~88M tokens: as fresh
    # str objects that is several GB, but every id below is one shared int
    # object out of `vocab`, so a repeated n-gram costs a single list slot.
    vocab: dict[str, int] = {}
    corpus: list[list[int]] = []
    for offset in range(0, count, _GET_BATCH):
        page = collection.get(
            limit=_GET_BATCH, offset=offset, include=["documents", "metadatas"]
        )
        for doc, meta in zip(page.get("documents") or [], page.get("metadatas") or []):
            keys.append(chunk_key(meta or {}))
            corpus.append(
                [vocab.setdefault(t, len(vocab)) for t in tokenize(doc or "", lo, hi)]
            )
    if not keys:
        return None
    bm25 = bm25s.BM25()
    bm25.index((corpus, vocab), show_progress=False)
    return bm25, keys


# --- persistence ------------------------------------------------------------

def _cache_dir(collection_name: str | None) -> Path:
    name = collection_name or settings.chroma_collection_name
    return Path(settings.chroma_persist_dir) / "bm25" / name


def _fingerprint(collection, count: int) -> dict:
    import bm25s

    ids: list[str] = []
    for offset in range(0, count, _IDS_BATCH):
        ids.extend(collection.get(limit=_IDS_BATCH, offset=offset, include=[])["ids"])
    return {
        "format": _FORMAT_VERSION,
        "count": count,
        "ids_sha256": hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest(),
        "ngram_min": settings.ngram_min,
        "ngram_max": settings.ngram_max,
        "bm25s": getattr(bm25s, "__version__", "?"),
    }


def _load(directory: Path, fingerprint: dict) -> _Index | None:
    """The saved index if it exists, is intact and matches `fingerprint`."""
    import bm25s

    try:
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        logger.warning("BM25 cache metadata unreadable, rebuilding", exc_info=True)
        return None
    if meta.get("fingerprint") != fingerprint:
        return None
    try:
        keys = json.loads((directory / "keys.json").read_text(encoding="utf-8"))
        # mmap: the score arrays are paged in on demand rather than copied
        # into this process's heap, and are shared by every worker.
        bm25 = bm25s.BM25.load(str(directory / "index"), mmap=True)
    except Exception:
        logger.warning("BM25 cache unreadable, rebuilding", exc_info=True)
        return None
    if len(keys) != fingerprint["count"]:
        return None
    return _Index(bm25, keys, fingerprint["count"], fingerprint["ngram_min"], fingerprint["ngram_max"])


def _save(directory: Path, bm25, keys: list[str], fingerprint: dict) -> None:
    """Write to a temp dir and rename into place; meta.json goes last, so a
    directory without it (an interrupted write) is never mistaken for a cache."""
    directory.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f"{directory.name}.tmp-", dir=directory.parent))
    try:
        bm25.save(str(tmp / "index"))
        (tmp / "keys.json").write_text(json.dumps(keys), encoding="utf-8")
        (tmp / "meta.json").write_text(json.dumps({"fingerprint": fingerprint}), encoding="utf-8")
        shutil.rmtree(directory, ignore_errors=True)
        os.replace(tmp, directory)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


@contextmanager
def _build_lock(directory: Path) -> Iterator[None]:
    """Exclusive cross-process lock so concurrent workers build the index once.

    The lock file sits beside the cache directory, not in it: _save replaces
    the directory. Best effort: if it cannot be taken, build unlocked (the
    worst case is the duplicated work this lock exists to avoid).
    """
    handle = None
    if fcntl is not None:
        try:
            directory.parent.mkdir(parents=True, exist_ok=True)
            handle = open(directory.parent / f"{directory.name}.lock", "w")
            fcntl.flock(handle, fcntl.LOCK_EX)
        except OSError:
            logger.warning("BM25 build lock unavailable, building unlocked", exc_info=True)
    try:
        yield
    finally:
        if handle is not None:
            handle.close()  # closing releases the flock


def _load_or_build(collection, collection_name: str | None, count: int) -> _Index | None:
    directory = _cache_dir(collection_name)
    fingerprint = _fingerprint(collection, count)
    index = _load(directory, fingerprint)
    if index is not None:
        return index

    with _build_lock(directory):
        # Another worker may have built it while this one waited on the lock.
        index = _load(directory, fingerprint)
        if index is not None:
            return index
        started = time.monotonic()
        built = _build(collection, count)
        if built is None:
            return None
        bm25, keys = built
        in_memory = _Index(bm25, keys, count, settings.ngram_min, settings.ngram_max)
        try:
            _save(directory, bm25, keys, fingerprint)
        except Exception:
            logger.warning("could not save BM25 cache, serving from memory", exc_info=True)
            return in_memory
        logger.info(
            "built BM25 index: %d chunks in %.0fs -> %s",
            count, time.monotonic() - started, directory,
        )
        # Serve from the mmap'd copy, same as every later start, so the
        # build-time heap (3.5 GB on the production corpus) can be released.
        return _load(directory, fingerprint) or in_memory


def _get_index(collection_name: str | None) -> _Index | None:
    collection = (
        get_collection() if collection_name is None else get_named_collection(collection_name)
    )
    count = collection.count()
    if count == 0:
        return None
    cache_key = collection_name or ""
    index = _indexes.get(cache_key)
    if index is not None and index.count == count:
        return index
    with _lock:
        index = _indexes.get(cache_key)
        if index is None or index.count != count:
            index = _load_or_build(collection, collection_name, count)
            if index is None:
                _indexes.pop(cache_key, None)
                return None
            _indexes[cache_key] = index
        return index


def search(
    query: str, k: int, collection_name: str | None = None
) -> list[tuple[str, float]]:
    """Top-k chunks by BM25 as (chunk_key, score), best first.

    Returns [] for an empty collection, an empty or token-less query, or when
    nothing matches (zero-score rows are dropped: they carry no signal and
    would otherwise rank arbitrary chunks into the fusion). Never raises on an
    odd query.
    """
    if k <= 0:
        return []
    index = _get_index(collection_name)
    if index is None:
        return []
    # Dedupe, order preserved: a repeated query term must not out-weigh the rest.
    tokens = list(dict.fromkeys(tokenize(query, index.ngram_min, index.ngram_max)))
    if not tokens:
        return []
    ids, scores = index.bm25.retrieve(
        [tokens], k=min(k, len(index.keys)), show_progress=False
    )
    return [
        (index.keys[int(i)], float(s))
        for i, s in zip(ids[0], scores[0])
        if float(s) > 0.0
    ]


def reset() -> None:
    """Forget the in-memory indexes (tests). The saved cache is left alone."""
    with _lock:
        _indexes.clear()


def rebuild(collection_name: str | None = None) -> int:
    """Delete the saved index and rebuild it from Chroma. Returns chunk count.

    For after an ingest the fingerprint cannot detect (see module docstring).
    """
    with _lock:
        _indexes.pop(collection_name or "", None)
        shutil.rmtree(_cache_dir(collection_name), ignore_errors=True)
    index = _get_index(collection_name)
    return 0 if index is None else index.count


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m app.rag.lexical",
        description="Rebuild the saved BM25 index from the Chroma collection.",
    )
    parser.add_argument("--collection", default=None,
                        help="collection name (default: settings.chroma_collection_name)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    started = time.monotonic()
    n = rebuild(args.collection)
    print(f"BM25 index ready: {n} chunks in {time.monotonic() - started:.0f}s "
          f"-> {_cache_dir(args.collection)}")
