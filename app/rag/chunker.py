"""Bengali-aware text chunking.  NEW FILE.

Previously ingest.py embedded one whole DB page as one vector. Median page in
your corpus is 2,451 chars and the largest is 23,123 -- far too much meaning
per vector, and past MiniLM's 256-token ceiling entirely.

Splitting priority: paragraph -> Bengali danda -> Latin period -> space.
The danda separator matters: without it Bengali sentences break mid-clause.
"""

from __future__ import annotations

import re
import unicodedata

CHUNK_SIZE = 900        # characters. ~250-300 Bengali words -- one coherent idea.
CHUNK_OVERLAP = 150     # keeps a sentence that straddles a boundary findable.

_SEPARATORS = ["\n\n", "\n", "। ", "।", "? ", "! ", ". ", " "]
_WS = re.compile(r"[ \t]+")
_BLANKS = re.compile(r"\n{3,}")


def normalize(text: str) -> str:
    """Unicode NFC + whitespace cleanup.

    Bengali conjuncts can be stored decomposed (e.g. ক + ্ + ষ) or composed.
    Without NFC on BOTH sides, the same word produces different vectors and
    exact-match lookups silently fail. Apply this at ingest AND at query time.
    """
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u200c", "").replace("\u200d", "")  # ZWNJ / ZWJ
    text = _WS.sub(" ", text)
    return _BLANKS.sub("\n\n", text).strip()


def _split_once(text: str, limit: int) -> list[str]:
    if len(text) <= limit:
        return [text]
    for sep in _SEPARATORS:
        if sep not in text:
            continue
        parts, buf = [], ""
        for piece in text.split(sep):
            candidate = piece if not buf else buf + sep + piece
            if len(candidate) <= limit:
                buf = candidate
            else:
                if buf:
                    parts.append(buf)
                buf = piece
        if buf:
            parts.append(buf)
        if all(len(p) <= limit for p in parts):
            return parts
        return [sub for p in parts for sub in _split_once(p, limit)]
    # No separator helped -- hard cut.
    return [text[i : i + limit] for i in range(0, len(text), limit)]


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    text = normalize(text)
    if not text:
        return []

    pieces = _split_once(text, chunk_size)

    merged: list[str] = []
    for piece in pieces:
        if merged and len(merged[-1]) + len(piece) + 1 <= chunk_size:
            merged[-1] = merged[-1] + " " + piece
        else:
            merged.append(piece)

    if overlap <= 0 or len(merged) < 2:
        return merged

    out = [merged[0]]
    for prev, cur in zip(merged, merged[1:]):
        out.append((prev[-overlap:] + " " + cur).strip())
    return out


def build_document(book: str, chapter: str, body: str) -> str:
    """Text that actually gets embedded.

    The header is in Bengali on purpose. The old ingest.py prefixed every chunk
    with English "Book: / Chapter: / Content:", which (a) burned tokens and
    (b) pushed every document toward the same English-ish region of the space.
    Keeping the header means a question naming a book still matches its chunks.
    """
    return f"বই: {book}\nঅধ্যায়: {chapter}\n\n{body}"