"""BM25 lexical (keyword) search.  NEW FILE.

Dense search (embedder.py + Chroma) finds passages that MEAN the same thing
as the question, but this corpus is full of exact names and numbers that
carry very little "meaning" on their own: Quran sura names (সূরা আল মুমিনুন vs
the similarly-spelled but completely different সূরা আল মু'মিন), verse/ayah
numbers, hijri years, and rare technical terms such as "শায়খাইন" (a specific
dual title for two named hadith scholars). A bi-encoder compresses a whole
chunk into one vector and tends to blur names like this together; a user
typing the exact sura name or verse number deserves an exact-word match, not
an approximate one.

BM25 is the classic complement: pure term-frequency / inverse-document-
frequency scoring, no embedding involved, and it rewards a rare word (like
"শায়খাইন") far more than a common one -- exactly backwards from where dense
search is weak.

How it is combined with dense search (retriever.py's _rrf_fuse):
  - BM25 contributes its own top bm25_fetch_k-ranked chunks alongside dense
    search's own top fetch_k. The two ranked lists are merged with
    Reciprocal Rank Fusion and the UNION is truncated back down to fetch_k
    before reranking -- so enabling BM25 changes WHICH candidates reach the
    cross-encoder, never how many. The cross-encoder and min_rerank_score
    gate are completely unchanged: every BM25 candidate still has to clear
    the same relevance bar dense candidates always did, so a coincidental
    keyword overlap cannot by itself make the model answer from an
    irrelevant excerpt (retriever.py §"hybrid retrieval").

Index source: built directly from Chroma's own stored documents/metadatas
(the same store dense search reads), not from Postgres -- so the BM25
corpus can never drift out of sync with what is actually searchable, and
nothing here needs a second ingest path.

Rebuilding: like the retrieval cache (retriever.py), there is no signal from
app/rag/ingest.py (a separate process) back into a running server, so a
built index is trusted for settings.bm25_rebuild_interval_seconds and then
rebuilt lazily on the next search -- bounded staleness, not perfect
invalidation, the same trade-off the rest of this module's neighbours make.

Indexed per collection_name (None = production), mirroring
chroma_client.get_named_collection(): scripts/eval_chunking.py points
retrieval at temporary chunk_eval_* collections to score a candidate
chunking config, and must never have that search silently fall back to
scoring against the production BM25 index.
"""

from __future__ import annotations

import threading
import time
import unicodedata
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from app.core.config import settings
from app.rag.chroma_client import get_collection, get_named_collection
from app.rag.chunker import normalize

# NOT re.compile(r"\w+"): Python's regex \w (and str.isalnum()) excludes
# Unicode COMBINING MARKS (category Mn/Mc) -- which is exactly what Bengali
# vowel signs (matras) and Arabic diacritics ARE. "সূরা" is four codepoints
# (স + the vowel sign ূ + র + the vowel sign া); \w+ only matches the base
# letters and splits at every vowel sign, tokenizing it as ['স', 'র']
# instead of ['সূরা'] -- silently scoring individual consonants instead of
# real words. Scanning by Unicode category and keeping Letter/Mark/Number
# runs together avoids this for Bengali, Arabic AND Latin uniformly.
def _is_word_char(ch: str) -> bool:
    return unicodedata.category(ch)[0] in ("L", "N", "M") or ch == "_"


def _tokenize(text: str) -> list[str]:
    text = normalize(text).lower()  # .lower() only affects Latin; Bengali/Arabic have no case
    tokens: list[str] = []
    buf: list[str] = []
    for ch in text:
        if _is_word_char(ch):
            buf.append(ch)
        elif buf:
            tokens.append("".join(buf))
            buf = []
    if buf:
        tokens.append("".join(buf))
    return tokens


def _key(meta: dict) -> str:
    row_key = meta.get("row_key") or f"page_{meta.get('page_id')}"
    return f"{row_key}:{meta.get('chunk_index', 0)}"


@dataclass
class _Index:
    bm25: BM25Okapi | None
    keys: list[str]
    docs: list[str]
    metas: list[dict]


_indexes: dict[str | None, _Index] = {}
_built_at: dict[str | None, float] = {}
_lock = threading.Lock()


def _build(collection_name: str | None) -> _Index:
    collection = get_collection() if collection_name is None else get_named_collection(collection_name)
    result = collection.get(include=["documents", "metadatas"])
    docs = result.get("documents") or []
    metas = result.get("metadatas") or []
    keys = [_key(meta) for meta in metas]
    tokenized = [_tokenize(d) for d in docs]
    bm25 = BM25Okapi(tokenized) if tokenized else None
    return _Index(bm25=bm25, keys=keys, docs=docs, metas=metas)


def _get_index(collection_name: str | None) -> _Index | None:
    now = time.monotonic()
    stale = now - _built_at.get(collection_name, 0.0) > settings.bm25_rebuild_interval_seconds
    if collection_name not in _indexes or stale:
        with _lock:
            now = time.monotonic()
            stale = now - _built_at.get(collection_name, 0.0) > settings.bm25_rebuild_interval_seconds
            if collection_name not in _indexes or stale:
                _indexes[collection_name] = _build(collection_name)
                _built_at[collection_name] = now
    return _indexes[collection_name]


def search(
    query: str, top_n: int, collection_name: str | None = None
) -> list[tuple[str, str, dict, float]]:
    """Best-first (key, doc, meta, bm25_score) for `query`.

    Zero-score hits are dropped: no shared term at all is noise, not a
    match, and letting it through would only cost the cross-encoder a slot
    for free. BM25Okapi.get_scores() scores every document in the corpus
    (a few ms for this corpus's ~12k chunks -- no ANN index needed at this
    scale), so this is a plain sort + slice, not an approximate search.
    """
    index = _get_index(collection_name)
    if index is None or index.bm25 is None:
        return []
    tokens = _tokenize(query)
    if not tokens:
        return []
    scores = index.bm25.get_scores(tokens)
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    out: list[tuple[str, str, dict, float]] = []
    for i in ranked[:top_n]:
        if scores[i] <= 0:
            break  # descending order -- nothing after this can be > 0 either
        out.append((index.keys[i], index.docs[i], index.metas[i], float(scores[i])))
    return out


def reset_index(collection_name: str | None = None) -> None:
    """Drop the built index for one collection (default: production), so the
    next search() rebuilds it from Chroma's current content.

    Used by tests and by scripts/eval_chunking.py right after it rebuilds a
    chunk_eval_* collection from scratch -- otherwise a config name reused
    within one script run could search a stale index built against that
    same collection name's PREVIOUS content (mirrors
    retriever.clear_retrieval_cache(), called at the same call site).
    """
    with _lock:
        _indexes.pop(collection_name, None)
        _built_at.pop(collection_name, None)
