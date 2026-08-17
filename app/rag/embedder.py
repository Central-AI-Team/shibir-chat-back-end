"""Embedding layer.

CHANGED: all-MiniLM-L6-v2 -> BAAI/bge-m3.

Why: all-MiniLM-L6-v2 uses the bert-base-uncased WordPiece vocabulary, which
contains no Bengali codepoints. Every Bengali word tokenised to [UNK], so all
~4,149 documents collapsed to nearly the same vector and retrieval was random.

bge-m3 is multilingual (100+ languages incl. Bengali), handles 8192 tokens, and
does cross-lingual matching -- which also helps Banglish queries.

Cost: 1024 dims (was 384) and ~2.2 GB of RAM. If that is too heavy, the fallback
is intfloat/multilingual-e5-base (768 dims, ~1.1 GB) -- still far better than
MiniLM for Bengali, but you must then prefix documents with "passage: " and
queries with "query: ".
"""

from __future__ import annotations

from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app.core.config import settings


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    # Loaded lazily so importing this module (e.g. in tests) is cheap.
    # device="cpu": bge-m3 is ~2.2 GB, which doesn't fit alongside the
    # reranker on small GPUs (e.g. a 2 GB card) and OOMs on the default
    # CUDA autodetect. Sized for RAM, not VRAM -- see module docstring.
    return SentenceTransformer(settings.embedding_model_name, device="cpu")


def embed_text(text: str) -> list[float]:
    """Embed a single string. Kept for backwards compatibility."""
    return embed_texts([text])[0]


def embed_texts(texts: list[str], batch_size: int = 16) -> list[list[float]]:
    """Embed a batch. Much faster than looping embed_text() during ingest.

    normalize_embeddings=True is REQUIRED -- the Chroma collection is created
    with hnsw:space=cosine, and cosine only behaves correctly on unit vectors.
    """
    return _model().encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).tolist()


def embed_query(query: str) -> list[float]:
    """Embed a search query.

    bge-m3 needs no instruction prefix -- queries and documents go through the
    same encoder. (This is NOT true of bge-large-en or e5; if you swap models,
    revisit this function.)
    """
    return embed_texts([query])[0]
