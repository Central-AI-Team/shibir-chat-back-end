"""Embedding layer.

CHANGED: all-MiniLM-L6-v2 -> BAAI/bge-m3.

Why: all-MiniLM-L6-v2 uses the bert-base-uncased WordPiece vocabulary, which
contains no Bengali codepoints. Every Bengali word tokenised to [UNK], so all
~4,149 documents collapsed to nearly the same vector and retrieval was random.

bge-m3 is multilingual (100+ languages incl. Bengali), handles 8192 tokens, and
does cross-lingual matching -- which also helps Banglish queries.

GPU mode: with settings.gpu_service_url set, embedding runs on the external
GPU service (see gpu_client.py) and nothing below loads a local model.

Cost: 1024 dims (was 384) and ~2.2 GB of RAM (local mode). If that is too heavy, the fallback
is intfloat/multilingual-e5-base (768 dims, ~1.1 GB) -- still far better than
MiniLM for Bengali, but you must then prefix documents with "passage: " and
queries with "query: ".
"""

from __future__ import annotations

import threading
from functools import lru_cache
from typing import TYPE_CHECKING

from app.core.config import settings
from app.rag import gpu_client

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

# GPU mode (settings.gpu_service_url set): /embed on the external GPU service.
# The Chroma collection holds bge-m3 vectors, so a service answering with any
# other model or dimension would silently corrupt retrieval -- checked on
# every response.
_EXPECTED_DIM = 1024
_GPU_MAX_TEXTS = 256    # service-side limit per /embed request
_GPU_BATCH_SIZE = 64    # the service's own encode() batch size

# The API serves requests from a threadpool, so two /chat calls can reach this
# module at once. lru_cache does NOT lock: two concurrent first calls both
# miss and each load a separate ~2.2 GB copy, which on a small box pushes it
# into swap and stalls every request for minutes. Concurrent encode() calls
# on CPU also just fight over the same cores (torch already uses all of them
# per call), so serialising costs nothing and keeps memory bounded. One lock
# covers both the load and the inference.
_lock = threading.Lock()


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    # Imported and loaded lazily so importing this module -- and anything at
    # all in GPU mode -- never pulls in torch.
    from sentence_transformers import SentenceTransformer

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

    In GPU mode `batch_size` is ignored: texts go out in slices of
    _GPU_MAX_TEXTS and the service encodes each in batches of _GPU_BATCH_SIZE.
    """
    if not texts:
        return []
    if gpu_client.is_enabled():
        return _embed_texts_gpu(texts)
    with _lock:
        return _model().encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()


def _embed_texts_gpu(texts: list[str]) -> list[list[float]]:
    vectors: list[list[float]] = []
    for i in range(0, len(texts), _GPU_MAX_TEXTS):
        chunk = texts[i : i + _GPU_MAX_TEXTS]
        data = gpu_client.post(
            "/embed",
            {"texts": chunk, "normalize": True, "batch_size": _GPU_BATCH_SIZE},
        )
        if data.get("model") != settings.embedding_model_name:
            raise gpu_client.GPUServiceError(
                f"GPU service /embed returned model {data.get('model')!r}, "
                f"expected {settings.embedding_model_name!r}"
            )
        if data.get("dim") != _EXPECTED_DIM:
            raise gpu_client.GPUServiceError(
                f"GPU service /embed returned dim {data.get('dim')!r}, expected {_EXPECTED_DIM}"
            )
        got = data.get("vectors") or []
        if len(got) != len(chunk) or any(len(v) != _EXPECTED_DIM for v in got):
            raise gpu_client.GPUServiceError(
                f"GPU service /embed returned {len(got)} vector(s) for {len(chunk)} text(s) "
                "or a vector of the wrong length"
            )
        vectors.extend(got)
    return vectors


def embed_query(query: str) -> list[float]:
    """Embed a search query.

    bge-m3 needs no instruction prefix -- queries and documents go through the
    same encoder. (This is NOT true of bge-large-en or e5; if you swap models,
    revisit this function.)
    """
    return embed_texts([query])[0]
