"""Cross-encoder reranking.  NEW FILE.

A bi-encoder (bge-m3) compresses query and document into separate vectors, so
it can only approximate relevance. A cross-encoder reads the pair together and
scores it directly -- far more accurate, but too slow to run over 4,149 docs.

Standard pattern, and usually the single biggest quality jump after fixing the
embedding model: fetch a wide candidate set (top 25) with the bi-encoder, then
rerank down to the 5 you actually send to the LLM.

GPU mode: with settings.gpu_service_url set, scoring runs on the external GPU
service (see gpu_client.py) and no local model is loaded.

bge-reranker-v2-m3 is the matching reranker for bge-m3 and supports Bengali.
"""

from __future__ import annotations

import threading
from functools import lru_cache
from typing import TYPE_CHECKING

from app.core.config import settings
from app.rag import gpu_client

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

# Same truncation as the local CrossEncoder below, so GPU and local scores match.
_MAX_LENGTH = 1024
_GPU_MAX_DOCS = 200  # service-side limit per /rerank request

# Same reason as embedder._lock: serialise the lazy load (lru_cache doesn't
# lock, so concurrent first requests would each load a ~2.2 GB copy) and the
# CPU-bound predict() calls.
_lock = threading.Lock()


@lru_cache(maxsize=1)
def _model() -> CrossEncoder:
    # Lazy import: GPU mode must never pull in torch.
    from sentence_transformers import CrossEncoder

    # device="cpu": see embedder.py -- the embedding model alone already
    # doesn't fit small GPUs, so default CUDA autodetect OOMs here too.
    return CrossEncoder(settings.reranker_model_name, max_length=_MAX_LENGTH, device="cpu")


def rerank(query: str, docs: list[str], top_n: int) -> list[tuple[int, float]]:
    """Return [(original_index, score), ...] sorted best-first, truncated to top_n.

    Scores are Sigmoid-activated, in [0, 1] -- sentence-transformers applies
    that activation by default for a num_labels=1 model like
    bge-reranker-v2-m3, and this is what settings.min_rerank_score is tuned
    against (see config.py / PROJECT.md's "Prompt contract" section, smoke-
    tested there at 0.94-0.99 on-topic vs 0.011 off-topic). NOT the cosine
    similarity used at the retrieval stage -- don't compare the two directly.
    """
    if not docs:
        return []
    if gpu_client.is_enabled():
        return _rerank_gpu(query, docs, top_n)
    with _lock:
        scores = _model().predict([(query, d) for d in docs])
    ranked = sorted(enumerate(float(s) for s in scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_n]


def _rerank_gpu(query: str, docs: list[str], top_n: int) -> list[tuple[int, float]]:
    """/rerank on the GPU service, in slices of <= _GPU_MAX_DOCS, merged.

    Uses each result's `prob` (sigmoid, [0, 1]) -- NOT `score`, the raw
    logit -- so settings.min_rerank_score keeps the meaning it was tuned
    with on the local CrossEncoder.
    """
    merged: list[tuple[int, float]] = []
    for offset in range(0, len(docs), _GPU_MAX_DOCS):
        chunk = docs[offset : offset + _GPU_MAX_DOCS]
        data = gpu_client.post(
            "/rerank",
            {
                "query": query,
                "documents": chunk,
                # Per-slice top_n is enough: the global top_n can't include
                # more than top_n items from any one slice.
                "top_k": min(top_n, len(chunk)),
                "max_length": _MAX_LENGTH,
            },
        )
        # Same guard as embedder: min_rerank_score is tuned against this
        # model's scores, so a different one would silently skew the gate.
        if data.get("model") != settings.reranker_model_name:
            raise gpu_client.GPUServiceError(
                f"GPU service /rerank returned model {data.get('model')!r}, "
                f"expected {settings.reranker_model_name!r}"
            )
        try:
            for r in data["results"]:
                idx = int(r["index"])
                if not 0 <= idx < len(chunk):
                    raise ValueError(f"index {idx} out of range")
                merged.append((offset + idx, float(r["prob"])))
        except (KeyError, TypeError, ValueError) as e:
            raise gpu_client.GPUServiceError(f"GPU service /rerank: malformed results ({e})") from e
    merged.sort(key=lambda x: x[1], reverse=True)
    return merged[:top_n]
