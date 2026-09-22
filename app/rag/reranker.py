"""Cross-encoder reranking.  NEW FILE.

A bi-encoder (bge-m3) compresses query and document into separate vectors, so
it can only approximate relevance. A cross-encoder reads the pair together and
scores it directly -- far more accurate, but too slow to run over 4,149 docs.

Standard pattern, and usually the single biggest quality jump after fixing the
embedding model: fetch a wide candidate set (top 25) with the bi-encoder, then
rerank down to the 5 you actually send to Gemini.

bge-reranker-v2-m3 is the matching reranker for bge-m3 and supports Bengali.
"""

from __future__ import annotations

import threading
from functools import lru_cache

from sentence_transformers import CrossEncoder

from app.core.config import settings

# Same reason as embedder._lock: serialise the lazy load (lru_cache doesn't
# lock, so concurrent first requests would each load a ~2.2 GB copy) and the
# CPU-bound predict() calls.
_lock = threading.Lock()


@lru_cache(maxsize=1)
def _model() -> CrossEncoder:
    # device="cpu": see embedder.py -- the embedding model alone already
    # doesn't fit small GPUs, so default CUDA autodetect OOMs here too.
    return CrossEncoder(settings.reranker_model_name, max_length=1024, device="cpu")


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
    with _lock:
        scores = _model().predict([(query, d) for d in docs])
    ranked = sorted(enumerate(float(s) for s in scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_n]