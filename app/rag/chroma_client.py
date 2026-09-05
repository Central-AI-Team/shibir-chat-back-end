"""Chroma client.

CHANGED: the collection is now created with hnsw:space="cosine".

Chroma's default is L2 (squared euclidean). With normalized bge-m3 vectors you
want cosine -- and, more practically, cosine gives you a distance in [0, 2]
that you can actually threshold on. That is what makes the "no good match ->
return no sources" logic in qa_service.py possible.

NOTE: an existing collection keeps whatever metric it was created with. You
must delete chroma_db/ and re-ingest for this to take effect. You have to do
that anyway -- bge-m3 is 1024-dim and the old index is 384-dim, so Chroma would
raise a dimension mismatch otherwise.

ADDED for scripts/eval_chunking.py: get_named_collection() / delete_collection()
so that script can build temporary chunk_eval_* indices (one per chunking
config it wants to compare) without ever touching the cached production
collection object returned by get_collection(). They share the one
PersistentClient instance -- Chroma supports many collections per client --
so get_collection()'s behavior/caching is completely unchanged.
"""

from functools import lru_cache

import chromadb

from app.core.config import settings


@lru_cache(maxsize=1)
def _client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=settings.chroma_persist_dir)


@lru_cache
def get_collection():
    return _client().get_or_create_collection(
        settings.chroma_collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def get_named_collection(name: str):
    """Get-or-create an arbitrary named collection. Uncached (a metadata
    lookup, not a real cost) -- unlike get_collection(), which the API server
    hits on every request and must stay a stable cached object."""
    return _client().get_or_create_collection(name, metadata={"hnsw:space": "cosine"})


def delete_collection(name: str) -> None:
    """Drop a named collection if it exists. For scripts/eval_chunking.py's
    temp chunk_eval_* collections ONLY -- never pass
    settings.chroma_collection_name here, there is no guard against it beyond
    this docstring."""
    try:
        _client().delete_collection(name)
    except Exception:
        pass
