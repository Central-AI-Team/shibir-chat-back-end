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
"""

from functools import lru_cache

import chromadb

from app.core.config import settings


@lru_cache
def get_collection():
    client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    return client.get_or_create_collection(
        settings.chroma_collection_name,
        metadata={"hnsw:space": "cosine"},
    )