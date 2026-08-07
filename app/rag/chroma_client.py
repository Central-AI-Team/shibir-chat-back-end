from functools import lru_cache

import chromadb

from app.core.config import settings


@lru_cache
def get_collection():
    client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    return client.get_or_create_collection(settings.chroma_collection_name)
