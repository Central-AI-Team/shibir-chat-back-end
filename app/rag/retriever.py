from app.core.config import settings
from app.rag.chroma_client import get_collection
from app.rag.embedder import embed_text
from app.schemas.query import Citation


def retrieve_relevant_docs(query: str, top_k: int | None = None) -> list[Citation]:
    top_k = top_k or settings.top_k
    query_emb = embed_text(query)

    result = get_collection().query(
        query_embeddings=[query_emb],
        n_results=top_k,
        include=["documents", "metadatas"],
    )

    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]

    return [
        Citation(
            book=meta.get("book", "Unknown"),
            chapter=meta.get("chapter", "Unknown"),
            source_db=meta.get("source_db", "unknown"),
            content=doc,
        )
        for doc, meta in zip(documents, metadatas)
    ]
