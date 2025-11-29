import chromadb
from .embedder import embed_text

chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection("documents")

def retrieve_relevant_docs(query: str, top_k: int = 3):
    query_emb = embed_text(query)

    result = collection.query(
        query_embeddings=[query_emb],
        n_results=top_k
    )

    docs = result.get("documents", [[]])[0]
    return docs
