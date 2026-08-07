import sqlite3
from dataclasses import dataclass

from bs4 import BeautifulSoup

from app.core.config import settings
from app.rag.chroma_client import get_collection
from app.rag.embedder import embed_text


@dataclass(frozen=True)
class DataSource:
    key: str
    db_path: str


SOURCES = [
    DataSource(key="tarun", db_path=settings.tarun_db_path),
    DataSource(key="nobin", db_path=settings.nobin_db_path),
]

PAGES_QUERY = """
    SELECT
        p.id,
        b.name as book_name,
        c.name as chapter_name,
        p.content
    FROM pages p
    LEFT JOIN books b ON p.book = b.id
    LEFT JOIN chapters c ON p.chapter = c.id
    WHERE p.content IS NOT NULL
"""


def clean_html(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    return soup.get_text(separator="\n").strip()


def ingest_source(source: DataSource, collection) -> int:
    conn = sqlite3.connect(source.db_path)
    cursor = conn.cursor()
    cursor.execute(PAGES_QUERY)
    rows = cursor.fetchall()

    print(f"[{source.key}] Found {len(rows)} pages. Starting ingestion...")

    ids = []
    documents = []
    metadatas = []
    embeddings = []
    count = 0

    for row in rows:
        page_id, book_name, chapter_name, content = row

        clean_text = clean_html(content)
        if not clean_text:
            continue

        full_text = f"Book: {book_name}\nChapter: {chapter_name}\nContent:\n{clean_text}"
        embedding = embed_text(full_text)

        ids.append(f"{source.key}_{page_id}")
        documents.append(full_text)
        metadatas.append({
            "book": book_name or "Unknown",
            "chapter": chapter_name or "Unknown",
            "page_id": page_id,
            "source_db": source.key,
        })
        embeddings.append(embedding)
        count += 1

        if len(ids) >= 100:
            collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings,
            )
            print(f"[{source.key}] Ingested batch ending at page {page_id}")
            ids, documents, metadatas, embeddings = [], [], [], []

    if ids:
        collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

    conn.close()
    print(f"[{source.key}] Ingestion complete: {count} chunks.")
    return count


def ingest_all() -> None:
    collection = get_collection()
    total = sum(ingest_source(source, collection) for source in SOURCES)
    print(f"All sources ingested. Total chunks: {total}")


if __name__ == "__main__":
    ingest_all()
