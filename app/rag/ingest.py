from datetime import datetime, timezone

from sqlalchemy import or_
from sqlalchemy.orm import joinedload

from app.db.models import Article, ContentStatus, Page
from app.db.session import SessionLocal
from app.rag.chroma_client import get_collection
from app.rag.embedder import embed_text

BATCH_SIZE = 100

# Rows never embedded, or edited since their last embedding.
_NEEDS_EMBEDDING = or_(Page.embedded_at.is_(None), Page.updated_at > Page.embedded_at)


def _due_pages(session):
    return (
        session.query(Page)
        .options(joinedload(Page.book), joinedload(Page.chapter))
        .filter(Page.status == ContentStatus.published)
        .filter(_NEEDS_EMBEDDING)
        .all()
    )


def _due_articles(session):
    return (
        session.query(Article)
        .filter(Article.status == ContentStatus.published)
        .filter(or_(Article.embedded_at.is_(None), Article.updated_at > Article.embedded_at))
        .all()
    )


def _upsert_batch(collection, ids, documents, metadatas, embeddings) -> None:
    collection.upsert(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)


def ingest_pages(session, collection) -> int:
    pages = _due_pages(session)
    print(f"[pages] {len(pages)} published page(s) need (re-)embedding.")

    ids, documents, metadatas, embeddings, batch = [], [], [], [], []
    count = 0

    for page in pages:
        book_name = page.book.name if page.book else "Unknown"
        chapter_name = page.chapter.name if page.chapter else "Unknown"
        full_text = f"Book: {book_name}\nChapter: {chapter_name}\nContent:\n{page.content}"

        ids.append(f"page_{page.id}")
        documents.append(full_text)
        metadatas.append({
            "book": book_name,
            "chapter": chapter_name,
            "page_id": page.id,
            "source_db": page.source_db or "unknown",
        })
        embeddings.append(embed_text(full_text))
        batch.append(page)
        count += 1

        if len(ids) >= BATCH_SIZE:
            _upsert_batch(collection, ids, documents, metadatas, embeddings)
            now = datetime.now(timezone.utc)
            for p in batch:
                p.embedded_at = now
            session.commit()
            print(f"[pages] embedded {count} so far...")
            ids, documents, metadatas, embeddings, batch = [], [], [], [], []

    if ids:
        _upsert_batch(collection, ids, documents, metadatas, embeddings)
        now = datetime.now(timezone.utc)
        for p in batch:
            p.embedded_at = now
        session.commit()

    print(f"[pages] ingestion complete: {count} page(s) embedded.")
    return count


def ingest_articles(session, collection) -> int:
    articles = _due_articles(session)
    print(f"[articles] {len(articles)} published article(s) need (re-)embedding.")

    ids, documents, metadatas, embeddings, batch = [], [], [], [], []
    count = 0

    for article in articles:
        full_text = f"Title: {article.title}\nContent:\n{article.content}"

        ids.append(f"article_{article.id}")
        documents.append(full_text)
        metadatas.append({
            "book": article.title,
            "chapter": "Article",
            "page_id": article.id,
            "source_db": "articles",
        })
        embeddings.append(embed_text(full_text))
        batch.append(article)
        count += 1

        if len(ids) >= BATCH_SIZE:
            _upsert_batch(collection, ids, documents, metadatas, embeddings)
            now = datetime.now(timezone.utc)
            for a in batch:
                a.embedded_at = now
            session.commit()
            print(f"[articles] embedded {count} so far...")
            ids, documents, metadatas, embeddings, batch = [], [], [], [], []

    if ids:
        _upsert_batch(collection, ids, documents, metadatas, embeddings)
        now = datetime.now(timezone.utc)
        for a in batch:
            a.embedded_at = now
        session.commit()

    print(f"[articles] ingestion complete: {count} article(s) embedded.")
    return count


def ingest_all() -> None:
    collection = get_collection()
    session = SessionLocal()
    try:
        total = ingest_pages(session, collection) + ingest_articles(session, collection)
        print(f"All sources ingested. Total chunks: {total}")
    finally:
        session.close()


if __name__ == "__main__":
    ingest_all()
