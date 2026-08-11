"""Ingestion.

CHANGES vs the original:
  1. Pages are CHUNKED. Previously one page = one vector; your median page is
     2,451 chars and the largest 23,123, so each vector was averaging together
     several unrelated ideas.
  2. Batch embedding (embed_texts) instead of one encode() call per row. On
     ~4,100 pages -> ~15,000 chunks this is the difference between minutes and
     a very long afternoon.
  3. Old chunks for a page are deleted before re-upsert. Chunk COUNT changes
     when content is edited, so upsert alone leaves orphaned stale chunks
     behind that keep showing up in search results forever.
  4. Bengali header (chunker.build_document) instead of English "Book:/Chapter:".
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import or_
from sqlalchemy.orm import joinedload

from app.db.models import Article, ContentStatus, Page
from app.db.session import SessionLocal
from app.rag.chroma_client import get_collection
from app.rag.chunker import build_document, chunk_text
from app.rag.embedder import embed_texts

BATCH_SIZE = 64  # chunks per Chroma upsert / embedding batch

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


def _drop_stale(collection, prefix: str, row_id: int) -> None:
    """Remove any existing chunks for this row before writing new ones."""
    try:
        collection.delete(where={"row_key": f"{prefix}_{row_id}"})
    except Exception:
        pass


def _flush(collection, ids, docs, metas) -> None:
    if not ids:
        return
    collection.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embed_texts(docs))


def _ingest(session, collection, rows, prefix: str, extract) -> int:
    ids, docs, metas, touched = [], [], [], []
    n_chunks = 0

    for row in rows:
        book, chapter, body, source_db = extract(row)
        pieces = chunk_text(body or "")
        if not pieces:
            continue

        _drop_stale(collection, prefix, row.id)

        for i, piece in enumerate(pieces):
            ids.append(f"{prefix}_{row.id}_c{i}")
            docs.append(build_document(book, chapter, piece))
            metas.append({
                "book": book,
                "chapter": chapter,
                "page_id": row.id,
                "chunk_index": i,
                "row_key": f"{prefix}_{row.id}",
                "source_db": source_db,
            })
            n_chunks += 1

        touched.append(row)

        if len(ids) >= BATCH_SIZE:
            _flush(collection, ids, docs, metas)
            now = datetime.now(timezone.utc)
            for r in touched:
                r.embedded_at = now
            session.commit()
            print(f"[{prefix}] {n_chunks} chunks embedded...")
            ids, docs, metas, touched = [], [], [], []

    _flush(collection, ids, docs, metas)
    now = datetime.now(timezone.utc)
    for r in touched:
        r.embedded_at = now
    session.commit()

    print(f"[{prefix}] done: {n_chunks} chunks.")
    return n_chunks


def ingest_all() -> None:
    collection = get_collection()
    session = SessionLocal()
    try:
        pages = _due_pages(session)
        articles = _due_articles(session)
        print(f"{len(pages)} page(s), {len(articles)} article(s) need embedding.")

        total = _ingest(
            session, collection, pages, "page",
            lambda p: (
                p.book.name if p.book else "Unknown",
                p.chapter.name if p.chapter else "Unknown",
                p.content,
                p.source_db or "unknown",
            ),
        )
        total += _ingest(
            session, collection, articles, "article",
            lambda a: (a.title, "প্রবন্ধ", a.content, "articles"),
        )
        print(f"All sources ingested. Total chunks: {total}")
    finally:
        session.close()


if __name__ == "__main__":
    ingest_all()