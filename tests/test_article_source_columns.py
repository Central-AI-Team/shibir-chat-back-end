"""Smoke test for the source_type/source_ref/source_metadata columns added to
the articles table (see scripts/add_source_type_to_articles.py). Confirms a
non-web-article source (e.g. a PDF) round-trips correctly.
"""

from __future__ import annotations

from app.db.models import Article, ContentStatus
from app.db.session import SessionLocal


def test_article_pdf_source_round_trips():
    session = SessionLocal()
    try:
        article = Article(
            title="Sample PDF Article",
            content="নমুনা বিষয়বস্তু।",
            language="bn",
            status=ContentStatus.draft,
            source_type="pdf",
            source_ref="/data/uploads/sample.pdf",
            source_metadata={"page_number": 12, "extracted_by": "pdfminer"},
        )
        session.add(article)
        session.commit()
        article_id = article.id

        session.expunge_all()  # force a fresh read from the DB, not the identity map

        fetched = session.get(Article, article_id)
        assert fetched is not None
        assert fetched.source_type == "pdf"
        assert fetched.source_ref == "/data/uploads/sample.pdf"
        assert fetched.source_metadata == {"page_number": 12, "extracted_by": "pdfminer"}
    finally:
        session.query(Article).filter(Article.title == "Sample PDF Article").delete()
        session.commit()
        session.close()
