"""app/rag/ingest.sweep_unpublished: chunks of rows that are embedded but no
longer published (or excluded) are deleted from Chroma. Runs on the test DB
(rolled back) with a fake collection -- no Chroma, no embedding."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.rag import ingest

EMBEDDED = datetime(2026, 1, 1, tzinfo=timezone.utc)
UPDATED = datetime(2025, 12, 31, tzinfo=timezone.utc)


class FakeCollection:
    def __init__(self, fail: bool = False):
        self.deleted: list[str] = []
        self.fail = fail

    def delete(self, where):
        if self.fail:
            raise RuntimeError("chroma is down")
        self.deleted.extend(where["row_key"]["$in"])


@pytest.fixture
def corpus(db_session):
    from app.db.models import Article, Book, ContentStatus, Page

    book = Book(name="সুইপ পরীক্ষা", language="bn")
    db_session.add(book)
    db_session.flush()

    def page(status, embedded=True, excluded=False):
        return Page(book_id=book.id, content="লেখা", language="bn", status=status,
                    excluded_from_rag=excluded, embedded_at=EMBEDDED if embedded else None,
                    updated_at=UPDATED)

    def article(status, embedded=True):
        return Article(title="প্রবন্ধ", content="লেখা", language="bn", status=status,
                       embedded_at=EMBEDDED if embedded else None, updated_at=UPDATED)

    rows = {
        "live": page(ContentStatus.published),
        "drafted": page(ContentStatus.draft),
        "excluded": page(ContentStatus.published, excluded=True),
        "never_embedded": page(ContentStatus.draft, embedded=False),
        "live_article": article(ContentStatus.published),
        "drafted_article": article(ContentStatus.draft),
    }
    db_session.add_all(rows.values())
    db_session.flush()
    return rows


def _ours(keys, corpus):
    """Restrict to this test's rows (the test DB may hold others)."""
    mine = {f"page_{r.id}" for n, r in corpus.items() if "article" not in n}
    mine |= {f"article_{r.id}" for n, r in corpus.items() if "article" in n}
    return sorted(k for k in keys if k in mine)


def test_sweeps_drafted_and_excluded_rows_only(db_session, corpus):
    collection = FakeCollection()

    ingest.sweep_unpublished(db_session, collection)
    db_session.expire_all()

    assert _ours(collection.deleted, corpus) == sorted([
        f"page_{corpus['drafted'].id}",
        f"page_{corpus['excluded'].id}",
        f"article_{corpus['drafted_article'].id}",
    ])
    for name in ("drafted", "excluded", "drafted_article"):
        assert corpus[name].embedded_at is None
        assert corpus[name].updated_at == UPDATED  # not a content change
    for name in ("live", "live_article"):
        assert corpus[name].embedded_at == EMBEDDED


def test_second_sweep_is_a_no_op(db_session, corpus):
    ingest.sweep_unpublished(db_session, FakeCollection())
    again = FakeCollection()

    ingest.sweep_unpublished(db_session, again)

    assert _ours(again.deleted, corpus) == []


def test_republished_row_is_re_embedded_after_a_sweep(db_session, corpus):
    from app.db.models import ContentStatus

    ingest.sweep_unpublished(db_session, FakeCollection())
    corpus["drafted"].status = ContentStatus.published
    db_session.flush()

    assert corpus["drafted"].id in {p.id for p in ingest._due_pages(db_session)}


def test_chroma_failure_keeps_embedded_at_so_the_sweep_retries(db_session, corpus):
    with pytest.raises(RuntimeError):
        ingest.sweep_unpublished(db_session, FakeCollection(fail=True))
    db_session.expire_all()

    assert corpus["drafted"].embedded_at == EMBEDDED
