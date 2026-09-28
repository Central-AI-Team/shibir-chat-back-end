"""scripts/load_articles.py: cleaning/filtering (no DB) and the upsert against
the test DB (rolled back)."""

from __future__ import annotations

from datetime import datetime

import pytest

from app.loaders.sources import RawDoc
from scripts import load_articles as la

CS = la.SOURCES["cs-posts"]
PP = la.SOURCES["pp-articles"]


def _prose(tag: str) -> str:
    """Unique, non-repetitive Bengali text comfortably over MIN_CHARS."""
    return (
        f"{tag} নম্বর লেখাটি সমাজ ও শিক্ষা নিয়ে। লেখক এখানে ইতিহাসের কয়েকটি ঘটনা ব্যাখ্যা করেছেন। "
        f"শেষে তিনি ভবিষ্যতের জন্য কিছু পরামর্শ দিয়েছেন যা পাঠকের কাজে লাগবে।"
    )


def _cs_dump(tmp_path, rows) -> str:
    """rows: (id, title, writer, post_html, news_date) -> path of a cs-posts dump."""
    def q(v):
        return "NULL" if v is None else "'" + v.replace("\\", "\\\\").replace("'", "\\'") + "'"

    lines = [
        "INSERT INTO `posts` (`id`, `title`, `writer`, `designation`, `post`, `news_date`) VALUES"
    ]
    body = [f"({i}, {q(t)}, {q(w)}, NULL, {q(p)}, {q(d)})" for i, t, w, p, d in rows]
    lines.append(",\n".join(body) + ";")
    path = tmp_path / "cs.sql"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _doc(doc_id, html, **meta):
    return RawDoc(source="cs-posts", doc_id=str(doc_id), title=f"শিরোনাম {doc_id}", html=html,
                  meta={"writer": "লেখক", "designation": "", "date": "2024-01-17", **meta})


# ---------- prepare (no DB) ----------


def test_prepare_skips_short_repetitive_and_in_file_duplicates():
    long = f"<p>{_prose('এক')}</p>"
    docs = [
        _doc(1, long),
        _doc(2, "<p>খুব ছোট</p>"),
        _doc(3, "Best Travel Guide Books 2020 " * 40),
        _doc(4, f"<div>{_prose('এক')}</div>"),  # same text, different markup
    ]
    stats = la.PrepareStats()
    rows = la.prepare(docs, CS, set(), stats)

    assert [r.source_ref for r in rows] == ["1"]
    assert stats.parsed == 4
    assert stats.skipped == {"short": 1, "repetitive": 1, "duplicate": 1}


def test_prepare_cleans_and_maps_fields():
    html = (
        '[caption id="attachment_9" caption="ছবি"]<img src="x.jpg">[/caption]'
        f"<p>{_prose('দুই')}</p><p>{_prose('দুই')}</p>"  # body pasted twice
    )
    [row] = la.prepare([_doc(5, html, designation="সম্পাদক")], CS, set(), la.PrepareStats())

    assert "caption" not in row.content
    assert row.content == _prose("দুই")  # drop_repeated_sentences kept one copy
    assert (row.title, row.author, row.language) == ("শিরোনাম 5", "লেখক", "bn")
    assert row.published_at.isoformat() == "2024-01-17T00:00:00+06:00"
    assert row.source_metadata == {"designation": "সম্পাদক", "raw_date": "2024-01-17"}


def test_prepare_maps_pp_articles_byline_and_english():
    text = (
        "An English article about transport policy in Dhaka. It compares bus routes "
        "with rail plans. The author closes with three proposals for the city."
    )
    doc = RawDoc(source="pp-articles", doc_id="7", title="Traffic", html=f"<p>{text}</p>",
                 meta={"date": "24-05-2023 13:16:00", "byline": "perspectivebd.com",
                       "video_url": ""})
    [row] = la.prepare([doc], PP, set(), la.PrepareStats())
    assert (row.author, row.language) == ("perspectivebd.com", "en")
    assert row.published_at.isoformat() == "2023-05-24T13:16:00+06:00"
    assert row.source_metadata == {"raw_date": "24-05-2023 13:16:00"}


@pytest.mark.parametrize("raw", ["", "0000-00-00", "32-13-2020", "yesterday"])
def test_parse_date_rejects_invalid(raw):
    assert la.parse_date(raw) is None


# ---------- upsert (test DB, rolled back) ----------


def _articles(session, source_type="cs-post"):
    from app.db.models import Article

    return {a.source_ref: a for a in session.query(Article).filter_by(source_type=source_type)}


def test_rerun_is_idempotent(db_session, tmp_path):
    path = _cs_dump(tmp_path, [(1, "ক", "লেখক", _prose("১"), "2024-01-17"),
                               (2, "খ", "লেখক", _prose("২"), "0000-00-00")])

    _, _, first, _ = la.load(db_session, CS, path, publish=False)
    _, _, second, _ = la.load(db_session, CS, path, publish=False)

    assert (first.inserted, first.updated) == (2, 0)
    assert (second.inserted, second.updated, second.unchanged) == (0, 0, 2)
    rows = _articles(db_session)
    assert sorted(rows) == ["1", "2"]
    assert rows["2"].published_at is None


def test_dedupes_against_existing_published_page(db_session, tmp_path):
    from app.db.models import Book, ContentStatus, Page

    book = Book(name="পরীক্ষার বই", language="bn")
    db_session.add(book)
    db_session.flush()
    db_session.add_all([
        Page(book_id=book.id, content=_prose("পুরনো"), language="bn",
             status=ContentStatus.published),
        Page(book_id=book.id, content=_prose("খসড়া"), language="bn",
             status=ContentStatus.draft),  # drafts are not dedupe targets
    ])
    db_session.flush()
    path = _cs_dump(tmp_path, [(1, "ক", "লেখক", f"<p>{_prose('পুরনো')}</p>", "2024-01-17"),
                               (2, "খ", "লেখক", _prose("খসড়া"), "2024-01-17")])

    _, prep, ups, _ = la.load(db_session, CS, path, publish=False)

    assert prep.skipped["duplicate"] == 1
    assert ups.inserted == 1
    assert sorted(_articles(db_session)) == ["2"]


def test_draft_by_default_publish_promotes_and_supersedes_legacy(db_session, tmp_path):
    from app.db.models import Article, ContentStatus

    def legacy(ref):
        return Article(title="পুরনো", content=_prose("১"), language="bn",
                       status=ContentStatus.published, source_type=CS.legacy_type,
                       source_ref=ref)

    in_file, not_in_file = legacy("cs_post_1"), legacy("cs_post_999")
    db_session.add_all([in_file, not_in_file])
    db_session.flush()
    # identical text to the legacy copy: must NOT be skipped as its duplicate
    path = _cs_dump(tmp_path, [(1, "ক", "লেখক", _prose("১"), "2024-01-17")])

    _, prep, ups, _ = la.load(db_session, CS, path, publish=False)
    assert prep.skipped["duplicate"] == 0 and ups.inserted == 1
    assert _articles(db_session)["1"].status == ContentStatus.draft
    assert in_file.status == ContentStatus.published  # draft runs leave legacy alone

    _, _, ups, _ = la.load(db_session, CS, path, publish=True)
    db_session.expire_all()
    assert (ups.updated, ups.superseded) == (1, 1)
    assert _articles(db_session)["1"].status == ContentStatus.published
    assert db_session.get(Article, in_file.id).status == ContentStatus.draft
    assert db_session.get(Article, not_in_file.id).status == ContentStatus.published


def test_ingest_uses_title_and_source_type_for_articles(monkeypatch):
    from types import SimpleNamespace

    from app.rag import ingest

    captured = {}

    def fake_ingest(session, collection, rows, prefix, extract):
        if prefix == "article":
            captured["meta"] = extract(SimpleNamespace(
                title="শিরোনাম", content="লেখা", source_type="cs-post"))
        return 0

    class _Session:
        def close(self):
            pass

    monkeypatch.setattr(ingest, "get_collection", lambda: None)
    monkeypatch.setattr(ingest, "SessionLocal", _Session)
    monkeypatch.setattr(ingest, "sweep_unpublished", lambda s, c: 0)
    monkeypatch.setattr(ingest, "_due_pages", lambda s: [])
    monkeypatch.setattr(ingest, "_due_articles", lambda s: [])
    monkeypatch.setattr(ingest, "_ingest", fake_ingest)
    ingest.ingest_all()

    assert captured["meta"] == ("শিরোনাম", "প্রবন্ধ", "লেখা", "cs-post", None, None)


def test_published_at_is_timezone_aware():
    assert la.parse_date("2024-01-17").utcoffset() is not None
    assert isinstance(la.parse_date("22-10-2022 13:16:00"), datetime)
