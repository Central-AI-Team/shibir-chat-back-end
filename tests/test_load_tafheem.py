"""scripts/load_tafheem.py against a tiny SQLite fixture shaped like
data/tafheemul_quran.db (TEXT ids, literal backslash-n, glued Bengali-digit
footnote markers that restart per sura, negative ids for "(ক)" sub-notes)."""

from __future__ import annotations

import re
import sqlite3
from datetime import datetime, timezone

import pytest

from scripts import load_tafheem as lt

ARABIC = "بِسْمِ ٱللَّهِ"


@pytest.fixture
def tafheem_db(tmp_path):
    path = tmp_path / "tafheem.db"
    con = sqlite3.connect(path)
    con.executescript(
        """
        CREATE TABLE alquran (ruku_id, page_id, para_id, sura_id, ayat_id, arabic_text,
                              bangla_trans, arabic_search TEXT, english_trans TEXT);
        CREATE TABLE expl (expl_id, sura_id, expels);
        CREATE TABLE vumika_sura (sura_id, vumika);
        CREATE TABLE surah_name (_id INTEGER, sura_name TEXT);
        CREATE TABLE users (id, email, password);
        INSERT INTO users VALUES ('1', 'a@b.c', 'hash');
        """
    )
    con.executemany("INSERT INTO surah_name VALUES (?, ?)", [(1, "আল-ফাতিহা"), (2, "আল-বাক্বারাহ")])
    con.executemany(
        "INSERT INTO vumika_sura VALUES (?, ?)",
        [("1", "(০১-ফাতিহা) : নামকরণ:\\n ফাতিহার ভূমিকা এক।"),
         ("2", "বাকারার ভূমিকা।"),
         ("1", "ফাতিহার ভূমিকা দুই।")],
    )
    con.executemany(
        "INSERT INTO alquran (sura_id, ayat_id, arabic_text, bangla_trans) VALUES (?, ?, ?, ?)",
        [
            ("1", "1", ARABIC,
             "পরম করুণাময় আল্লাহর নামে<font color='blue'><sup>১</sup></font>"
             "\\n\\n * সুরা ফাতিহার ভূমিকা:<a href='#://0'>১</a>"),
            ("1", "2", ARABIC, "প্রশংসা আল্লাহর জন্য২ যিনি রব,"),
            ("2", "0", ARABIC, "পরম করুণাময়\\n\\n * ভূমিকা:<a href='#://3'> ২</a>"),
            ("2", "1", ARABIC, "আলিফ লাম মীম।১ তারপর১(ক) শেষ"),
            ("2", "2", ARABIC, "এটি কিতাব।২ হিদায়াত।৩"),
        ],
    )
    con.executemany(
        "INSERT INTO expl VALUES (?, ?, ?)",
        [
            ("1", "1", "\\n[[টিকা: ১) সূরা একের প্রথম টীকা।]]"),
            # columns are (expl_id, sura_id, expels)
            ("1", "2", "টিকা: 1)\\nসূরা দুইয়ের প্রথম টীকা, দেখুন <a href='9:112'>আত্ তাওবা ১১২</a>।"),
            ("-1", "2", "\\n[[টিকা:১)(ক)) সূরা দুইয়ের উপটীকা।]]"),
            ("2", "1", "সূরা একের দ্বিতীয় টীকা।"),
            ("2", "2", "সূরা দুইয়ের দ্বিতীয় টীকা।"),
            ("2", "2", "সূরা দুইয়ের দ্বিতীয় টীকা।"),  # the duplicated row
            ("7", "2", "কোনো আয়াতে এর চিহ্ন নেই।"),
        ],
    )
    con.commit()
    con.close()
    return path


def _pages(suras) -> dict[int, str]:
    return {p.source_page_id: p.content for s in suras for p in s.pages}


# ---------- reading & cleaning (no DB) ----------


def test_footnote_joins_on_sura_and_number_not_number_alone(tafheem_db):
    suras, _ = lt.read_tafheem(tafheem_db)
    pages = _pages(suras)

    # Footnote ১ exists in BOTH suras; each ayah gets its own sura's.
    assert "টীকা ১: সূরা একের প্রথম টীকা।" in pages[1001]
    assert "সূরা দুইয়ের" not in pages[1001]
    assert "টীকা ১: সূরা দুইয়ের প্রথম টীকা" in pages[2001]
    assert "সূরা একের" not in pages[2001]
    # negative expl_id = the "(ক)" sub-note, attached after footnote ১
    assert pages[2001].endswith("টীকা ১(ক): সূরা দুইয়ের উপটীকা।")
    assert pages[1002] == (
        "সূরা আল-ফাতিহা ১:২\nপ্রশংসা আল্লাহর জন্য যিনি রব,\n\nটীকা ২: সূরা একের দ্বিতীয় টীকা।"
    )


def test_cleaning_removes_html_markers_navigation_and_arabic(tafheem_db):
    suras, _ = lt.read_tafheem(tafheem_db)
    pages = _pages(suras)
    everything = "\n".join(pages.values())

    assert not re.search(r"<[a-zA-Z/]", everything)
    assert "\\n" not in everything
    assert "[[" not in everything and "]]" not in everything
    assert ARABIC not in everything
    assert "ভূমিকা:" not in pages[1001]  # the "* ... ভূমিকা:" nav line is gone
    assert pages[1001].split("\n")[1] == "পরম করুণাময় আল্লাহর নামে"  # <sup> marker stripped
    assert pages[2001].split("\n")[1] == "আলিফ লাম মীম। তারপর শেষ"
    # the footnote's own "টিকা: 1)" prefix is replaced, a cross-reference link keeps its text
    assert "টীকা ১: সূরা দুইয়ের প্রথম টীকা, দেখুন আত্ তাওবা ১১২।" in pages[2001]


def test_intro_is_page_one_and_ayah_zero_is_skipped(tafheem_db):
    suras, report = lt.read_tafheem(tafheem_db)
    assert [(s.number, s.name) for s in suras] == [(1, "আল-ফাতিহা"), (2, "আল-বাক্বারাহ")]
    assert [p.source_page_id for p in suras[0].pages] == [1000, 1001, 1002]
    assert [p.source_page_id for p in suras[1].pages] == [2000, 2001, 2002]
    # both vumika rows for sura 1, in table order, with the literal \n decoded
    assert suras[0].pages[0].content == (
        "(০১-ফাতিহা) : নামকরণ:\nফাতিহার ভূমিকা এক।\n\nফাতিহার ভূমিকা দুই।"
    )
    assert report.skipped_ayah0 == 1


def test_report_lists_duplicates_unreferenced_footnotes_and_orphan_markers(tafheem_db):
    _, report = lt.read_tafheem(tafheem_db)
    assert report.n_footnotes == 6
    assert report.duplicate_footnotes == [(2, "২")]
    assert report.unreferenced_footnotes == [(2, "৭")]
    assert report.orphan_markers == [(2, 2, "৩")]


def test_reads_only_the_four_corpus_tables(tafheem_db, monkeypatch):
    statements: list[str] = []
    real_connect = lt._connect

    def tracing_connect(path):
        con = real_connect(path)
        con.set_trace_callback(statements.append)
        return con

    monkeypatch.setattr(lt, "_connect", tracing_connect)
    lt.read_tafheem(tafheem_db)

    tables = {t for s in statements for t in re.findall(r"\bFROM\s+(\w+)", s, re.IGNORECASE)}
    assert tables == set(lt.TABLES)
    assert not any("arabic_text" in s for s in statements)


# ---------- Postgres upsert (test DB, rolled back) ----------


def _tafheem_pages(session):
    from app.db.models import Page

    return {p.source_page_id: p for p in session.query(Page).filter_by(source_db="tafheem")}


def test_rerun_is_idempotent(db_session, tafheem_db):
    from app.db.models import Book, Category, Chapter

    suras, _ = lt.read_tafheem(tafheem_db)
    first = lt.upsert_tafheem(db_session, suras, publish=False)
    second = lt.upsert_tafheem(db_session, suras, publish=False)

    assert (first.inserted, first.updated) == (6, 0)
    assert (second.inserted, second.updated, second.unchanged) == (0, 0, 6)
    assert len(_tafheem_pages(db_session)) == 6
    assert db_session.query(Category).filter_by(name=lt.CATEGORY_NAME).count() == 1
    [book] = db_session.query(Book).filter_by(name=lt.BOOK_NAME).all()
    chapters = db_session.query(Chapter).filter_by(book_id=book.id).order_by(Chapter.position)
    assert [(c.position, c.name) for c in chapters] == [(1, "আল-ফাতিহা"), (2, "আল-বাক্বারাহ")]


def test_content_change_updates_only_that_page_and_bumps_updated_at(db_session, tafheem_db):
    suras, _ = lt.read_tafheem(tafheem_db)
    lt.upsert_tafheem(db_session, suras, publish=False)
    old = datetime(2000, 1, 1, tzinfo=timezone.utc)
    for page in _tafheem_pages(db_session).values():
        page.updated_at = old
    db_session.flush()

    suras[0].pages[1].content += " (সংশোধিত)"
    stats = lt.upsert_tafheem(db_session, suras, publish=False)
    db_session.expire_all()

    assert (stats.updated, stats.unchanged) == (1, 5)
    pages = _tafheem_pages(db_session)
    assert pages[1001].content.endswith(" (সংশোধিত)")
    assert pages[1001].updated_at > old
    assert pages[1002].updated_at == old  # untouched rows are not re-marked for embedding


def test_draft_by_default_publish_promotes_and_supersedes_legacy(db_session, tafheem_db):
    from app.db.models import Book, ContentStatus, Page

    legacy_book = Book(name=lt.BOOK_NAME, language="bn")
    db_session.add(legacy_book)
    db_session.flush()
    legacy = Page(book_id=legacy_book.id, content="আয়াত 1: পুরনো", language="bn",
                  status=ContentStatus.published, source_db=lt.LEGACY_SOURCE_DB,
                  source_page_id=1)
    db_session.add(legacy)
    db_session.flush()
    suras, _ = lt.read_tafheem(tafheem_db)

    lt.upsert_tafheem(db_session, suras, publish=False)
    assert {p.status for p in _tafheem_pages(db_session).values()} == {ContentStatus.draft}
    assert legacy.status == ContentStatus.published

    stats = lt.upsert_tafheem(db_session, suras, publish=True)
    db_session.expire_all()
    assert stats.updated == 6 and stats.superseded == 1
    assert {p.status for p in _tafheem_pages(db_session).values()} == {ContentStatus.published}
    assert db_session.get(Page, legacy.id).status == ContentStatus.draft
    # the legacy book is adopted, not duplicated
    assert db_session.query(Book).filter_by(name=lt.BOOK_NAME).count() == 1
    assert {p.book_id for p in _tafheem_pages(db_session).values()} == {legacy_book.id}
