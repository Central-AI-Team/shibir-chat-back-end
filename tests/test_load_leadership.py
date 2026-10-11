"""scripts/load_leadership: CSV reader, rendering and upsert idempotence (test DB, rolled back)."""

from __future__ import annotations

from scripts import load_leadership

CSV = (
    "শিরোনাম,,,,,\n"
    "ক্রম,সেশন,শুরু ,শেষ,সভাপতির নাম,সেক্রেটারি জেনারেলের নাম\n"
    "1,১৯৭৭,৬ ফেব্রুয়ারি ১৯৭৭,সেপ্টেম্বর ১৯৭৭,শহীদ মীর কাসেম আলী,ড. আব্দুল বারী\n"
    "2,২০২৫,জানুয়ারি ২০২৫,-,জাহিদুল ইসলাম,নুূরুল ইসলাম সাদ্দাম\n"
)


def _terms(tmp_path):
    p = tmp_path / "leaders.csv"
    p.write_text(CSV, encoding="utf-8")
    return load_leadership.read_terms(p)


def test_reader_skips_title_and_header(tmp_path):
    terms, skipped = _terms(tmp_path)

    assert (len(terms), skipped) == (2, 2)
    assert terms[0].president == "শহীদ মীর কাসেম আলী"


def test_render_names_roles_and_marks_open_term(tmp_path):
    text = load_leadership.render(_terms(tmp_path)[0])

    assert "কেন্দ্রীয় সভাপতি: শহীদ মীর কাসেম আলী" in text
    assert "(জানুয়ারি ২০২৫ থেকে চলমান)" in text
    assert text.startswith(load_leadership.TITLE)


def test_upsert_is_idempotent_and_publish_flips_status(db_session, tmp_path):
    from app.db.models import Article, ContentStatus

    terms = _terms(tmp_path)[0]

    assert load_leadership.upsert_article(db_session, terms, publish=False) == "inserted"
    assert load_leadership.upsert_article(db_session, terms, publish=False) == "unchanged"
    assert load_leadership.upsert_article(db_session, terms, publish=True) == "updated"
    rec = db_session.query(Article).filter_by(source_type=load_leadership.SOURCE_TYPE).one()
    assert rec.status == ContentStatus.published
