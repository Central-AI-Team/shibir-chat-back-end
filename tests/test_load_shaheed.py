"""scripts/load_shaheed: pg_dump COPY reader (no DB), upsert idempotence and the
ingest sweep/rendering for the shaheed table (test DB, rolled back)."""

from __future__ import annotations

from datetime import date, datetime, timezone

from app.loaders.sources import iter_copy_rows
from scripts import load_shaheed

DUMP = (
    "COPY public.shaheed (id, serial_no, name, shahadat_date, shahadat_date_text, "
    "shahadat_place, organizational_rank) FROM stdin;\n"
    "1\t9001\tশহীদ সাব্বির আহমদ\t1982-03-11\t১১ মার্চ, ১৯৮২\tরাজশাহী\tসদস্য প্রার্থী\n"
    "2\t9002\tশহীদ আব্দুল হামিদ\t\\N\t১৯৮২\tঠাকুরগাঁও\tকর্মী\n"
    "3\tbad\t\t\t\t\t\n"
    "\\.\n"
    "COPY public.other (id) FROM stdin;\n9\n\\.\n"
)


def _dump(tmp_path):
    p = tmp_path / "shaheed.sql"
    p.write_text(DUMP, encoding="utf-8")
    return p


def test_copy_reader_parses_rows_nulls_and_stops_at_terminator(tmp_path):
    rows = list(iter_copy_rows(_dump(tmp_path), "shaheed"))

    assert [r["serial_no"] for r in rows] == ["9001", "9002", "bad"]
    assert rows[0]["name"] == "শহীদ সাব্বির আহমদ"
    assert rows[1]["shahadat_date"] is None


def test_upsert_is_idempotent_and_skips_malformed_rows(db_session, tmp_path):
    from app.db.models import ContentStatus, Shaheed

    rows = load_shaheed.read_shaheed(_dump(tmp_path))

    first = load_shaheed.upsert_shaheed(db_session, rows, publish=False)
    again = load_shaheed.upsert_shaheed(db_session, rows, publish=False)
    published = load_shaheed.upsert_shaheed(db_session, rows, publish=True)

    assert (first.inserted, first.skipped) == (2, 1)
    assert (again.inserted, again.updated, again.unchanged) == (0, 0, 2)
    assert published.updated == 2
    rec = db_session.query(Shaheed).filter_by(serial_no=9001).one()
    assert rec.status == ContentStatus.published
    assert rec.shahadat_date == date(1982, 3, 11)


def test_render_and_sweep(db_session):
    from app.db.models import ContentStatus, Shaheed
    from app.rag import ingest

    ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    live = Shaheed(serial_no=9101, name="শহীদ ক", shahadat_date_text="১ জানুয়ারি, ২০০০",
                   shahadat_place="ঢাকা", organizational_rank="সাথী",
                   status=ContentStatus.published, embedded_at=ts, updated_at=ts)
    drafted = Shaheed(serial_no=9102, name="শহীদ খ", shahadat_date_text="x",
                      shahadat_place="y", organizational_rank="z",
                      status=ContentStatus.draft, embedded_at=ts, updated_at=ts)
    db_session.add_all([live, drafted])
    db_session.flush()

    text = ingest.render_shaheed(live)
    assert "শাহাদাতের স্থান: ঢাকা" in text and "শহীদ ক" in text

    deleted: list[str] = []

    class Fake:
        def delete(self, where):
            deleted.extend(where["row_key"]["$in"])

    ingest.sweep_unpublished(db_session, Fake())

    assert f"shaheed_{drafted.id}" in deleted
    assert f"shaheed_{live.id}" not in deleted
    assert drafted.embedded_at is None
