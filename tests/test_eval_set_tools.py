"""scripts/eval tooling: the v2 validator rules (each one), v1 compatibility."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import validate_eval_set as v

LIVE = {"page_1", "page_2", "page_3", "article_9"}


def rec(**over) -> dict:
    base = {
        "id": "bn-0001", "slice": "answerable_bn", "category": "কুরআন ও তাফসীর", "source_db": "tafheem",
        "language": "bn", "question": "সূরা ফাতিহায় সিরাতুল মুসতাকীম বলতে কোন পথ বোঝানো হয়েছে?",
        "previous_turn": None, "expected_row_keys": ["page_1"], "answerable": True,
        "variant_of": None, "review": "accepted", "notes": "",
    }
    base.update(over)
    return base


NEG = rec(id="neg-0001", slice="unanswerable_on_theme", category=None, source_db=None,
          question="ইসলামে মহাকাশ ভ্রমণের বিধান কী?", expected_row_keys=[], answerable=False)
OFF = rec(id="off-0001", slice="off_topic", category=None, source_db=None, language="en",
          question="Who won the cricket world cup in 2023?", expected_row_keys=[], answerable=False)
FU = rec(id="fu-0001", slice="followup", question="এটা কোন সূরায় আছে?",
         previous_turn="সিরাতুল মুসতাকীম কী?", expected_row_keys=["page_1"])
BL = rec(id="bl-0001", slice="answerable_banglish", language="banglish",
         question="surah fatiha te siratul mustaqim bolte kon poth bojhano hoyeche?", variant_of="bn-0001")
EN = rec(id="en-0001", slice="answerable_en", language="en",
         question="Which path does the straight path in Surah Fatiha refer to?", variant_of="bn-0001")


def write(tmp_path: Path, records) -> Path:
    p = tmp_path / "set.json"
    p.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    return p


@pytest.fixture(autouse=True)
def _no_corpus(monkeypatch):
    monkeypatch.setattr(v, "live_row_keys", lambda: set(LIVE))
    monkeypatch.setattr(v, "bad_page_ids", lambda: {3})


def errors_for(tmp_path, records, **kw) -> list[str]:
    hard, _, _ = v.validate(write(tmp_path, records), **kw)
    return hard


def test_a_complete_v2_set_passes(tmp_path):
    assert errors_for(tmp_path, [rec(), BL, EN, FU, NEG, OFF]) == []


# -- each rule -----------------------------------------------------------------

def test_duplicate_ids(tmp_path):
    errs = errors_for(tmp_path, [rec(), rec(question="অন্য একটি প্রশ্ন কী?")])
    assert any("duplicate id 'bn-0001'" in e for e in errs)


def test_slice_must_come_from_the_fixed_list(tmp_path):
    errs = errors_for(tmp_path, [rec(slice="hard_questions")])
    assert any("slice 'hard_questions' not in" in e for e in errs)


def test_id_prefix_matches_the_slice(tmp_path):
    errs = errors_for(tmp_path, [rec(id="x-0001")])
    assert any("id must start with 'bn'-" in e for e in errs)


def test_language_is_fixed_by_the_slice(tmp_path):
    assert any("requires language 'bn'" in e for e in errors_for(tmp_path, [rec(language="en")]))
    assert any("language 'de' not in" in e for e in errors_for(tmp_path, [rec(language="de")]))
    # negatives may be in any language
    assert errors_for(tmp_path, [dict(NEG, language="banglish")]) == []


def test_answerable_follows_the_slice_and_the_keys(tmp_path):
    assert any("requires answerable=True" in e for e in errors_for(tmp_path, [rec(answerable=False, expected_row_keys=[])]))
    assert any("answerable=false but expected_row_keys is non-empty" in e
               for e in errors_for(tmp_path, [dict(NEG, expected_row_keys=["page_1"])]))
    assert any("answerable=true but expected_row_keys is empty" in e
               for e in errors_for(tmp_path, [rec(expected_row_keys=[])]))


def test_expected_keys_must_exist_in_chroma_unless_skipped(tmp_path):
    errs = errors_for(tmp_path, [rec(expected_row_keys=["page_404"])])
    assert any("NOT in the live Chroma collection" in e and "page_404" in e for e in errs)
    assert errors_for(tmp_path, [rec(expected_row_keys=["page_404"])], use_chroma=False) == []


def test_known_bad_pages_and_malformed_keys_are_rejected(tmp_path):
    assert any("KNOWN-BAD" in e for e in errors_for(tmp_path, [rec(expected_row_keys=["page_3"])]))
    assert any("not a 'page_<int>'" in e for e in errors_for(tmp_path, [rec(expected_row_keys=["chunk_1"])]))


def test_follow_ups_need_previous_turn_and_an_anaphor(tmp_path):
    assert any("need a non-empty previous_turn" in e for e in errors_for(tmp_path, [rec(), dict(FU, previous_turn=None)]))
    assert any("no anaphor" in e for e in errors_for(tmp_path, [dict(FU, question="সূরা ফাতিহা কোথায় নাজিল হয়?")]))
    assert any("previous_turn must be null" in e for e in errors_for(tmp_path, [rec(previous_turn="x")]))


def test_variants_point_at_a_bn_record_with_the_same_keys(tmp_path):
    assert any("must name an existing record" in e for e in errors_for(tmp_path, [BL]))
    assert any("differ from variant_of" in e
               for e in errors_for(tmp_path, [rec(), dict(BL, expected_row_keys=["page_2"])]))
    assert any("is not an answerable_bn record" in e
               for e in errors_for(tmp_path, [rec(), FU, dict(BL, variant_of="fu-0001")]))
    assert any("variant_of must be null" in e for e in errors_for(tmp_path, [rec(variant_of="bn-0001")]))


def test_review_values_and_negative_metadata(tmp_path):
    assert any("review 'maybe' not in" in e for e in errors_for(tmp_path, [rec(review="maybe")]))
    assert any("negatives must have category and source_db null" in e
               for e in errors_for(tmp_path, [dict(NEG, category="x")]))
    assert any("answerable records need a category" in e for e in errors_for(tmp_path, [rec(category=None)]))
    assert any("source_db 'mysql' not in" in e for e in errors_for(tmp_path, [rec(source_db="mysql")]))


def test_near_duplicates_within_a_language_only(tmp_path):
    dup = rec(id="bn-0002", question="সূরা ফাতিহায় সিরাতুল মুসতাকীম বলতে কোন পথ বোঝানো হয়?")
    assert any("near-duplicate" in e for e in errors_for(tmp_path, [rec(), dup]))
    assert errors_for(tmp_path, [rec(), BL, EN]) == []  # translations are not duplicates


def test_missing_field_is_reported(tmp_path):
    r = rec()
    del r["notes"]
    assert any("missing field(s) ['notes']" in e for e in errors_for(tmp_path, [r]))


def test_accepted_only_filters_the_summary_and_checks(tmp_path):
    pending = rec(id="bn-0002", question="অন্য প্রশ্ন কী?", expected_row_keys=["page_404"], review="pending")
    assert errors_for(tmp_path, [rec(), pending], accepted_only=True) == []
    assert errors_for(tmp_path, [rec(), pending]) != []


# -- v1 compatibility -----------------------------------------------------------

def test_v1_file_still_validates_and_maps_to_legacy_slice(tmp_path):
    v1 = [
        {"_README": "x", "query": "যাকাত কী?", "answerable": True, "relevant_ids": ["page_1"], "lang": "bn",
         "difficulty": "direct", "reference": "r"},
        {"query": "mitochondria?", "answerable": False, "relevant_ids": [], "lang": "en", "difficulty": "direct",
         "reference": ""},
    ]
    hard, records, is_v2 = v.validate(write(tmp_path, v1))
    assert hard == [] and is_v2 is False
    assert [r["slice"] for r in records] == ["legacy", "legacy"]
    assert records[0]["expected_row_keys"] == ["page_1"] and records[0]["language"] == "bn"
    assert records[0]["review"] == "accepted"
    assert v.slice_report(records)[0][0] == "answerable_bn"


def test_v1_rules_still_fire(tmp_path):
    errs = errors_for(tmp_path, [{"query": "x?", "answerable": True, "relevant_ids": []}])
    assert any("answerable=true but relevant_ids is empty" in e for e in errs)


def test_cli_exit_codes(tmp_path, capsys):
    assert v.main([str(write(tmp_path, [rec(), NEG]))]) == 0
    assert "answerable_bn" in capsys.readouterr().out
    assert v.main([str(write(tmp_path, [rec(expected_row_keys=["page_404"])]))]) == 1
    assert v.main([str(write(tmp_path, [rec(expected_row_keys=["page_404"])])), "--no-chroma"]) == 0


# -- drafting -------------------------------------------------------------------

from scripts import draft_eval_questions as d  # noqa: E402

PAGE = {"row_key": "page_1", "book": "বই", "chapter": "অধ্যায়", "category": "কুরআন ও তাফসীর",
        "source_db": "tafheem",
        "content": "যাকাত ইসলামের পাঁচটি স্তম্ভের একটি এবং এটি নিসাব পরিমাণ সম্পদের মালিকের উপর ফরজ। "
                   "নিসাবের পরিমাণ সাড়ে সাত ভরি সোনা অথবা সাড়ে বায়ান্ন ভরি রুপা।"}


def test_verbatim_reuse_is_rejected_but_paraphrase_is_kept():
    assert d.draft_ok("যাকাত ইসলামের পাঁচটি স্তম্ভের একটি এবং এটি কাদের উপর ফরজ?", PAGE["content"]) \
        == "reuses 6+ consecutive words of the page"
    assert d.draft_ok("কী পরিমাণ সম্পদ থাকলে একজন মুসলমানের উপর যাকাত আবশ্যক হয়?", PAGE["content"]) is None


def test_non_bengali_short_and_non_question_drafts_are_dropped():
    assert d.draft_ok("What is the nisab of zakat for gold?", PAGE["content"]) == "not Bengali"
    assert d.draft_ok("যাকাত কী?", PAGE["content"]).startswith("length")
    assert d.draft_ok("যাকাতের নিসাব কত তা এই পৃষ্ঠায় বলা আছে।", PAGE["content"]) == "not a question"


def test_record_shape_and_id_sequence():
    existing = [{"id": "bn-0007"}, {"id": "neg-0100"}, {"id": "bn-0003"}]
    assert d.next_index(existing) == 8 and d.next_index([]) == 1
    r = d.make_record(8, PAGE, "  কী পরিমাণ সম্পদে যাকাত  ফরজ হয়?  ", "নিসাবের পরিমাণ")
    assert r["id"] == "bn-0008" and r["slice"] == "answerable_bn" and r["review"] == "pending"
    assert r["expected_row_keys"] == ["page_1"] and r["source_db"] == "tafheem"
    assert r["question"] == "কী পরিমাণ সম্পদে যাকাত ফরজ হয়?"
    assert set(r) == set(v.FIELDS)


def test_interleave_spreads_over_category_and_source():
    pages = [dict(PAGE, row_key=f"page_{i}", category=c, source_db=s)
             for i, (c, s) in enumerate([("A", "x")] * 3 + [("B", "y")] * 2 + [("A", "z")])]
    order = [(p["category"], p["source_db"]) for p in d.interleave(pages)]
    assert order[:3] == [("A", "x"), ("B", "y"), ("A", "z")]


def test_estimate_and_dry_run_make_no_call(tmp_path, monkeypatch, capsys):
    sample = tmp_path / "s.json"
    sample.write_text(json.dumps([PAGE] * 3, ensure_ascii=False), encoding="utf-8")
    out = tmp_path / "seed.json"
    called = []
    monkeypatch.setattr(d, "complete", lambda *a, **k: called.append(1))
    assert d.main(["--sample", str(sample), "--out", str(out)]) == 0
    assert "re-run with --yes" in capsys.readouterr().out and not called and not out.exists()
    est = d.estimate([PAGE])
    assert est["pages"] == 1 and est["usd"] > 0


def test_yes_writes_pending_records_and_skips_drafted_pages(tmp_path, monkeypatch):
    sample = tmp_path / "s.json"
    sample.write_text(json.dumps([PAGE, dict(PAGE, row_key="page_2")], ensure_ascii=False), encoding="utf-8")
    out = tmp_path / "seed.json"
    out.write_text(json.dumps([d.make_record(1, dict(PAGE, row_key="page_2"), "আগের একটি প্রশ্ন কি আছে?", "")]))
    reply = {"questions": [
        {"question": "কী পরিমাণ সম্পদ থাকলে একজন মুসলমানের উপর যাকাত আবশ্যক হয়?", "answer_in_page": "নিসাব"},
        {"question": "যাকাত ইসলামের পাঁচটি স্তম্ভের একটি এবং এটি কী?", "answer_in_page": "x"},  # verbatim
    ]}
    from types import SimpleNamespace
    calls = []

    def fake_complete(task, messages, **kw):
        calls.append(task)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(reply, ensure_ascii=False)))])

    monkeypatch.setattr(d, "complete", fake_complete)
    assert d.main(["--sample", str(sample), "--out", str(out), "--yes"]) == 0
    assert calls == ["eval_draft"]  # page_2 already drafted -> one call
    records = json.loads(out.read_text(encoding="utf-8"))
    assert [r["id"] for r in records] == ["bn-0001", "bn-0002"]
    assert records[1]["expected_row_keys"] == ["page_1"] and records[1]["review"] == "pending"
