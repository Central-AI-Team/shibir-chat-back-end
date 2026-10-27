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


# -- split ----------------------------------------------------------------------

from collections import Counter  # noqa: E402

from scripts import split_eval_set as sp  # noqa: E402


def _accepted_set() -> list[dict]:
    out = []
    for i in range(1, 21):
        cat = "A" if i <= 12 else "B"
        bn = rec(id=f"bn-{i:04d}", category=cat, question=f"প্রশ্ন নম্বর {i} এর বিষয়বস্তু কী বলা হয়েছে?",
                 expected_row_keys=[f"page_{i}"])
        out.append(bn)
        if i % 2 == 0:
            out.append(rec(id=f"bl-{i:04d}", slice="answerable_banglish", language="banglish", category=cat,
                           question=f"proshno number {i} er bishoy ki?", expected_row_keys=[f"page_{i}"],
                           variant_of=bn["id"]))
    for i in range(1, 11):
        out.append(dict(NEG, id=f"neg-{i:04d}", question=f"ইসলামে বিষয় {i} সম্পর্কে কী বিধান আছে?"))
    out.append(rec(id="bn-0099", question="এটি পর্যালোচনায় বাদ পড়া একটি প্রশ্ন কি?", review="pending"))
    out.append(rec(id="bn-0098", question="এটি প্রত্যাখ্যাত একটি প্রশ্ন কি?", review="rejected"))
    return out


def test_split_uses_accepted_only_is_deterministic_and_stratified():
    records = _accepted_set()
    train, test = sp.split(records)
    assert sp.split(records) == (train, test)  # same seed, same files
    ids = {r["id"] for r in train + test}
    assert "bn-0099" not in ids and "bn-0098" not in ids
    assert len(train) + len(test) == 40
    # ~30 % per stratum: bn/A 12 families -> 4 test; bn/B 8 -> 2; neg 10 -> 3
    by = Counter((r["slice"], r["category"]) for r in test if r["slice"] != "answerable_banglish")
    assert by[("answerable_bn", "A")] == 4 and by[("answerable_bn", "B")] == 2
    assert by[("unanswerable_on_theme", None)] == 3
    assert [r["id"] for r in test] == sorted(r["id"] for r in test)


def test_variants_stay_with_their_bn_record():
    train, test = sp.split(_accepted_set())
    side = {r["id"]: "test" for r in test} | {r["id"]: "train" for r in train}
    for r in train + test:
        if r.get("variant_of"):
            assert side[r["variant_of"]] == side[r["id"]], r["id"]


def test_split_changes_with_the_seed_and_single_strata_go_to_train():
    records = _accepted_set()
    assert sp.split(records, seed=1)[1] != sp.split(records, seed=2)[1]
    lonely = [rec(id="bn-0001", category="Z")]
    train, test = sp.split(lonely)
    assert len(train) == 1 and test == []


def test_split_cli_writes_both_files(tmp_path, capsys):
    seed = tmp_path / "seed.json"
    seed.write_text(json.dumps(_accepted_set(), ensure_ascii=False), encoding="utf-8")
    train, test = tmp_path / "train.json", tmp_path / "test.json"
    assert sp.main(["--seed-file", str(seed), "--train", str(train), "--test", str(test)]) == 0
    assert "answerable_bn" in capsys.readouterr().out
    assert len(json.loads(train.read_text())) + len(json.loads(test.read_text())) == 40
    empty = tmp_path / "empty.json"
    empty.write_text("[]")
    assert sp.main(["--seed-file", str(empty), "--train", str(train), "--test", str(test)]) == 1


# -- eval script breakdowns -------------------------------------------------------

from scripts import eval_retrieval as er, tune_threshold as tt  # noqa: E402


def test_both_eval_loaders_read_v1_and_v2(tmp_path):
    v2 = write(tmp_path, [rec(), NEG])
    loaded = er._load_questions(v2)
    assert loaded[0]["query"] == rec()["question"] and loaded[0]["relevant_ids"] == ["page_1"]
    assert loaded[0]["slice"] == "answerable_bn" and loaded[1]["slice"] == "unanswerable_on_theme"
    assert tt._load_questions(v2)[0]["category"] == "কুরআন ও তাফসীর"
    v1 = tmp_path / "v1.json"
    v1.write_text(json.dumps([{"query": "যাকাত কী?", "answerable": True, "relevant_ids": ["page_1"], "lang": "bn"}]))
    assert er._load_questions(v1)[0]["slice"] == "legacy"
    assert tt._load_questions(v1)[0]["slice"] == "legacy"


def _record(slice_, cat, answerable, gold, ranked, top):
    return {
        "slice": slice_, "category": cat, "answerable": answerable, "relevant_ids": gold,
        "scorable": bool(answerable and gold), "top_rerank_score": top,
        "metrics": {"final": {5: er._score_one(ranked, gold, 5)}},
    }


def test_retrieval_breakdown_groups_without_changing_the_metric(monkeypatch):
    monkeypatch.setattr(er.settings, "min_rerank_score", 0.5)
    records = [
        _record("answerable_bn", "A", True, ["page_1"], ["page_1", "page_9"], 0.9),   # hit, MRR 1
        _record("answerable_bn", "A", True, ["page_2"], ["page_9", "page_2"], 0.7),   # hit, MRR .5
        _record("answerable_bn", "B", True, ["page_3"], ["page_9"], 0.2),             # miss, false refusal
        _record("off_topic", None, False, [], ["page_9"], 0.6),                       # false answer
        _record("unanswerable_on_theme", None, False, [], [], None),                  # true refusal
    ]
    by_slice = er._breakdown(records, "slice", "final", 5)
    bn = by_slice["answerable_bn"]
    assert bn["n"] == 3 and bn["n_scored"] == 3
    assert bn["recall"] == pytest.approx(2 / 3) and bn["mrr"] == pytest.approx(0.5)
    assert bn["false_refusals"] == 1 and bn["false_answers"] == 0 and bn["gate_recall"] == pytest.approx(2 / 3)
    assert by_slice["off_topic"]["false_answers"] == 1 and by_slice["off_topic"]["recall"] is None
    assert by_slice["off_topic"]["gate_precision"] == 0.0  # one cleared, none of them answerable
    assert by_slice["unanswerable_on_theme"]["gate_precision"] is None  # nothing cleared the gate
    by_cat = er._breakdown(records, "category", "final", 5)
    assert by_cat["A"]["recall"] == 1.0 and by_cat["B"]["recall"] == 0.0
    # the whole-set gate numbers match tune_threshold's confusion at the same threshold
    tt_rows = [{"answerable": r["answerable"], "top_score": r["top_rerank_score"], "top_is_gold": None}
               for r in records]
    whole = er._gate_counts(records)
    m = tt._metrics_at(0.5, tt_rows)
    assert (whole["false_refusals"], whole["false_answers"]) == (m["false_refusals"], m["false_answers"])
    assert whole["gate_precision"] == pytest.approx(m["precision"], abs=1e-4)  # tune_threshold rounds to 4 dp


def test_tune_threshold_breakdown_is_metrics_at_per_group():
    rows = [
        {"slice": "answerable_bn", "category": "A", "answerable": True, "top_score": 0.9, "top_is_gold": True},
        {"slice": "answerable_bn", "category": "A", "answerable": True, "top_score": 0.3, "top_is_gold": False},
        {"slice": "off_topic", "category": None, "answerable": False, "top_score": 0.8, "top_is_gold": None},
    ]
    table = tt._breakdown_at(0.5, rows, "slice")
    assert table["answerable_bn"]["n"] == 2 and table["answerable_bn"]["false_refusals"] == 1
    assert table["off_topic"]["false_answers"] == 1
    assert table["answerable_bn"] == {"n": 2, **tt._metrics_at(0.5, rows[:2])}

# -- follow-up eval ---------------------------------------------------------------

from scripts import eval_followups as ef  # noqa: E402


def test_followup_eval_detects_the_history_parameter_and_loads_the_slice(tmp_path):
    assert ef.pipeline_has_history() is True
    p = write(tmp_path, [rec(), FU, NEG])
    loaded = ef.load_followups(p)
    assert [r["id"] for r in loaded] == ["fu-0001"]
    results = [{"without": {"metrics": {5: {"recall": 0.0, "mrr": 0.0, "hit_rate": 0.0}}},
                "with": {"metrics": {5: {"recall": 1.0, "mrr": 0.5, "hit_rate": 1.0}}}}] * 2
    agg = ef.aggregate(results, [5])
    assert agg["without"][5]["recall"] == 0.0 and agg["with"][5]["mrr"] == 0.5
