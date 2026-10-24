"""Validate an eval question file before it is trusted.

Two formats are accepted:

  v2 (scripts/eval/*.json; schema in scripts/eval/schema.md): id, slice,
     category, source_db, language, question, previous_turn, expected_row_keys,
     answerable, variant_of, review, notes.
  v1 (scripts/retrieval_eval_questions.json): query, answerable, relevant_ids,
     lang, difficulty, reference. Converted with to_v2() and given slice
     "legacy"; the v1 rules still apply to it.

Hard errors (exit 1):

  1. JSON parses and is a flat list.
  2. v2: every field present with the right type; ids unique; slice in
     SLICES; language and answerable consistent with the slice; review in
     REVIEWS; previous_turn only (and always) on follow-ups, which must
     contain an anaphor (app.services.intent_classifier.ANAPHORA);
     variant_of only on banglish/en variants, pointing at an existing
     answerable_bn record with the same expected_row_keys.
  3. answerable <-> expected_row_keys non-empty; every key is a well-formed
     row_key AND resolves to a chunk in the LIVE Chroma collection
     (settings.chroma_collection_name; read-only). --no-chroma skips the live
     check (tests, machines without the index).
  4. No expected key is a KNOWN-BAD page (the cleanup CSVs in the repo root).
  5. No duplicate or near-duplicate questions within one language.
  6. The file still loads through eval_retrieval._load_questions.

Then a summary: counts per slice against the targets in scripts/eval/schema.md
(drift is a warning), per category / source_db for the answerable slices, and
review status counts.

Usage:
    python -m scripts.validate_eval_set [path] [--no-chroma] [--accepted-only]

Run from a shell with the app's deps + .env (same as `python -m app.rag.ingest`).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent.parent
DEFAULT_FILE = Path(__file__).parent / "retrieval_eval_questions.json"

ROW_KEY_RE = re.compile(r"^(page|article)_\d+$")
NEAR_DUP_JACCARD = 0.72  # token-set overlap above this = flagged as near-dup

SLICES = (
    "answerable_bn",
    "answerable_banglish",
    "answerable_en",
    "followup",
    "unanswerable_on_theme",
    "off_topic",
)
LEGACY_SLICE = "legacy"
LANGUAGES = ("bn", "banglish", "en")
REVIEWS = ("pending", "accepted", "rejected")
SOURCE_DBS = ("tarun", "nobin", "tafheem", "books_sql", "shaheed", "cs-post", "pp-article",
              "central-leadership")
ID_PREFIX = {
    "answerable_bn": "bn", "answerable_banglish": "bl", "answerable_en": "en",
    "followup": "fu", "unanswerable_on_theme": "neg", "off_topic": "off",
}
SLICE_LANGUAGE = {"answerable_bn": "bn", "followup": "bn", "answerable_banglish": "banglish",
                  "answerable_en": "en"}  # negatives: any
SLICE_ANSWERABLE = {"answerable_bn": True, "answerable_banglish": True, "answerable_en": True,
                    "followup": True, "unanswerable_on_theme": False, "off_topic": False}
VARIANT_SLICES = ("answerable_banglish", "answerable_en")
# docs/roadmap-q4-2026.md §3, for a 250-question set.
SLICE_TARGETS = {
    "answerable_bn": 0.40, "answerable_banglish": 0.15, "answerable_en": 0.05,
    "followup": 0.15, "unanswerable_on_theme": 0.15, "off_topic": 0.10,
}
FIELDS = ("id", "slice", "category", "source_db", "language", "question", "previous_turn",
          "expected_row_keys", "answerable", "variant_of", "review", "notes")


class Fail(Exception):
    pass


# ---------------------------------------------------------------------------
# loading and format detection
# ---------------------------------------------------------------------------

def load_raw(path: Path) -> list[dict]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise Fail(f"JSON does not parse: {e}")
    if not isinstance(data, list):
        raise Fail("top level must be a LIST (a dict breaks tune_threshold.py / "
                   "eval_responses.py, which iterate the file directly)")
    if not data:
        raise Fail("dataset is empty")
    return data


def is_v2(entries: list[dict]) -> bool:
    return any(isinstance(e, dict) and "question" in e and "slice" in e for e in entries)


def to_v2(entry: dict, index: int) -> dict:
    """A v1 record as a v2 record with slice "legacy" (shared by the eval
    scripts, so per-slice tables can mix both files)."""
    if "question" in entry and "slice" in entry:
        return {**{f: None for f in FIELDS}, **entry}
    return {
        "id": entry.get("id") or f"v1-{index:04d}",
        "slice": LEGACY_SLICE,
        "category": entry.get("category"),
        "source_db": entry.get("source_db"),
        "language": entry.get("lang") or entry.get("language") or "bn",
        "question": entry.get("query", ""),
        "previous_turn": None,
        "expected_row_keys": list(entry.get("relevant_ids") or []),
        "answerable": entry.get("answerable"),
        "variant_of": None,
        "review": "accepted",
        "notes": entry.get("reference", "") or "",
    }


def normalize_records(entries: list[dict]) -> list[dict]:
    return [to_v2(e, i) for i, e in enumerate(entries) if isinstance(e, dict)]


# ---------------------------------------------------------------------------
# schema rules
# ---------------------------------------------------------------------------

def _anaphora():
    from app.services.intent_classifier import ANAPHORA

    return ANAPHORA


def check_schema(entries: list[dict], v2: bool) -> list[str]:
    """Hard errors for the record shapes (empty = ok). v1 files get the v1 rules."""
    errors: list[str] = []
    if not v2:
        for i, q in enumerate(entries):
            tag = f"entry {i}"
            if not isinstance(q, dict):
                errors.append(f"{tag}: not an object")
                continue
            if not isinstance(q.get("query"), str) or not q["query"].strip():
                errors.append(f"{tag}: missing/blank 'query'")
            if not isinstance(q.get("answerable"), bool):
                errors.append(f"{tag}: 'answerable' must be a bool")
            ids = q.get("relevant_ids", [])
            if not isinstance(ids, list) or any(not isinstance(x, str) for x in ids):
                errors.append(f"{tag}: 'relevant_ids' must be a list of strings")
                ids = []
            errors += _key_rules(tag, q.get("answerable"), ids, q.get("query"))
        return errors

    seen_ids: Counter = Counter()
    by_id: dict[str, dict] = {}
    for i, q in enumerate(entries):
        tag = f"entry {i}"
        if not isinstance(q, dict):
            errors.append(f"{tag}: not an object")
            continue
        missing = [f for f in FIELDS if f not in q]
        if missing:
            errors.append(f"{tag} ({q.get('id')}): missing field(s) {missing}")
        if isinstance(q.get("id"), str) and q["id"]:
            seen_ids[q["id"]] += 1
            by_id[q["id"]] = q
    for dup, n in seen_ids.items():
        if n > 1:
            errors.append(f"duplicate id {dup!r} ({n} records)")

    anaphora = None
    for q in entries:
        if not isinstance(q, dict):
            continue
        tag = f"{q.get('id')!r}"
        slc = q.get("slice")
        if slc not in SLICES:
            errors.append(f"{tag}: slice {slc!r} not in {list(SLICES)}")
            continue
        if not isinstance(q.get("id"), str) or not q["id"].startswith(ID_PREFIX[slc] + "-"):
            errors.append(f"{tag}: id must start with {ID_PREFIX[slc]!r}- for slice {slc}")
        if not isinstance(q.get("question"), str) or not q["question"].strip():
            errors.append(f"{tag}: missing/blank 'question'")
        lang = q.get("language")
        if lang not in LANGUAGES:
            errors.append(f"{tag}: language {lang!r} not in {list(LANGUAGES)}")
        elif slc in SLICE_LANGUAGE and lang != SLICE_LANGUAGE[slc]:
            errors.append(f"{tag}: slice {slc} requires language {SLICE_LANGUAGE[slc]!r}, got {lang!r}")
        if q.get("answerable") is not SLICE_ANSWERABLE[slc]:
            errors.append(f"{tag}: slice {slc} requires answerable={SLICE_ANSWERABLE[slc]}")
        if q.get("review") not in REVIEWS:
            errors.append(f"{tag}: review {q.get('review')!r} not in {list(REVIEWS)}")
        if q.get("notes") is not None and not isinstance(q["notes"], str):
            errors.append(f"{tag}: notes must be a string")
        keys = q.get("expected_row_keys")
        if not isinstance(keys, list) or any(not isinstance(x, str) for x in keys):
            errors.append(f"{tag}: expected_row_keys must be a list of strings")
            keys = []
        errors += _key_rules(tag, q.get("answerable"), keys, q.get("question"), field="expected_row_keys")
        if SLICE_ANSWERABLE[slc]:
            if q.get("source_db") is not None and q["source_db"] not in SOURCE_DBS:
                errors.append(f"{tag}: source_db {q['source_db']!r} not in {list(SOURCE_DBS)}")
            if not isinstance(q.get("category"), str) or not q["category"]:
                errors.append(f"{tag}: answerable records need a category")
        else:
            if q.get("category") is not None or q.get("source_db") is not None:
                errors.append(f"{tag}: negatives must have category and source_db null")
        # follow-ups
        prev = q.get("previous_turn")
        if slc == "followup":
            if not isinstance(prev, str) or not prev.strip():
                errors.append(f"{tag}: follow-ups need a non-empty previous_turn")
            anaphora = anaphora or _anaphora()
            if isinstance(q.get("question"), str) and not anaphora.search(q["question"]):
                errors.append(f"{tag}: follow-up has no anaphor (ANAPHORA): {q['question']!r}")
        elif prev is not None:
            errors.append(f"{tag}: previous_turn must be null outside the followup slice")
        # variants
        parent_id = q.get("variant_of")
        if slc in VARIANT_SLICES:
            parent = by_id.get(parent_id) if isinstance(parent_id, str) else None
            if parent is None:
                errors.append(f"{tag}: variant_of must name an existing record (got {parent_id!r})")
            else:
                if parent.get("slice") != "answerable_bn":
                    errors.append(f"{tag}: variant_of {parent_id!r} is not an answerable_bn record")
                if set(parent.get("expected_row_keys") or []) != set(keys):
                    errors.append(f"{tag}: expected_row_keys differ from variant_of {parent_id!r}")
        elif parent_id is not None:
            errors.append(f"{tag}: variant_of must be null outside the banglish/en slices")
    return errors


def _key_rules(tag: str, answerable, keys: list[str], question, field: str = "relevant_ids") -> list[str]:
    errors = []
    if answerable is True and not keys:
        errors.append(f"{tag}: answerable=true but {field} is empty ({question!r})")
    if answerable is False and keys:
        errors.append(f"{tag}: answerable=false but {field} is non-empty ({question!r})")
    for rk in keys:
        if not ROW_KEY_RE.match(rk):
            errors.append(f"{tag}: '{rk}' is not a 'page_<int>'/'article_<int>' row_key")
    return errors


# ---------------------------------------------------------------------------
# corpus checks
# ---------------------------------------------------------------------------

def live_row_keys() -> set[str]:
    from app.rag.chroma_client import get_collection

    got = get_collection().get(include=["metadatas"])
    return {m["row_key"] for m in got["metadatas"] if m.get("row_key")}


def bad_page_ids() -> set[int]:
    bad: set[int] = set()
    with open(ROOT / "arabic_placeholder_pages.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            bad.add(int(row["page_id"]))
    with open(ROOT / "book203_cross_contamination.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            bad.add(int(row["loser_page_id"]))
    return bad


def check_ids(records: list[dict], *, keys: set[str] | None, bad: set[int] | None = None) -> tuple[list[str], int]:
    """`keys` = the live collection's row keys (None = skip the existence check)."""
    gold = sorted({rk for q in records for rk in q.get("expected_row_keys", [])})
    bad = bad_page_ids() if bad is None else bad
    errors: list[str] = []
    if keys is not None:
        missing = [rk for rk in gold if rk not in keys]
        if missing:
            errors.append(
                f"{len(missing)} expected id(s) NOT in the live Chroma collection "
                f"(settings.chroma_collection_name): {', '.join(missing)}"
            )
    hit_bad = [rk for rk in gold if rk.startswith("page_") and int(rk.split("_")[1]) in bad]
    if hit_bad:
        errors.append(
            f"{len(hit_bad)} expected id(s) are KNOWN-BAD pages (in the cleanup "
            f"CSVs) and must not be used: {', '.join(hit_bad)}"
        )
    return errors, len(gold)


# ---------------------------------------------------------------------------
# near-duplicate questions
# ---------------------------------------------------------------------------

def _tokens(s: str) -> set[str]:
    return set(re.findall(r"\w+", s.lower()))


def check_near_dups(records: list[dict]) -> list[str]:
    """Exact and near-duplicate questions within one language (a Banglish
    variant of a Bengali question is a different string by design)."""
    errors: list[str] = []
    toks = [(_tokens(q["question"]), q["question"], q.get("language")) for q in records
            if isinstance(q.get("question"), str)]
    for i in range(len(toks)):
        a, qa, la = toks[i]
        for j in range(i + 1, len(toks)):
            b, qb, lb = toks[j]
            if not a or not b or la != lb:
                continue
            if qa.strip() == qb.strip():
                errors.append(f"exact duplicate question: {qa!r}")
                continue
            jac = len(a & b) / len(a | b)
            if jac >= NEAR_DUP_JACCARD:
                errors.append(f"near-duplicate (jaccard={jac:.2f}):\n    {qa!r}\n    {qb!r}")
    return errors


# ---------------------------------------------------------------------------
# downstream-compat check
# ---------------------------------------------------------------------------

def check_downstream(path: Path) -> list[str]:
    """The eval scripts must still be able to read the file."""
    try:
        from scripts.eval_retrieval import _load_questions
    except Exception as e:  # pragma: no cover
        return [f"could not import scripts.eval_retrieval._load_questions: {e}"]
    try:
        _load_questions(path)
    except Exception as e:
        return [f"eval_retrieval._load_questions rejected the file: {e}"]
    return []


def check_downstream_v2(records: list[dict]) -> list[str]:
    """v2 files reach the eval scripts through normalize_records(); make sure
    every record carries what they consume."""
    errors = []
    for q in records:
        if not isinstance(q.get("question"), str) or not isinstance(q.get("answerable"), bool):
            errors.append(f"{q.get('id')!r}: not consumable by the eval scripts (question/answerable)")
    return errors


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------

def slice_report(records: list[dict]) -> list[tuple[str, int, float, float]]:
    """(slice, count, observed share, target share) for every target slice."""
    n = len(records) or 1
    counts = Counter(q.get("slice") for q in records)
    return [(s, counts.get(s, 0), counts.get(s, 0) / n, SLICE_TARGETS[s]) for s in SLICES]


def summary(records: list[dict], n_gold: int, v2: bool) -> None:
    n = len(records)
    print("\n" + "=" * 70)
    print(f"SUMMARY  ({n} records, {n_gold} distinct expected ids, format {'v2' if v2 else 'v1'})")
    print("=" * 70)
    ans = sum(1 for q in records if q["answerable"])
    print(f"  answerable    : {ans:>3}  ({ans/n:.0%})")
    print(f"  unanswerable  : {n-ans:>3}  ({(n-ans)/n:.0%})")

    print("  by slice (target share for a 250-question set):")
    for s, c, share, target in slice_report(records):
        drift = "  ! off target" if abs(share - target) > 0.05 else ""
        print(f"    {s:<22}: {c:>3}  ({share:.0%}, target {target:.0%}){drift}")
    legacy = sum(1 for q in records if q.get("slice") == LEGACY_SLICE)
    if legacy:
        print(f"    {LEGACY_SLICE:<22}: {legacy:>3}  (v1 records, no slice)")

    print("  by language:")
    for k, c in Counter(q.get("language") for q in records).most_common():
        print(f"    {str(k):<9}: {c:>3}")

    answerable = [q for q in records if q["answerable"]]
    if answerable:
        print("  answerable by category:")
        for k, c in Counter(q.get("category") for q in answerable).most_common():
            print(f"    {str(k):<30}: {c:>3}")
        print("  answerable by source_db:")
        for k, c in Counter(q.get("source_db") for q in answerable).most_common():
            print(f"    {str(k):<20}: {c:>3}")
    if v2:
        print("  by review:")
        for k, c in Counter(q.get("review") for q in records).most_common():
            print(f"    {str(k):<9}: {c:>3}")
    print(f"\n  id-existence: {n_gold} expected ids checked.  OK")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def validate(path: Path, *, use_chroma: bool = True, accepted_only: bool = False) -> tuple[list[str], list[dict], bool]:
    """(hard errors, normalized records, is_v2). Raises Fail when the file
    cannot be read at all."""
    entries = load_raw(path)
    v2 = is_v2(entries)
    hard = check_schema(entries, v2)
    if hard and any(k in e for e in hard for k in ("not an object", "missing field", "'query'", "'answerable'",
                                                   "expected_row_keys must", "'relevant_ids'")):
        return hard, [], v2
    records = normalize_records(entries)
    if accepted_only:
        records = [q for q in records if q.get("review") == "accepted"]
    id_errors, n_gold = check_ids(records, keys=live_row_keys() if use_chroma else None)
    hard += id_errors
    hard += check_near_dups(records)
    hard += check_downstream(path) if not v2 else check_downstream_v2(records)
    validate.n_gold = n_gold  # type: ignore[attr-defined]
    return hard, records, v2


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m scripts.validate_eval_set")
    ap.add_argument("path", nargs="?", type=Path, default=DEFAULT_FILE)
    ap.add_argument("--no-chroma", action="store_true", help="skip the live row_key existence check")
    ap.add_argument("--accepted-only", action="store_true", help="summarise review=accepted records only")
    args = ap.parse_args(argv)
    print(f"validating {args.path}")
    try:
        hard, records, v2 = validate(args.path, use_chroma=not args.no_chroma, accepted_only=args.accepted_only)
    except Fail as e:
        print(f"\nFAIL: {e}")
        return 1
    if hard:
        print("\n" + "!" * 70)
        print(f"VALIDATION FAILED  ({len(hard)} error(s))")
        print("!" * 70)
        for e in hard:
            print(f"  - {e}")
        return 1
    print(f"\nall hard checks passed ({len(records)} records).")
    summary(records, getattr(validate, "n_gold", 0), v2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
