"""Validate scripts/retrieval_eval_questions.json before it is trusted.

Checks, in order, and FAILS LOUDLY (exit 1) on any hard error:

  1. JSON parses and is a flat list (the shape tune_threshold.py /
     eval_responses.py iterate).
  2. Every entry has the right schema: query:str, answerable:bool,
     relevant_ids:list[str]; answerable<->relevant_ids consistency.
  3. Every relevant_id is a well-formed row_key ('page_<int>' / 'article_<int>')
     AND resolves to a real chunk in the LIVE Chroma collection the eval
     harness queries (settings.chroma_collection_name). Any id not found is
     listed and the script exits 1 -- this is the check that matters most.
  4. No gold id is a KNOWN-BAD page (arabic_placeholder_pages.csv,
     book203_cross_contamination.csv, orphaned_source_pages.csv).
  5. No duplicate or near-duplicate queries (normalised token Jaccard).
  6. The file still loads cleanly through eval_retrieval._load_questions.

Then it PRINTS a summary (counts by answerable / language / difficulty,
answerable & language ratios vs the dataset's stated targets, and the
id-existence result). Ratio drift is a warning, not a failure.

Usage:
    python -m scripts.validate_eval_set [path/to/questions.json]

Run from a shell with the app's deps + .env (same as `python -m app.rag.ingest`).
"""

from __future__ import annotations

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

# Rough targets, only used for the drift warning in the summary.
TARGET_ANSWERABLE = (0.60, 0.75)
TARGET_LANG = {"bn": 0.55, "banglish": 0.35, "en": 0.10}


class Fail(Exception):
    pass


# ---------------------------------------------------------------------------
# loading + schema
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


def check_schema(entries: list[dict]) -> list[str]:
    """Return the list of hard errors (empty = ok)."""
    errors: list[str] = []
    for i, q in enumerate(entries):
        tag = f"entry {i}"
        if not isinstance(q, dict):
            errors.append(f"{tag}: not an object")
            continue
        if "query" not in q or not isinstance(q["query"], str) or not q["query"].strip():
            errors.append(f"{tag}: missing/blank 'query'")
        if "answerable" not in q or not isinstance(q["answerable"], bool):
            errors.append(f"{tag}: 'answerable' must be a bool")
        ids = q.get("relevant_ids", [])
        if not isinstance(ids, list) or any(not isinstance(x, str) for x in ids):
            errors.append(f"{tag}: 'relevant_ids' must be a list of strings")
            ids = []
        if q.get("answerable") is True and not ids:
            errors.append(f"{tag}: answerable=true but relevant_ids is empty "
                          f"({q.get('query')!r})")
        if q.get("answerable") is False and ids:
            errors.append(f"{tag}: answerable=false but relevant_ids is non-empty "
                          f"({q.get('query')!r})")
        for rk in ids:
            if not ROW_KEY_RE.match(rk):
                errors.append(f"{tag}: '{rk}' is not a 'page_<int>'/'article_<int>' "
                              "row_key")
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


def check_ids(entries: list[dict]) -> tuple[list[str], int]:
    gold = sorted({rk for q in entries for rk in q.get("relevant_ids", [])})
    keys = live_row_keys()
    bad = bad_page_ids()

    errors: list[str] = []
    missing = [rk for rk in gold if rk not in keys]
    if missing:
        errors.append(
            f"{len(missing)} gold id(s) NOT in the live Chroma collection "
            f"(settings.chroma_collection_name): {', '.join(missing)}"
        )
    hit_bad = [rk for rk in gold
               if rk.startswith("page_") and int(rk.split("_")[1]) in bad]
    if hit_bad:
        errors.append(
            f"{len(hit_bad)} gold id(s) are KNOWN-BAD pages (in the cleanup "
            f"CSVs) and must not be used: {', '.join(hit_bad)}"
        )
    return errors, len(gold)


# ---------------------------------------------------------------------------
# near-duplicate queries
# ---------------------------------------------------------------------------

def _tokens(s: str) -> set[str]:
    return set(re.findall(r"\w+", s.lower()))


def check_near_dups(entries: list[dict]) -> list[str]:
    errors: list[str] = []
    toks = [(_tokens(q["query"]), q["query"]) for q in entries]
    for i in range(len(toks)):
        a, qa = toks[i]
        for j in range(i + 1, len(toks)):
            b, qb = toks[j]
            if not a or not b:
                continue
            if qa.strip() == qb.strip():
                errors.append(f"exact duplicate query: {qa!r}")
                continue
            jac = len(a & b) / len(a | b)
            if jac >= NEAR_DUP_JACCARD:
                errors.append(f"near-duplicate (jaccard={jac:.2f}):\n"
                              f"    {qa!r}\n    {qb!r}")
    return errors


# ---------------------------------------------------------------------------
# downstream-compat check
# ---------------------------------------------------------------------------

def check_downstream(path: Path) -> list[str]:
    try:
        from scripts.eval_retrieval import _load_questions
    except Exception as e:  # pragma: no cover
        return [f"could not import scripts.eval_retrieval._load_questions: {e}"]
    try:
        _load_questions(path)
    except Exception as e:
        return [f"eval_retrieval._load_questions rejected the file: {e}"]
    return []


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------

def summary(entries: list[dict], n_gold: int) -> None:
    n = len(entries)
    ans = [q for q in entries if q["answerable"]]
    unans = [q for q in entries if not q["answerable"]]
    lang = Counter(q.get("lang", "?") for q in entries)
    diff = Counter(q.get("difficulty", "?") for q in entries)

    print("\n" + "=" * 70)
    print(f"SUMMARY  ({n} entries, {n_gold} distinct gold ids)")
    print("=" * 70)
    print(f"  answerable    : {len(ans):>3}  ({len(ans)/n:.0%})")
    print(f"  unanswerable  : {len(unans):>3}  ({len(unans)/n:.0%})")
    lo, hi = TARGET_ANSWERABLE
    if not lo <= len(ans) / n <= hi:
        print(f"    ! answerable ratio outside target {lo:.0%}-{hi:.0%}")

    print("  by language:")
    for k in ("bn", "banglish", "en"):
        c = lang.get(k, 0)
        want = TARGET_LANG[k]
        drift = "  ! off target" if abs(c / n - want) > 0.1 else ""
        print(f"    {k:<9}: {c:>3}  ({c/n:.0%}, target {want:.0%}){drift}")
    for k in lang:
        if k not in TARGET_LANG:
            print(f"    {k:<9}: {lang[k]:>3}   (unexpected lang tag)")

    print("  by difficulty:")
    for k, c in diff.most_common():
        print(f"    {k:<10}: {c:>3}")

    print("  by difficulty x answerable:")
    da = Counter((q.get("difficulty", "?"), q["answerable"]) for q in entries)
    for (d, a), c in sorted(da.items()):
        print(f"    {d:<10} {'answerable' if a else 'unanswerable':<12}: {c:>3}")

    print(f"\n  id-existence: all {n_gold} gold ids resolve in the live "
          "collection.  OK")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_FILE
    print(f"validating {path}")

    hard: list[str] = []
    try:
        entries = load_raw(path)
    except Fail as e:
        print(f"\nFAIL: {e}")
        sys.exit(1)

    # _README rides on the first real entry; it is a normal entry otherwise.
    hard += check_schema(entries)
    if any("row_key" in e or "relevant_ids" in e or "'query'" in e or
           "'answerable'" in e for e in hard):
        # schema is too broken to run the corpus checks meaningfully
        _report_and_exit(hard)

    id_errors, n_gold = check_ids(entries)
    hard += id_errors
    hard += check_near_dups(entries)
    hard += check_downstream(path)

    if hard:
        _report_and_exit(hard)

    print(f"\nall hard checks passed ({len(entries)} entries).")
    summary(entries, n_gold)


def _report_and_exit(errors: list[str]) -> None:
    print("\n" + "!" * 70)
    print(f"VALIDATION FAILED  ({len(errors)} error(s))")
    print("!" * 70)
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)


if __name__ == "__main__":
    main()
