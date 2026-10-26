"""Follow-up slice: does condensation + carry-over find the right page?

    python -m scripts.eval_followups [scripts/eval/test.json] [--k 1,3,5,10]
                                     [--report out.json]

For every `followup` record (schema: scripts/eval/schema.md) the first-turn
question (previous_turn) is run through retrieve_stages() to build a
realistic history -- the first retrieved excerpt stands in for the
assistant's answer text, and the turn's top pages are its cited sources --
then the follow-up is retrieved two ways through the SAME retrieve_stages():

  without  -- the follow-up text as typed, no history (what the pipeline did
              before roadmap §1);
  with     -- classify_intent(follow-up, history) primes the standalone
              question from the intent call, condense() reads it, and the
              previous turn's row_keys are carried into the rerank pool
              (extra_row_keys), exactly as qa_service._prepare() does.

Reports recall@k / MRR of the expected row_keys for both, per k, plus the
delta, and lists every follow-up whose standalone query was the raw message
(skip rule or fallback). No answers are generated; retrieval is not
duplicated here. Needs the conversational-QA pipeline: exits with a message
if qa_service.answer_question has no `history` parameter.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from datetime import datetime
from pathlib import Path

from app.core.config import settings

DEFAULT_FILE = Path(__file__).parent / "eval" / "test.json"
REPORT_DIR = Path(__file__).parent.parent / "eval_reports"
DEFAULT_KS = (1, 3, 5, 10)
ASSISTANT_CHARS = 600


def pipeline_has_history() -> bool:
    from app.services import qa_service

    return "history" in inspect.signature(qa_service.answer_question).parameters


def load_followups(path: Path) -> list[dict]:
    from scripts.validate_eval_set import normalize_records

    records = normalize_records(json.loads(path.read_text(encoding="utf-8")))
    return [r for r in records if r.get("slice") == "followup" and r.get("previous_turn")]


def _pages(chunks) -> list[str]:
    seen, out = set(), []
    for c in chunks:
        if c.row_key not in seen:
            seen.add(c.row_key)
            out.append(c.row_key)
    return out


def _score(ranked: list[str], gold: list[str], k: int) -> dict[str, float]:
    from scripts.eval_retrieval import _score_one

    return _score_one(ranked, gold, k)


def build_history(previous_turn: str, max_k: int) -> tuple[list[dict], list[str]]:
    """Run turn one through retrieval and shape it as the stored history."""
    from app.rag.retriever import retrieve_stages

    _, final = retrieve_stages(previous_turn, rerank_top_n=max(max_k, settings.top_k))
    production = final[: settings.top_k]
    sources = [{"book": c.book, "chapter": c.chapter, "row_key": c.row_key, "content": c.content}
               for c in production]
    answer_proxy = production[0].content[:ASSISTANT_CHARS] if production else "বইগুলোতে এ বিষয়ে তথ্য পাওয়া যায়নি।"
    history = [{"role": "user", "content": previous_turn},
               {"role": "assistant", "content": answer_proxy, "sources": sources}]
    return history, _pages(production)


def run_one(record: dict, ks: list[int]) -> dict:
    from app.rag.retriever import retrieve_stages
    from app.services.intent_classifier import classify_intent, condense
    from app.services.qa_service import previous_row_keys

    gold = record["expected_row_keys"]
    max_k = max(ks)
    history, prev_pages = build_history(record["previous_turn"], max_k)
    followup = record["question"]

    _, final_without = retrieve_stages(followup, rerank_top_n=max(max_k, settings.top_k))
    without = _pages(final_without)

    classify_intent(followup, False, history)  # primes the standalone query (one intent call)
    standalone = condense(followup, history)
    extra = previous_row_keys(history)
    _, final_with = retrieve_stages(
        standalone, rerank_top_n=max(max_k, settings.top_k),
        **({"extra_row_keys": extra} if extra else {}),
    )
    with_ = _pages(final_with)
    return {
        "id": record.get("id"), "previous_turn": record["previous_turn"], "question": followup,
        "standalone_query": standalone, "condensed": standalone != followup,
        "previous_pages": prev_pages, "expected_row_keys": gold,
        "without": {"pages": without, "metrics": {k: _score(without, gold, k) for k in ks}},
        "with": {"pages": with_, "metrics": {k: _score(with_, gold, k) for k in ks}},
    }


def aggregate(results: list[dict], ks: list[int]) -> dict:
    out = {}
    for arm in ("without", "with"):
        out[arm] = {k: {m: (sum(r[arm]["metrics"][k][m] for r in results) / len(results) if results else 0.0)
                        for m in ("recall", "mrr", "hit_rate")} for k in ks}
    return out


def print_report(results: list[dict], agg: dict, ks: list[int]) -> None:
    print(f"\nFOLLOW-UP SLICE  n={len(results)}  (final stage, distinct pages)")
    print(f"  {'k':>4}  {'recall w/o':>10} {'recall with':>11} {'delta':>7}  {'MRR w/o':>8} {'MRR with':>8}")
    for k in ks:
        a, b = agg["without"][k], agg["with"][k]
        print(f"  {k:>4}  {a['recall']:>10.3f} {b['recall']:>11.3f} {b['recall']-a['recall']:>+7.3f}  "
              f"{a['mrr']:>8.3f} {b['mrr']:>8.3f}")
    raw = [r for r in results if not r["condensed"]]
    print(f"\n  standalone == raw follow-up (skip rule or fallback): {len(raw)}/{len(results)}")
    for r in raw:
        print(f"    {r['id']}: {r['question']!r}")
    print("\n  per follow-up (hit@top_k without -> with):")
    for r in results:
        k = settings.top_k if settings.top_k in ks else ks[0]
        a = "HIT " if r["without"]["metrics"][k]["hit_rate"] else "MISS"
        b = "HIT " if r["with"]["metrics"][k]["hit_rate"] else "MISS"
        print(f"    [{a} -> {b}] {r['id']}  {r['question']!r}  ->  {r['standalone_query']!r}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m scripts.eval_followups")
    ap.add_argument("questions", nargs="?", type=Path, default=DEFAULT_FILE)
    ap.add_argument("--k", default=",".join(map(str, DEFAULT_KS)))
    ap.add_argument("--report", type=Path)
    args = ap.parse_args(argv)

    if not pipeline_has_history():
        print("qa_service.answer_question has no `history` parameter: the conversational-QA "
              "pipeline (feat/conversational-qa, roadmap §1) is not on this branch. Nothing to measure.")
        return 2
    ks = sorted({int(x) for x in args.k.split(",") if x.strip()})
    followups = load_followups(args.questions)
    if not followups:
        print(f"no followup records in {args.questions}")
        return 1

    results = [run_one(r, ks) for r in followups]
    agg = aggregate(results, ks)
    print_report(results, agg, ks)

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": str(args.questions), "k": ks, "n": len(results),
        "config": {"top_k": settings.top_k, "fetch_k": settings.fetch_k,
                   "min_rerank_score": settings.min_rerank_score,
                   "qa_history_turns": settings.qa_history_turns,
                   "condense_min_words": settings.condense_min_words},
        "aggregate": {arm: {str(k): m for k, m in per.items()} for arm, per in agg.items()},
        "per_followup": [{**r, "without": {**r["without"], "metrics": {str(k): m for k, m in r["without"]["metrics"].items()}},
                          "with": {**r["with"], "metrics": {str(k): m for k, m in r["with"]["metrics"].items()}}}
                         for r in results],
    }
    path = args.report or (REPORT_DIR / f"followups_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nReport written to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
