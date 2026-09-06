"""Suggest a MIN_RERANK_SCORE by testing retrieval against known questions.

Usage:
    python -m scripts.tune_threshold [path/to/questions.json] [--report out.json]
                                     [--no-rewrite]

Input file: a JSON list of {"query": str, "answerable": bool}, optionally with
"relevant_ids": [row_key, ...] (the labeled retrieval eval set,
scripts/retrieval_eval_questions.json). See scripts/eval_questions.example.json
for the minimal shape. Extra keys (_README, lang, difficulty) are ignored.

WHAT THIS THRESHOLD DOES (app/services/qa_service.py):
    relevant = citations[0].rerank_score >= settings.min_rerank_score
The top reranked chunk's score is compared to the gate. Below it, the bot
answers grounded in NOTHING (empty sources); at/above, it answers grounded in
the retrieved sources. So the gate is a refusal decision with two error modes:

  FALSE REFUSAL  -- an answerable question scored below the bar. User wrongly
                    told "not found" when the corpus had the answer.
  FALSE ANSWER   -- an unanswerable question scored above the bar. Bot answers
                    from weak / irrelevant context -> hallucination risk. For a
                    religious-content bot this is the more dangerous error, so
                    the "conservative" policy below drives it to zero first.

For each question it retrieves the top candidate via the real pipeline (query
rewrite + bi-encoder + cross-encoder rerank -- the same path /chat uses) and
records the top rerank_score AND the top page's row_key. It then sweeps every
candidate threshold and reports the error split, precision/recall/F1/accuracy,
and -- when relevant_ids are present -- how many "answered" answerable queries
were grounded in the RIGHT page vs a wrong one (a grounded-but-wrong answer is
not a win even though it clears the gate).

Scores are the sigmoid-activated cross-encoder outputs in [0, 1] (see the note
on min_rerank_score in app/core/config.py). Determinism: the only
non-deterministic step is the query-rewrite LLM call; pass --no-rewrite to sweep
the raw-query path for a fully reproducible run.

Run this from a shell where the app's dependencies and .env are available
(same environment as `python -m app.rag.ingest`).
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from app.core.config import settings
from app.rag.retriever import retrieve_stages

# Distributions whose middles sit closer than this are called "overlapping":
# there is then no clean separating threshold and the fix is upstream, not here.
OVERLAP_MEDIAN_GAP = 0.15


# ---------------------------------------------------------------------------
# input
# ---------------------------------------------------------------------------

def _load_questions(path: Path) -> list[dict]:
    """Read the dataset, tolerating the _README-on-first-entry convention and
    entries with or without relevant_ids."""
    with open(path, encoding="utf-8") as f:
        questions = json.load(f)
    if isinstance(questions, dict):  # tolerated: {"questions": [...]}
        questions = questions["questions"]

    out: list[dict] = []
    for q in questions:
        if "query" not in q or "answerable" not in q:
            raise ValueError(f"bad entry, expected query/answerable keys: {q}")
        ids = q.get("relevant_ids") or []
        if not isinstance(ids, list) or any(not isinstance(i, str) for i in ids):
            raise ValueError(f"relevant_ids must be a list of row_key strings: {q}")
        out.append({"query": q["query"], "answerable": bool(q["answerable"]),
                    "relevant_ids": ids})
    return out


# ---------------------------------------------------------------------------
# retrieval
# ---------------------------------------------------------------------------

def _top_result(query: str, use_rewrite: bool) -> tuple[float | None, str | None]:
    """Return (top_rerank_score, top_row_key) for the production top-1.

    This is exactly what qa_service gates on: retrieve_stages' reranked list,
    best first; qa_service compares its [0].rerank_score to min_rerank_score.
    """
    _candidates, final = retrieve_stages(query, use_rewrite=use_rewrite)
    if not final:
        return None, None
    top = final[0]
    return top.rerank_score, top.row_key


def _gather(questions: list[dict], use_rewrite: bool) -> list[dict]:
    records = []
    for q in questions:
        score, row_key = _top_result(q["query"], use_rewrite)
        gold = set(q["relevant_ids"])
        # Only answerable+labeled queries can be judged "grounded in the right
        # page". For those, was the top page (the one whose score is gated) gold?
        top_is_gold = (row_key in gold) if (gold and row_key) else None
        records.append({
            "query": q["query"],
            "answerable": q["answerable"],
            "labeled": bool(gold),
            "top_score": score,
            "top_row_key": row_key,
            "top_is_gold": top_is_gold,
        })
        state = "answerable" if q["answerable"] else "unanswerable"
        gold_tag = ""
        if q["answerable"] and gold:
            gold_tag = f" top_is_gold={top_is_gold}"
        s = "None" if score is None else f"{score:.4f}"
        print(f"[{state:>12}] top_rerank={s} page={row_key}{gold_tag}  {q['query']!r}")
    return records


# ---------------------------------------------------------------------------
# distributions
# ---------------------------------------------------------------------------

def _quantiles(scores: list[float]) -> dict:
    s = sorted(scores)
    n = len(s)

    def q(p: float) -> float:
        if n == 1:
            return s[0]
        idx = p * (n - 1)
        lo = int(idx)
        frac = idx - lo
        hi = min(lo + 1, n - 1)
        return s[lo] + frac * (s[hi] - s[lo])

    return {"min": s[0], "p25": q(0.25), "median": q(0.50),
            "p75": q(0.75), "max": s[-1]}


def _print_distribution(label: str, records: list[dict], answerable: bool) -> dict:
    scores = [r["top_score"] for r in records
              if r["answerable"] == answerable and r["top_score"] is not None]
    n_none = sum(1 for r in records
                 if r["answerable"] == answerable and r["top_score"] is None)
    total = sum(1 for r in records if r["answerable"] == answerable)
    print(f"\n{label} (n={total}, {n_none} retrieved nothing):")
    if not scores:
        print("  no scored results")
        return {"n": total, "n_none": n_none, "quantiles": None, "scores": []}
    qs = _quantiles(scores)
    print(f"  min={qs['min']:.4f}  p25={qs['p25']:.4f}  median={qs['median']:.4f}"
          f"  p75={qs['p75']:.4f}  max={qs['max']:.4f}")
    print("  scores:", ", ".join(f"{s:.4f}" for s in sorted(scores)))
    return {"n": total, "n_none": n_none, "quantiles": qs,
            "scores": sorted(scores)}


# ---------------------------------------------------------------------------
# threshold sweep
# ---------------------------------------------------------------------------

def _candidate_thresholds(all_scores: list[float]) -> list[float]:
    present = sorted(set(all_scores))
    if not present:
        return [0.0]
    midpoints = [(a + b) / 2 for a, b in zip(present, present[1:])]
    # also test just above the max and just below the min so "reject everything"
    # and "accept everything" are both reachable.
    return [round(present[0] - 0.01, 6), *[round(m, 6) for m in midpoints],
            round(present[-1] + 0.01, 6)]


def _metrics_at(threshold: float, records: list[dict]) -> dict:
    """Confusion matrix + derived metrics at one threshold.

    A query "clears the gate" when top_score is not None and >= threshold.
    Positive class = answerable (should be answered).
    """
    ans = [r for r in records if r["answerable"]]
    una = [r for r in records if not r["answerable"]]

    def cleared(r) -> bool:
        return r["top_score"] is not None and r["top_score"] >= threshold

    tp = [r for r in ans if cleared(r)]          # answerable, answered
    fn = [r for r in ans if not cleared(r)]      # answerable, refused = false refusal
    fp = [r for r in una if cleared(r)]          # unanswerable, answered = false answer
    tn = [r for r in una if not cleared(r)]      # unanswerable, refused

    n_tp, n_fn, n_fp, n_tn = len(tp), len(fn), len(fp), len(tn)
    total = n_tp + n_fn + n_fp + n_tn
    precision = n_tp / (n_tp + n_fp) if (n_tp + n_fp) else 0.0
    recall = n_tp / (n_tp + n_fn) if (n_tp + n_fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    accuracy = (n_tp + n_tn) / total if total else 0.0

    # Grounding-aware split of the answered answerable queries (labeled only):
    # cleared the gate but on the WRONG page -> a grounded-but-wrong answer.
    labeled_cleared = [r for r in tp if r["top_is_gold"] is not None]
    misgrounded = [r for r in labeled_cleared if r["top_is_gold"] is False]

    return {
        "threshold": round(threshold, 6),
        "false_refusals": n_fn,
        "false_answers": n_fp,
        "true_answers": n_tp,
        "true_refusals": n_tn,
        "misgrounded_answers": len(misgrounded),
        "n_labeled_cleared": len(labeled_cleared),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "total_errors": n_fn + n_fp,
    }


def _print_table(rows: list[dict], has_labels: bool) -> None:
    print("\nPER-THRESHOLD SWEEP")
    hdr = (f"  {'thresh':>7}  {'FalseRef':>8}  {'FalseAns':>8}  {'prec':>6}  "
           f"{'recall':>6}  {'F1':>6}  {'acc':>6}  {'errors':>6}")
    if has_labels:
        hdr += f"  {'misgnd':>6}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for m in rows:
        line = (f"  {m['threshold']:>7.4f}  {m['false_refusals']:>8}  "
                f"{m['false_answers']:>8}  {m['precision']:>6.3f}  "
                f"{m['recall']:>6.3f}  {m['f1']:>6.3f}  {m['accuracy']:>6.3f}  "
                f"{m['total_errors']:>6}")
        if has_labels:
            line += f"  {m['misgrounded_answers']:>6}"
        print(line)
    if has_labels:
        print("\n  misgnd = answerable queries that cleared the gate but whose "
              "top page was NOT the gold page\n  (a grounded-but-wrong answer -- "
              "not counted as an error here, but not a real win either).")


# ---------------------------------------------------------------------------
# recommendations
# ---------------------------------------------------------------------------

def _pick_balanced(rows: list[dict]) -> dict:
    # Max F1; ties broken by fewer total errors, then lower threshold (answers more).
    return max(rows, key=lambda m: (m["f1"], -m["total_errors"], -m["threshold"]))


def _pick_conservative(rows: list[dict]) -> dict:
    # Cap false answers at zero, then minimize false refusals; if zero-FA is
    # impossible, fall back to the minimum achievable false-answer count.
    zero_fa = [m for m in rows if m["false_answers"] == 0]
    pool = zero_fa or _min_fa_pool(rows)
    # fewest refusals, then lowest threshold that achieves it (answers the most).
    return min(pool, key=lambda m: (m["false_refusals"], m["threshold"]))


def _min_fa_pool(rows: list[dict]) -> list[dict]:
    fa_min = min(m["false_answers"] for m in rows)
    return [m for m in rows if m["false_answers"] == fa_min]


# ---------------------------------------------------------------------------
# overlap guardrail
# ---------------------------------------------------------------------------

def _overlap_report(ans_dist: dict, una_dist: dict) -> dict:
    """Is there a clean separating threshold? Reports the separation and, if the
    groups overlap, how many of each land in the shared band."""
    aq, uq = ans_dist["quantiles"], una_dist["quantiles"]
    if not aq or not uq:
        return {"separable": None, "reason": "one group has no scored results"}

    # A perfectly clean split exists iff min(answerable) > max(unanswerable).
    clean = aq["min"] > uq["max"]
    median_gap = aq["median"] - uq["median"]

    ans_scores = ans_dist["scores"]
    una_scores = una_dist["scores"]
    # Overlap band = [max(mins), min(maxes)] where both groups can appear.
    lo = max(aq["min"], uq["min"])
    hi = min(aq["max"], uq["max"])
    ans_in_band = sum(1 for s in ans_scores if lo <= s <= hi) if hi >= lo else 0
    una_in_band = sum(1 for s in una_scores if lo <= s <= hi) if hi >= lo else 0

    heavy = (not clean) and (median_gap < OVERLAP_MEDIAN_GAP)
    return {
        "clean_separation": clean,
        "answerable_min": round(aq["min"], 4),
        "unanswerable_max": round(uq["max"], 4),
        "median_gap": round(median_gap, 4),
        "overlap_band": [round(lo, 4), round(hi, 4)] if hi >= lo else None,
        "answerable_in_band": ans_in_band,
        "unanswerable_in_band": una_in_band,
        "heavy_overlap": heavy,
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m scripts.tune_threshold",
                                description="Tune MIN_RERANK_SCORE from a labeled set.")
    p.add_argument("questions", nargs="?",
                   default=str(Path(__file__).parent / "eval_questions.example.json"),
                   help="dataset JSON (default: the example file)")
    p.add_argument("--report", metavar="OUT.JSON",
                   help="dump per-threshold metrics + recommendations as JSON")
    p.add_argument("--no-rewrite", action="store_true",
                   help="sweep the raw-query path (fully deterministic)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    path = Path(args.questions)
    questions = _load_questions(path)
    use_rewrite = not args.no_rewrite

    n_ans = sum(1 for q in questions if q["answerable"])
    n_una = len(questions) - n_ans
    n_labeled = sum(1 for q in questions if q["answerable"] and q["relevant_ids"])
    has_labels = n_labeled > 0

    print("=" * 78)
    print(f"TUNING MIN_RERANK_SCORE  (current: {settings.min_rerank_score})")
    print(f"dataset: {path}")
    print(f"  {len(questions)} questions: {n_ans} answerable, {n_una} unanswerable"
          f"  |  rewrite={'ON' if use_rewrite else 'OFF'}")
    print(f"  answerable questions with gold labels: {n_labeled}")
    print("=" * 78)

    records = _gather(questions, use_rewrite)

    ans_dist = _print_distribution("Answerable questions", records, answerable=True)
    una_dist = _print_distribution("Unanswerable questions", records, answerable=False)

    ans_scored = [r for r in records if r["answerable"] and r["top_score"] is not None]
    una_scored = [r for r in records if not r["answerable"] and r["top_score"] is not None]
    if not ans_scored or not una_scored:
        print("\nNeed at least one scored answerable AND one scored unanswerable "
              "question to suggest a threshold.")
        return

    overlap = _overlap_report(ans_dist, una_dist)

    all_scores = [r["top_score"] for r in records if r["top_score"] is not None]
    thresholds = _candidate_thresholds(all_scores)
    rows = [_metrics_at(t, records) for t in thresholds]

    _print_table(rows, has_labels)

    balanced = _pick_balanced(rows)
    conservative = _pick_conservative(rows)

    # --- overlap guardrail ---
    print("\n" + "=" * 78)
    print("DISTRIBUTION SEPARATION")
    print("=" * 78)
    if overlap.get("clean_separation"):
        print(f"  CLEAN: every answerable score ({overlap['answerable_min']}) is above "
              f"every unanswerable score ({overlap['unanswerable_max']}).")
        print("  A threshold in that gap separates the two groups perfectly.")
    else:
        print(f"  answerable min      = {overlap['answerable_min']}")
        print(f"  unanswerable max    = {overlap['unanswerable_max']}")
        print(f"  median gap          = {overlap['median_gap']}")
        print(f"  overlap band        = {overlap['overlap_band']}  "
              f"(answerable in band: {overlap['answerable_in_band']}, "
              f"unanswerable in band: {overlap['unanswerable_in_band']})")
        if overlap["heavy_overlap"]:
            print("\n  ** HEAVY OVERLAP: the groups are not cleanly separable by any single")
            print("     threshold (median gap < "
                  f"{OVERLAP_MEDIAN_GAP}). No gate value fixes this -- the leak is")
            print("     upstream (retrieval / rerank / corpus). Treat the numbers below as")
            print("     the best available trade-off, not a clean operating point.")
        else:
            print("\n  Groups overlap at the edges but are mostly separated -- a threshold")
            print("  in the band below trades a few errors either way.")

    # --- recommendations ---
    print("\n" + "=" * 78)
    print("RECOMMENDATIONS  (eval set: "
          f"{n_ans} answerable / {n_una} unanswerable"
          + (f", {n_labeled} labeled)" if has_labels else ")"))
    print("=" * 78)
    print(f"\n  BALANCED (max F1)         = {balanced['threshold']:.4f}")
    print(f"    F1={balanced['f1']:.3f}  precision={balanced['precision']:.3f}  "
          f"recall={balanced['recall']:.3f}  accuracy={balanced['accuracy']:.3f}")
    print(f"    false answers={balanced['false_answers']}  "
          f"false refusals={balanced['false_refusals']}  "
          f"(total errors={balanced['total_errors']})")

    print(f"\n  CONSERVATIVE (0 false answers, then fewest refusals) "
          f"= {conservative['threshold']:.4f}")
    print(f"    F1={conservative['f1']:.3f}  precision={conservative['precision']:.3f}  "
          f"recall={conservative['recall']:.3f}  accuracy={conservative['accuracy']:.3f}")
    print(f"    false answers={conservative['false_answers']}  "
          f"false refusals={conservative['false_refusals']}  "
          f"(total errors={conservative['total_errors']})")
    if conservative["false_answers"] > 0:
        print("    NOTE: zero false answers was not achievable; this is the "
              "minimum-false-answer point.")

    n_small = len(questions) < 30
    if n_small:
        print(f"\n  CONFIDENCE: only {len(questions)} questions -- treat these as "
              "directional, not final.")

    # --- config diff (printed, NOT applied) ---
    print("\n" + "=" * 78)
    print("CONFIG DIFF  (not applied -- pick a policy, then edit app/core/config.py)")
    print("=" * 78)
    print("  - min_rerank_score: float = 0.5")
    print(f"  + min_rerank_score: float = <BALANCED {balanced['threshold']:.2f} "
          f"or CONSERVATIVE {conservative['threshold']:.2f}>")

    if args.report:
        report = {
            "dataset": str(path),
            "current_min_rerank_score": settings.min_rerank_score,
            "use_rewrite": use_rewrite,
            "counts": {"total": len(questions), "answerable": n_ans,
                       "unanswerable": n_una, "labeled": n_labeled},
            "distributions": {"answerable": ans_dist, "unanswerable": una_dist},
            "separation": overlap,
            "sweep": rows,
            "recommendations": {"balanced": balanced, "conservative": conservative},
            "per_query": records,
        }
        out = Path(args.report)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        print(f"\nReport written to {out}")


if __name__ == "__main__":
    main()
