"""Does expanding a query into more variants actually help retrieval, and at what cost?

app/rag/query_rewriter.expand_query() turns a query into a Bengali-script
canonical form ("bn") plus up to settings.max_variants alternate phrasings,
and app/rag/retriever.retrieve_stages() embeds + searches EVERY one of those
strings, then merges. More variants can only help recall (more chances to
match the corpus's own wording) but each extra variant costs one more embed +
Chroma query and can pull off-topic chunks into the pool the reranker sees.
This script answers "how many variants earn their keep?" empirically: it runs
the SAME rewriter/retriever at max_variants in {0,1,2,3,4} (configurable)
against the hand-labelled set and reports recall against cost, so a variant
count can be chosen from evidence instead of "more must be better".

n_variants=0 means: still rewrite (Banglish/English -> Bengali script), but
cap the tuple at just the canonical "bn" form. That is NOT the same as
--no-rewrite in scripts/eval_retrieval.py, which skips the LLM call entirely
and searches the raw string -- 0 here isolates "is expansion worth it at
all", not "is the rewrite itself worth it".

REUSED, not reinvented:
  - app/rag/retriever.retrieve_stages -- now takes an optional max_variants
    (see its docstring, point 7), forwarded straight into expand_query(). No
    second retrieval path; this sweeps the real pipeline.
  - scripts/eval_retrieval's own scoring primitives (_row_keys, _score_one,
    _aggregate, _mean) -- same recall/hit-rate/MRR math, so a number here
    means the same thing it means over there.

RETRIEVAL ONLY: never imports qa_service or generator, same as
scripts/eval_retrieval.py. The only non-deterministic step is the
query-rewrite LLM call inside expand_query(), which is lru_cached per
process -- a repeated (query, max_variants) pair costs one call, not five.

RAM: this holds bge-m3 + bge-reranker-v2-m3 resident, same as
scripts/eval_retrieval.py -- budget ~6 GB free, or it OOMs/swap-thrashes on a
small box. See that script's module docstring for the same note.

Usage:
    python -m scripts.eval_query_expansion [questions.json]
        [--variants 0,1,2,3,4] [--k 1,3,5] [--report out.json]

Run this from a shell where the app's dependencies and .env are available
(same environment as `python -m app.rag.ingest`).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from time import perf_counter

from app.core.config import settings
from app.rag.query_rewriter import expand_query
from app.rag.retriever import retrieve_stages
from scripts.eval_retrieval import _aggregate, _mean, _row_keys, _score_one

DEFAULT_FILE = Path(__file__).parent / "retrieval_eval_questions.json"
REPORT_DIR = Path(__file__).parent.parent / "eval_reports"
DEFAULT_KS = (1, 3, 5)
DEFAULT_VARIANTS = (0, 1, 2, 3, 4)
LANGS = ("bn", "banglish", "en")

# Below this many labelled (answerable + gold) questions in a bucket, its
# numbers are flagged as weak evidence rather than presented as a real signal.
MIN_LABELED_FOR_CONFIDENCE = 10
# recall@k improvement smaller than this going n -> n+1 counts as "no
# measurable gain" -- noise, not signal, on a ~40-question set.
RECALL_EPSILON = 0.02


# --------------------------------------------------------------------------
# input
# --------------------------------------------------------------------------

def _load_questions(path: Path) -> list[dict]:
    """Same {query, answerable, relevant_ids} shape scripts.eval_retrieval
    consumes, but keeps `lang` for the per-language breakdown this script
    reports (eval_retrieval's loader drops it)."""
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
        out.append({
            "query": q["query"],
            "answerable": bool(q["answerable"]),
            "relevant_ids": ids if q["answerable"] else [],
            "lang": q.get("lang", "unknown"),
        })
    return out


# --------------------------------------------------------------------------
# running one variant count
# --------------------------------------------------------------------------

def _n_query_strings(query: str, n_variants: int) -> int:
    """Actual number of distinct strings retrieve_stages will embed+search --
    mirrors its own `expand_query(query, max_variants=...) or (query,)`
    fallback, since dedupe can make this less than n_variants+1."""
    return len(expand_query(query, max_variants=n_variants) or (query,))


def _run(questions: list[dict], ks: list[int], n_variants: int) -> list[dict]:
    """Retrieve once per query at this variant count; score + cost + timing."""
    top_k = settings.top_k
    max_k = max(ks)
    records: list[dict] = []

    for q in questions:
        t0 = perf_counter()
        stage_a_chunks, stage_b_chunks = retrieve_stages(
            q["query"],
            rerank_top_n=max(max_k, top_k),
            use_rewrite=True,
            max_variants=n_variants,
        )
        latency = perf_counter() - t0
        n_strings = _n_query_strings(q["query"], n_variants)

        stage_a = _row_keys(stage_a_chunks)
        stage_b = _row_keys(stage_b_chunks)
        production = _row_keys(stage_b_chunks[:top_k])

        gold = q["relevant_ids"]
        scorable = bool(q["answerable"] and gold)
        records.append({
            "query": q["query"],
            "lang": q["lang"],
            "answerable": q["answerable"],
            "relevant_ids": gold,
            "scorable": scorable,
            "n_query_strings": n_strings,
            "latency_s": latency,
            "stage_a_ids": stage_a,
            "stage_b_ids": production,
            "metrics": {
                "candidates": {k: _score_one(stage_a, gold, k) for k in ks},
                "final": {k: _score_one(stage_b, gold, k) for k in ks},
            },
        })

        state = "----" if not scorable else (
            "HIT " if _score_one(production, gold, top_k)["hit_rate"] else "MISS"
        )
        print(f"  [{state}] n_str={n_strings} lat={latency:.2f}s {q['query']!r}")
    return records


def _summarize(n_variants: int, records: list[dict], ks: list[int]) -> dict:
    agg = {stage: _aggregate(records, stage, ks) for stage in ("candidates", "final")}
    return {
        "n_variants": n_variants,
        "n_total": len(records),
        "n_scored": sum(1 for r in records if r["scorable"]),
        "avg_query_strings": _mean([r["n_query_strings"] for r in records]),
        "avg_latency_s": _mean([r["latency_s"] for r in records]),
        "metrics": agg,
    }


# --------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------

def _print_sweep_table(summaries: list[dict], ks: list[int], stage: str) -> None:
    label = "STAGE A -- candidates (pre-rerank)" if stage == "candidates" \
        else "STAGE B -- final (post-rerank)"
    n_scored = summaries[0]["n_scored"] if summaries else 0
    print(f"\n{label}   n={n_scored} labelled answerable queries")
    hdr = f"  {'n_var':>5}" + "".join(f"  {'hit@'+str(k):>8}" for k in ks) \
        + "".join(f"  {'rec@'+str(k):>8}" for k in ks) \
        + f"  {'avg_strs':>8}  {'avg_lat_s':>9}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for s in summaries:
        m = s["metrics"][stage]
        row = f"  {s['n_variants']:>5}" \
            + "".join(f"  {m[k]['hit_rate']:>8.3f}" for k in ks) \
            + "".join(f"  {m[k]['recall']:>8.3f}" for k in ks) \
            + f"  {s['avg_query_strings']:>8.2f}  {s['avg_latency_s']:>9.3f}"
        print(row)


def _print_language_breakdown(
    variants: list[int], all_records: dict[int, list[dict]], ks: list[int]
) -> None:
    print("\n" + "=" * 78)
    print("PER-LANGUAGE BREAKDOWN  (Stage B / final, post-rerank)")
    print("=" * 78)
    print("Variants should help banglish/en far more than native bn -- that's the "
          "whole point of expansion into Bengali script.")
    for lang in LANGS:
        bucket_summaries = [
            _summarize(n, [r for r in all_records[n] if r["lang"] == lang], ks)
            for n in variants
        ]
        n_labeled = bucket_summaries[0]["n_scored"] if bucket_summaries else 0
        n_total = bucket_summaries[0]["n_total"] if bucket_summaries else 0
        flag = ""
        if n_labeled < MIN_LABELED_FOR_CONFIDENCE:
            flag = f"  ** WEAK EVIDENCE: only {n_labeled} labelled question(s) -- directional only **"
        print(f"\n[{lang}]  {n_total} question(s), {n_labeled} labelled answerable{flag}")
        if n_labeled == 0:
            print("  (nothing to score)")
            continue
        _print_sweep_table(bucket_summaries, ks, "final")


def _recommend(summaries: list[dict], k: int) -> tuple[dict, list[str]]:
    """Walk n_variants ascending; keep raising the recommendation only while
    each step buys a real (> RECALL_EPSILON) recall@k gain on Stage B."""
    lines = []
    best = summaries[0]
    for i, s in enumerate(summaries):
        recall = s["metrics"]["final"][k]["recall"]
        if i == 0:
            lines.append(
                f"n_variants={s['n_variants']}: recall@{k}={recall:.3f}  "
                f"avg_strings={s['avg_query_strings']:.2f}  "
                f"avg_latency={s['avg_latency_s']:.3f}s  (baseline)"
            )
            continue
        prev = summaries[i - 1]
        d_recall = recall - prev["metrics"]["final"][k]["recall"]
        d_strings = s["avg_query_strings"] - prev["avg_query_strings"]
        d_latency = s["avg_latency_s"] - prev["avg_latency_s"]
        gained = d_recall > RECALL_EPSILON
        verdict = "GAIN -- keep raising" if gained else "no measurable gain for the added cost"
        lines.append(
            f"n_variants={s['n_variants']}: recall@{k}={recall:.3f}  "
            f"(Δrecall={d_recall:+.3f} vs n={prev['n_variants']}, "
            f"Δavg_strings={d_strings:+.2f}, Δavg_latency={d_latency:+.3f}s) -> {verdict}"
        )
        if gained:
            best = s
    return best, lines


def _jsonable(summary: dict) -> dict:
    out = dict(summary)
    out["metrics"] = {
        stage: {str(k): m for k, m in per_k.items()}
        for stage, per_k in summary["metrics"].items()
    }
    return out


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------

def _parse_ks(raw: str) -> list[int]:
    ks = sorted({int(part) for part in raw.split(",") if part.strip()})
    if not ks or ks[0] < 1:
        raise argparse.ArgumentTypeError(f"--k must be positive integers: {raw!r}")
    return ks


def _parse_variants(raw: str) -> list[int]:
    vs = sorted({int(part) for part in raw.split(",") if part.strip()})
    if not vs or vs[0] < 0:
        raise argparse.ArgumentTypeError(f"--variants must be non-negative integers: {raw!r}")
    return vs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.eval_query_expansion",
        description="Sweep query-rewriter variant counts and measure recall vs. cost.",
    )
    parser.add_argument("questions", nargs="?", default=str(DEFAULT_FILE),
                        help="dataset JSON (default: scripts/retrieval_eval_questions.json)")
    parser.add_argument("--variants", type=_parse_variants,
                        default=list(DEFAULT_VARIANTS),
                        help="comma-separated max_variants values to sweep, default 0,1,2,3,4")
    parser.add_argument("--k", type=_parse_ks, default=list(DEFAULT_KS),
                        help="comma-separated cutoffs, default 1,3,5")
    parser.add_argument("--report", metavar="OUT.JSON",
                        help="where to write the JSON report "
                             "(default: eval_reports/query_expansion_<stamp>.json)")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    path = Path(args.questions)
    questions = _load_questions(path)
    ks = args.k
    variants = args.variants
    max_k = max(ks)

    n_ans = sum(1 for q in questions if q["answerable"])
    n_labeled = sum(1 for q in questions if q["answerable"] and q["relevant_ids"])
    lang_counts = dict(Counter(q["lang"] for q in questions))

    print("=" * 78)
    print("QUERY EXPANSION SWEEP")
    print("=" * 78)
    print(f"dataset: {path}")
    print(f"  {len(questions)} questions: {n_ans} answerable, {n_labeled} labelled "
          f"answerable (scorable)")
    print(f"  language mix: {lang_counts}")
    print(f"  sweeping max_variants in {variants}, k in {ks} "
          f"(current settings.max_variants={settings.max_variants})")

    if n_labeled == 0:
        print("\nNo question has relevant_ids, so nothing can be scored. Run "
              f"`python -m scripts.eval_retrieval --label {path}` first.")
        return

    all_records: dict[int, list[dict]] = {}
    summaries: list[dict] = []
    for n in variants:
        print(f"\n--- max_variants={n} ---")
        records = _run(questions, ks, n)
        all_records[n] = records
        summaries.append(_summarize(n, records, ks))

    _print_sweep_table(summaries, ks, "candidates")
    _print_sweep_table(summaries, ks, "final")
    _print_language_breakdown(variants, all_records, ks)

    best, rec_lines = _recommend(summaries, max_k)

    print("\n" + "=" * 78)
    print("RECOMMENDATION")
    print("=" * 78)
    for line in rec_lines:
        print(" ", line)

    small_set = len(questions) < 30 or n_labeled < 20
    print(f"\n  Eval set: {len(questions)} questions, {n_labeled} labelled answerable "
          f"(language mix: {lang_counts})")
    if small_set:
        print("  ** CONFIDENCE: small eval set -- treat this recommendation as "
              "directional, not final. **")

    print(f"\n  Recommended max_variants = {best['n_variants']}  "
          f"(recall@{max_k}={best['metrics']['final'][max_k]['recall']:.3f}, "
          f"avg_query_strings={best['avg_query_strings']:.2f}, "
          f"avg_latency={best['avg_latency_s']:.3f}s)")
    print("\n  CONFIG DIFF (not applied -- edit app/core/config.py yourself):")
    print(f"    - max_variants: int = {settings.max_variants}")
    print(f"    + max_variants: int = {best['n_variants']}"
          f"  # chosen {datetime.now().date().isoformat()} from a "
          f"{len(questions)}-question eval set ({n_labeled} labelled) via "
          "scripts/eval_query_expansion.py")

    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        REPORT_DIR.mkdir(exist_ok=True)
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        report_path = REPORT_DIR / f"query_expansion_{stamp}.json"

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": str(path),
        "k": ks,
        "variants_swept": variants,
        "n_questions": len(questions),
        "n_labeled": n_labeled,
        "language_mix": lang_counts,
        "current_max_variants": settings.max_variants,
        "config": {
            "top_k": settings.top_k,
            "fetch_k": settings.fetch_k,
            "min_similarity": settings.min_similarity,
            "embedding_model": settings.embedding_model_name,
            "reranker_model": settings.reranker_model_name,
        },
        "summaries": [_jsonable(s) for s in summaries],
        "language_breakdown": {
            lang: [
                _jsonable(_summarize(n, [r for r in all_records[n] if r["lang"] == lang], ks))
                for n in variants
            ]
            for lang in LANGS
        },
        "recommendation": {
            "max_variants": best["n_variants"],
            "lines": rec_lines,
            "small_eval_set": small_set,
        },
        "per_query": {str(n): all_records[n] for n in variants},
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nReport written to {report_path}")


if __name__ == "__main__":
    main()
