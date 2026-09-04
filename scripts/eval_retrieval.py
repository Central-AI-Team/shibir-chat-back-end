"""Retrieval-only quality check: are the right pages even being fetched?

scripts/eval_responses.py measures the whole /chat pipeline, and
scripts/tune_threshold.py measures one number (the top rerank score) so a
refusal threshold can be picked. Neither answers the question you actually
need first: when an answer is wrong, was the right source ever in front of the
model? Until you know that, "the LLM hallucinated" and "retrieval handed it
five irrelevant pages" look identical from the outside.

So this script generates NO answers. It never imports qa_service or
generator. It runs retrieval, compares the page ids that come back against a
hand-labelled gold set, and reports standard IR metrics.

It scores TWO stages separately, because they fail for different reasons and
have different fixes:

  STAGE A -- candidates: the fetch_k pool from Chroma, after the min_similarity
    filter, exactly as handed to the cross-encoder. This is the bi-encoder's
    ceiling: a gold page missing here can never be recovered downstream.
    Fix by looking at the embedder, the chunking, or the query rewriter.

  STAGE B -- final: the reranked top_k the user really gets.
    A gold page present in A but absent from B is the reranker throwing away
    a document the bi-encoder found. Fix by looking at the reranker.

The miss analysis makes that split explicit per query ("rerank loss" vs
"retrieval loss"), which is the whole point of the script.

UNIT OF RELEVANCE: row_key ("page_8219" / "article_1234", see
app/rag/ingest.py). Both stages are deduplicated to first-occurrence row_keys
before scoring, so "top 5" means five distinct pages -- the thing a human can
label -- not five chunks that might all come from the same page.

Usage:
    python -m scripts.eval_retrieval [path/to/questions.json] [--k 1,3,5,10]
                                     [--no-rewrite] [--ablation] [--report out.json]
    python -m scripts.eval_retrieval --label [path/to/questions.json]

Input file: the {"query": str, "answerable": bool} shape the other two eval
scripts consume, plus "relevant_ids": [row_key, ...]. See
scripts/retrieval_eval_questions.example.json, whose _README explains how to
fill the gold ids in. The file stays valid input for tune_threshold.py and
eval_responses.py.

--label prints, for every entry whose relevant_ids is empty, the top retrieved
candidates with an id and a text preview, so a 30-50 question gold set can be
built by reading instead of by grepping chroma_dump.csv.

Determinism: everything here is deterministic except the query-rewrite LLM
call, which is retrieval, not answer generation, and is what --no-rewrite /
--ablation exist to isolate. Rewrites are lru_cached per process.

Run this from a shell where the app's dependencies and .env are available
(same environment as `python -m app.rag.ingest`).
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from app.core.config import settings
from app.rag.retriever import RetrievedChunk, retrieve_stages

DEFAULT_FILE = Path(__file__).parent / "retrieval_eval_questions.example.json"
REPORT_DIR = Path(__file__).parent.parent / "eval_reports"
DEFAULT_KS = (1, 3, 5, 10)
PREVIEW_CHARS = 200
LABEL_CANDIDATES = 10


# --------------------------------------------------------------------------
# input
# --------------------------------------------------------------------------

def _load_questions(path: Path) -> list[dict]:
    """Read the dataset, tolerating entries written for the other eval scripts.

    relevant_ids is optional on disk -- a missing one is treated as empty, so
    a tune_threshold.py question file can be pointed at --label directly and
    grown into a retrieval dataset from there.
    """
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
        if q["answerable"] and not ids:
            print(f"  warning: answerable question has no relevant_ids, "
                  f"excluded from recall metrics: {q['query']!r}")
        if not q["answerable"] and ids:
            print(f"  warning: unanswerable question has relevant_ids, ignoring "
                  f"them: {q['query']!r}")
            ids = []
        out.append({"query": q["query"], "answerable": bool(q["answerable"]),
                    "relevant_ids": ids})
    return out


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------

def _row_keys(chunks: list[RetrievedChunk]) -> list[str]:
    """Chunk ranking -> page ranking, keeping each page's best position."""
    seen: set[str] = set()
    keys: list[str] = []
    for c in chunks:
        if c.row_key not in seen:
            seen.add(c.row_key)
            keys.append(c.row_key)
    return keys


def _score_one(ranked: list[str], gold: list[str], k: int) -> dict[str, float]:
    """IR metrics for a single query at cutoff k.

    precision@k divides by how many pages were actually returned, not by k --
    the pipeline caps out at top_k distinct pages, and dividing by a k it can
    never fill would report a shortfall the retriever is not responsible for.
    """
    top = ranked[:k]
    gold_set = set(gold)
    hits = [i for i, key in enumerate(top) if key in gold_set]
    return {
        "hit_rate": 1.0 if hits else 0.0,
        "recall": len(hits) / len(gold_set) if gold_set else 0.0,
        "mrr": 1.0 / (hits[0] + 1) if hits else 0.0,
        "precision": len(hits) / len(top) if top else 0.0,
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _aggregate(records: list[dict], stage: str, ks: list[int]) -> dict[int, dict]:
    """Average each metric over the scorable (answerable, labelled) queries."""
    scorable = [r for r in records if r["scorable"]]
    return {
        k: {
            metric: _mean([r["metrics"][stage][k][metric] for r in scorable])
            for metric in ("hit_rate", "recall", "mrr", "precision")
        }
        for k in ks
    }


# --------------------------------------------------------------------------
# running
# --------------------------------------------------------------------------

def _run(questions: list[dict], ks: list[int], use_rewrite: bool) -> list[dict]:
    """Retrieve once per query and score every stage/k off that one call."""
    top_k = settings.top_k
    max_k = max(ks)
    records: list[dict] = []

    for q in questions:
        # rerank_top_n beyond top_k costs nothing extra (the cross-encoder has
        # already scored the whole pool) and lets Stage B be measured at k
        # values above the production cutoff -- i.e. "what would raising top_k
        # buy me?". The production slice is kept separately below.
        stage_a_chunks, stage_b_chunks = retrieve_stages(
            q["query"],
            rerank_top_n=max(max_k, top_k),
            use_rewrite=use_rewrite,
        )
        stage_a = _row_keys(stage_a_chunks)
        stage_b = _row_keys(stage_b_chunks)
        # What the user actually receives: top_k CHUNKS, then deduped to pages.
        production = _row_keys(stage_b_chunks[:top_k])
        top_rerank = stage_b_chunks[0].rerank_score if stage_b_chunks else None

        gold = q["relevant_ids"]
        scorable = bool(q["answerable"] and gold)
        records.append({
            "query": q["query"],
            "answerable": q["answerable"],
            "relevant_ids": gold,
            "scorable": scorable,
            "n_candidates": len(stage_a),
            "n_final": len(stage_b_chunks),
            "top_rerank_score": top_rerank,
            "stage_a_ids": stage_a,
            "stage_b_ids": production,
            "metrics": {
                "candidates": {k: _score_one(stage_a, gold, k) for k in ks},
                "final": {k: _score_one(stage_b, gold, k) for k in ks},
            },
        })

        if not scorable:
            state = "----"  # unanswerable, or answerable but unlabelled
        elif _score_one(production, gold, top_k)["hit_rate"]:
            state = "HIT "
        else:
            state = "MISS"
        score = "None" if top_rerank is None else f"{top_rerank:.4f}"
        print(f"[{state}] {q['query']!r} candidates={len(stage_a)} "
              f"final={len(production)} top_rerank={score}")
    return records


def _miss_analysis(records: list[dict]) -> list[dict]:
    """Classify every answerable query that missed at the production cutoff.

    "rerank loss"    -- a gold page WAS in the Stage A pool and the
                        cross-encoder ranked it out of the top_k.
                        Look at the reranker.
    "retrieval loss" -- no gold page reached the pool at all (never returned by
                        the vector search, or cut by min_similarity).
                        Look at the embedder, the chunking, or the rewriter.
    """
    misses = []
    for r in records:
        if not r["scorable"]:
            continue
        gold = set(r["relevant_ids"])
        if gold & set(r["stage_b_ids"]):
            continue
        in_pool = [g for g in r["relevant_ids"] if g in r["stage_a_ids"]]
        misses.append({
            "query": r["query"],
            "relevant_ids": r["relevant_ids"],
            "cause": "rerank loss" if in_pool else "retrieval loss",
            "gold_in_candidate_pool": in_pool,
            "candidate_rank": (
                min(r["stage_a_ids"].index(g) + 1 for g in in_pool) if in_pool else None
            ),
            "retrieved_ids": r["stage_b_ids"],
        })
    return misses


def _false_positives(records: list[dict]) -> dict:
    """How often unanswerable queries surface something anyway.

    Reported two ways. "raw" is any chunk coming back at all, which is normal
    -- a vector search almost always returns its nearest neighbours. "gated"
    is the one that matters: the top rerank score cleared
    settings.min_rerank_score, so qa_service would have treated junk as a
    grounded source.
    """
    unanswerable = [r for r in records if not r["answerable"]]
    if not unanswerable:
        return {"n": 0, "raw": 0, "gated": 0, "raw_rate": 0.0, "gated_rate": 0.0,
                "threshold": settings.min_rerank_score, "queries": []}
    raw = [r for r in unanswerable if r["stage_b_ids"]]
    gated = [r for r in unanswerable
             if r["top_rerank_score"] is not None
             and r["top_rerank_score"] >= settings.min_rerank_score]
    return {
        "n": len(unanswerable),
        "raw": len(raw),
        "gated": len(gated),
        "raw_rate": len(raw) / len(unanswerable),
        "gated_rate": len(gated) / len(unanswerable),
        "threshold": settings.min_rerank_score,
        "queries": [{"query": r["query"], "top_rerank_score": r["top_rerank_score"],
                     "retrieved_ids": r["stage_b_ids"]} for r in gated],
    }


# --------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------

_STAGE_LABELS = {
    "candidates": f"STAGE A -- candidates (pre-rerank, fetch_k={settings.fetch_k})",
    "final": f"STAGE B -- final (post-rerank, top_k={settings.top_k})",
}


def _print_table(agg: dict[str, dict[int, dict]], ks: list[int], n_scored: int) -> None:
    for stage in ("candidates", "final"):
        print(f"\n{_STAGE_LABELS[stage]}   n={n_scored} labelled answerable queries")
        print(f"  {'k':>4}  {'hit_rate':>9}  {'recall':>9}  {'MRR':>9}  {'precision':>9}")
        print(f"  {'-'*4}  {'-'*9}  {'-'*9}  {'-'*9}  {'-'*9}")
        for k in ks:
            m = agg[stage][k]
            # Stage B cannot return more than top_k pages, so larger k is a
            # "what if top_k were raised" reading, not production behaviour.
            note = " *" if stage == "final" and k > settings.top_k else ""
            print(f"  {k:>4}  {m['hit_rate']:>9.3f}  {m['recall']:>9.3f}  "
                  f"{m['mrr']:>9.3f}  {m['precision']:>9.3f}{note}")
    if any(k > settings.top_k for k in ks):
        print(f"\n  * k > top_k ({settings.top_k}): diagnostic only -- shows what "
              "raising top_k would recover.")


def _print_misses(misses: list[dict], n_scored: int) -> None:
    print(f"\nMISS ANALYSIS -- answerable queries with no gold page in the "
          f"Stage B top_k ({settings.top_k})")
    if not misses:
        print(f"  none: all {n_scored} labelled queries retrieved at least one "
              "gold page.")
        return
    rerank_loss = [m for m in misses if m["cause"] == "rerank loss"]
    retrieval_loss = [m for m in misses if m["cause"] == "retrieval loss"]
    print(f"  {len(misses)}/{n_scored} missed  |  "
          f"rerank loss: {len(rerank_loss)}  retrieval loss: {len(retrieval_loss)}")
    for m in misses:
        print(f"\n  [{m['cause']}] {m['query']!r}")
        print(f"      gold:      {', '.join(m['relevant_ids'])}")
        print(f"      retrieved: {', '.join(m['retrieved_ids']) or '(nothing)'}")
        if m["cause"] == "rerank loss":
            print(f"      gold {', '.join(m['gold_in_candidate_pool'])} was in the "
                  f"candidate pool at rank {m['candidate_rank']} and the "
                  "cross-encoder dropped it -> look at the reranker.")
        else:
            print("      no gold page reached the candidate pool at all -> look at "
                  "the embedder / chunking / query rewrite.")
    if rerank_loss and not retrieval_loss:
        print("\n  Every miss is a rerank loss: the bi-encoder is finding the right "
              "pages and the cross-encoder is discarding them.")
    elif retrieval_loss and not rerank_loss:
        print("\n  Every miss is a retrieval loss: the reranker is not the problem "
              "-- the right pages never make it into the pool.")


def _print_false_positives(fp: dict) -> None:
    print("\nFALSE POSITIVES -- unanswerable queries that surfaced a 'relevant' chunk")
    if not fp["n"]:
        print("  no unanswerable questions in this set (add some: they are what "
              "keeps the gate honest).")
        return
    print(f"  returned anything at all:            {fp['raw']}/{fp['n']} "
          f"({fp['raw_rate']:.0%})  -- expected, a vector search always has neighbours")
    print(f"  cleared min_rerank_score={fp['threshold']}:     {fp['gated']}/{fp['n']} "
          f"({fp['gated_rate']:.0%})  -- these would be answered as if grounded")
    for q in fp["queries"]:
        print(f"    {q['query']!r} top_rerank={q['top_rerank_score']:.4f} "
              f"-> {', '.join(q['retrieved_ids'])}")


def _print_delta(on: dict, off: dict, ks: list[int]) -> None:
    print("\n" + "=" * 78)
    print("ABLATION -- query rewrite ON minus OFF (positive = the rewriter helps)")
    print("=" * 78)
    for stage in ("candidates", "final"):
        print(f"\n{_STAGE_LABELS[stage]}")
        print(f"  {'k':>4}  {'d hit_rate':>11}  {'d recall':>11}  {'d MRR':>11}")
        print(f"  {'-'*4}  {'-'*11}  {'-'*11}  {'-'*11}")
        for k in ks:
            print(f"  {k:>4}  "
                  f"{on[stage][k]['hit_rate'] - off[stage][k]['hit_rate']:>+11.3f}  "
                  f"{on[stage][k]['recall'] - off[stage][k]['recall']:>+11.3f}  "
                  f"{on[stage][k]['mrr'] - off[stage][k]['mrr']:>+11.3f}")


# --------------------------------------------------------------------------
# label mode
# --------------------------------------------------------------------------

def _preview(text: str) -> str:
    flat = " ".join(text.split())
    return flat[:PREVIEW_CHARS] + ("..." if len(flat) > PREVIEW_CHARS else "")


def _label(questions: list[dict], use_rewrite: bool) -> None:
    """Print candidate pages per unlabelled query so gold ids can be pasted in.

    Only entries with an empty relevant_ids are shown -- run it repeatedly as
    the set grows and it only ever asks about what is still unlabelled.
    """
    todo = [q for q in questions if q["answerable"] and not q["relevant_ids"]]
    if not todo:
        print("Nothing to label: every answerable question already has "
              "relevant_ids.")
        return

    print(f"Labelling {len(todo)} unlabelled answerable question(s). "
          f"Showing up to {LABEL_CANDIDATES} pages each, best first.\n")
    for q in todo:
        # Ask for more CHUNKS than the pages we want to show: several chunks
        # commonly come from the same page and collapse to one entry.
        _, final = retrieve_stages(
            q["query"], rerank_top_n=LABEL_CANDIDATES * 3, use_rewrite=use_rewrite
        )
        print("=" * 78)
        print(f"QUERY: {q['query']}")
        print("=" * 78)
        if not final:
            print("  nothing retrieved -- corpus may not cover this at all "
                  '(consider "answerable": false).\n')
            continue

        # One line per PAGE, not per chunk: relevance is labelled page-level.
        seen: set[str] = set()
        shown: list[str] = []
        for c in final:
            if c.row_key in seen:
                continue
            if len(shown) >= LABEL_CANDIDATES:
                break
            seen.add(c.row_key)
            shown.append(c.row_key)
            print(f"\n  {len(shown):>2}. {c.row_key}   "
                  f"rerank={c.rerank_score:.4f}  similarity={c.similarity:.4f}")
            print(f"      {c.book} / {c.chapter}")
            print(f"      {_preview(c.content)}")
        print("\n  paste the ids that actually answer it:")
        print(f'    {{"query": {json.dumps(q["query"], ensure_ascii=False)}, '
              f'"answerable": true, "relevant_ids": ["{shown[0]}"]}}')
        print(f"  (candidates: {', '.join(shown)})\n")


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------

def _parse_ks(raw: str) -> list[int]:
    ks = sorted({int(part) for part in raw.split(",") if part.strip()})
    if not ks or ks[0] < 1:
        raise argparse.ArgumentTypeError(f"--k must be positive integers: {raw!r}")
    return ks


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.eval_retrieval",
        description="Retrieval-only eval: recall@k / hit-rate / MRR, per stage.",
    )
    parser.add_argument("questions", nargs="?", default=str(DEFAULT_FILE),
                        help="dataset JSON (default: the example file)")
    parser.add_argument("--k", type=_parse_ks, default=list(DEFAULT_KS),
                        help="comma-separated cutoffs, default 1,3,5,10")
    parser.add_argument("--no-rewrite", action="store_true",
                        help="search the raw query; skip the Banglish->Bengali rewrite")
    parser.add_argument("--ablation", action="store_true",
                        help="run with rewrite ON and OFF and print the delta")
    parser.add_argument("--report", metavar="OUT.JSON",
                        help="where to write the JSON report "
                             "(default: eval_reports/retrieval_eval_<stamp>.json)")
    parser.add_argument("--label", action="store_true",
                        help="labelling helper: print candidate ids + previews for "
                             "every question with an empty relevant_ids")
    return parser.parse_args()


def _evaluate(questions: list[dict], ks: list[int], use_rewrite: bool) -> dict:
    print("\n" + "=" * 78)
    print(f"RUN: query rewrite {'ON' if use_rewrite else 'OFF'}")
    print("=" * 78)
    records = _run(questions, ks, use_rewrite)
    agg = {stage: _aggregate(records, stage, ks) for stage in ("candidates", "final")}
    n_scored = sum(1 for r in records if r["scorable"])

    _print_table(agg, ks, n_scored)
    misses = _miss_analysis(records)
    _print_misses(misses, n_scored)
    fp = _false_positives(records)
    _print_false_positives(fp)

    return {
        "use_rewrite": use_rewrite,
        "n_questions": len(records),
        "n_scored": n_scored,
        # int-keyed, so _print_delta can index it; _jsonable() stringifies the
        # k values on the way out (JSON object keys must be strings).
        "metrics": agg,
        "misses": misses,
        "false_positives": fp,
        "per_query": records,
    }


def _jsonable(run: dict) -> dict:
    """json.dump turns int keys into strings anyway; do it explicitly so the
    report's k values are unambiguous rather than an accident of the encoder."""
    out = dict(run)
    out["metrics"] = {
        stage: {str(k): m for k, m in per_k.items()}
        for stage, per_k in run["metrics"].items()
    }
    out["per_query"] = [
        {**r, "metrics": {stage: {str(k): m for k, m in per_k.items()}
                          for stage, per_k in r["metrics"].items()}}
        for r in run["per_query"]
    ]
    return out


def main() -> None:
    args = _parse_args()
    path = Path(args.questions)
    questions = _load_questions(path)
    use_rewrite = not args.no_rewrite

    if args.label:
        _label(questions, use_rewrite)
        return

    if not any(q["answerable"] and q["relevant_ids"] for q in questions):
        print("\nNo question has relevant_ids, so nothing can be scored. Run "
              f"`python -m scripts.eval_retrieval --label {path}` first.")

    if args.ablation:
        # --ablation always compares ON against OFF, so --no-rewrite alongside
        # it would otherwise silently run OFF twice.
        runs = [_evaluate(questions, args.k, use_rewrite=True),
                _evaluate(questions, args.k, use_rewrite=False)]
        _print_delta(runs[0]["metrics"], runs[1]["metrics"], args.k)
    else:
        runs = [_evaluate(questions, args.k, use_rewrite)]

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": str(path),
        "k": args.k,
        "config": {
            "top_k": settings.top_k,
            "fetch_k": settings.fetch_k,
            "min_similarity": settings.min_similarity,
            "min_rerank_score": settings.min_rerank_score,
            "embedding_model": settings.embedding_model_name,
            "reranker_model": settings.reranker_model_name,
            "collection": settings.chroma_collection_name,
        },
        "runs": [_jsonable(r) for r in runs],
    }

    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        REPORT_DIR.mkdir(exist_ok=True)
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        report_path = REPORT_DIR / f"retrieval_eval_{stamp}.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nReport written to {report_path}")


if __name__ == "__main__":
    main()
