"""RAGAS end-to-end quality harness (retrieve + generate), on the SAME labeled set.

scripts/eval_retrieval.py only scores retrieval (recall@k / hit-rate / MRR).
scripts/eval_generation_ab.py scores generation quality with a custom
faithfulness/relevance/citation/fluency judge. This script is the
STANDARDIZED version of that same idea -- RAGAS's four metrics
(faithfulness, answer_relevancy, context_precision, context_recall) over the
WHOLE pipeline (app.services.qa_service.answer_question(): retrieve + gate +
generate, exactly what a real user gets), so quality is trackable over time
on a repeatable, widely-recognized scale instead of only this project's own
judge prompt. Same eval set as everything else --
scripts/retrieval_eval_questions.json, extended with a `reference` field
(Part A of this task; see that file's _README) -- no disjoint dataset.

WHAT RAGAS NEEDS (per-question): {question, contexts (retrieved chunk
texts), answer (generated text), reference (ground-truth answer, for
context_recall)}. faithfulness and answer_relevancy don't need `reference`;
context_precision and context_recall do -- so those two are only scored on
the answerable+labeled questions that have a non-empty `reference`
(unanswerable/unlabeled rows get NaN for those two, not a crash).

===========================================================================
TWO PHASES, TWO VENVS -- READ THIS BEFORE RUNNING ANYTHING
===========================================================================
ragas 0.3.9 (the version this script is written against and pinned to --
see requirements-ragas.txt) transitively needs a langchain family that
predates langchain's 1.x migration (langchain-core/-community/-openai all
<1.0) or it fails to import at all (a bare `import ragas` 404s on
langchain_community.chat_models.vertexai, which the current
langchain-community release removed -- confirmed live, Sept 2026, on every
ragas release from 0.3.9 through the latest 0.4.3; ragas doesn't upper-bound
that dependency). That whole langchain stack must NOT live in the app's own
venv (it must stay LangChain-free, and installing it there once already
silently upgraded the pinned `openai` package -- a real regression risk to
production, caught and reverted while building this script). It also should
not live alongside the app's OWN heavy deps (sentence-transformers/torch/
chromadb) in a second copy -- this box has ~8GB of free disk, and a second
full ML stack doesn't fit (confirmed live: an attempt OOM'd the disk, not
just RAM).

So: a separate venv-ragas/ (repo root, gitignored) holds ONLY ragas +
datasets + the pinned pre-1.x langchain family (~1.3GB) -- see
requirements-ragas.txt, a `pip freeze` snapshot of a known-working install,
not requirements.txt. It also has pydantic-settings + python-dotenv (tiny,
no disk risk) so it can import app.core.config for the judge/embedding
model names WITHOUT importing anything else from app/ -- venv-ragas cannot
run retrieval (no sentence-transformers/chromadb there), by design.

Phase 1 (capture) -- run under the MAIN venv (has qa_service/retriever/the
    embedder+reranker): calls answer_question() once per question (real
    production retrieve+gate+generate), writes {question, contexts, answer,
    reference, ...} to a JSON file. No ragas import here at all.
Phase 2 (score) -- run under venv-ragas: reads that JSON file, builds a
    ragas EvaluationDataset, runs the four metrics with the judge LLM +
    embeddings from settings.ragas_judge_model/ragas_embedding_model, prints
    aggregate + per-question + a Bengali-judgment spot check. No app.rag /
    app.services import here at all (would ImportError -- sentence-
    transformers etc. aren't in this venv on purpose).

Usage:
    # Phase 1, from the MAIN venv:
    venv/bin/python -m scripts.eval_ragas capture [questions.json]
        [--limit N] [--seed N] [--out captured.json]

    # Phase 2, from venv-ragas, same repo root:
    venv-ragas/bin/python -m scripts.eval_ragas score captured.json
        [--yes] [--metrics faithfulness,answer_relevancy,context_precision,context_recall]
        [--report out.json]

VERSION GUARD: this script imports ragas lazily (only inside _score(), never
at module level) specifically so `capture` works under the main venv where
ragas isn't (and shouldn't be) installed. score() prints ragas.__version__
first thing and hard-fails with a clear message if it's not 0.3.x (the API
this script was written against -- LangchainLLMWrapper/LangchainEmbeddingsWrapper,
bypass_temperature=True, the user_input/retrieved_contexts/response/reference
SingleTurnSample field names -- all confirmed against 0.3.9 specifically;
ragas's own API "changes a lot across releases" per upstream's own framing,
so don't assume a different installed version behaves the same).

COST + TIME GATE: RAGAS is call-heavy -- each metric can issue several LLM
calls per question (statement extraction, NLI verdicts, reverse-question
generation for answer_relevancy, ...), and confirmed live latency for a
SINGLE metric call against gpt-5-mini is 60-90 seconds (reasoning-model
overhead, not network). score() prints a cost AND a wall-clock time
estimate and stops unless --yes is passed. Use --limit on capture for a
small first pass, and --metrics on score to run a subset.

Run capture from a shell where the app's dependencies and .env are available
(same as `python -m app.rag.ingest`). Run score from venv-ragas/'s python,
same repo root, same .env.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime
from pathlib import Path

DEFAULT_FILE = Path(__file__).parent / "retrieval_eval_questions.json"
REPORT_DIR = Path(__file__).parent.parent / "eval_reports"
DEFAULT_SEED = 20260906
DEFAULT_METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
N_BENGALI_VALIDATION = 5

# Approximate $/1M-token snapshot (Sept 2026); reconfirm before a large run.
_JUDGE_PRICING = {
    "gpt-5-mini": (0.25, 2.00),
    "gpt-5": (1.25, 10.00),
}
_EMBEDDING_PRICING = {"text-embedding-3-large": 0.13, "text-embedding-3-small": 0.02}  # $/1M tokens

# Confirmed live (Sept 2026): a single metric call against gpt-5-mini/gpt-5
# takes 60-90s (hidden reasoning, not network) -- multiple calls per metric
# per question. This is a rough per-(question,metric) wall-clock estimate.
_EST_SECONDS_PER_METRIC_CALL = 75
# Rough calls-per-metric (statement extraction + verdicts, reverse
# questions, etc.) -- approximate, varies with answer length.
_EST_CALLS_PER_METRIC = {
    "faithfulness": 2, "answer_relevancy": 4,
    "context_precision": 1, "context_recall": 1,
}
_EST_TOKENS_PER_CALL_IN, _EST_TOKENS_PER_CALL_OUT = 1200, 600


# --------------------------------------------------------------------------
# shared input loading (no ragas or app import needed for this part)
# --------------------------------------------------------------------------

def _load_questions(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        questions = json.load(f)
    if isinstance(questions, dict):
        questions = questions["questions"]
    out = []
    for q in questions:
        if "query" not in q or "answerable" not in q:
            raise ValueError(f"bad entry, expected query/answerable keys: {q}")
        out.append({
            "query": q["query"], "answerable": bool(q["answerable"]),
            "relevant_ids": (q.get("relevant_ids") or []) if q["answerable"] else [],
            "lang": q.get("lang", "unknown"), "reference": q.get("reference", "") or "",
        })
    return out


# --------------------------------------------------------------------------
# phase 1: capture (main venv -- real retrieve + generate)
# --------------------------------------------------------------------------

def _capture(questions: list[dict], limit: int | None, seed: int) -> list[dict]:
    from app.services.qa_service import answer_question  # deferred: main venv only

    if limit is not None and limit < len(questions):
        questions = random.Random(seed).sample(questions, limit)

    records = []
    for i, q in enumerate(questions):
        t0 = time.perf_counter()
        response = answer_question(q["query"])
        latency = time.perf_counter() - t0
        records.append({
            "question": q["query"],
            "contexts": [c.content for c in response.sources],
            "answer": response.answer,
            "reference": q["reference"],
            "answerable": q["answerable"],
            "lang": q["lang"],
            "latency_s": latency,
        })
        print(f"  [{i+1}/{len(questions)}] n_contexts={len(response.sources)} "
              f"lat={latency:.2f}s {q['query'][:60]!r}")
    return records


def _run_capture(args: argparse.Namespace) -> None:
    path = Path(args.questions)
    questions = _load_questions(path)
    print(f"CAPTURE: {len(questions)} questions from {path}"
          + (f", sampling {args.limit}" if args.limit else ""))
    records = _capture(questions, args.limit, args.seed)

    out_path = Path(args.out) if args.out else REPORT_DIR / f"ragas_captured_{datetime.now():%Y-%m-%d_%H-%M}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    n_labeled = sum(1 for r in records if r["answerable"] and r["reference"])
    print(f"\nCaptured {len(records)} question(s), {n_labeled} with a reference "
          f"(scorable for context_precision/context_recall).")
    print(f"Written to {out_path}")
    print(f"\nNext: venv-ragas/bin/python -m scripts.eval_ragas score {out_path}")


# --------------------------------------------------------------------------
# phase 2: score (venv-ragas -- ragas metrics, no app.rag/app.services import)
# --------------------------------------------------------------------------

def _print_cost_time_estimate(records: list[dict], metrics: list[str],
                               judge_model: str, embedding_model: str) -> None:
    n = len(records)
    n_scorable = sum(1 for r in records if r["answerable"] and r["reference"])
    price_in, price_out = _JUDGE_PRICING.get(judge_model, (0.25, 2.00))
    emb_price = _EMBEDDING_PRICING.get(embedding_model, 0.13)

    print("\n" + "=" * 78)
    print("COST + TIME ESTIMATE (approximate -- pricing/call-counts are rough; "
          "RAGAS's actual call count depends on answer/context length)")
    print("=" * 78)
    total_cost, total_seconds = 0.0, 0.0
    for m in metrics:
        n_rows = n_scorable if m in ("context_precision", "context_recall") else n
        calls = _EST_CALLS_PER_METRIC.get(m, 2)
        cost = n_rows * calls * (_EST_TOKENS_PER_CALL_IN * price_in
                                  + _EST_TOKENS_PER_CALL_OUT * price_out) / 1e6
        seconds = n_rows * calls * _EST_SECONDS_PER_METRIC_CALL
        total_cost += cost
        total_seconds += seconds
        print(f"  {m:<20} n={n_rows:>3}  ~{calls} call(s)/row  "
              f"est_$={cost:>7.3f}  est_time={seconds/60:>6.1f} min (sequential)")
    embed_cost = n * 300 * emb_price / 1e6  # rough: question + answer embedded for answer_relevancy
    total_cost += embed_cost
    print(f"  {'embeddings (' + embedding_model + ')':<20} est_$={embed_cost:.4f}")
    print(f"\n  ESTIMATED TOTAL: ${total_cost:.3f}")
    print(f"  ESTIMATED TIME: {total_seconds/60:.1f} min if run sequentially "
          "(ragas parallelizes some of this, so real wall-clock is usually "
          "less -- but budget for the sequential figure, not the best case)")
    print(f"\n  eval set: {n} questions captured, {n_scorable} answerable+labeled "
          "(context_precision/context_recall only score these)")


def _validate_bengali(samples: list, llm, n: int) -> None:
    """Print RAGAS's decomposed claims + verdicts for a few Bengali samples,
    so the Faithfulness score can be sanity-checked against your own reading
    instead of trusted blind. Reaches into Faithfulness's internal helper
    methods (_create_statements/_create_verdicts) -- these are private and
    could change on a ragas upgrade; that's an acceptable risk for a
    one-off transparency check, not a public API this script depends on
    structurally."""
    import asyncio
    from ragas.metrics import Faithfulness

    print("\n" + "=" * 78)
    print(f"BENGALI VALIDATION -- RAGAS's claim decomposition + verdicts on "
          f"{min(n, len(samples))} example(s) (read this before trusting the aggregates)")
    print("=" * 78)
    metric = Faithfulness(llm=llm)
    for i, sample in enumerate(samples[:n]):
        print(f"\n--- [{i+1}] {sample.user_input[:70]!r} ---")
        print(f"  answer: {sample.response[:200]!r}")
        try:
            row = sample.to_dict()
            statements = asyncio.run(metric._create_statements(row, callbacks=None))
            print(f"  decomposed claims ({len(statements.statements)}):")
            for s in statements.statements:
                print(f"    - {s}")
            verdicts = asyncio.run(metric._create_verdicts(row, statements.statements, callbacks=None))
            for v in verdicts.statements:
                print(f"    [{'FAITHFUL' if v.verdict else 'UNSUPPORTED'}] {v.statement}")
                print(f"      reason: {v.reason}")
        except Exception as e:
            print(f"  (validation call failed, not fatal to the main run: {e})")
    print("\n  If the claim splits or verdicts above look wrong for Bengali "
          "(bad sentence boundaries, wrong entity handling, etc.), treat the "
          "aggregate scores below as DIRECTIONAL ONLY -- lean on per-question "
          "inspection and scripts/eval_generation_ab.py's human review sheet instead.")


def _score(records: list[dict], metric_names: list[str], judge_model: str,
           embedding_model: str) -> dict:
    import ragas
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas import evaluate
    from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness
    from ragas.run_config import RunConfig

    print(f"ragas version: {ragas.__version__}")
    if not ragas.__version__.startswith("0.3."):
        raise SystemExit(
            f"This script was written against ragas 0.3.x (LangchainLLMWrapper, "
            f"bypass_temperature=True, user_input/retrieved_contexts/response/"
            f"reference field names). Installed: {ragas.__version__}. Re-verify "
            f"that API still matches before trusting this script's output -- see "
            f"the module docstring's VERSION GUARD section."
        )

    from app.core.config import settings  # safe here: only pydantic-settings, no heavy deps

    api_key = settings.openai_api_key
    chat = ChatOpenAI(model=judge_model, api_key=api_key, max_completion_tokens=4000)
    # bypass_temperature=True is REQUIRED for gpt-5-family judges -- ragas's
    # own prompts pass a low default temperature these models reject outright
    # (confirmed live: a 400 without this flag). See config.py's comment.
    llm = LangchainLLMWrapper(chat, bypass_temperature=True)
    embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model=embedding_model, api_key=api_key))

    metric_map = {
        "faithfulness": Faithfulness(llm=llm),
        "answer_relevancy": AnswerRelevancy(llm=llm, embeddings=embeddings),
        "context_precision": ContextPrecision(llm=llm),
        "context_recall": ContextRecall(llm=llm),
    }
    # RunConfig: default timeout=180s/max_workers=16 caused a timeout cascade
    # in testing -- 16 concurrent gpt-5-mini calls at 60-90s/call each blows
    # well past 180s once the API starts queuing them. Fewer concurrent
    # workers + a longer per-call timeout fixed it (confirmed live).
    run_config = RunConfig(timeout=600, max_workers=4)

    # reference-dependent metrics (context_precision/context_recall) MUST
    # NOT see a row with no reference -- SingleTurnSample.to_dict() drops a
    # None field entirely, so ragas's own metric code does row['reference']
    # and raises a bare KeyError (confirmed live) instead of skipping that
    # row gracefully. So these two metrics get their OWN dataset, built only
    # from rows that actually have a reference -- not the same dataset
    # filtered after the fact.
    no_ref_metrics = [metric_map[m] for m in metric_names if m in ("faithfulness", "answer_relevancy")]
    ref_metrics = [metric_map[m] for m in metric_names if m in ("context_precision", "context_recall")]

    def _sample(r: dict) -> SingleTurnSample:
        return SingleTurnSample(
            user_input=r["question"],
            retrieved_contexts=r["contexts"] or ["(কোনো প্রসঙ্গ পাওয়া যায়নি)"],
            response=r["answer"],
            reference=r["reference"] or "N/A",
        )

    all_samples = [_sample(r) for r in records]
    ref_samples = [_sample(r) for r in records if r["reference"]]

    _validate_bengali([s for s, r in zip(all_samples, records) if r["answerable"] and r["reference"]],
                       llm, N_BENGALI_VALIDATION)

    results = {}
    if no_ref_metrics:
        print(f"\nScoring {[m.name for m in no_ref_metrics]} on all {len(all_samples)} rows...")
        results["no_ref"] = evaluate(dataset=EvaluationDataset(samples=all_samples), metrics=no_ref_metrics,
                                      show_progress=True, raise_exceptions=False, run_config=run_config)
    if ref_metrics:
        print(f"\nScoring {[m.name for m in ref_metrics]} on {len(ref_samples)} referenced rows...")
        results["ref"] = evaluate(dataset=EvaluationDataset(samples=ref_samples), metrics=ref_metrics,
                                   show_progress=True, raise_exceptions=False, run_config=run_config)
    return results


def _mean_median_min(values: list[float]) -> dict:
    vs = sorted(v for v in values if v == v)  # drop NaN (v==v is False for NaN)
    if not vs:
        return {"n": 0, "mean": None, "median": None, "min": None}
    return {
        "n": len(vs), "mean": sum(vs) / len(vs),
        "median": vs[len(vs) // 2], "min": vs[0],
    }


def _print_results(results: dict, records: list[dict], metric_names: list[str], worst_n: int = 5) -> None:
    """results: {"no_ref": EvaluationResult | None, "ref": EvaluationResult | None} --
    see _score()'s split (reference-dependent metrics run on a separate,
    smaller dataset than faithfulness/answer_relevancy)."""
    dfs = {k: v.to_pandas() for k, v in results.items()}
    print("\n" + "=" * 78)
    print("AGGREGATE")
    print("=" * 78)
    for m in metric_names:
        df = next((d for d in dfs.values() if m in d.columns), None)
        if df is None:
            continue
        stats = _mean_median_min(df[m].tolist())
        print(f"  {m:<20} n={stats['n']:>3}  mean={stats['mean'] and round(stats['mean'],3)}  "
              f"median={stats['median'] and round(stats['median'],3)}  "
              f"min={stats['min'] and round(stats['min'],3)}")

        worst = df.nsmallest(worst_n, m)[["user_input", m]] if stats["n"] else None
        if worst is not None and len(worst):
            print(f"    worst {min(worst_n, len(worst))} by {m}:")
            for _, row in worst.iterrows():
                score = row[m]
                if score == score:  # not NaN
                    print(f"      {score:.3f}  {row['user_input'][:70]!r}")


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------

def _parse_metrics(raw: str) -> list[str]:
    names = [n.strip() for n in raw.split(",") if n.strip()]
    bad = [n for n in names if n not in DEFAULT_METRICS]
    if bad:
        raise argparse.ArgumentTypeError(f"unknown metric(s) {bad}, choose from {DEFAULT_METRICS}")
    return names


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m scripts.eval_ragas")
    sub = p.add_subparsers(dest="phase", required=True)

    cap = sub.add_parser("capture", help="run under the MAIN venv")
    cap.add_argument("questions", nargs="?", default=str(DEFAULT_FILE))
    cap.add_argument("--limit", type=int, default=None)
    cap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    cap.add_argument("--out", metavar="OUT.JSON")

    sc = sub.add_parser("score", help="run under venv-ragas")
    sc.add_argument("captured", help="JSON file written by the capture phase")
    sc.add_argument("--metrics", type=_parse_metrics, default=list(DEFAULT_METRICS))
    sc.add_argument("--yes", action="store_true")
    sc.add_argument("--report", metavar="OUT.JSON")

    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if args.phase == "capture":
        _run_capture(args)
        return

    # score
    from app.core.config import settings  # safe: pydantic-settings only

    records = json.loads(Path(args.captured).read_text(encoding="utf-8"))
    n_scorable = sum(1 for r in records if r["answerable"] and r["reference"])
    print(f"SCORE: {len(records)} captured question(s) from {args.captured}, "
          f"{n_scorable} answerable+labeled")

    _print_cost_time_estimate(records, args.metrics, settings.ragas_judge_model,
                               settings.ragas_embedding_model)
    if not args.yes:
        print("\nStopping before any paid API call. Re-run with --yes to actually score.")
        return

    results = _score(records, args.metrics, settings.ragas_judge_model, settings.ragas_embedding_model)
    _print_results(results, records, args.metrics)

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    report_path = Path(args.report) if args.report else REPORT_DIR / f"ragas_score_{stamp}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    dfs = {k: v.to_pandas() for k, v in results.items()}
    aggregate = {}
    per_question = {}
    for m in args.metrics:
        df = next((d for d in dfs.values() if m in d.columns), None)
        if df is None:
            continue
        aggregate[m] = _mean_median_min(df[m].tolist())
        per_question[m] = df[["user_input", m]].to_dict(orient="records")
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "captured_file": args.captured, "metrics": args.metrics,
        "judge_model": settings.ragas_judge_model, "embedding_model": settings.ragas_embedding_model,
        "n_questions": len(records), "n_scorable": n_scorable,
        "aggregate": aggregate,
        "per_question": per_question,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\nReport written to {report_path}")


if __name__ == "__main__":
    main()
