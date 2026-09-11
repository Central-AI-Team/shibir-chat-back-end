"""Validate a cheap/fast model for the MECHANICAL tasks before routing to it.

app/core/llm.py now lets any call site route per-task
(get_model(task)/complete(task, ...)), driven by settings.model_by_task --
see that file and scripts/eval_generation_ab.py (which did the same thing
for the GENERATION tier). This script does the other half: intent
classification and query rewriting run on EVERY request, before the user
sees anything, so a small fast model there is attractive for latency -- but
a cheaper model is often WORSE at strict JSON / instruction-following, and
downgrading a mechanical task on vibes alone can quietly raise parse-failure
/ misroute rates. This script proves or disproves that per task, per
candidate, with real numbers -- it does not assume a cheap model is fine.

TWO CANDIDATES are checked (config.py doesn't hardcode either; both stay in
this script's CANDIDATES list, not settings.model_by_task, until you decide):
  gpt-5-nano          -- same provider/client as gpt-5-mini, same
                          reasoning-model quirks (fixed temperature,
                          max_completion_tokens) -- simplest to adopt.
  openai/gpt-oss-20b  -- different provider (Groq, OpenAI-compatible API),
                          open-weight, cheaper still. NOTE: llama-3.1-8b-instant
                          was the original pick here but is confirmed (live,
                          Sept 2026) to 404 on this project's GROQ_API_KEY
                          ("does not exist or you do not have access to it"),
                          despite Groq's own docs listing it as self-serve --
                          gpt-oss-20b is the confirmed-working substitute.
                          It's ALSO a reasoning model under the hood
                          (nonzero hidden reasoning tokens even for a
                          one-line reply), unlike a genuine plain-chat model.

THREE CHECKS, in increasing cost/resource weight:
  1. INTENT accuracy -- reuses scripts/eval_intent_routing.py's question set
     and classify_intent() UNCHANGED; only settings.model_by_task["intent"]
     is swapped between runs (via _routed_to() below), restored after.
  2. REWRITE parse-failure rate + a variant-quality sample -- calls
     expand_query() directly (NOT retrieve_stages), so this needs no local
     model (bge-m3/reranker) -- cheap and fast. app/rag/query_rewriter.py
     gained a small additive-only fallback counter for this
     (get_fallback_count() / _reset_fallback_count()); expand_query's
     lru_cache is model-blind (keyed on (query, max_variants) only), so this
     ALWAYS calls expand_query.cache_clear() before/after measuring a model,
     or a repeat query would silently reuse a different model's cached
     result.
  3. REWRITE recall delta -- the real "does retrieval still work" number,
     via retrieve_stages() (reusing scripts/eval_retrieval.py's scoring:
     _row_keys/_score_one/_aggregate). This DOES need bge-m3 + the reranker
     resident, same RAM footprint as scripts/eval_retrieval.py itself --
     gated behind --full-retrieval (off by default) so a plain run of this
     script never pays that cost or risk by accident.

GENERATION tier is NOT re-validated here -- that was scripts/eval_generation_ab.py's
job; this script only prints a reminder to go read its result.

Every candidate/task swap is TEMPORARY: _routed_to() is a context manager
that restores settings.model_by_task to whatever it was (including "not
present at all") on exit, so a crash mid-run can't leave production routing
pointed at an unvalidated model.

COST GATE: prints an estimate and stops unless --yes is passed, same
convention as scripts/eval_generation_ab.py. Phase 3 (--full-retrieval) has
its own separate gate (a print + confirmation is not enough to protect
against an OOM on a small box -- see that flag's help text).

Usage:
    python -m scripts.eval_task_routing [--yes] [--full-retrieval]
        [--candidates gpt-5-nano,openai/gpt-oss-20b]
        [--report out.json]

Run this from a shell where the app's dependencies and .env are available
(same environment as `python -m app.rag.ingest`), and where GROQ_API_KEY is
set (openai/gpt-oss-20b).
"""

from __future__ import annotations

import argparse
import json
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from app.core.config import settings
from app.core.llm import MODEL_ADAPTERS
from app.rag.query_rewriter import expand_query, get_fallback_count, _reset_fallback_count
from app.services.intent_classifier import classify_intent
from scripts.eval_intent_routing import _REAL_MODES
from scripts.eval_intent_routing import _load_questions as _load_intent_questions
from scripts.eval_retrieval import _aggregate, _row_keys, _score_one
from scripts.eval_retrieval import _load_questions as _load_retrieval_questions

INTENT_FILE = Path(__file__).parent / "chat_intent_test_questions.json"
RETRIEVAL_FILE = Path(__file__).parent / "retrieval_eval_questions.json"
REPORT_DIR = Path(__file__).parent.parent / "eval_reports"
DEFAULT_CANDIDATES = ("gpt-5-nano", "openai/gpt-oss-20b")
RETRIEVAL_KS = (1, 3, 5)

# Approximate $/1M-token snapshot (researched at design time, Sept 2026);
# reconfirm before trusting a large run. Not in MODEL_ADAPTERS (that table
# is provider/param FACTS, not pricing).
PRICING = {
    "gpt-5-mini": (0.25, 2.00),
    "gpt-5-nano": (0.05, 0.40),
    "openai/gpt-oss-20b": (0.075, 0.30),
}
_EST_INTENT_IN, _EST_INTENT_OUT = 220, 30
_EST_REWRITE_IN, _EST_REWRITE_OUT = 400, 500


@contextmanager
def _routed_to(task: str, model_name: str | None):
    """Temporarily point settings.model_by_task[task] at model_name (None =
    unset -> falls back to settings.openai_model, i.e. today's baseline).
    Always restores the exact prior state on exit, key-presence included."""
    had_key = task in settings.model_by_task
    prev = settings.model_by_task.get(task)
    if model_name is None:
        settings.model_by_task.pop(task, None)
    else:
        settings.model_by_task[task] = model_name
    try:
        yield
    finally:
        if had_key:
            settings.model_by_task[task] = prev
        else:
            settings.model_by_task.pop(task, None)


# --------------------------------------------------------------------------
# check 1: intent accuracy
# --------------------------------------------------------------------------

def _run_intent(questions: list[dict], model_name: str | None) -> list[dict]:
    results = []
    with _routed_to("intent", model_name):
        for q in questions:
            t0 = time.perf_counter()
            actual = classify_intent(q["text"], has_active_roleplay_session=False).lower()
            latency = time.perf_counter() - t0
            expected = q["expected_intent"]
            is_real = expected in _REAL_MODES
            results.append({
                "text": q["text"], "expected": expected, "actual": actual,
                "is_real_mode": is_real,
                "passed": (actual == expected) if is_real else None,
                "latency_s": latency,
            })
    return results


def _summarize_intent(results: list[dict]) -> dict:
    real = [r for r in results if r["is_real_mode"]]
    passed = sum(1 for r in real if r["passed"])
    return {
        "n_total": len(results), "n_real": len(real), "n_passed": passed,
        "accuracy": passed / len(real) if real else 0.0,
        "avg_latency_s": sum(r["latency_s"] for r in results) / len(results) if results else 0.0,
    }


# --------------------------------------------------------------------------
# check 2: rewrite parse-failure rate + variant sample (no retrieval)
# --------------------------------------------------------------------------

def _run_rewrite_cheap(questions: list[dict], model_name: str | None) -> tuple[list[dict], int]:
    expand_query.cache_clear()
    _reset_fallback_count()
    records = []
    with _routed_to("rewrite", model_name):
        for q in questions:
            t0 = time.perf_counter()
            variants = expand_query(q["query"])
            latency = time.perf_counter() - t0
            records.append({
                "query": q["query"], "lang": q.get("lang"),
                "n_variants": len(variants), "variants": list(variants),
                "latency_s": latency,
            })
    n_fail = get_fallback_count()
    expand_query.cache_clear()  # leave no stale cross-model cache behind
    return records, n_fail


def _summarize_rewrite_cheap(records: list[dict], n_fail: int) -> dict:
    return {
        "n_questions": len(records),
        "n_parse_failures": n_fail,
        "parse_failure_rate": n_fail / len(records) if records else 0.0,
        "avg_n_variants": sum(r["n_variants"] for r in records) / len(records) if records else 0.0,
        "avg_latency_s": sum(r["latency_s"] for r in records) / len(records) if records else 0.0,
    }


# --------------------------------------------------------------------------
# check 3 (heavy, --full-retrieval only): rewrite recall delta
# --------------------------------------------------------------------------

def _run_rewrite_recall(questions: list[dict], model_name: str | None, ks: list[int]) -> list[dict]:
    from app.rag.retriever import retrieve_stages  # deferred: only imported

    expand_query.cache_clear()
    records = []
    with _routed_to("rewrite", model_name):
        for q in questions:
            t0 = time.perf_counter()
            stage_a_chunks, stage_b_chunks = retrieve_stages(q["query"], use_rewrite=True)
            latency = time.perf_counter() - t0
            stage_a, stage_b = _row_keys(stage_a_chunks), _row_keys(stage_b_chunks)
            gold = q["relevant_ids"]
            scorable = bool(q["answerable"] and gold)
            records.append({
                "query": q["query"], "scorable": scorable, "latency_s": latency,
                "metrics": {
                    "candidates": {k: _score_one(stage_a, gold, k) for k in ks},
                    "final": {k: _score_one(stage_b, gold, k) for k in ks},
                },
            })
            print(f"    {q['query'][:60]!r} latency={latency:.1f}s")
    expand_query.cache_clear()
    return records


# --------------------------------------------------------------------------
# cost estimate
# --------------------------------------------------------------------------

def _cost(model: str, n_calls: int, avg_in: int, avg_out: int) -> float:
    price_in, price_out = PRICING.get(model, (0.25, 2.00))
    return n_calls * (avg_in * price_in + avg_out * price_out) / 1e6


def _print_cost_estimate(n_intent: int, n_rewrite: int, candidates: list[str],
                          full_retrieval: bool) -> None:
    print("\n" + "=" * 78)
    print("COST ESTIMATE (approximate; intent has cheap regex shortcuts that skip "
          "the LLM entirely for some questions -- the real run will show fewer "
          "actual LLM calls than this upper bound)")
    print("=" * 78)
    models = ["gpt-5-mini", *candidates]  # baseline + each candidate
    total = 0.0
    for m in models:
        c_intent = _cost(m, n_intent, _EST_INTENT_IN, _EST_INTENT_OUT)
        c_rewrite = _cost(m, n_rewrite, _EST_REWRITE_IN, _EST_REWRITE_OUT)
        total += c_intent + c_rewrite
        print(f"  {m:<24} intent(<={n_intent} calls)=${c_intent:.4f}  "
              f"rewrite({n_rewrite} calls)=${c_rewrite:.4f}")
    print(f"\n  ESTIMATED TOTAL (checks 1+2): ${total:.4f}")
    if full_retrieval:
        print("  --full-retrieval requested: check 3 also loads bge-m3 + the reranker "
              "(same RAM footprint as scripts/eval_retrieval.py) and runs "
              f"{n_rewrite} retrievals PER model ({len(models)} models) -- budget "
              "real time and ~6GB free RAM for that part, separate from the $ above.")


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

def _print_intent_table(summaries: dict[str, dict]) -> None:
    print("\nINTENT  (chat_intent_test_questions.json, "
          f"{summaries['gpt-5-mini']['n_real']} real-mode of "
          f"{summaries['gpt-5-mini']['n_total']} total)")
    print(f"  {'model':<24} {'accuracy':>9} {'avg_lat_s':>10}")
    for m, s in summaries.items():
        print(f"  {m:<24} {s['accuracy']:>9.2%} {s['avg_latency_s']:>10.3f}")


def _print_rewrite_table(summaries: dict[str, dict]) -> None:
    print(f"\nREWRITE parse-failure + cost  ({next(iter(summaries.values()))['n_questions']} questions)")
    print(f"  {'model':<24} {'parse_fail':>10} {'avg_variants':>13} {'avg_lat_s':>10}")
    for m, s in summaries.items():
        print(f"  {m:<24} {s['parse_failure_rate']:>10.2%} {s['avg_n_variants']:>13.2f} "
              f"{s['avg_latency_s']:>10.3f}")


def _print_recall_table(summaries: dict[str, dict], ks: list[int]) -> None:
    print("\nREWRITE recall delta (Stage B / final, post-rerank)")
    hdr = f"  {'model':<24}" + "".join(f"  {'rec@'+str(k):>8}" for k in ks) + f"  {'avg_lat_s':>10}"
    print(hdr)
    for m, (agg, avg_lat) in summaries.items():
        row = f"  {m:<24}" + "".join(f"  {agg[k]['recall']:>8.3f}" for k in ks) + f"  {avg_lat:>10.3f}"
        print(row)


def _verdict(baseline: dict, candidate: dict, metric: str, epsilon: float) -> str:
    delta = candidate[metric] - baseline[metric]
    if delta < -epsilon:
        return f"WORSE ({delta:+.3f}) -- keep gpt-5-mini for this task"
    return f"within noise ({delta:+.3f}) -- candidate holds up"


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m scripts.eval_task_routing",
        description="Validate cheap/fast model candidates for the mechanical tasks.",
    )
    p.add_argument("--candidates", default=",".join(DEFAULT_CANDIDATES))
    p.add_argument("--yes", action="store_true", help="proceed past the cost estimate")
    p.add_argument("--full-retrieval", action="store_true",
                   help="also run check 3 (rewrite recall delta) -- loads bge-m3 + "
                        "the reranker, same RAM footprint as eval_retrieval.py")
    p.add_argument("--report", metavar="OUT.JSON")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    candidates = [c.strip() for c in args.candidates.split(",") if c.strip()]
    missing = [c for c in candidates if c not in MODEL_ADAPTERS]
    if missing:
        raise SystemExit(f"no adapter for {missing} in app.core.llm.MODEL_ADAPTERS -- "
                          "add one before validating it.")

    intent_questions = _load_intent_questions(INTENT_FILE)
    rewrite_questions = _load_retrieval_questions(RETRIEVAL_FILE)

    print("=" * 78)
    print("TASK ROUTING VALIDATION (mechanical tasks: intent, rewrite)")
    print("=" * 78)
    print(f"baseline: gpt-5-mini  |  candidates: {candidates}")
    print(f"intent set: {len(intent_questions)} questions ({INTENT_FILE.name})")
    print(f"rewrite set: {len(rewrite_questions)} questions ({RETRIEVAL_FILE.name})")

    _print_cost_estimate(len(intent_questions), len(rewrite_questions), candidates,
                          args.full_retrieval)
    if not args.yes:
        print("\nStopping before any paid API call. Re-run with --yes to actually validate.")
        return

    models = [None, *candidates]  # None = baseline (gpt-5-mini via fallback)
    model_labels = ["gpt-5-mini", *candidates]

    print("\n--- check 1: intent accuracy ---")
    intent_summaries = {}
    for label, model in zip(model_labels, models):
        print(f"  running intent with model={label}...")
        intent_summaries[label] = _summarize_intent(_run_intent(intent_questions, model))
    _print_intent_table(intent_summaries)

    print("\n--- check 2: rewrite parse-failure rate ---")
    rewrite_cheap_summaries = {}
    for label, model in zip(model_labels, models):
        print(f"  running rewrite (no retrieval) with model={label}...")
        records, n_fail = _run_rewrite_cheap(rewrite_questions, model)
        rewrite_cheap_summaries[label] = _summarize_rewrite_cheap(records, n_fail)
    _print_rewrite_table(rewrite_cheap_summaries)

    recall_summaries = None
    if args.full_retrieval:
        print("\n--- check 3: rewrite recall delta (loading bge-m3 + reranker) ---")
        recall_summaries = {}
        for label, model in zip(model_labels, models):
            print(f"  running full retrieval with model={label}...")
            records = _run_rewrite_recall(rewrite_questions, model, list(RETRIEVAL_KS))
            agg = _aggregate(records, "final", list(RETRIEVAL_KS))
            avg_lat = sum(r["latency_s"] for r in records) / len(records) if records else 0.0
            recall_summaries[label] = (agg, avg_lat)
        _print_recall_table(recall_summaries, list(RETRIEVAL_KS))

    print("\n" + "=" * 78)
    print("VERDICT PER TASK")
    print("=" * 78)
    for label in candidates:
        print(f"\n  {label}:")
        print(f"    intent accuracy:     {_verdict(intent_summaries['gpt-5-mini'], intent_summaries[label], 'accuracy', 0.05)}")
        base_pf = rewrite_cheap_summaries['gpt-5-mini']['parse_failure_rate']
        cand_pf = rewrite_cheap_summaries[label]['parse_failure_rate']
        pf_verdict = "OK" if cand_pf <= base_pf + 0.05 else "WORSE -- keep gpt-5-mini for rewrite"
        print(f"    rewrite parse-fail:  {cand_pf:.2%} vs baseline {base_pf:.2%} -> {pf_verdict}")
        if recall_summaries:
            base_r5 = recall_summaries['gpt-5-mini'][0][5]['recall']
            cand_r5 = recall_summaries[label][0][5]['recall']
            r_delta = cand_r5 - base_r5
            r_verdict = "WORSE -- keep gpt-5-mini for rewrite" if r_delta < -0.05 else "OK"
            print(f"    rewrite recall@5:    {cand_r5:.3f} vs baseline {base_r5:.3f} "
                  f"(Δ{r_delta:+.3f}) -> {r_verdict}")
        else:
            print("    rewrite recall@5:    not measured (pass --full-retrieval)")

    print("\n  GENERATION tier: not re-validated here -- see scripts/eval_generation_ab.py's "
          "own result/recommendation for qa/note/suggest/roleplay.")

    print("\n" + "=" * 78)
    print("CONFIG DIFF (not applied -- edit app/core/config.py yourself)")
    print("=" * 78)
    print("  model_by_task = {}")
    print(f"  # decided {datetime.now().date().isoformat()} from intent "
          f"({len(intent_questions)} Qs) + rewrite ({len(rewrite_questions)} Qs"
          f"{', full retrieval' if args.full_retrieval else ', parse-failure only'}) "
          "via scripts/eval_task_routing.py -- fill in only the tasks whose "
          "verdict above says OK, using the model that verdict was measured against.")

    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        REPORT_DIR.mkdir(exist_ok=True)
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        report_path = REPORT_DIR / f"task_routing_{stamp}.json"
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "baseline": "gpt-5-mini", "candidates": candidates,
        "intent": intent_summaries, "rewrite_parse_failure": rewrite_cheap_summaries,
        "rewrite_recall": (
            {m: {"metrics": {str(k): v for k, v in agg.items()}, "avg_latency_s": lat}
             for m, (agg, lat) in recall_summaries.items()}
            if recall_summaries else None
        ),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nReport written to {report_path}")


if __name__ == "__main__":
    main()
