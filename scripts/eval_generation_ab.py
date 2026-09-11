"""A/B different generation models for Bengali answer QUALITY (not retrieval).

app/rag/generator.py's grounded-answer prompt currently only ever runs through
gpt-5-mini (app/core/llm.get_model()). This script answers a different
question than every other eval script here: given the SAME retrieved context,
does a stronger or differently-sourced model produce a meaningfully better
Bengali answer -- more faithful to the context, better cited, more fluent,
correctly refusing when it should -- and is that gain worth its extra cost
and latency? Unlike scripts/eval_retrieval.py / eval_query_expansion.py, this
CALLS THE GENERATOR, so it spends real API money per model. See the COST GATE
below before running this for real.

ISOLATING THE VARIABLE (the whole point of this script): every candidate
model answers from IDENTICAL context, retrieved/looked-up ONCE per question,
never re-retrieved per model. Two context modes:
  --context retrieved (default) -- production retrieval (app/rag/retriever
      .retrieve_relevant_docs), gated exactly like qa_service.answer_question
      does (citations[0].rerank_score >= settings.min_rerank_score, else empty
      grounding). Realistic, but a model can inherit retrieval's mistakes.
  --context gold -- the labeled gold page(s) fetched straight from Postgres
      (full page text, not chunks), for answerable+labeled questions only;
      empty grounding for unanswerable ones (nothing SHOULD be retrieved for
      those). Isolates pure generation ability from retrieval quality.
All candidates get the SAME app/rag/generator._SYSTEM/_USER prompt in this
first pass -- per-model prompt tuning is a real follow-up, not done here.

CANDIDATE MODELS (see app/core/config.settings.generation_candidates):
  gpt-5-mini  -- baseline, always included, same client/params as production.
  gpt-5       -- larger OpenAI reasoning model. Same client/key as the
                 baseline; assumed to share its "fixed temperature,
                 max_completion_tokens" quirk (see app/rag/generator.py) --
                 if that assumption is wrong the API 400s and this script
                 records it as an error for that model, not a crash.
  openai/gpt-oss-120b -- different PROVIDER (Groq, OpenAI-compatible API,
                 different base_url -- app.core.llm.get_client_for), open-
                 weight, genuinely multilingual. NOTE: confirmed live that
                 this project's GROQ_API_KEY has no access to either Llama
                 model on Groq (404, despite Groq's own docs listing them as
                 self-serve) -- gpt-oss is the confirmed-working Groq
                 candidate instead. It's ALSO a reasoning model under the
                 hood (nonzero hidden reasoning tokens even for a one-line
                 reply) -- same token-budget consideration as gpt-5-mini/
                 gpt-5, just spent through Groq's max_tokens param name.
These are looked up in MODEL_REGISTRY below, which is ONLY the per-model
wiring (client/provider/token-param/pricing) -- kept in this script, not
app/core/config.py, per the task's "isolate provider adapter code to this
script" guidance. config.py holds the POLICY (which names to compare, which
name judges) so changing the comparison set doesn't require code changes here
unless the new name also needs a new registry entry.

JUDGE MODEL (settings.generation_judge_model, default qwen/qwen3.6-27b via
Groq): deliberately NOT a candidate and a different family from every
candidate (OpenAI reasoning models + Meta Llama), to avoid a model favouring
its own or a sibling's answers. The judge is LISTWISE per question -- shown
all candidates' answers at once, ANONYMIZED (labeled A/B/C.., real model names
never sent) and ORDER-RANDOMIZED (seeded per question, so a rerun with the
same seed reproduces the same shuffle) -- so it cannot infer identity from
answer order either. Judge scores are explicitly a FILTER, not the verdict:
LLM-judge scores are known to be noisy and verbosity-biased. See "human sheet"
below for the real tiebreaker.

SCORING, two layers:
  1. Objective checks (no judge, deterministic): citation validity (every
     [১]/[১2]/[1] the answer emits maps to a real provided chunk -- generator
     .py's prompt EXAMPLE uses Bengali digits but its actual context labels
     use plain "[1]"/"[2]"; both are accepted here), refusal correctness
     (when no grounding was given, did it avoid inventing citations instead
     of admitting it doesn't know), an approximate uncited-claim ratio, and
     Bengali-language / length sanity.
  2. LLM-as-judge (see above): faithfulness / relevance / citation
     appropriateness / fluency, 1-5 each, one short justification per model.

HUMAN REVIEW SHEET: a self-contained HTML file, ~15-20 questions, each
model's answer shown NEXT TO the context it was given -- real model names
shown here (the anonymization is only for the automatic judge; there is no
self-preference risk in a human reading their own eval). This is the real
tiebreaker for the fluency call, not the judge's numbers.

REUSED, not reinvented: app/rag/generator._SYSTEM/_USER/_format_context (same
prompt every model gets -- called directly rather than through
generate_answer() because this script needs the raw API response for
per-call token usage/cost, which generate_answer()'s str-only return
deliberately doesn't expose); app/rag/retriever.retrieve_relevant_docs
(production retrieval); app/core/llm.get_client/get_client_for (client
construction, including the Groq override); qa_service's exact gate
arithmetic (reproduced here rather than imported, since qa_service.
answer_question() is hardwired to the single production model).

COST GATE: prints an estimate (questions x models x rough avg tokens x
researched-at-design-time pricing -- reconfirm current rates before trusting
this for a large run) and STOPS. Nothing calls a paid API until you pass
--yes. Use --limit N for a small paid smoke test and --models to drop a
candidate.

Usage:
    python -m scripts.eval_generation_ab [questions.json]
        [--context retrieved|gold] [--models name1,name2] [--limit N]
        [--yes] [--report out.json] [--sheet out.html] [--seed N]

Run this from a shell where the app's dependencies and .env are available
(same environment as `python -m app.rag.ingest`), and where GROQ_API_KEY is
set if openai/gpt-oss-120b or the default judge is in play.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from sqlalchemy.orm import joinedload

from app.core.config import settings
from app.core.llm import MODEL_ADAPTERS
from app.core.llm import _client_for_provider as _client_for
from app.db.models import Article, Page
from app.db.session import SessionLocal
from app.rag.generator import _SYSTEM, _USER, _format_context
from app.rag.query_rewriter import _looks_bengali
from app.rag.retriever import retrieve_relevant_docs
from app.schemas.query import Citation
from scripts.eval_responses import _FONT_STACK

DEFAULT_FILE = Path(__file__).parent / "retrieval_eval_questions.json"
REPORT_DIR = Path(__file__).parent.parent / "eval_reports"
DEFAULT_SEED = 20260906
SHEET_SIZE = 18

# recall@k-style epsilon, but for a 1-5 judge score: a candidate's mean score
# has to beat the baseline's by more than this to count as a real gain, not
# judge noise.
JUDGE_SCORE_EPSILON = 0.3


# --------------------------------------------------------------------------
# model registry -- pricing + eval-tuned params ONLY. provider/client wiring
# is NOT duplicated here: it's sourced from app.core.llm.MODEL_ADAPTERS (the
# same table app/core/llm.complete() uses for production per-task routing),
# via _provider_for() below, so the eval and production can never silently
# diverge on which provider a model needs. Pricing is an approximate
# $/1M-token snapshot researched at design time (Sept 2026); reconfirm
# before a large paid run.
# --------------------------------------------------------------------------

def _provider_for(name: str) -> str:
    return MODEL_ADAPTERS[name].provider if name in MODEL_ADAPTERS else "openai"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    provider: str  # "openai" | "groq" -- see _provider_for()
    extra_params: dict
    price_in_per_mtok: float
    price_out_per_mtok: float


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "gpt-5-mini": ModelSpec(
        name="gpt-5-mini", provider=_provider_for("gpt-5-mini"),
        # gpt-5-mini/gpt-5 are reasoning models: fixed temperature (omit it),
        # max_completion_tokens (not max_tokens) -- see app/rag/generator.py.
        extra_params={"max_completion_tokens": 1500},
        price_in_per_mtok=0.25, price_out_per_mtok=2.00,
    ),
    "gpt-5": ModelSpec(
        name="gpt-5", provider=_provider_for("gpt-5"),
        extra_params={"max_completion_tokens": 1500},
        price_in_per_mtok=1.25, price_out_per_mtok=10.00,
    ),
    "openai/gpt-oss-120b": ModelSpec(
        name="openai/gpt-oss-120b", provider=_provider_for("openai/gpt-oss-120b"),
        # Groq-hosted, but ALSO a reasoning model under the hood (confirmed
        # live: nonzero completion_tokens_details.reasoning_tokens even for a
        # one-line reply) -- give it real headroom, same reasoning as
        # gpt-5-mini/gpt-5's token_budget, just through Groq's max_tokens name.
        extra_params={"max_tokens": 1500, "temperature": 0.3},
        price_in_per_mtok=0.15, price_out_per_mtok=0.60,
    ),
}

JUDGE_SPEC = ModelSpec(
    name=settings.generation_judge_model, provider=_provider_for(settings.generation_judge_model),
    extra_params={"max_tokens": 900, "temperature": 0.0,
                  "response_format": {"type": "json_object"}},
    price_in_per_mtok=0.60, price_out_per_mtok=3.00,
)


def _resolve_specs(names: list[str]) -> list[ModelSpec]:
    baseline = settings.openai_model
    ordered = [baseline] + [n for n in names if n != baseline]
    missing = [n for n in ordered if n not in MODEL_REGISTRY]
    if missing:
        raise SystemExit(
            f"no adapter config for {missing} -- add an entry to MODEL_REGISTRY "
            f"in this script (provider, token/temperature params, pricing)."
        )
    return [MODEL_REGISTRY[n] for n in ordered]


# --------------------------------------------------------------------------
# input
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
            "query": q["query"],
            "answerable": bool(q["answerable"]),
            "relevant_ids": (q.get("relevant_ids") or []) if q["answerable"] else [],
            "lang": q.get("lang", "unknown"),
        })
    return out


# --------------------------------------------------------------------------
# context: retrieve/look up ONCE per question, shared by every model
# --------------------------------------------------------------------------

def _fetch_gold_citations(session, row_keys: list[str]) -> list[Citation]:
    """Full page/article text for labeled gold ids -- same book/chapter/
    source_db extraction app/rag/ingest.py uses, so a Citation built here
    means the same thing one built from Chroma metadata would."""
    out = []
    for rk in row_keys:
        prefix, _, id_str = rk.partition("_")
        row_id = int(id_str)
        if prefix == "page":
            p = (session.query(Page)
                 .options(joinedload(Page.book), joinedload(Page.chapter))
                 .get(row_id))
            if p is None:
                continue
            out.append(Citation(
                book=p.book.name if p.book else "Unknown",
                chapter=p.chapter.name if p.chapter else "Unknown",
                source_db=p.source_db or "unknown", content=p.content,
            ))
        elif prefix == "article":
            a = session.query(Article).get(row_id)
            if a is None:
                continue
            out.append(Citation(book=a.title, chapter="প্রবন্ধ",
                                 source_db="articles", content=a.content))
    return out


def _get_context(q: dict, mode: str, session) -> tuple[list[Citation], float | None]:
    """Return (grounding citations, top_rerank_score or None). Called ONCE
    per question regardless of how many models are being compared."""
    if mode == "gold":
        if not q["answerable"] or not q["relevant_ids"]:
            return [], None
        return _fetch_gold_citations(session, q["relevant_ids"]), None

    citations = retrieve_relevant_docs(q["query"])
    # Exactly qa_service.answer_question()'s gate -- so this harness's
    # answerable/refusal behaviour matches what a real user would see.
    relevant = bool(citations) and citations[0].rerank_score >= settings.min_rerank_score
    top = citations[0].rerank_score if citations else None
    return (citations if relevant else []), top


# --------------------------------------------------------------------------
# per-model generation call
# --------------------------------------------------------------------------

def _call_model(spec: ModelSpec, query: str, citations: list[Citation]) -> dict:
    """One answer + real usage/cost/latency, or an error record. Reuses
    generator.py's exact prompt pieces (see module docstring for why this
    doesn't go through generate_answer() itself)."""
    client = _client_for(spec.provider)
    t0 = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=spec.name,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": _USER.format(
                    context=_format_context(citations), query=query
                )},
            ],
            **spec.extra_params,
        )
        latency = time.perf_counter() - t0
        answer = response.choices[0].message.content or ""
        usage = response.usage
        in_tok = getattr(usage, "prompt_tokens", 0) or 0
        out_tok = getattr(usage, "completion_tokens", 0) or 0
        cost = (in_tok * spec.price_in_per_mtok + out_tok * spec.price_out_per_mtok) / 1e6
        return {"model": spec.name, "answer": answer, "latency_s": latency,
                "in_tokens": in_tok, "out_tokens": out_tok, "cost_usd": cost,
                "error": None}
    except Exception as e:
        return {"model": spec.name, "answer": "", "latency_s": time.perf_counter() - t0,
                "in_tokens": 0, "out_tokens": 0, "cost_usd": 0.0, "error": str(e)}


# --------------------------------------------------------------------------
# objective checks (no judge)
# --------------------------------------------------------------------------

_BENGALI_DIGIT_TRANS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
# generator.py's prompt EXAMPLE shows Bengali-digit brackets ([১]/[২]) but
# _format_context() actually labels chunks with plain "[1]"/"[2]" -- accept
# whichever digit style a model actually produced.
_CITATION_RE = re.compile(r"\[([০-৯0-9]+)\]")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[।.!?])\s+")


def _citation_numbers(answer: str) -> list[int]:
    return [int(m.translate(_BENGALI_DIGIT_TRANS)) for m in _CITATION_RE.findall(answer)]


def _uncited_claim_ratio(answer: str) -> float:
    """Cheap heuristic, not real NLP: share of non-trivial sentences with no
    citation marker anywhere in them. Approximate by design."""
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(answer) if s.strip()]
    if not sentences:
        return 0.0
    uncited = [s for s in sentences if not _CITATION_RE.search(s) and len(s.split()) >= 4]
    return len(uncited) / len(sentences)


def _objective_checks(answer: str, citations: list[Citation]) -> dict:
    nums = _citation_numbers(answer)
    n = len(citations)
    invalid = [x for x in nums if x < 1 or x > n]
    words = answer.split()
    # refusal_correct is only meaningful when NO grounding was given -- there
    # citing anything at all is invention by definition. n/a (None) otherwise.
    refusal_correct = (len(invalid) == 0 and len(nums) == 0) if n == 0 else None
    return {
        "n_citations_used": len(nums),
        "n_invalid_citations": len(invalid),
        "citation_valid": len(invalid) == 0,
        "refusal_correct": refusal_correct,
        "uncited_claim_ratio": round(_uncited_claim_ratio(answer), 3),
        "is_bengali": _looks_bengali(answer) if answer else False,
        "n_words": len(words),
        "empty": len(words) == 0,
    }


# --------------------------------------------------------------------------
# LLM-as-judge -- listwise, anonymized, order-randomized
# --------------------------------------------------------------------------

_JUDGE_SYSTEM = """You are an impartial evaluator of Bengali RAG (retrieval-augmented \
generation) answers. You will see a user's question, the source context every \
candidate answer was grounded in, and several candidate answers labeled with \
single letters in RANDOM order. You do NOT know which system produced which \
answer -- judge each purely on its own merit, never by guessing its source.

For EACH candidate, score 1-5 (integers only) on:
  faithfulness - is every claim supported by the given context? No \
hallucinated facts, no invented or misattributed citations.
  relevance    - does it actually answer the question asked?
  citation     - are the [N] citation markers present, correct, and not \
misused (missing where needed, or citing a chunk that doesn't say that)?
  fluency      - is the Bengali natural, correct, and well-formed -- not \
awkward, not Banglish, not a stiff literal translation?

Then give ONE short (under 20 words) justification per candidate, naming its \
weakest scored dimension.

Respond with ONLY a JSON object, one key per candidate label, no other text:
{{"A": {{"faithfulness": int, "relevance": int, "citation": int, "fluency": int, \
"justification": str}}, "B": {{...}}, ...}}"""

_JUDGE_USER = """প্রশ্ন (question): {query}

উৎস প্রসঙ্গ (source context -- every candidate was given exactly this):
{context}

প্রার্থী উত্তরসমূহ (candidate answers, unlabeled source, random order):

{candidates}"""


def _judge_one(session_specs: list[dict], query: str, citations: list[Citation],
                rng: random.Random) -> tuple[dict | None, dict]:
    """session_specs: [{"model": name, "answer": text}, ...] for one question.
    Returns (parsed_scores_by_real_model_name_or_None, call_record)."""
    labels = [chr(ord("A") + i) for i in range(len(session_specs))]
    shuffled = session_specs[:]
    rng.shuffle(shuffled)
    label_to_model = dict(zip(labels, (s["model"] for s in shuffled)))
    candidates_block = "\n\n".join(
        f"[{label}]\n{s['answer']}" for label, s in zip(labels, shuffled)
    )

    client = _client_for(JUDGE_SPEC.provider)
    t0 = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=JUDGE_SPEC.name,
            messages=[
                {"role": "system", "content": _JUDGE_SYSTEM.format(n=len(labels))},
                {"role": "user", "content": _JUDGE_USER.format(
                    query=query, context=_format_context(citations),
                    candidates=candidates_block,
                )},
            ],
            **JUDGE_SPEC.extra_params,
        )
        latency = time.perf_counter() - t0
        raw = response.choices[0].message.content.strip()
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        parsed = json.loads(raw)
        by_model = {label_to_model[label]: scores for label, scores in parsed.items()
                    if label in label_to_model}
        usage = response.usage
        in_tok = getattr(usage, "prompt_tokens", 0) or 0
        out_tok = getattr(usage, "completion_tokens", 0) or 0
        cost = (in_tok * JUDGE_SPEC.price_in_per_mtok + out_tok * JUDGE_SPEC.price_out_per_mtok) / 1e6
        return by_model, {"latency_s": latency, "in_tokens": in_tok, "out_tokens": out_tok,
                           "cost_usd": cost, "error": None}
    except Exception as e:
        return None, {"latency_s": time.perf_counter() - t0, "in_tokens": 0,
                       "out_tokens": 0, "cost_usd": 0.0, "error": str(e)}


# --------------------------------------------------------------------------
# cost estimate (printed BEFORE any paid call)
# --------------------------------------------------------------------------

# Rough averages for the estimate only -- top_k=5 chunks x ~900 chars each
# (settings.top_k / chunker.CHUNK_SIZE), Bengali script, plus prompt overhead.
# Reasoning models spend additional (billed) hidden tokens on top of the
# visible answer; non-reasoning Groq models don't.
_EST_CONTEXT_TOKENS = 2000
_EST_QUERY_TOKENS = 30
_EST_ANSWER_TOKENS_REASONING = 800
_EST_ANSWER_TOKENS_PLAIN = 350
_EST_JUDGE_OVERHEAD_TOKENS = 1200
_EST_JUDGE_OUT_TOKENS = 500


def _print_cost_estimate(n_questions: int, n_judged: int, specs: list[ModelSpec]) -> float:
    print("\n" + "=" * 78)
    print("COST ESTIMATE (approximate -- pricing researched at design time, "
          "reconfirm before trusting a large run)")
    print("=" * 78)
    print(f"  {n_questions} questions x {len(specs)} model(s) for generation, "
          f"{n_judged} judged (answerable + labeled)")
    print(f"\n  {'model':<28} {'calls':>6} {'avg_in':>7} {'avg_out':>8} {'est_$':>9}")
    total = 0.0
    for spec in specs:
        avg_out = _EST_ANSWER_TOKENS_REASONING if spec.provider == "openai" else _EST_ANSWER_TOKENS_PLAIN
        avg_in = _EST_CONTEXT_TOKENS + _EST_QUERY_TOKENS
        cost = n_questions * (avg_in * spec.price_in_per_mtok + avg_out * spec.price_out_per_mtok) / 1e6
        total += cost
        print(f"  {spec.name:<28} {n_questions:>6} {avg_in:>7} {avg_out:>8} {cost:>9.4f}")

    judge_avg_out_per_model = sum(
        _EST_ANSWER_TOKENS_REASONING if s.provider == "openai" else _EST_ANSWER_TOKENS_PLAIN
        for s in specs
    )
    judge_in = _EST_CONTEXT_TOKENS + judge_avg_out_per_model + _EST_JUDGE_OVERHEAD_TOKENS
    judge_cost = n_judged * (judge_in * JUDGE_SPEC.price_in_per_mtok
                              + _EST_JUDGE_OUT_TOKENS * JUDGE_SPEC.price_out_per_mtok) / 1e6
    total += judge_cost
    print(f"  {JUDGE_SPEC.name + ' (judge)':<28} {n_judged:>6} {judge_in:>7} "
          f"{_EST_JUDGE_OUT_TOKENS:>8} {judge_cost:>9.4f}")
    print(f"\n  ESTIMATED TOTAL: ${total:.3f}")
    print("  (real cost is computed from actual token usage and reported after the run)")
    return total


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------

def _run(questions: list[dict], specs: list[ModelSpec], context_mode: str, seed: int) -> dict:
    session = SessionLocal() if context_mode == "gold" else None
    # order randomization is seeded per-question below (random.Random(seed + i)),
    # so there is no run-level rng here.
    per_query: list[dict] = []
    try:
        for i, q in enumerate(questions):
            citations, top_rerank = _get_context(q, context_mode, session)
            calls = [_call_model(spec, q["query"], citations) for spec in specs]
            for c in calls:
                c["objective"] = _objective_checks(c["answer"], citations) if not c["error"] else None

            judge_scores, judge_meta = (None, {"error": "no scorable answers"})
            scorable = q["answerable"] and bool(q["relevant_ids"]) and bool(citations)
            if scorable:
                ok_calls = [c for c in calls if not c["error"]]
                if len(ok_calls) >= 2:
                    judge_scores, judge_meta = _judge_one(
                        [{"model": c["model"], "answer": c["answer"]} for c in ok_calls],
                        q["query"], citations, random.Random(seed + i),
                    )

            rec = {
                "query": q["query"], "lang": q["lang"], "answerable": q["answerable"],
                "relevant_ids": q["relevant_ids"], "scorable": scorable,
                "n_context_chunks": len(citations), "top_rerank_score": top_rerank,
                "calls": calls, "judge_scores": judge_scores, "judge_meta": judge_meta,
            }
            per_query.append(rec)

            tags = " ".join(
                f"{c['model']}={'ERR' if c['error'] else ('OK' if c['objective']['citation_valid'] else 'BADCITE')}"
                for c in calls
            )
            print(f"  [{i+1}/{len(questions)}] {tags}  {q['query'][:60]!r}")
    finally:
        if session is not None:
            session.close()
    return {"per_query": per_query}


# --------------------------------------------------------------------------
# summary table + recommendation
# --------------------------------------------------------------------------

def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _summarize(per_query: list[dict], specs: list[ModelSpec]) -> dict:
    out = {}
    for spec in specs:
        calls = [r for q in per_query for r in q["calls"] if r["model"] == spec.name]
        ok = [c for c in calls if not c["error"]]
        objs = [c["objective"] for c in ok]
        judge_rows = [
            q["judge_scores"][spec.name] for q in per_query
            if q["judge_scores"] and spec.name in q["judge_scores"]
        ]
        out[spec.name] = {
            "n_calls": len(calls),
            "n_errors": len(calls) - len(ok),
            "citation_valid_rate": _mean([1.0 if o["citation_valid"] else 0.0 for o in objs]),
            "refusal_correct_rate": _mean([
                1.0 if o["refusal_correct"] else 0.0 for o in objs if o["refusal_correct"] is not None
            ]),
            "n_refusal_scored": sum(1 for o in objs if o["refusal_correct"] is not None),
            "bengali_rate": _mean([1.0 if o["is_bengali"] else 0.0 for o in objs]),
            "avg_uncited_claim_ratio": _mean([o["uncited_claim_ratio"] for o in objs]),
            "avg_latency_s": _mean([c["latency_s"] for c in ok]),
            "avg_cost_usd": _mean([c["cost_usd"] for c in ok]),
            "total_cost_usd": sum(c["cost_usd"] for c in ok),
            "n_judged": len(judge_rows),
            "judge_faithfulness": _mean([j["faithfulness"] for j in judge_rows]),
            "judge_relevance": _mean([j["relevance"] for j in judge_rows]),
            "judge_citation": _mean([j["citation"] for j in judge_rows]),
            "judge_fluency": _mean([j["fluency"] for j in judge_rows]),
        }
    return out


def _print_summary_table(summary: dict, specs: list[ModelSpec]) -> None:
    print("\n" + "=" * 100)
    print("SUMMARY  (judge scores are 1-5, a FILTER not a verdict -- see human sheet)")
    print("=" * 100)
    hdr = (f"  {'model':<24} {'cite_ok':>7} {'refuse_ok':>9} {'bengali':>7} "
           f"{'faith':>6} {'rel':>5} {'cite_j':>6} {'flnc':>5} "
           f"{'lat_s':>6} {'$/ans':>7} {'total_$':>8}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for spec in specs:
        s = summary[spec.name]
        print(f"  {spec.name:<24} {s['citation_valid_rate']:>7.2f} "
              f"{s['refusal_correct_rate']:>9.2f} {s['bengali_rate']:>7.2f} "
              f"{s['judge_faithfulness']:>6.2f} {s['judge_relevance']:>5.2f} "
              f"{s['judge_citation']:>6.2f} {s['judge_fluency']:>5.2f} "
              f"{s['avg_latency_s']:>6.2f} {s['avg_cost_usd']:>7.4f} {s['total_cost_usd']:>8.3f}")
        if s["n_errors"]:
            print(f"    ** {s['n_errors']}/{s['n_calls']} call(s) errored -- see --report for messages **")


def _recommend(summary: dict, specs: list[ModelSpec], n_judged: int) -> list[str]:
    baseline = specs[0].name
    base = summary[baseline]
    base_overall = _mean([base["judge_faithfulness"], base["judge_relevance"],
                           base["judge_citation"], base["judge_fluency"]])
    lines = [f"baseline {baseline}: overall judge={base_overall:.2f}, "
             f"citation_valid={base['citation_valid_rate']:.2f}, "
             f"avg_cost=${base['avg_cost_usd']:.4f}, avg_latency={base['avg_latency_s']:.2f}s"]

    winner = baseline
    for spec in specs[1:]:
        s = summary[spec.name]
        overall = _mean([s["judge_faithfulness"], s["judge_relevance"],
                          s["judge_citation"], s["judge_fluency"]])
        d_overall = overall - base_overall
        cost_mult = (s["avg_cost_usd"] / base["avg_cost_usd"]) if base["avg_cost_usd"] else float("inf")
        contradiction = (d_overall > JUDGE_SCORE_EPSILON
                          and s["citation_valid_rate"] < base["citation_valid_rate"] - 0.05)
        if contradiction:
            verdict = ("CONTRADICTION: higher judge score but WORSE citation validity "
                       "than baseline -- don't declare this a clean win")
        elif d_overall > JUDGE_SCORE_EPSILON:
            verdict = f"GAIN over baseline (+{d_overall:.2f}), at {cost_mult:.1f}x the cost/answer"
            winner = spec.name
        else:
            verdict = "within judge-score noise of baseline -- not a measurable quality gain"
        lines.append(f"{spec.name}: overall judge={overall:.2f} (Δ{d_overall:+.2f} vs baseline), "
                      f"citation_valid={s['citation_valid_rate']:.2f}, "
                      f"avg_cost=${s['avg_cost_usd']:.4f} ({cost_mult:.1f}x baseline) -> {verdict}")

    if winner == baseline:
        lines.append(f"\n  RECOMMENDATION: keep {baseline} -- no candidate showed a measurable "
                      "quality gain worth its extra cost/latency.")
    else:
        lines.append(f"\n  RECOMMENDATION: {winner} shows a measurable gain. Read the human "
                      "review sheet before switching -- judge scores are a filter, not the verdict.")
    lines.append(f"  (based on {n_judged} judged question(s) -- see the eval-set size note above)")
    return lines


# --------------------------------------------------------------------------
# human review sheet
# --------------------------------------------------------------------------

def _render_sheet(per_query: list[dict], specs: list[ModelSpec]) -> str:
    import html as html_mod

    rows = [q for q in per_query if q["scorable"]][:SHEET_SIZE]
    cards = []
    for q in rows:
        answers_html = "".join(
            f"<div class='answer-col'><h4>{html_mod.escape(c['model'])}</h4>"
            f"<div class='answer'>{html_mod.escape(c['answer']) if not c['error'] else '<em>ERROR: ' + html_mod.escape(c['error']) + '</em>'}</div></div>"
            for c in q["calls"]
        )
        cards.append(f"""
        <div class="card">
          <div class="query">{html_mod.escape(q['query'])}</div>
          <div class="answers">{answers_html}</div>
        </div>""")

    return f"""<!DOCTYPE html>
<html lang="bn">
<head><meta charset="utf-8"><title>Generation A/B -- human review</title>
<style>
  body {{ font-family: {_FONT_STACK}; background: #f4f5f7; color: #1a1a1a; margin: 0; padding: 2rem; }}
  .card {{ background: #fff; border-radius: 8px; padding: 1rem 1.25rem; margin-bottom: 1.25rem;
           box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
  .query {{ font-weight: 700; font-size: 1.05rem; margin-bottom: 0.75rem; }}
  .answers {{ display: flex; gap: 1rem; flex-wrap: wrap; }}
  .answer-col {{ flex: 1; min-width: 260px; }}
  .answer-col h4 {{ margin: 0 0 0.35rem; color: #444; font-size: 0.85rem; }}
  .answer {{ white-space: pre-wrap; line-height: 1.6; background: #fafafa; border-radius: 4px;
             padding: 0.6rem 0.8rem; font-size: 0.95rem; }}
</style></head>
<body>
  <h1>Generation A/B -- human review ({len(rows)} question(s))</h1>
  <p>Real model names shown (this sheet is for you, not the judge). Judge scores are a filter --
     this is the real tiebreaker for fluency/naturalness.</p>
  {"".join(cards)}
</body></html>"""


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m scripts.eval_generation_ab",
        description="A/B generation models for Bengali answer quality (spends API money).",
    )
    p.add_argument("questions", nargs="?", default=str(DEFAULT_FILE))
    p.add_argument("--context", choices=("retrieved", "gold"), default="retrieved")
    p.add_argument("--models", default=",".join(settings.generation_candidates),
                   help="comma-separated candidate names (baseline is always included)")
    p.add_argument("--limit", type=int, default=None,
                   help="seeded random sample of N questions instead of the full set")
    p.add_argument("--yes", action="store_true",
                   help="proceed past the cost estimate and actually call the APIs")
    p.add_argument("--report", metavar="OUT.JSON")
    p.add_argument("--sheet", metavar="OUT.HTML")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    path = Path(args.questions)
    questions = _load_questions(path)

    if args.limit is not None and args.limit < len(questions):
        questions = random.Random(args.seed).sample(questions, args.limit)

    names = [n.strip() for n in args.models.split(",") if n.strip()]
    specs = _resolve_specs(names)

    n_ans = sum(1 for q in questions if q["answerable"])
    n_labeled = sum(1 for q in questions if q["answerable"] and q["relevant_ids"])

    print("=" * 78)
    print("GENERATION MODEL A/B")
    print("=" * 78)
    print(f"dataset: {path}  ({len(questions)} questions: {n_ans} answerable, "
          f"{n_labeled} labeled -- these are what the judge scores)")
    print(f"context mode: {args.context}")
    print(f"models: {[s.name for s in specs]}  |  judge: {JUDGE_SPEC.name}")
    if len(questions) < 30 or n_labeled < 20:
        print("** CONFIDENCE: small eval set -- any recommendation below is provisional. **")

    _print_cost_estimate(len(questions), n_labeled, specs)

    if not args.yes:
        print("\nStopping before any paid API call. Re-run with --yes to actually generate, "
              "or --limit N for a smaller paid smoke test first.")
        return

    result = _run(questions, specs, args.context, args.seed)
    summary = _summarize(result["per_query"], specs)
    _print_summary_table(summary, specs)

    rec_lines = _recommend(summary, specs, n_labeled)
    print("\n" + "=" * 78)
    print("RECOMMENDATION")
    print("=" * 78)
    for line in rec_lines:
        print(" ", line)

    baseline = specs[0].name
    print("\n  CONFIG DIFF (not applied -- edit app/core/config.py yourself):")
    print(f"    - openai_model: str = {baseline!r}")
    print(f"    + openai_model: str = <see recommendation above>"
          f"  # decided {datetime.now().date().isoformat()} from a "
          f"{len(questions)}-question eval ({n_labeled} labeled) via "
          "scripts/eval_generation_ab.py")
    print("\n  REMINDER: switching openai_model does NOT change the retrieval gate "
          "(min_rerank_score is computed pre-generation) -- but the grounded prompt "
          "may need light re-tuning for a new model, and spot-check a few real "
          "end-to-end answers before trusting this in production.")

    REPORT_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    sheet_path = Path(args.sheet) if args.sheet else REPORT_DIR / f"generation_ab_sheet_{stamp}.html"
    sheet_path.parent.mkdir(parents=True, exist_ok=True)
    sheet_path.write_text(_render_sheet(result["per_query"], specs), encoding="utf-8")
    print(f"\nHuman review sheet written to {sheet_path}")

    report_path = Path(args.report) if args.report else REPORT_DIR / f"generation_ab_{stamp}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": str(path), "context_mode": args.context,
        "models": [s.name for s in specs], "judge_model": JUDGE_SPEC.name,
        "n_questions": len(questions), "n_answerable": n_ans, "n_labeled": n_labeled,
        "summary": summary, "recommendation": rec_lines, "per_query": result["per_query"],
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Report written to {report_path}")


if __name__ == "__main__":
    main()
