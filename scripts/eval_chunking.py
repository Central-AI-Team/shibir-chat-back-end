"""Empirically compare chunking configs against the labeled retrieval eval set.

WHY this script exists: CHUNK_SIZE=900 / CHUNK_OVERLAP=150 (app/rag/chunker.py)
was a reasonable starting guess, never measured. This is the tradeoff it
encodes and never tested:

  SMALLER chunks -> the embedding represents one focused idea -> more precise
    retrieval, but each chunk carries less surrounding context.
  LARGER chunks -> more context per chunk, but the embedding becomes an
    "average" of several ideas, which dilutes the signal and can hurt
    retrieval precision.

There is no theoretical best size for a given corpus/embedder -- it is
empirical. This script re-chunks the cleaned, published corpus at several
configs, embeds each into its OWN temporary Chroma collection, and scores
every collection with the exact same metrics as scripts/eval_retrieval.py
(hit_rate@k, recall@k, MRR, both pre- and post-rerank), fetch_k/top_k held
fixed at production values so the comparison is apples-to-apples.

REUSED, not reinvented:
  - app/rag/chunker.chunk_text -- the real splitter, called with each
    config's (chunk_size, overlap). Confirmed it already accepts both as
    params (see chunker.py); no change needed there.
  - app/rag/embedder.embed_texts -- same embedding call ingest.py uses.
  - app/rag/retriever.retrieve_stages -- now takes an optional
    collection_name (added in this change, defaulting to None = production;
    every existing caller is unaffected). That is the ONLY way this script
    reaches a temp collection instead of production.
  - scripts/eval_retrieval's own scoring primitives (_load_questions,
    _row_keys, _score_one, _aggregate) -- same recall/hit-rate/MRR math, so a
    number here means the same thing it means over there.

NON-DESTRUCTIVE: production (settings.chroma_collection_name) is never
opened for writing here -- only get_named_collection()/delete_collection()
for temp names like "chunk_eval_900_150" (see app/rag/chroma_client.py).
Postgres is read-only from this script.

COST GUARDRAIL: re-embedding the whole corpus once PER config is the
expensive part (bge-m3 on CPU -- see app/rag/embedder.py). Without --sample,
this script prints a chunk-count + measured-throughput time estimate and
STOPS; pass --apply to actually run it. --sample N builds each index from
all gold pages + N random distractor pages instead, for a fast first pass
that does not need --apply -- but a small distractor pool makes recall look
better than production, where every other page is a potential distractor.
Treat --sample numbers as directional; make the real call from a full-corpus
run (or the largest N you can afford).

RAM: the scoring phase holds bge-m3 and bge-reranker-v2-m3 resident at the
same time (~4.5 GB of weights + torch/Chroma overhead). Budget ~6 GB free.
A 7-8 GB box swap-thrashes here and the run gets OOM-killed mid-scoring --
run this where there is real headroom, or a GPU.

Usage:
    python -m scripts.eval_chunking [questions.json] [--sample N] [--apply]
        [--configs 900_150,700_120] [--k 1,3,5,10] [--no-rewrite]
        [--report out.json] [--cleanup] [--seed N]

Run from a shell where the app's dependencies and .env are available (same
environment as `python -m app.rag.ingest`).
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from sqlalchemy.orm import joinedload

from app.core.config import settings
from app.db.models import Article, ContentStatus, Page
from app.db.session import SessionLocal
from app.rag.chroma_client import delete_collection, get_named_collection
from app.rag.chunker import build_document, chunk_text
from app.rag.embedder import embed_texts
from app.rag.retriever import retrieve_stages
from scripts.eval_retrieval import _aggregate, _load_questions, _row_keys, _score_one

DEFAULT_FILE = Path(__file__).parent / "retrieval_eval_questions.json"
REPORT_DIR = Path(__file__).parent.parent / "eval_reports"
DEFAULT_KS = (1, 3, 5, 10)
COLLECTION_PREFIX = "chunk_eval_"
BATCH_SIZE = 64            # chunks per embed/upsert call, matches ingest.py
CALIBRATION_CHUNKS = 32    # sample size for the live throughput measurement

# --------------------------------------------------------------------------
# configs to test -- edit this list to try others. Overlap kept ~15-20% of
# chunk_size, matching the ratio of the current 900/150 baseline (~17%).
# --------------------------------------------------------------------------
CONFIGS = [
    {"name": "900_150", "chunk_size": 900, "overlap": 150, "baseline": True},
    {"name": "700_120", "chunk_size": 700, "overlap": 120, "baseline": False},
    {"name": "500_100", "chunk_size": 500, "overlap": 100, "baseline": False},
    {"name": "350_70", "chunk_size": 350, "overlap": 70, "baseline": False},
]
# A sentence-window variant (chunk_size capped, overlap forced to 0 so no
# chunk ever straddles a sentence boundary) was in scope as an optional
# extra. Skipped here: chunk_text's overlap is applied as a trailing
# character slice of the previous chunk, which CAN land mid-sentence -- doing
# this properly needs a real "carry whole trailing sentences" mode in
# chunker.py, not a config tweak, so it did not qualify as "cheap." Add it as
# a fifth CONFIGS entry once that splitting mode exists.


@dataclass
class CorpusRow:
    prefix: str        # "page" | "article"
    row_id: int
    book: str
    chapter: str
    body: str
    source_db: str


@dataclass
class ConfigResult:
    name: str
    chunk_size: int
    overlap: int
    collection: str
    n_pages: int
    n_articles: int
    n_chunks: int
    avg_chunks_per_page: float
    build_seconds: float
    n_scored: int = 0
    metrics: dict = field(default_factory=dict)   # {stage: {k: {...}}}


# --------------------------------------------------------------------------
# corpus loading (read-only)
# --------------------------------------------------------------------------

def _gold_page_ids(questions: list[dict]) -> set[int]:
    ids = set()
    for q in questions:
        for rk in q.get("relevant_ids") or []:
            if rk.startswith("page_"):
                ids.add(int(rk.split("_", 1)[1]))
    return ids


def _load_corpus_rows(session, sample: int | None, seed: int, gold_ids: set[int]) -> list[CorpusRow]:
    """Published, non-excluded pages + published articles -- the same set
    app/rag/ingest.py would (re-)embed, minus its "needs (re-)embedding"
    filter (we always rebuild fresh here). Deterministic sampling: sorted by
    id, then random.Random(seed).sample() over the distractor pool only --
    every gold page is always included.
    """
    pages_q = (
        session.query(Page)
        .options(joinedload(Page.book), joinedload(Page.chapter))
        .filter(Page.status == ContentStatus.published)
        .filter(Page.excluded_from_rag.is_(False))
        .order_by(Page.id)
        .all()
    )
    # Articles are omitted entirely from --sample mode: the gold set has no
    # article-level labels, so they only ever act as distractors, and all 234
    # of them (see corpus-cleanup notes) would swamp a "fast first pass" meant
    # to be cheap. Full-corpus runs include them for production fidelity.
    articles_q = (
        session.query(Article)
        .filter(Article.status == ContentStatus.published)
        .order_by(Article.id)
        .all()
    ) if sample is None else []

    if sample is not None:
        gold_pages = [p for p in pages_q if p.id in gold_ids]
        found_gold = {p.id for p in gold_pages}
        missing_gold = gold_ids - found_gold
        if missing_gold:
            print(f"  WARNING: {len(missing_gold)} gold page id(s) not found in the "
                  f"published/non-excluded pool (excluded, unpublished, or deleted): "
                  f"{sorted(missing_gold)[:10]}")
        distractor_pool = [p for p in pages_q if p.id not in gold_ids]
        n_pick = min(sample, len(distractor_pool))
        distractors = random.Random(seed).sample(distractor_pool, n_pick)
        selected_pages = gold_pages + distractors
        print(f"  --sample {sample}: {len(gold_pages)} gold page(s) + "
              f"{n_pick} random distractor page(s) (seed={seed}), "
              f"out of {len(distractor_pool)} available distractors. "
              "Articles excluded from --sample (see module docstring) -- a "
              "full-corpus run also scores those as distractors.")
    else:
        selected_pages = pages_q

    rows: list[CorpusRow] = []
    for p in selected_pages:
        rows.append(CorpusRow(
            prefix="page", row_id=p.id,
            book=p.book.name if p.book else "Unknown",
            chapter=p.chapter.name if p.chapter else "Unknown",
            body=p.content or "", source_db=p.source_db or "unknown",
        ))
    for a in articles_q:
        rows.append(CorpusRow(
            prefix="article", row_id=a.id,
            book=a.title, chapter="প্রবন্ধ", body=a.content or "", source_db="articles",
        ))
    return rows


# --------------------------------------------------------------------------
# chunk-count preview (cheap: pure string splitting, no embedding)
# --------------------------------------------------------------------------

def _chunk_counts(rows: list[CorpusRow], chunk_size: int, overlap: int) -> dict:
    n_page_chunks = 0
    n_article_chunks = 0
    n_pages = n_articles = 0
    for row in rows:
        n = len(chunk_text(row.body, chunk_size=chunk_size, overlap=overlap))
        if row.prefix == "page":
            n_page_chunks += n
            n_pages += 1
        else:
            n_article_chunks += n
            n_articles += 1
    total = n_page_chunks + n_article_chunks
    return {
        "n_pages": n_pages, "n_articles": n_articles,
        "n_chunks_pages": n_page_chunks, "n_chunks_articles": n_article_chunks,
        "n_chunks": total,
        "avg_chunks_per_page": (n_page_chunks / n_pages) if n_pages else 0.0,
    }


# --------------------------------------------------------------------------
# cost estimate
# --------------------------------------------------------------------------

def _measure_embed_rate(sample_texts: list[str]) -> float:
    """chunks/sec on THIS machine, right now. Warms the model with one
    throwaway call first so the ~2.2GB one-time model load (embedder.py)
    doesn't get counted as embedding throughput."""
    if not sample_texts:
        return 0.0
    embed_texts(sample_texts[:1])
    t0 = time.perf_counter()
    embed_texts(sample_texts)
    elapsed = time.perf_counter() - t0
    return len(sample_texts) / elapsed if elapsed > 0 else float("inf")


def _fmt_seconds(s: float) -> str:
    if s < 90:
        return f"{s:.0f}s"
    m = s / 60
    if m < 90:
        return f"{m:.1f}min"
    return f"{m / 60:.1f}h"


def _print_cost_estimate(counts: list[dict], rate: float | None) -> int:
    print("\n" + "=" * 78)
    print("COST ESTIMATE")
    print("=" * 78)
    print(f"  {'config':>10}  {'pages':>6}  {'articles':>8}  {'chunks':>8}  "
          f"{'avg/page':>8}  {'est. embed time':>16}")
    print(f"  {'-'*10}  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*16}")
    total_chunks = 0
    for c in counts:
        total_chunks += c["n_chunks"]
        est = _fmt_seconds(c["n_chunks"] / rate) if rate else "?"
        print(f"  {c['name']:>10}  {c['n_pages']:>6}  {c['n_articles']:>8}  "
              f"{c['n_chunks']:>8}  {c['avg_chunks_per_page']:>8.2f}  {est:>16}")
    if rate:
        print(f"\n  measured throughput this run: {rate:.2f} chunks/sec "
              f"(batch_size=16, CPU, bge-m3 -- see app/rag/embedder.py)")
        print(f"  TOTAL across {len(counts)} config(s): {total_chunks} chunks, "
              f"~{_fmt_seconds(total_chunks / rate)}")
    print("  (rerank compute during eval is separate: fetch_k candidates x "
          "n_questions x n_configs cross-encoder pairs.)")
    print("\n  PEAK RAM: the scoring phase holds bge-m3 (~2.2 GB) AND "
          "bge-reranker-v2-m3 (~2.2 GB) resident at once, plus torch/Chroma "
          "overhead -- budget ~6 GB free. On a box with less, the scoring "
          "phase will swap-thrash or get OOM-killed (the embed phase, bge-m3 "
          "only, is lighter). Same models scripts/eval_retrieval.py loads.")
    return total_chunks


# --------------------------------------------------------------------------
# build + embed one config's temp collection
# --------------------------------------------------------------------------

def _build_index(rows: list[CorpusRow], chunk_size: int, overlap: int, collection_name: str) -> tuple[int, float]:
    """Rebuild collection_name from scratch: delete if present, then chunk_text
    + embed_texts + upsert every row, exactly like app/rag/ingest.py's
    _ingest(), just against a temp collection instead of production."""
    delete_collection(collection_name)   # idempotent: clean slate every run
    collection = get_named_collection(collection_name)

    t0 = time.perf_counter()
    ids, docs, metas = [], [], []
    n_chunks = 0

    def flush():
        if ids:
            collection.upsert(ids=list(ids), documents=list(docs),
                               metadatas=list(metas), embeddings=embed_texts(docs))
            ids.clear(); docs.clear(); metas.clear()

    for row in rows:
        pieces = chunk_text(row.body, chunk_size=chunk_size, overlap=overlap)
        for i, piece in enumerate(pieces):
            ids.append(f"{row.prefix}_{row.row_id}_c{i}")
            docs.append(build_document(row.book, row.chapter, piece))
            metas.append({
                "book": row.book, "chapter": row.chapter, "page_id": row.row_id,
                "chunk_index": i, "row_key": f"{row.prefix}_{row.row_id}",
                "source_db": row.source_db,
            })
            n_chunks += 1
        if len(ids) >= BATCH_SIZE:
            flush()
    flush()

    return n_chunks, time.perf_counter() - t0


# --------------------------------------------------------------------------
# retrieval scoring -- reuses eval_retrieval's metric primitives verbatim
# --------------------------------------------------------------------------

def _score_config(questions: list[dict], ks: list[int], use_rewrite: bool,
                   collection_name: str) -> tuple[dict, int]:
    """Same per-query logic as scripts.eval_retrieval._run(), parameterized by
    collection_name (which _run does not take). Metrics computed with
    eval_retrieval's own _row_keys/_score_one/_aggregate -- not reimplemented."""
    top_k = settings.top_k
    max_k = max(ks)
    records = []
    for q in questions:
        stage_a_chunks, stage_b_chunks = retrieve_stages(
            q["query"], rerank_top_n=max(max_k, top_k),
            use_rewrite=use_rewrite, collection_name=collection_name,
        )
        stage_a = _row_keys(stage_a_chunks)
        stage_b = _row_keys(stage_b_chunks)
        gold = q["relevant_ids"]
        scorable = bool(q["answerable"] and gold)
        records.append({
            "query": q["query"], "scorable": scorable,
            "metrics": {
                "candidates": {k: _score_one(stage_a, gold, k) for k in ks},
                "final": {k: _score_one(stage_b, gold, k) for k in ks},
            },
        })
    agg = {stage: _aggregate(records, stage, ks) for stage in ("candidates", "final")}
    n_scored = sum(1 for r in records if r["scorable"])
    return agg, n_scored


# --------------------------------------------------------------------------
# comparison output
# --------------------------------------------------------------------------

_STAGE_LABELS = {
    "candidates": f"STAGE A -- candidates (pre-rerank pool, fetch_k={settings.fetch_k})",
    "final": f"STAGE B -- final (post-rerank, top_k={settings.top_k})",
}
_METRICS = ("hit_rate", "recall", "mrr")


def _print_comparison(results: list[ConfigResult], ks: list[int]) -> None:
    for stage in ("candidates", "final"):
        print("\n" + "=" * 78)
        print(f"COMPARISON -- {_STAGE_LABELS[stage]}")
        print("=" * 78)
        for metric in _METRICS:
            print(f"\n{metric}@k")
            header = f"  {'config':>10}" + "".join(f"  {'k='+str(k):>8}" for k in ks)
            print(header)
            print("  " + "-" * (len(header) - 2))
            # best value per k, across configs, for the * marker
            best = {k: max(r.metrics[stage][k][metric] for r in results) for k in ks}
            for r in results:
                cells = []
                for k in ks:
                    v = r.metrics[stage][k][metric]
                    mark = "*" if v == best[k] and v > 0 else " "
                    cells.append(f"{mark}{v:>7.3f}")
                tag = " (base)" if r.name == _baseline_name() else ""
                print(f"  {r.name:>10}" + "".join(f"  {c}" for c in cells) + tag)
        print("\n  * = best in that column")
        if stage == "final" and any(k > settings.top_k for k in ks):
            print(f"  k > top_k ({settings.top_k}) is diagnostic only: Stage B never "
                  f"returns more than top_k pages in production -- it shows what "
                  "raising top_k would recover, same as scripts/eval_retrieval.py.")


def _baseline_name() -> str:
    for c in CONFIGS:
        if c.get("baseline"):
            return c["name"]
    return CONFIGS[0]["name"]


def _print_index_cost(results: list[ConfigResult]) -> None:
    print("\n" + "=" * 78)
    print("INDEX SIZE / BUILD COST")
    print("=" * 78)
    print(f"  {'config':>10}  {'pages':>6}  {'articles':>8}  {'chunks':>8}  "
          f"{'avg/page':>8}  {'build time':>10}")
    print(f"  {'-'*10}  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*10}")
    for r in results:
        print(f"  {r.name:>10}  {r.n_pages:>6}  {r.n_articles:>8}  {r.n_chunks:>8}  "
              f"{r.avg_chunks_per_page:>8.2f}  {_fmt_seconds(r.build_seconds):>10}")
    print("\n  smaller chunks -> more vectors -> bigger index + slower reranking "
          "(the cross-encoder cost at query time is fixed at fetch_k candidates "
          "regardless of chunk size, but a bigger index costs more disk/RAM and "
          "a slower HNSW search).")


NOISE = 0.02  # smaller than one query flipping in a ~30-40 question set


def _decide_winner(results: list[ConfigResult], ks: list[int]) -> tuple[ConfigResult | None, int, float, float]:
    """Pick the config that beats the baseline on Stage B hit_rate@k5/recall@k5
    by more than NOISE. Returns (winner_or_None, k5, base_hit_rate, base_recall)
    -- winner is None if nothing clears the bar, which is the "baseline is
    fine, don't churn it" outcome the task explicitly asks to report honestly.
    """
    base = next(r for r in results if r.name == _baseline_name())
    others = [r for r in results if r.name != base.name]
    k5 = 5 if 5 in ks else ks[len(ks) // 2]

    def hit(r: ConfigResult) -> float:
        return r.metrics["final"][k5]["hit_rate"]

    def rec(r: ConfigResult) -> float:
        return r.metrics["final"][k5]["recall"]

    base_hit, base_rec = hit(base), rec(base)
    best = max(others, key=lambda r: (hit(r), rec(r)), default=None)
    if best is not None and (hit(best) - base_hit > NOISE or rec(best) - base_rec > NOISE):
        return best, k5, base_hit, base_rec
    return None, k5, base_hit, base_rec


def _recommend(results: list[ConfigResult], ks: list[int], sampled: bool) -> tuple[ConfigResult | None, str]:
    base = next(r for r in results if r.name == _baseline_name())
    winner, k5, base_hit, base_rec = _decide_winner(results, ks)

    lines = ["\n" + "=" * 78, "RECOMMENDATION", "=" * 78]
    if sampled:
        lines.append(
            "  RAN WITH --sample: a small distractor pool makes recall look "
            "better than production, where every other page in the corpus is a "
            "potential distractor. Treat this as a directional first pass -- "
            "confirm with a full-corpus run before changing anything."
        )

    if winner is not None:
        w_hit = winner.metrics["final"][k5]["hit_rate"]
        w_rec = winner.metrics["final"][k5]["recall"]
        lines.append(
            f"  {winner.name} beats the {base.name} baseline on Stage B "
            f"hit_rate@{k5} ({w_hit:.3f} vs {base_hit:.3f}) and "
            f"recall@{k5} ({w_rec:.3f} vs {base_rec:.3f}), "
            f"at {winner.n_chunks} vs {base.n_chunks} vectors "
            f"({winner.n_chunks / base.n_chunks:+.0%} index size)."
        )
        if not sampled:
            lines.append(f"  -> RECOMMEND switching to CHUNK_SIZE={winner.chunk_size}, "
                          f"CHUNK_OVERLAP={winner.overlap}.")
        else:
            lines.append("  -> Looks promising; re-run without --sample (or with the "
                          "largest N you can afford) before committing to it.")
    else:
        lines.append(
            f"  No config beats the {base.name} baseline on Stage B "
            f"hit_rate@{k5}/recall@{k5} by more than noise (+{NOISE:.2f}) for "
            f"this eval set. The current CHUNK_SIZE={base.chunk_size}, "
            f"CHUNK_OVERLAP={base.overlap} is FINE -- do not churn it on the "
            "strength of this run."
        )
    return winner, "\n".join(lines)


def _print_diff(chosen_size: int, chosen_overlap: int, tentative: bool) -> None:
    print("\n" + "=" * 78)
    print("CHUNKER.PY DIFF" + (" (TENTATIVE -- from --sample, confirm on full corpus)"
                                if tentative else "") + "  (printed, NOT applied)")
    print("=" * 78)
    print(f"  - CHUNK_SIZE = 900")
    print(f"  - CHUNK_OVERLAP = 150")
    print(f"  + CHUNK_SIZE = {chosen_size}")
    print(f"  + CHUNK_OVERLAP = {chosen_overlap}")
    print("\n  If you apply this:")
    print("    1. Edit app/rag/chunker.py, then run `python -m app.rag.ingest` "
          "for a FULL re-ingest (chunk boundaries change for every page).")
    print("    2. Re-tune scripts/tune_threshold.py -- MIN_RERANK_SCORE was fit "
          "against the OLD chunk boundaries; a different average chunk length "
          "shifts the rerank-score distribution.")
    print("    3. Spot-check a few real answers with scripts/eval_responses.py "
          "to confirm the generator still gets enough context per chunk.")


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------

def _parse_ks(raw: str) -> list[int]:
    ks = sorted({int(p) for p in raw.split(",") if p.strip()})
    if not ks or ks[0] < 1:
        raise argparse.ArgumentTypeError(f"--k must be positive integers: {raw!r}")
    return ks


def _parse_configs(raw: str | None) -> list[dict]:
    if not raw:
        return CONFIGS
    wanted = {n.strip() for n in raw.split(",") if n.strip()}
    by_name = {c["name"]: c for c in CONFIGS}
    unknown = wanted - set(by_name)
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown config name(s): {sorted(unknown)}; edit CONFIGS in "
            f"scripts/eval_chunking.py to add one. known: {sorted(by_name)}"
        )
    if _baseline_name() not in wanted:
        print(f"note: baseline ({_baseline_name()}) is always included as the "
              "comparison point; adding it to --configs.")
        wanted.add(_baseline_name())
    return [c for c in CONFIGS if c["name"] in wanted]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m scripts.eval_chunking",
        description="Compare chunking configs against the labeled retrieval eval set.",
    )
    p.add_argument("questions", nargs="?", default=str(DEFAULT_FILE),
                   help="dataset JSON (default: scripts/retrieval_eval_questions.json)")
    p.add_argument("--configs", metavar="NAME,NAME",
                   help=f"comma-separated subset of {[c['name'] for c in CONFIGS]} "
                        "(default: all)")
    p.add_argument("--sample", type=int, metavar="N",
                   help="fast pass: build each index from all gold pages + N random "
                        "distractor pages instead of the full corpus. Proceeds without "
                        "--apply (see module docstring on why a small distractor pool "
                        "inflates recall).")
    p.add_argument("--apply", action="store_true",
                   help="required (without --sample) to actually re-embed the full "
                        "corpus per config, after the cost estimate is printed")
    p.add_argument("--k", type=_parse_ks, default=list(DEFAULT_KS),
                   help="comma-separated cutoffs, default 1,3,5,10")
    p.add_argument("--no-rewrite", action="store_true",
                   help="search the raw query; skip the Banglish->Bengali rewrite "
                        "(fully deterministic; the rewriter's output does not depend "
                        "on chunking, so this mainly matters for reproducibility)")
    p.add_argument("--seed", type=int, default=20260904,
                   help="distractor sampling seed for --sample (default matches the "
                        "corpus sampler used to build the gold set)")
    p.add_argument("--report", metavar="OUT.JSON",
                   help="where to write the full JSON report")
    p.add_argument("--cleanup", action="store_true",
                   help="drop this run's temp chunk_eval_* collections when done "
                        "(or, given alone with no --sample/--apply, just drop the "
                        "selected configs' temp collections and exit)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    configs = _parse_configs(args.configs)
    collection_names = {c["name"]: f"{COLLECTION_PREFIX}{c['name']}" for c in configs}

    if args.cleanup and args.sample is None and not args.apply:
        print("Dropping temp collections (no run requested):")
        for name, coll in collection_names.items():
            delete_collection(coll)
            print(f"  dropped {coll}")
        return

    path = Path(args.questions)
    questions = _load_questions(path)
    use_rewrite = not args.no_rewrite
    gold_ids = _gold_page_ids(questions)

    print("=" * 78)
    print("CHUNKING CONFIG COMPARISON")
    print("=" * 78)
    print(f"dataset: {path}  ({len(questions)} questions, "
          f"{sum(1 for q in questions if q['answerable'] and q['relevant_ids'])} scorable)")
    print(f"configs: {[c['name'] for c in configs]}")
    print(f"fetch_k={settings.fetch_k}  top_k={settings.top_k}  "
          f"(held fixed at production values for every config)")

    session = SessionLocal()
    try:
        rows = _load_corpus_rows(session, args.sample, args.seed, gold_ids)
    finally:
        session.close()
    n_pages = sum(1 for r in rows if r.prefix == "page")
    n_articles = sum(1 for r in rows if r.prefix == "article")
    print(f"corpus rows loaded: {n_pages} pages + {n_articles} articles = {len(rows)}")

    counts = [dict(_chunk_counts(rows, c["chunk_size"], c["overlap"]), name=c["name"])
              for c in configs]

    # Calibrate on real text from the smallest chunk_size (most chunks -> the
    # binding constraint) so the throughput number is a fair worst case.
    smallest = min(configs, key=lambda c: c["chunk_size"])
    calib_pieces: list[str] = []
    for row in rows:
        calib_pieces.extend(chunk_text(row.body, chunk_size=smallest["chunk_size"],
                                        overlap=smallest["overlap"]))
        if len(calib_pieces) >= CALIBRATION_CHUNKS:
            break
    print(f"\nMeasuring embedding throughput on this machine "
          f"({min(len(calib_pieces), CALIBRATION_CHUNKS)} real chunks, "
          "one-time model load excluded)...")
    rate = _measure_embed_rate(calib_pieces[:CALIBRATION_CHUNKS]) if calib_pieces else None

    total_chunks = _print_cost_estimate(counts, rate)

    sampled = args.sample is not None
    if not sampled and not args.apply:
        print("\nNot proceeding: this would re-embed the FULL corpus "
              f"{len(configs)} time(s) ({total_chunks} chunk embeddings total). "
              "Re-run with --apply to go ahead, or --sample N for a fast first pass.")
        return

    # --- build + score every config ---
    results: list[ConfigResult] = []
    for c in configs:
        coll = collection_names[c["name"]]
        print("\n" + "-" * 78)
        print(f"[{c['name']}] building {coll} (chunk_size={c['chunk_size']}, "
              f"overlap={c['overlap']})...")
        n_chunks, build_s = _build_index(rows, c["chunk_size"], c["overlap"], coll)
        print(f"[{c['name']}] {n_chunks} chunks embedded in {_fmt_seconds(build_s)}. "
              f"Scoring against {path.name}...")
        metrics, n_scored = _score_config(questions, args.k, use_rewrite, coll)

        cnt = next(x for x in counts if x["name"] == c["name"])
        results.append(ConfigResult(
            name=c["name"], chunk_size=c["chunk_size"], overlap=c["overlap"],
            collection=coll, n_pages=cnt["n_pages"], n_articles=cnt["n_articles"],
            n_chunks=n_chunks, avg_chunks_per_page=cnt["avg_chunks_per_page"],
            build_seconds=build_s, n_scored=n_scored, metrics=metrics,
        ))

    _print_comparison(results, args.k)
    _print_index_cost(results)
    winner, recommendation = _recommend(results, args.k, sampled)
    print(recommendation)
    if winner is not None:
        _print_diff(winner.chunk_size, winner.overlap, tentative=sampled)

    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        REPORT_DIR.mkdir(exist_ok=True)
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        report_path = REPORT_DIR / f"eval_chunking_{stamp}.json"
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": str(path),
        "k": args.k,
        "sample": args.sample,
        "seed": args.seed,
        "use_rewrite": use_rewrite,
        "config": {
            "top_k": settings.top_k, "fetch_k": settings.fetch_k,
            "embedding_model": settings.embedding_model_name,
            "reranker_model": settings.reranker_model_name,
        },
        "measured_embed_rate_chunks_per_sec": rate,
        "results": [
            {
                "name": r.name, "chunk_size": r.chunk_size, "overlap": r.overlap,
                "collection": r.collection, "n_pages": r.n_pages,
                "n_articles": r.n_articles, "n_chunks": r.n_chunks,
                "avg_chunks_per_page": round(r.avg_chunks_per_page, 3),
                "build_seconds": round(r.build_seconds, 1), "n_scored": r.n_scored,
                "metrics": {stage: {str(k): m for k, m in per_k.items()}
                            for stage, per_k in r.metrics.items()},
            }
            for r in results
        ],
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nReport written to {report_path}")

    if args.cleanup:
        print("\n--cleanup: dropping this run's temp collections:")
        for r in results:
            delete_collection(r.collection)
            print(f"  dropped {r.collection}")


if __name__ == "__main__":
    main()
