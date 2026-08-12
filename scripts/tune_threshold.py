"""Suggest a MIN_RERANK_SCORE by testing retrieval against known questions.

Usage:
    python -m scripts.tune_threshold [path/to/questions.json]

Input file: a JSON list of {"query": str, "answerable": bool}. See
scripts/eval_questions.example.json for the shape -- fill in ~30 real
questions (20 answerable from the corpus, 10 that are not) and point this
script at that file.

For each question, retrieves the top candidate via the real retrieval
pipeline (query rewriting + bi-encoder + cross-encoder rerank -- the same
path /ask uses) and records its rerank_score. Then scans candidate
thresholds and reports which one best separates the answerable group from
the unanswerable group, plus the false-refusal / false-answer rate at that
threshold.

Run this from a shell where the app's dependencies and .env are available
(same environment as `python -m app.rag.ingest`).
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

from app.rag.retriever import retrieve_relevant_docs

DEFAULT_FILE = Path(__file__).parent / "eval_questions.example.json"


def _load_questions(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        questions = json.load(f)
    for q in questions:
        if "query" not in q or "answerable" not in q:
            raise ValueError(f"bad entry, expected query/answerable keys: {q}")
    return questions


def _top_score(query: str) -> float | None:
    citations = retrieve_relevant_docs(query)
    if not citations:
        return None
    return citations[0].rerank_score


def _print_distribution(label: str, scores: list[float | None]) -> None:
    present = sorted(s for s in scores if s is not None)
    n_none = sum(1 for s in scores if s is None)
    print(f"\n{label} (n={len(scores)}, {n_none} retrieved nothing):")
    if not present:
        print("  no scored results")
        return
    print(f"  min={present[0]:.4f}  max={present[-1]:.4f}  "
          f"mean={statistics.mean(present):.4f}  median={statistics.median(present):.4f}")
    print("  scores:", ", ".join(f"{s:.4f}" for s in present))


def _candidate_thresholds(all_scores: list[float | None]) -> list[float]:
    present = sorted(set(s for s in all_scores if s is not None))
    if not present:
        return [0.0]
    midpoints = [(a + b) / 2 for a, b in zip(present, present[1:])]
    # also test just above the max and just below the min, so "reject everything"
    # and "accept everything" are both reachable.
    return [present[0] - 0.01, *midpoints, present[-1] + 0.01]


def _evaluate(threshold: float, answerable_scores: list[float | None],
              unanswerable_scores: list[float | None]) -> tuple[int, int]:
    """Return (false_refusals, false_answers) at this threshold.

    False refusal: an answerable question whose top score fell below the bar
    (or nothing was retrieved at all) -- the user gets refused when the
    corpus actually had the answer.
    False answer: an unanswerable question whose top score cleared the bar --
    the user gets an answer built from irrelevant chunks.
    """
    false_refusals = sum(
        1 for s in answerable_scores if s is None or s < threshold
    )
    false_answers = sum(
        1 for s in unanswerable_scores if s is not None and s >= threshold
    )
    return false_refusals, false_answers


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_FILE
    questions = _load_questions(path)

    answerable_scores: list[float | None] = []
    unanswerable_scores: list[float | None] = []

    for q in questions:
        score = _top_score(q["query"])
        bucket = answerable_scores if q["answerable"] else unanswerable_scores
        bucket.append(score)
        label = "answerable" if q["answerable"] else "unanswerable"
        print(f"[{label}] {q['query']!r} -> top_rerank_score={score}")

    _print_distribution("Answerable questions", answerable_scores)
    _print_distribution("Unanswerable questions", unanswerable_scores)

    if not answerable_scores or not unanswerable_scores:
        print("\nNeed at least one answerable and one unanswerable question "
              "to suggest a threshold.")
        return

    all_scores = answerable_scores + unanswerable_scores
    best_threshold = None
    best_errors = None
    for t in _candidate_thresholds(all_scores):
        fr, fa = _evaluate(t, answerable_scores, unanswerable_scores)
        errors = fr + fa
        if best_errors is None or errors < best_errors:
            best_errors = errors
            best_threshold = t

    fr, fa = _evaluate(best_threshold, answerable_scores, unanswerable_scores)
    fr_rate = fr / len(answerable_scores)
    fa_rate = fa / len(unanswerable_scores)

    print(f"\nSuggested MIN_RERANK_SCORE: {best_threshold:.4f}")
    print(f"  false-refusal rate:  {fr}/{len(answerable_scores)} ({fr_rate:.0%}) "
          "-- answerable questions that would be wrongly refused")
    print(f"  false-answer rate:   {fa}/{len(unanswerable_scores)} ({fa_rate:.0%}) "
          "-- unanswerable questions that would wrongly get an answer")

    if fr_rate > 0 or fa_rate > 0:
        print("\n  Non-zero error rate at the best achievable threshold means "
              "the two groups overlap -- no single cutoff perfectly separates "
              "them on this question set. Consider adding more questions or "
              "accepting the trade-off above.")


if __name__ == "__main__":
    main()
