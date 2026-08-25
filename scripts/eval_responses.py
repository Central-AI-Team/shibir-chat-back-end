"""End-to-end quality check for the /ask pipeline. NEW FILE.

scripts/tune_threshold.py only exercises retrieval -- it calls
retrieve_relevant_docs() directly and looks at rerank scores. It never calls
generate_answer(), so it cannot tell you what the user actually sees: the
answer text, which sources got cited, or how long a real request took. The
only way to check that today is manually hitting POST /ask and reading raw
(escaped) JSON one question at a time, with no way to compare runs or see an
aggregate pass rate.

This script closes that gap by calling app.services.qa_service.answer_question()
in-process (same as tune_threshold.py imports retrieve_relevant_docs() instead
of going over HTTP) for every question in an input file, and renders the
results as a single self-contained HTML report plus a terminal summary.

NOTE on "did it answer": qa_service.py no longer has a hardcoded refusal
string to compare against -- below-threshold queries still call
generate_answer() and get back free-form Bengali text (varied wording, not a
fixed constant), and greetings get one of a few canned conversational
replies. Neither is safe to string-match. So "got_answer" here means
"the answer was grounded in retrieved book excerpts", i.e. response.sources
is non-empty -- the same rerank-score gate tune_threshold.py already tunes
against, just observed through the real pipeline this time.

Usage:
    python -m scripts.eval_responses [path/to/questions.json]

Input file: same {"query": str, "answerable": bool} shape as
scripts/eval_questions.example.json (see tune_threshold.py's docstring).

Run this from a shell where the app's dependencies and .env are available
(same environment as `python -m app.rag.ingest`).
"""

from __future__ import annotations

import html
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path

from app.services.qa_service import answer_question

DEFAULT_FILE = Path(__file__).parent / "eval_questions.example.json"
REPORT_DIR = Path(__file__).parent.parent / "eval_reports"

_FONT_STACK = (
    "'Noto Sans Bengali', 'Hind Siliguri', 'SolaimanLipi', 'Kalpurush', "
    "'Segoe UI', sans-serif"
)


def _load_questions(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        questions = json.load(f)
    for q in questions:
        if "query" not in q or "answerable" not in q:
            raise ValueError(f"bad entry, expected query/answerable keys: {q}")
    return questions


def _run(questions: list[dict]) -> list[dict]:
    results = []
    for q in questions:
        response = answer_question(q["query"])
        got_answer = bool(response.sources)
        expected_match = got_answer == q["answerable"]

        results.append({
            "query": q["query"],
            "answerable": q["answerable"],
            "answer": response.answer,
            "sources": [
                {
                    "book": c.book,
                    "chapter": c.chapter,
                    "rerank_score": c.rerank_score,
                    "similarity": c.similarity,
                }
                for c in response.sources
            ],
            "response_time_ms": response.response_time_ms,
            "got_answer": got_answer,
            "expected_match": expected_match,
        })

        tag = "match" if expected_match else "MISMATCH"
        print(
            f"[{tag}] {q['query']!r} -> answerable={q['answerable']} "
            f"got_answer={got_answer} n_sources={len(response.sources)} "
            f"time={response.response_time_ms:.1f}ms"
        )
    return results


def _print_summary(results: list[dict]) -> None:
    total = len(results)
    matched = sum(1 for r in results if r["expected_match"])
    pass_rate = matched / total if total else 0.0
    avg_time = statistics.mean(r["response_time_ms"] for r in results) if results else 0.0

    print(f"\n{matched}/{total} matched expectation ({pass_rate:.0%})")
    print(f"average response time: {avg_time:.1f}ms")
    if matched < total:
        print(f"{total - matched} mismatch(es) -- see report for details.")


def _render_card(r: dict) -> str:
    status_class = "match" if r["expected_match"] else "mismatch"
    answerable_label = "হ্যাঁ" if r["answerable"] else "না"
    got_answer_label = "হ্যাঁ" if r["got_answer"] else "না"

    if r["sources"]:
        sources_html = "".join(
            f'<li><span class="src-book">{html.escape(s["book"])}</span>'
            f' &mdash; {html.escape(s["chapter"])}'
            f' <span class="src-score">rerank={s["rerank_score"]:.3f}'
            f'{f", similarity={s['similarity']:.3f}" if s["similarity"] is not None else ""}'
            f'</span></li>'
            for s in r["sources"]
        )
        sources_html = f"<ul class='sources'>{sources_html}</ul>"
    else:
        sources_html = "<p class='no-sources'>কোনো উৎস পাওয়া যায়নি।</p>"

    return f"""
    <div class="card {status_class}">
      <div class="card-header">
        <span class="query">{html.escape(r["query"])}</span>
        <span class="time">{r["response_time_ms"]:.1f}ms</span>
      </div>
      <div class="expectation">
        <span>প্রত্যাশিত উত্তরযোগ্য: <strong>{answerable_label}</strong></span>
        <span>প্রকৃত উত্তর পাওয়া গেছে: <strong>{got_answer_label}</strong></span>
      </div>
      <div class="answer">{html.escape(r["answer"])}</div>
      {sources_html}
    </div>
    """


def _render_html(results: list[dict]) -> str:
    total = len(results)
    matched = sum(1 for r in results if r["expected_match"])
    pass_rate = matched / total if total else 0.0
    avg_time = statistics.mean(r["response_time_ms"] for r in results) if results else 0.0

    cards = "\n".join(_render_card(r) for r in results)

    return f"""<!DOCTYPE html>
<html lang="bn">
<head>
<meta charset="utf-8">
<title>Eval Report</title>
<style>
  body {{
    font-family: {_FONT_STACK};
    background: #f4f5f7;
    color: #1a1a1a;
    margin: 0;
    padding: 2rem;
  }}
  h1 {{ margin-top: 0; }}
  .summary {{
    background: #fff;
    border-radius: 8px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1.5rem;
    display: flex;
    gap: 2.5rem;
    flex-wrap: wrap;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
  }}
  .summary div {{ display: flex; flex-direction: column; }}
  .summary .label {{ font-size: 0.8rem; color: #666; }}
  .summary .value {{ font-size: 1.5rem; font-weight: 700; }}
  .card {{
    background: #fff;
    border-left: 6px solid #ccc;
    border-radius: 6px;
    padding: 1rem 1.25rem;
    margin-bottom: 1rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
  }}
  .card.match {{ border-left-color: #2ea043; }}
  .card.mismatch {{ border-left-color: #d1242f; }}
  .card-header {{
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    gap: 1rem;
  }}
  .query {{ font-size: 1.1rem; font-weight: 700; }}
  .time {{ color: #666; font-size: 0.85rem; white-space: nowrap; }}
  .expectation {{
    display: flex;
    gap: 1.5rem;
    font-size: 0.9rem;
    color: #444;
    margin: 0.5rem 0;
  }}
  .answer {{
    white-space: pre-wrap;
    line-height: 1.6;
    background: #fafafa;
    border-radius: 4px;
    padding: 0.75rem 1rem;
    margin: 0.5rem 0;
  }}
  .sources {{ margin: 0.5rem 0 0; padding-left: 1.25rem; font-size: 0.9rem; }}
  .sources li {{ margin-bottom: 0.25rem; }}
  .src-book {{ font-weight: 600; }}
  .src-score {{ color: #666; font-size: 0.85rem; }}
  .no-sources {{ color: #888; font-style: italic; margin: 0.5rem 0 0; }}
</style>
</head>
<body>
  <h1>Eval Report</h1>
  <div class="summary">
    <div><span class="label">Total questions</span><span class="value">{total}</span></div>
    <div><span class="label">Matched expectation</span><span class="value">{matched}</span></div>
    <div><span class="label">Pass rate</span><span class="value">{pass_rate:.0%}</span></div>
    <div><span class="label">Avg response time</span><span class="value">{avg_time:.1f}ms</span></div>
  </div>
  {cards}
</body>
</html>
"""


def _ensure_gitignored() -> None:
    gitignore = Path(__file__).parent.parent / ".gitignore"
    if not gitignore.exists():
        return
    text = gitignore.read_text(encoding="utf-8")
    if "eval_reports/" in text:
        return
    with open(gitignore, "a", encoding="utf-8") as f:
        f.write("\n# Generated eval reports (scripts/eval_responses.py)\neval_reports/\n")


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_FILE
    questions = _load_questions(path)

    results = _run(questions)
    _print_summary(results)

    REPORT_DIR.mkdir(exist_ok=True)
    _ensure_gitignored()

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    report_path = REPORT_DIR / f"eval_{stamp}.html"
    report_path.write_text(_render_html(results), encoding="utf-8")
    print(f"\nReport written to {report_path}")


if __name__ == "__main__":
    main()
