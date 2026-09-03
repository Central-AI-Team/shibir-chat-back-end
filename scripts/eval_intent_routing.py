"""Intent-routing accuracy check for /chat.

/chat (app/api/router.py) classifies every incoming message with
app.services.intent_classifier.classify_intent(message, has_active_roleplay_session)
before dispatching to NOTE / ROLEPLAY / SUGGESTION / QA. That function is a
plain, importable Python function -- no live server needed -- so this script
calls it in-process the same way scripts/tune_threshold.py calls
retrieve_relevant_docs() directly instead of going over HTTP.

Each test entry is treated as a standalone first-turn message, so
has_active_roleplay_session is always passed as False: that isolates what
the text alone routes to, uncontaminated by sticky-roleplay session state
from a previous turn (see classify_intent's docstring/logic for that
stickiness rule).

Usage:
    python -m scripts.eval_intent_routing [path/to/questions.json]

Input file: a JSON list of {"text": str, "expected_intent": str, "note": str
(optional)}. expected_intent is one of the four real modes -- "qa", "note",
"roleplay", "suggestion" -- or one of two edge-case labels with no single
correct routing target: "ambiguous" or "none-of-the-four". See
scripts/chat_intent_test_questions.json for the shape.

Scoring: pass/fail is only computed for the four real-mode entries (exact
match against actual_intent). "ambiguous" / "none-of-the-four" entries are
never scored -- they're reported separately, informationally, showing what
actual_intent came back so a human can judge whether that's reasonable.

Run this from a shell where the app's dependencies and .env are available
(same environment as `python -m app.rag.ingest`).
"""

from __future__ import annotations

import html
import json
import sys
from datetime import datetime
from pathlib import Path

from app.services.intent_classifier import classify_intent

DEFAULT_FILE = Path(__file__).parent / "chat_intent_test_questions.json"
REPORT_DIR = Path(__file__).parent.parent / "eval_reports"

_FONT_STACK = (
    "'Noto Sans Bengali', 'Hind Siliguri', 'SolaimanLipi', 'Kalpurush', "
    "'Segoe UI', sans-serif"
)

# The four modes classify_intent can actually return. Anything else in the
# test file ("ambiguous", "none-of-the-four") is an edge case, not scored.
_REAL_MODES = ("qa", "note", "roleplay", "suggestion")


def _load_questions(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        questions = json.load(f)
    for q in questions:
        if "text" not in q or "expected_intent" not in q:
            raise ValueError(f"bad entry, expected text/expected_intent keys: {q}")
    return questions


def _run(questions: list[dict]) -> list[dict]:
    results = []
    for q in questions:
        actual = classify_intent(q["text"], has_active_roleplay_session=False).lower()
        expected = q["expected_intent"]
        is_real_mode = expected in _REAL_MODES
        passed = (actual == expected) if is_real_mode else None

        results.append({
            "text": q["text"],
            "expected_intent": expected,
            "actual_intent": actual,
            "note": q.get("note", ""),
            "is_real_mode": is_real_mode,
            "passed": passed,
        })

        if is_real_mode:
            tag = "match" if passed else "MISMATCH"
        else:
            tag = "info"
        print(f"[{tag}] {q['text']!r} -> expected={expected} actual={actual}")
    return results


def _confusion_matrix(real_results: list[dict]) -> dict[str, dict[str, int]]:
    matrix = {expected: dict.fromkeys(_REAL_MODES, 0) for expected in _REAL_MODES}
    for r in real_results:
        matrix[r["expected_intent"]][r["actual_intent"]] += 1
    return matrix


def _print_summary(results: list[dict]) -> None:
    real = [r for r in results if r["is_real_mode"]]
    edge = [r for r in results if not r["is_real_mode"]]
    passed = sum(1 for r in real if r["passed"])
    total = len(real)
    pass_rate = passed / total if total else 0.0

    print(f"\n{passed}/{total} passed ({pass_rate:.0%}) on real-mode entries")
    if edge:
        print(f"{len(edge)} edge case(s) reported informationally (no pass/fail)")

    matrix = _confusion_matrix(real)
    col_width = max(len(m) for m in _REAL_MODES) + 4
    print("\nConfusion matrix (rows=expected, cols=actual):")
    print("expected".ljust(12) + "".join(m.ljust(col_width) for m in _REAL_MODES))
    for expected in _REAL_MODES:
        row = expected.ljust(12) + "".join(
            str(matrix[expected][actual]).ljust(col_width) for actual in _REAL_MODES
        )
        print(row)


def _render_confusion_matrix_html(matrix: dict[str, dict[str, int]]) -> str:
    header_cells = "".join(f"<th>{m}</th>" for m in _REAL_MODES)
    body_rows = []
    for expected in _REAL_MODES:
        cells = []
        for actual in _REAL_MODES:
            count = matrix[expected][actual]
            if count == 0:
                cls = "cm-zero"
            elif expected == actual:
                cls = "cm-diag"
            else:
                cls = "cm-off"
            cells.append(f'<td class="{cls}">{count}</td>')
        body_rows.append(f"<tr><th>{expected}</th>{''.join(cells)}</tr>")
    return f"""
    <table class="confusion">
      <thead><tr><th>expected \\ actual</th>{header_cells}</tr></thead>
      <tbody>{"".join(body_rows)}</tbody>
    </table>
    """


def _render_result_card(r: dict) -> str:
    status_class = "match" if r["passed"] else "mismatch"
    return f"""
    <div class="card {status_class}">
      <div class="card-header">
        <span class="query">{html.escape(r["text"])}</span>
      </div>
      <div class="expectation">
        <span>প্রত্যাশিত: <strong>{html.escape(r["expected_intent"])}</strong></span>
        <span>প্রাপ্ত: <strong>{html.escape(r["actual_intent"])}</strong></span>
      </div>
    </div>
    """


def _render_edge_card(r: dict) -> str:
    note_html = f'<div class="note">{html.escape(r["note"])}</div>' if r["note"] else ""
    return f"""
    <div class="card info">
      <div class="card-header">
        <span class="query">{html.escape(r["text"])}</span>
      </div>
      <div class="expectation">
        <span>লেবেল: <strong>{html.escape(r["expected_intent"])}</strong></span>
        <span>actual_intent: <strong>{html.escape(r["actual_intent"])}</strong></span>
      </div>
      {note_html}
    </div>
    """


def _render_html(results: list[dict]) -> str:
    real = [r for r in results if r["is_real_mode"]]
    edge = [r for r in results if not r["is_real_mode"]]
    passed = sum(1 for r in real if r["passed"])
    total = len(real)
    pass_rate = passed / total if total else 0.0

    matrix = _confusion_matrix(real)
    confusion_html = _render_confusion_matrix_html(matrix)
    real_cards = "\n".join(_render_result_card(r) for r in real)
    edge_cards = "\n".join(_render_edge_card(r) for r in edge) or (
        "<p class='no-sources'>কোনো edge case পাওয়া যায়নি।</p>"
    )

    return f"""<!DOCTYPE html>
<html lang="bn">
<head>
<meta charset="utf-8">
<title>Intent Routing Eval Report</title>
<style>
  body {{
    font-family: {_FONT_STACK};
    background: #f4f5f7;
    color: #1a1a1a;
    margin: 0;
    padding: 2rem;
  }}
  h1 {{ margin-top: 0; }}
  h2 {{ margin-top: 2.5rem; }}
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
  table.confusion {{
    border-collapse: collapse;
    background: #fff;
    border-radius: 8px;
    overflow: hidden;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    margin-bottom: 1.5rem;
  }}
  table.confusion th, table.confusion td {{
    padding: 0.5rem 1rem;
    text-align: center;
    border: 1px solid #e5e5e5;
  }}
  table.confusion thead th {{ background: #f0f0f0; }}
  table.confusion tbody th {{ background: #f9f9f9; text-align: left; }}
  td.cm-diag {{ background: #e6f7ec; font-weight: 700; color: #1a7a3d; }}
  td.cm-off {{ background: #fdecea; font-weight: 700; color: #a02622; }}
  td.cm-zero {{ color: #bbb; }}
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
  .card.info {{ border-left-color: #8a63d2; }}
  .card-header {{
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    gap: 1rem;
  }}
  .query {{ font-size: 1.1rem; font-weight: 700; }}
  .expectation {{
    display: flex;
    gap: 1.5rem;
    font-size: 0.9rem;
    color: #444;
    margin: 0.5rem 0;
  }}
  .note {{
    font-size: 0.85rem;
    color: #555;
    background: #f5f2fc;
    border-radius: 4px;
    padding: 0.5rem 0.75rem;
    margin-top: 0.25rem;
  }}
  .no-sources {{ color: #888; font-style: italic; margin: 0.5rem 0 0; }}
  .edge-note {{ color: #666; font-size: 0.9rem; margin-top: -0.5rem; }}
</style>
</head>
<body>
  <h1>Intent Routing Eval Report</h1>
  <div class="summary">
    <div><span class="label">Real-mode entries</span><span class="value">{total}</span></div>
    <div><span class="label">Passed</span><span class="value">{passed}</span></div>
    <div><span class="label">Pass rate</span><span class="value">{pass_rate:.0%}</span></div>
    <div><span class="label">Edge cases</span><span class="value">{len(edge)}</span></div>
  </div>

  <h2>Confusion matrix</h2>
  {confusion_html}

  <h2>Real-mode entries ({total})</h2>
  {real_cards}

  <h2>Edge cases (informational only, not pass/fail)</h2>
  <p class="edge-note">"ambiguous" / "none-of-the-four" entries -- no single correct routing target, shown here for human review only.</p>
  {edge_cards}
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
    report_path = REPORT_DIR / f"intent_routing_{stamp}.html"
    report_path.write_text(_render_html(results), encoding="utf-8")
    print(f"\nReport written to {report_path}")


if __name__ == "__main__":
    main()
