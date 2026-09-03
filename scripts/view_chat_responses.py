"""Full-response read-through for /chat.

Unlike scripts/eval_intent_routing.py (which only checks routing accuracy by
calling classify_intent() directly), this script exercises the whole /chat
pipeline end to end -- routing AND generation -- so you can actually read
what each mode produces.

Investigation notes (app/api/router.py, app/schemas/query.py):
  - ChatResponse.answer holds the full generated reply text.
  - ChatResponse.mode holds which of the four modes was used ("qa", "note",
    "roleplay", "suggestion" -- already lowercased by the router).
  - ChatResponse.response_time_ms is present, so it's captured here too.
  - The route handler `chat()` in app.api.router is a plain async function
    that takes a ChatRequest and returns a ChatResponse -- no live HTTP
    server needed, it can be awaited directly in-process (same in-process
    preference as eval_intent_routing.py and eval_responses.py).

Usage:
    python -m scripts.view_chat_responses [path/to/questions.json]

Input file: a JSON list of {"text": str, "expected_intent": str, "note": str
(optional)}, same shape as scripts/chat_intent_test_questions.json.

This is deliberately NOT a scoring script -- no pass/fail, no confusion
matrix (see eval_intent_routing.py for that). It just renders every full
response for human review. Each real /chat call goes through retrieval,
reranking, and one or more real LLM calls, so this can take a while to run
against a full question set.

Run this from a shell where the app's dependencies and .env are available
(same environment as `python -m app.rag.ingest`).
"""

from __future__ import annotations

import asyncio
import html
import json
import sys
from datetime import datetime
from pathlib import Path

from fastapi import HTTPException

from app.api.router import chat
from app.schemas.query import ChatRequest

DEFAULT_FILE = Path(__file__).parent / "chat_intent_test_questions.json"
REPORT_DIR = Path(__file__).parent.parent / "eval_reports"

_FONT_STACK = (
    "'Noto Sans Bengali', 'Hind Siliguri', 'SolaimanLipi', 'Kalpurush', "
    "'Segoe UI', sans-serif"
)

# The four modes ChatResponse.mode can actually hold. Anything else in the
# test file ("ambiguous", "none-of-the-four") has no single correct routing
# target, so it never counts as a mismatch -- see _card_class().
_REAL_MODES = ("qa", "note", "roleplay", "suggestion")


def _load_questions(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        questions = json.load(f)
    for q in questions:
        if "text" not in q or "expected_intent" not in q:
            raise ValueError(f"bad entry, expected text/expected_intent keys: {q}")
    return questions


async def _call_chat(message: str):
    return await chat(ChatRequest(message=message))


def _run(questions: list[dict]) -> list[dict]:
    results = []
    for i, q in enumerate(questions, start=1):
        text = q["text"]
        try:
            response = asyncio.run(_call_chat(text))
            actual_intent = response.mode
            answer = response.answer
            response_time_ms = getattr(response, "response_time_ms", None)
        except HTTPException as e:
            # LLM call failed (quota, bad key, upstream outage, ...) -- record
            # it as the "answer" instead of crashing the whole read-through.
            actual_intent = None
            answer = f"[HTTP {e.status_code}] {e.detail}"
            response_time_ms = None

        results.append({
            "text": text,
            "expected_intent": q["expected_intent"],
            "actual_intent": actual_intent,
            "note": q.get("note", ""),
            "answer": answer,
            "response_time_ms": response_time_ms,
        })
        print(f"[{i}/{len(questions)}] {text!r} -> mode={actual_intent}")
    return results


def _card_class(r: dict) -> str:
    expected, actual = r["expected_intent"], r["actual_intent"]
    if expected not in _REAL_MODES or actual not in _REAL_MODES:
        # "ambiguous" / "none-of-the-four" entries, and calls that errored
        # out, have no single correct target -- neutral, not a mismatch.
        return "neutral"
    return "match" if expected == actual else "mismatch"


def _render_card(r: dict) -> str:
    css_class = _card_class(r)
    actual_tag = (
        f'<span class="tag actual">actual: {html.escape(r["actual_intent"])}</span>'
        if r["actual_intent"] is not None
        else '<span class="tag actual unknown">actual: (error)</span>'
    )
    time_tag = (
        f'<span class="tag time">{r["response_time_ms"]:.0f}ms</span>'
        if r["response_time_ms"] is not None
        else ""
    )
    note_html = f'<div class="note">{html.escape(r["note"])}</div>' if r["note"] else ""
    return f"""
    <div class="card {css_class}">
      <div class="card-header">
        <span class="query">{html.escape(r["text"])}</span>
      </div>
      <div class="tags">
        <span class="tag expected">expected: {html.escape(r["expected_intent"])}</span>
        {actual_tag}
        {time_tag}
      </div>
      {note_html}
      <div class="answer">{html.escape(r["answer"])}</div>
    </div>
    """


def _render_html(results: list[dict]) -> str:
    cards = "\n".join(_render_card(r) for r in results)
    return f"""<!DOCTYPE html>
<html lang="bn">
<head>
<meta charset="utf-8">
<title>Chat Response Report</title>
<style>
  body {{
    font-family: {_FONT_STACK};
    background: #f4f5f7;
    color: #1a1a1a;
    margin: 0;
    padding: 2rem;
  }}
  h1 {{ margin-top: 0; }}
  .summary {{ color: #555; margin-bottom: 1.5rem; }}
  .card {{
    background: #fff;
    border-left: 6px solid #ccc;
    border-radius: 6px;
    padding: 1rem 1.25rem;
    margin-bottom: 1rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
  }}
  .card.match {{ border-left-color: #2ea043; }}
  .card.mismatch {{ border-left-color: #d69e2e; }}
  .card.neutral {{ border-left-color: #9aa0a6; }}
  .card-header {{ margin-bottom: 0.4rem; }}
  .query {{ font-size: 1.1rem; font-weight: 700; }}
  .tags {{ display: flex; gap: 0.5rem; flex-wrap: wrap; margin-bottom: 0.5rem; }}
  .tag {{
    font-size: 0.78rem;
    padding: 0.15rem 0.55rem;
    border-radius: 999px;
    background: #eef0f3;
    color: #444;
    white-space: nowrap;
  }}
  .tag.expected {{ background: #e7ecfb; color: #2a4b9b; }}
  .tag.actual {{ background: #eef7ee; color: #256b34; }}
  .tag.actual.unknown {{ background: #fdecea; color: #a02622; }}
  .tag.time {{ background: #f2f2f2; color: #777; }}
  .note {{
    font-size: 0.85rem;
    color: #555;
    background: #f5f2fc;
    border-radius: 4px;
    padding: 0.5rem 0.75rem;
    margin-bottom: 0.5rem;
  }}
  .answer {{
    white-space: pre-wrap;
    line-height: 1.7;
    background: #fafafa;
    border-radius: 4px;
    padding: 0.85rem 1rem;
  }}
</style>
</head>
<body>
  <h1>Chat Response Report</h1>
  <p class="summary">{len(results)}টি প্রশ্নের সম্পূর্ণ /chat উত্তর -- এটি routing accuracy পরীক্ষা করে না, শুধু response পড়ার জন্য (দেখুন eval_intent_routing.py routing accuracy-র জন্য)।</p>
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

    REPORT_DIR.mkdir(exist_ok=True)
    _ensure_gitignored()

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    report_path = REPORT_DIR / f"chat_responses_{stamp}.html"
    report_path.write_text(_render_html(results), encoding="utf-8")

    print(f"\nProcessed {len(results)} question(s).")
    print(f"Report written to {report_path}")


if __name__ == "__main__":
    main()
