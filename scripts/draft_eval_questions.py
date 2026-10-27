"""Draft eval questions from sampled corpus pages, for HUMAN review.

    python -m scripts.sample_corpus_pages --n 150              # pages (stratified by category)
    python -m scripts.draft_eval_questions [--sample scripts/_corpus_sample.json]
        [--pages 150] [--per-page 2] [--out scripts/eval/seed.json] [--yes]

For each sampled page (interleaved by category x source_db so both spread),
ONE LLM call (task "eval_draft", app.core.llm.complete) drafts 1-2 Bengali
questions that THIS page answers. A draft is kept only if it is Bengali
(>= 80 % Bengali letters), a question of sane length, and shares no run of
N_GRAM consecutive words with the page -- verbatim phrasing inflates recall
and is exactly what reviewers would reject. Kept drafts are appended to
seed.json as `answerable_bn` records with review="pending", the page's
row_key as expected_row_keys and the page's category / source_db. Nothing is
ever marked accepted here: review is human (scripts/eval/README.md).

Before any call the script prints an estimate (pages, tokens, USD at the
PRICE_* constants) and stops unless --yes is given. Re-running skips pages
that already have a pending/accepted record in the output file.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from app.core.llm import complete
from app.rag.query_rewriter import _is_bengali_output

DEFAULT_SAMPLE = Path(__file__).parent / "_corpus_sample.json"
DEFAULT_OUT = Path(__file__).parent / "eval" / "seed.json"
PAGE_CHARS = 3500        # page text sent to the model (pages are ~2000 chars; tafheem pages longer)
N_GRAM = 6               # a shared run of this many words = verbatim reuse -> rejected
MIN_WORDS, MAX_WORDS = 4, 40
TOKEN_BUDGET = 1500
# gpt-5-mini list prices (USD per 1M tokens) at the time of writing. The
# estimate is only as good as these two numbers: check the pricing page.
PRICE_INPUT_PER_M = 0.25
PRICE_OUTPUT_PER_M = 2.00
# Bengali tokenizes badly (~3-4x English); 1 token per 1.5 characters is a
# conservative input estimate for the page text.
CHARS_PER_TOKEN = 1.5
PROMPT_OVERHEAD_TOKENS = 450
OUTPUT_TOKENS_PER_CALL = 700  # includes the reasoning tokens a gpt-5 model spends

_SYSTEM = """তুমি একটি বাংলা ইসলামিক গ্রন্থাগারের মূল্যায়ন প্রশ্ন তৈরি করছো। নিচে একটি বইয়ের একটি পৃষ্ঠা দেওয়া
আছে। এমন {n}টি প্রশ্ন লেখো যেগুলোর উত্তর **শুধু এই পৃষ্ঠাতেই** সম্পূর্ণভাবে আছে, এবং যা একজন সাধারণ পাঠক
নিজের ভাষায় জিজ্ঞাসা করতে পারে।

নিয়ম:
1. প্রমিত বাংলায়, একটি বাক্যে, প্রশ্নবোধক চিহ্ন দিয়ে শেষ।
2. পৃষ্ঠার কোনো বাক্য বা বাক্যাংশ হুবহু তুলে দেবে না; নিজের শব্দে, প্রয়োজনে প্রতিশব্দ ব্যবহার করে লেখো।
3. প্রশ্নে বইয়ের নাম বা অধ্যায়ের নাম উল্লেখ করবে না।
4. প্রশ্নটি পৃষ্ঠা না দেখেও বোধগম্য হতে হবে (কোনো "এই", "উপরের" ইত্যাদি ইঙ্গিত নয়)।
5. প্রতিটি প্রশ্নের সাথে এক লাইনে লেখো পৃষ্ঠার কোন তথ্য দিয়ে উত্তরটি দেওয়া যায়।

শুধুমাত্র একটি JSON অবজেক্ট দাও: {{"questions": [{{"question": "...", "answer_in_page": "..."}}]}}"""


# ---------------------------------------------------------------------------
# text checks
# ---------------------------------------------------------------------------

def _words(text: str) -> list[str]:
    # \w alone breaks Bengali words apart (vowel signs are combining marks),
    # so the Bengali block is spelled out.
    return re.findall(r"[\w\u0980-\u09ff]+", text)


def ngrams(text: str, n: int = N_GRAM) -> set[tuple[str, ...]]:
    w = _words(text)
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def reuses_page_text(question: str, page_text: str, n: int = N_GRAM) -> bool:
    """True when the question shares a run of `n` consecutive words with the page."""
    return bool(ngrams(question, n) & ngrams(page_text, n))


def draft_ok(question: str, page_text: str) -> str | None:
    """None when the draft is acceptable, else the reason it is dropped."""
    q = " ".join(question.split())
    words = _words(q)
    if not (MIN_WORDS <= len(words) <= MAX_WORDS):
        return f"length {len(words)} words"
    if not _is_bengali_output(q):
        return "not Bengali"
    if not q.endswith(("?", "？")):
        return "not a question"
    if reuses_page_text(q, page_text):
        return f"reuses {N_GRAM}+ consecutive words of the page"
    return None


# ---------------------------------------------------------------------------
# records
# ---------------------------------------------------------------------------

def next_index(existing: list[dict], prefix: str = "bn") -> int:
    nums = [int(r["id"].split("-")[1]) for r in existing
            if isinstance(r.get("id"), str) and r["id"].startswith(prefix + "-") and r["id"].split("-")[1].isdigit()]
    return (max(nums) + 1) if nums else 1


def make_record(index: int, page: dict, question: str, answer_in_page: str) -> dict:
    return {
        "id": f"bn-{index:04d}",
        "slice": "answerable_bn",
        "category": page.get("category"),
        "source_db": page.get("source_db"),
        "language": "bn",
        "question": " ".join(question.split()),
        "previous_turn": None,
        "expected_row_keys": [page["row_key"]],
        "answerable": True,
        "variant_of": None,
        "review": "pending",
        "notes": f"drafted from {page.get('book')} › {page.get('chapter')}; page says: "
                 f"{' '.join(str(answer_in_page).split())[:240]}",
    }


def interleave(pages: list[dict]) -> list[dict]:
    """Round-robin over (category, source_db) groups so the first N pages
    taken are spread over both, in the sample's own (seeded) order."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for p in pages:
        groups[(p.get("category"), p.get("source_db"))].append(p)
    out: list[dict] = []
    while any(groups.values()):
        for key in list(groups):
            if groups[key]:
                out.append(groups[key].pop(0))
    return out


def attach_source_db(pages: list[dict]) -> None:
    """sample_corpus_pages does not write source_db; read it from Postgres."""
    missing = [p for p in pages if not p.get("source_db") and p.get("row_key", "").startswith("page_")]
    if not missing:
        return
    from app.db.models import Page
    from app.db.session import SessionLocal

    ids = [int(p["row_key"].split("_")[1]) for p in missing]
    with SessionLocal() as s:
        rows = dict(s.query(Page.id, Page.source_db).filter(Page.id.in_(ids)).all())
    for p in missing:
        p["source_db"] = rows.get(int(p["row_key"].split("_")[1]))


# ---------------------------------------------------------------------------
# the call
# ---------------------------------------------------------------------------

def draft_for_page(page: dict, per_page: int) -> list[dict]:
    """Raw drafts from the model: [{"question", "answer_in_page"}]; [] on bad JSON."""
    text = (page.get("content") or "")[:PAGE_CHARS]
    response = complete(
        "eval_draft",
        [
            {"role": "system", "content": _SYSTEM.format(n=per_page)},
            {"role": "user", "content": f"বই: {page.get('book')}\nঅধ্যায়: {page.get('chapter')}\n\n{text}"},
        ],
        token_budget=TOKEN_BUDGET,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content
    try:
        data = json.loads(content) if isinstance(content, str) else {}
    except ValueError:
        return []
    items = data.get("questions") if isinstance(data, dict) else None
    return [i for i in (items or []) if isinstance(i, dict) and isinstance(i.get("question"), str)]


def estimate(pages: list[dict]) -> dict:
    in_tokens = sum(PROMPT_OVERHEAD_TOKENS + len((p.get("content") or "")[:PAGE_CHARS]) / CHARS_PER_TOKEN
                    for p in pages)
    out_tokens = OUTPUT_TOKENS_PER_CALL * len(pages)
    usd = in_tokens / 1e6 * PRICE_INPUT_PER_M + out_tokens / 1e6 * PRICE_OUTPUT_PER_M
    return {"pages": len(pages), "input_tokens": int(in_tokens), "output_tokens": int(out_tokens),
            "usd": round(usd, 3)}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m scripts.draft_eval_questions")
    ap.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE, help="sample_corpus_pages output")
    ap.add_argument("--pages", type=int, default=150, help="pages to draft from (default 150)")
    ap.add_argument("--per-page", type=int, default=2, choices=(1, 2))
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--yes", action="store_true", help="accept the cost estimate and call the model")
    args = ap.parse_args(argv)

    pages = json.loads(args.sample.read_text(encoding="utf-8"))
    existing: list[dict] = json.loads(args.out.read_text(encoding="utf-8")) if args.out.exists() else []
    done = {rk for r in existing for rk in r.get("expected_row_keys", []) if r.get("review") != "rejected"}
    attach_source_db(pages)
    todo = [p for p in interleave(pages) if p["row_key"] not in done][: args.pages]

    est = estimate(todo)
    print(f"sample: {args.sample} ({len(pages)} pages); already drafted: {len(done)}; to draft: {len(todo)}")
    print(f"estimate: ~{est['input_tokens']:,} input + ~{est['output_tokens']:,} output tokens "
          f"= ~${est['usd']} at ${PRICE_INPUT_PER_M}/${PRICE_OUTPUT_PER_M} per 1M (gpt-5-mini list price)")
    by_group = defaultdict(int)
    for p in todo:
        by_group[(p.get("category"), p.get("source_db"))] += 1
    for (cat, src), n in sorted(by_group.items(), key=lambda kv: -kv[1]):
        print(f"  {n:>3}  {cat} / {src}")
    if not todo:
        return 0
    if not args.yes:
        print("\nnot calling the model: re-run with --yes to accept the estimate.")
        return 0

    index = next_index(existing)
    kept, dropped = [], defaultdict(int)
    for i, page in enumerate(todo, start=1):
        try:
            drafts = draft_for_page(page, args.per_page)
        except Exception as e:  # keep going; the summary says what failed
            print(f"  [{i}/{len(todo)}] {page['row_key']}: call failed: {e}")
            dropped["call failed"] += 1
            continue
        if not drafts:
            dropped["bad JSON / no questions"] += 1
        for d in drafts[: args.per_page]:
            reason = draft_ok(d["question"], page.get("content") or "")
            if reason:
                dropped[reason] += 1
                continue
            kept.append(make_record(index, page, d["question"], d.get("answer_in_page", "")))
            index += 1
        print(f"  [{i}/{len(todo)}] {page['row_key']} {page.get('category')}: kept {len(kept)} so far")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(existing + kept, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"\nwrote {len(kept)} pending drafts to {args.out} ({len(existing) + len(kept)} records total)")
    for reason, n in sorted(dropped.items(), key=lambda kv: -kv[1]):
        print(f"  dropped {n:>3}: {reason}")
    print("Next: review them (scripts/eval/README.md), then python -m scripts.validate_eval_set "
          f"{args.out} and python -m scripts.split_eval_set.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
