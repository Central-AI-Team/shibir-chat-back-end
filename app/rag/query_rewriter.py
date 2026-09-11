"""Query normalization + expansion.  NEW FILE.

Fixes: "Banglish likhle information khuje pay na".

Even with bge-m3, romanized Bengali ("namajer gurutto ki") sits in a different
region of the vector space than the Bengali-script corpus. No amount of
retrieval tuning fixes a script mismatch -- the query has to be converted
before it is embedded.

This costs one extra Gemini Flash call (~200ms, negligible price). Results are
cached, because users repeat the same questions constantly.
"""

from __future__ import annotations

import json
from functools import lru_cache

from app.core.config import settings
from app.core.llm import complete
from app.rag.chunker import normalize

_REWRITE_PROMPT = """তুমি একটি বাংলা ইসলামিক গ্রন্থাগার সার্চ সিস্টেমের query প্রসেসর।

ব্যবহারকারীর প্রশ্নটি নিচের যেকোনো রূপে থাকতে পারে:
- বাংলা লিপিতে
- রোমান হরফে বাংলা (Banglish), যেমন "namajer gurutto ki"
- ইংরেজিতে

তোমার কাজ:
১. প্রশ্নটি শুদ্ধ, স্বাভাবিক বাংলা লিপিতে রূপান্তর করা (এটি "bn")।
২. সার্চের জন্য সর্বোচ্চ {max_variants}টি বিকল্প রূপ তৈরি করা, যেন প্রতিটি রূপ
   ভিন্ন কোণ থেকে একই তথ্য খুঁজে বের করার চেষ্টা করে -- শুধু শব্দ এদিক-ওদিক
   করা নয়:
   - মূল বিশেষ্য/পরিভাষার সমার্থক বা প্রচলিত বিকল্প বাংলা শব্দ ব্যবহার করে একটি রূপ,
   - একটি প্রাতিষ্ঠানিক/পারিভাষিক (formal) রূপ এবং একটি কথ্য/সহজ (colloquial) রূপ,
   - প্রশ্নটি Banglish বা ইংরেজিতে হলে, অন্তত একটি সম্পূর্ণ বিশুদ্ধ বাংলা লিপির রূপ
     অবশ্যই অন্তর্ভুক্ত করা,
   - আরবি-মূলীয় ইসলামি পরিভাষা থাকলে (যেমন নামায/সালাত, রোযা/সাওম, ওযু/অজু) তার
     প্রচলিত ভিন্ন বানান/প্রতিবর্ণীকরণ ব্যবহার করে একটি রূপ।

কঠোর নিয়ম: প্রশ্নের অর্থ পাল্টাবে না, প্রশ্নের উত্তর দেবে না -- শুধু সার্চ-স্ট্রিং
তৈরি করবে। প্রতিটি রূপ একটি সম্পূর্ণ প্রশ্ন/বাক্যাংশ হতে হবে, বিচ্ছিন্ন শব্দ নয়।

শুধুমাত্র নিচের JSON ফরম্যাটে উত্তর দাও, অন্য কোনো লেখা নয়:
{{"bn": "বাংলা লিপিতে মূল প্রশ্ন", "variants": ["বিকল্প ১", "বিকল্প ২", ...]}}

প্রশ্ন: {q}"""


def _looks_bengali(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    bengali = sum(1 for c in letters if "\u0980" <= c <= "\u09ff")
    return bengali / len(letters) > 0.5


# Eval-only instrumentation for scripts/eval_task_routing.py: how often the
# LLM call or JSON parse below fails and expand_query falls back to the raw
# query. Purely additive -- nothing in expand_query's behavior or return
# shape changes; this just counts an event that already happens. A cached
# expand_query() hit skips the try/except entirely (see lru_cache below), so
# measuring a genuine failure RATE for a given model requires
# expand_query.cache_clear() before the run, not just _reset_fallback_count().
_fallback_count = 0


def _reset_fallback_count() -> None:
    global _fallback_count
    _fallback_count = 0


def get_fallback_count() -> int:
    return _fallback_count


@lru_cache(maxsize=2048)
def expand_query(query: str, max_variants: int | None = None) -> tuple[str, ...]:
    """Return every string that should be embedded and searched.

    Includes the raw query when there's room -- book titles and
    Arabic-derived terms are sometimes written in Latin script in the corpus
    too, so the original is worth searching. It's appended last, so a small
    max_variants that's already full of LLM variants can drop it; that's the
    hard cap working as intended, not a bug.

    max_variants caps the tuple at max_variants+1 total (the canonical "bn"
    form plus at most max_variants more), regardless of how many the prompt
    -- or a model ignoring the prompt's own count -- produces. Defaults to
    settings.max_variants. This is also the hook
    scripts/eval_query_expansion.py uses to sweep variant counts against the
    SAME rewriter/prompt -- it's a real parameter (not test-only) so the
    lru_cache key correctly varies with it instead of serving a stale count.
    """
    max_variants = settings.max_variants if max_variants is None else max_variants
    query = normalize(query)
    if not query:
        return ()

    try:
        # Routed via complete("rewrite", ...) -- app/core/llm.py -- so the
        # actual model comes from settings.model_by_task, not hardcoded here.
        # token_budget=2000: gpt-5-mini spends completion-token budget on
        # internal "thinking" before emitting visible output. 300 was too
        # low -- the call hit finish_reason="length" mid-JSON and silently
        # fell back to the raw (un-rewritten) query on every call. complete()
        # turns this into whichever token-cap kwarg the routed model actually
        # needs, so this budget survives "rewrite" being routed to a
        # different model family later.
        resp = complete(
            "rewrite",
            [{"role": "user",
              "content": _REWRITE_PROMPT.format(q=query, max_variants=max_variants)}],
            token_budget=2000,
        )
        raw = resp.choices[0].message.content.strip()
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        data = json.loads(raw)
        out = [data["bn"], *data.get("variants", [])]
    except Exception:
        # Never let rewriting break the request -- fall back to the raw query.
        global _fallback_count
        _fallback_count += 1
        out = []

    if not _looks_bengali(query) or not out:
        out.append(query)

    seen, result = set(), []
    for q in out:
        q = normalize(q)
        if q and q not in seen:
            seen.add(q)
            result.append(q)
            # Hard cap: bn (or the raw-query fallback) always first, plus at
            # most max_variants more -- even if the model over-produces.
            if len(result) >= max_variants + 1:
                break
    return tuple(result)