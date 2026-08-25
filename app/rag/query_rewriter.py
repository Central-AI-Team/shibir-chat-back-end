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

from app.core.llm import get_client, get_model
from app.rag.chunker import normalize

_REWRITE_PROMPT = """তুমি একটি বাংলা সার্চ সিস্টেমের query প্রসেসর।

ব্যবহারকারীর প্রশ্নটি নিচের যেকোনো রূপে থাকতে পারে:
- বাংলা লিপিতে
- রোমান হরফে বাংলা (Banglish), যেমন "namajer gurutto ki"
- ইংরেজিতে

তোমার কাজ: প্রশ্নটি বাংলা লিপিতে রূপান্তর করা, এবং সার্চের জন্য
২টি বিকল্প রূপ তৈরি করা (সমার্থক শব্দ বা ভিন্নভাবে লেখা)।

শুধুমাত্র নিচের JSON ফরম্যাটে উত্তর দাও, অন্য কোনো লেখা নয়:
{{"bn": "বাংলা লিপিতে মূল প্রশ্ন", "variants": ["বিকল্প ১", "বিকল্প ২"]}}

প্রশ্ন: {q}"""


def _looks_bengali(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    bengali = sum(1 for c in letters if "\u0980" <= c <= "\u09ff")
    return bengali / len(letters) > 0.5


@lru_cache(maxsize=2048)
def expand_query(query: str) -> tuple[str, ...]:
    """Return every string that should be embedded and searched.

    Always includes the raw query -- book titles and Arabic-derived terms are
    sometimes written in Latin script in the corpus too, so the original is
    still worth searching.
    """
    query = normalize(query)
    if not query:
        return ()

    try:
        resp = get_client().chat.completions.create(
            model=get_model(),
            messages=[{"role": "user", "content": _REWRITE_PROMPT.format(q=query)}],
            # gpt-5-mini only supports the default temperature (1) -- passing
            # any other value is a 400.
            #
            # It also spends completion-token budget on internal "thinking"
            # before emitting visible output, same as the Gemini model this
            # was tuned against. 300 was too low -- the call hit
            # finish_reason="length" mid-JSON and silently fell back to the
            # raw (un-rewritten) query on every call. Also note: reasoning
            # models take max_completion_tokens, not max_tokens.
            max_completion_tokens=2000,
        )
        raw = resp.choices[0].message.content.strip()
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        data = json.loads(raw)
        out = [data["bn"], *data.get("variants", [])]
    except Exception:
        # Never let rewriting break the request -- fall back to the raw query.
        out = []

    if not _looks_bengali(query) or not out:
        out.append(query)

    seen, result = set(), []
    for q in out:
        q = normalize(q)
        if q and q not in seen:
            seen.add(q)
            result.append(q)
    return tuple(result)