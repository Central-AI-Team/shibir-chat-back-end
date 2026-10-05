"""Bengali Islamic RAG — query_rewriter.py

Simple version: input may be Bengali, Banglish (Roman-script Bengali),
English, or Arabic -- one LLM call translates/normalizes it into a single,
direct Bengali string. No route detection, no per-route prompts, no variant
expansion, no preserved-original field.

Public API:
    expand_query(query, max_variants=None) -> tuple[str, ...]
    detect_language(text) -> "BENGALI" | "BANGLISH" | "ENGLISH_ARABIC"
    get_fallback_count() -> int
    _reset_fallback_count() -> None

`detect_language()` and its helpers are kept as standalone utilities (used by
scripts/debug_pipeline.py for display, and tested directly) -- they are no
longer used inside expand_query() itself.
"""

from __future__ import annotations

import re
import threading
from collections import OrderedDict
from functools import lru_cache

from app.core import timing
from app.core.config import settings
from app.core.llm import complete
from app.rag.chunker import normalize


# ══════════════════════════════════════════════════════════════════════════════
# Language detection (pure Python, no LLM) -- standalone utility, not used by
# expand_query() below.
# ══════════════════════════════════════════════════════════════════════════════

def _looks_bengali(text: str) -> bool:
    """True when >50 % of alphabetic chars are Bengali Unicode (U+0980–U+09FF)."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    bengali = sum(1 for c in letters if "ঀ" <= c <= "৿")
    return bengali / len(letters) > 0.50


def mostly_bengali(text: str, ratio: float | None = None) -> bool:
    """True when >= `ratio` (default settings.rewrite_skip_bengali_ratio) of the
    alphabetic chars are Bengali script -- such a query needs no translation."""
    ratio = settings.rewrite_skip_bengali_ratio if ratio is None else ratio
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    return sum(1 for c in letters if "ঀ" <= c <= "৿") / len(letters) >= ratio


def _looks_arabic(text: str) -> bool:
    """True when >30 % of alphabetic chars are Arabic Unicode (U+0600–U+077F)."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    arabic = sum(
        1 for c in letters
        if "؀" <= c <= "ۿ" or "ݐ" <= c <= "ݿ"
    )
    return arabic / len(letters) > 0.30


# Banglish structural words — verb/question particles that only appear in
# romanised Bengali, never in real English sentences.
_BANGLISH_STRUCTURAL = re.compile(
    r"\b(?:ami|tumi|apni|ki|ke|na|hoy|ache|achen|achis|"
    r"bhalo|thaki|thako|thaken|kora|kori|koren|"
    r"rakha|rakhar|niyom|gurutto|tathyo|kemon|koto|"
    r"bolun|bolen|bolo|jani|janin|janbo|"
    r"korar|kivabe|ki_vabe|kothay|kobe|keno)\b",
    re.IGNORECASE,
)


def _is_banglish(text: str) -> bool:
    """
    Banglish = ≥70 % Latin chars AND at least one Bengali structural word.

    An Islamic noun alone ("salat importance") is NOT enough — it could be
    an English phrase. A structural Banglish connector must also be present.
    """
    if _looks_bengali(text) or _looks_arabic(text):
        return False
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    if sum(1 for c in letters if c.isascii()) / len(letters) < 0.70:
        return False
    return bool(_BANGLISH_STRUCTURAL.search(text))


def detect_language(text: str) -> str:
    """
    Classify a query into one of three categories (display/logging only).

    Priority (order matters):
      1. Bengali script  → "BENGALI"        (Unicode range check)
      2. Arabic script   → "ENGLISH_ARABIC" (Unicode range check)
      3. Banglish        → "BANGLISH"        (Latin + structural heuristic)
      4. Fallback        → "ENGLISH_ARABIC"  (pure English or unknown)
    """
    if _looks_bengali(text):
        return "BENGALI"
    if _looks_arabic(text):
        return "ENGLISH_ARABIC"
    if _is_banglish(text):
        return "BANGLISH"
    return "ENGLISH_ARABIC"


# ══════════════════════════════════════════════════════════════════════════════
# One universal prompt: translate/normalize whatever comes in into plain,
# direct Bengali.
# ══════════════════════════════════════════════════════════════════════════════

# Shared with the combined intent+rewrite prompt in intent_classifier.py.
# Incident 2026-10-05: "How can I improve my daily prayers?" was translated
# literally ("দৈনন্দিন প্রার্থনা উন্নত করা"): wrong term (the books say নামাজ)
# and wrong phrasing (they say খুশু/মনোযোগ, not "উন্নত করা"), so the top rerank
# score was 0.08 and the gate refused. Bengali-script queries skip the rewrite,
# so this only affects English/Banglish input.
SEARCH_RULES = """\
ইসলামি পরিভাষা বইয়ের প্রচলিত বাংলা শব্দে লিখবে: prayer/salah/salat → নামাজ (কখনো "প্রার্থনা" নয়),
fasting/sawm → রোজা, charity/zakat → যাকাত, supplication/dua → দোয়া, pilgrimage → হজ্জ।
রূপান্তরটি হবে বইয়ে খোঁজার উপযোগী প্রশ্ন: আক্ষরিক অনুবাদ নয়, বরং বইয়ে সাধারণত যেসব শব্দে বিষয়টি
আলোচিত হয় সেগুলো ব্যবহার করবে (যেমন "improve my prayers" → নামাজে খুশু ও মনোযোগ অর্জনের উপায়)।
প্রশ্নের মূল বিষয় বজায় রাখবে, উত্তর বা নতুন তথ্য যোগ করবে না, সর্বোচ্চ এক বা দুই বাক্য।"""

_PROMPT_TRANSLATE = """\
তুমি একটি বাংলা ইসলামিক গ্রন্থাগার সার্চ সিস্টেমের query প্রসেসর।
ব্যবহারকারীর প্রশ্নটি বাংলা, Banglish (রোমান হরফে বাংলা), ইংরেজি, বা আরবি -- যে ভাষাতেই হোক।

তোমার একমাত্র কাজ: প্রশ্নটিকে শুদ্ধ, সহজ, প্রমিত বাংলায় রূপান্তর করে ফেরত দেওয়া।
প্রশ্নের উত্তর দেবে না, কেবল বাংলা রূপান্তরটি দেবে।

নিয়ম:
  • অর্থ পরিবর্তন করবে না, মূল ভাবটি অক্ষুণ্ণ রাখো।
  • প্রশ্নটি যদি ইতিমধ্যে বাংলায় থাকে, শুধু শুদ্ধ বাংলায় রাখো (রূপান্তরের প্রয়োজন নেই)।
  • কেবল বাংলা টেক্সট ফেরত দেবে -- অন্য কোনো ব্যাখ্যা, উদ্ধৃতি চিহ্ন, বা JSON নয়।

""" + SEARCH_RULES + """

প্রশ্ন: {q}"""


# ══════════════════════════════════════════════════════════════════════════════
# Eval instrumentation
# ══════════════════════════════════════════════════════════════════════════════

_fallback_count: int = 0

def _reset_fallback_count() -> None:
    global _fallback_count
    _fallback_count = 0

def get_fallback_count() -> int:
    return _fallback_count


# ══════════════════════════════════════════════════════════════════════════════
# Rewrites produced elsewhere (the combined intent+rewrite call in
# intent_classifier.py), consumed once by expand_query() below. Bounded so a
# primed-but-never-retrieved query (e.g. a NOTE request) cannot leak memory.
# ══════════════════════════════════════════════════════════════════════════════

_primed: OrderedDict[str, str] = OrderedDict()
_primed_lock = threading.Lock()
_PRIMED_MAX = 256


def prime_rewrite(query: str, bengali: str) -> None:
    query, bengali = normalize(query), normalize(bengali)
    if not query or not bengali:
        return
    with _primed_lock:
        _primed[query] = bengali
        while len(_primed) > _PRIMED_MAX:
            _primed.popitem(last=False)


# ══════════════════════════════════════════════════════════════════════════════
# Main entry point (signature unchanged for retriever.py / eval scripts)
# ══════════════════════════════════════════════════════════════════════════════

@lru_cache(maxsize=2048)
def expand_query(query: str, max_variants: int | None = None) -> tuple[str, ...]:
    """Translate/normalize the input into a single, direct Bengali string.

      • Returns a 1-element tuple: (bengali_translation,).
      • Falls back to (raw_query,) on any LLM failure.

    `max_variants` is accepted only for call-site compatibility
    (app/rag/retriever.py passes it through) and is unused.
    """
    query = normalize(query)
    if not query:
        return ()

    if mostly_bengali(query):
        # Already Bengali: the LLM rewrite only re-spelled it (3-4 s per call).
        timing.mark("query_rewrite", "skipped: already Bengali")
        return (query,)
    with _primed_lock:
        primed = _primed.pop(query, None)
    if primed:
        timing.mark("query_rewrite", "skipped: rewritten in intent call")
        return (primed,)

    prompt = _PROMPT_TRANSLATE.format(q=query)
    try:
        resp = complete("rewrite", [{"role": "user", "content": prompt}], token_budget=2000)
        bn = normalize(resp.choices[0].message.content.strip())
        if not bn:
            raise ValueError("empty translation")
    except Exception:
        global _fallback_count
        _fallback_count += 1
        return (query,)

    return (bn,)
