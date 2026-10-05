"""Intent classification for /chat.

Mirrors qa_service.py's _is_conversational() approach: cheap, deterministic
regex matching first (anchored, covers Bengali script + common Banglish
spellings), and only falls back to an LLM call when nothing matches.
"""

from __future__ import annotations

import logging
import re

import json

from app.core import timing, tracing
from app.core.config import settings
from app.core.llm import complete
from app.rag.query_rewriter import SEARCH_RULES, mostly_bengali, prime_rewrite

logger = logging.getLogger(__name__)

_NOTE_RE = re.compile(
    r"(?:নোটস?\s*(?:বানা|তৈরি|লিখ|করে|দাও|দিন|চাই|দরকার)|"
    r"(?:সারাংশ|সারসংক্ষেপ)\s*(?:করো|কর|বানা|চাই|দাও|দিন|লিখ)|সংক্ষেপ\s*(?:করো|কর)|"
    r"(?:অধ্যায়|চ্যাপ্টার)\S*\s+(?:সংক্ষেপে|সারাংশ)|"
    r"note\s*(?:banao|banan|likho|toiri|kore|dao|den|chai)|"
    r"summary\s*(?:koro|kore|banao|dao|den|chai)|summarize|"
    r"(?:make|write|create|generate|give\s+me)\s+(?:a\s+)?(?:chapter\s+)?(?:notes?|summary))",
    re.IGNORECASE,
)

_ROLEPLAY_RE = re.compile(
    r"(?:রোল\s*প্লে|রোলপ্লে|তুমি\s+এখন\s+.+\s+হয়ে\s+যাও|তুমি\s+.+\s+হও|অভিনয়\s*(?:করো|করুন|কর)|"
    r"চরিত্রে\s*(?:অভিনয়|থেকে)|ভূমিকায়\s+(?:থেকে|আমার|আমাকে|কথা|অভিনয়)|ভান\s*করো|"
    r"(?:ধরো|ধরুন)\s+তুমি|"
    r"role\s*play|roleplay|act\s*(?:as|like)|abhinoy\s*koro|pretend\s*(?:to\s*be|you\s*are))",
    re.IGNORECASE,
)

_ROLEPLAY_EXIT_RE = re.compile(
    r"(?:রোল\s*প্লে\s*বন্ধ|রোলপ্লে\s*বন্ধ|রোল\s*প্লে\s*(?:থেকে\s*)?বের|অভিনয়\s*বন্ধ|"
    r"স্বাভাবিক\s*হয়ে\s*যাও|"
    r"stop\s*roleplay|exit\s*roleplay|end\s*roleplay|quit\s*roleplay)",
    re.IGNORECASE,
)

_SUGGESTION_RE = re.compile(
    r"(?:পরামর্শ\s*(?:দাও|দিন|দেন|দেবেন|চাই)|আমার\s*কি\s*করা\s*উচিত|কি\s*করা\s*উচিত|"
    r"কী\s*করা\s*উচিত|মতামত\s*(?:দাও|দিন)|সাজেশন|সাজেস্ট|সুপারিশ\s*(?:করো|করুন|দাও)|"
    r"উচিত\s*[?？।]?\s*$|"
    r"suggestion\s*(?:dao|den)?|suggest\s*(?:me)?|recommend|poramorsho|"
    r"ki\s*kora\s*uchit|advice\s*(?:dao|den)?)",
    re.IGNORECASE,
)

_INTENT_PATTERNS = (
    ("NOTE", _NOTE_RE),
    ("ROLEPLAY", _ROLEPLAY_RE),
    ("SUGGESTION", _SUGGESTION_RE),
)

# Order matters here: if the LLM ever ignores the "one word only" instruction
# and its reply contains more than one of these as a substring, _classify_
# with_llm's `if intent in raw` loop picks whichever comes first. A set has no
# guaranteed iteration order (varies with Python's per-process hash seed), so
# that pick would be non-deterministic across runs for the identical raw
# response -- a tuple pins it to this priority order instead.
_VALID_INTENTS = ("NOTE", "ROLEPLAY", "SUGGESTION", "QA")

_CLASSIFIER_SYSTEM = """তুমি একজন ইনটেন্ট ক্লাসিফায়ার। ব্যবহারকারীর বার্তাটি পড়ে
নিচের চারটি ক্যাটাগরির মধ্যে ঠিক একটি বেছে নাও:

NOTE - ব্যবহারকারী কোনো অধ্যায় বা বইয়ের নোট/সারাংশ চাইছে।
ROLEPLAY - ব্যবহারকারী তোমাকে কোনো চরিত্রে অভিনয় করতে বলছে।
SUGGESTION - ব্যবহারকারী পরামর্শ/মতামত/সুপারিশ চাইছে।
QA - ব্যবহারকারী সরাসরি কোনো তথ্যভিত্তিক প্রশ্ন জিজ্ঞাসা করছে।

শুধুমাত্র একটি শব্দ দিয়ে উত্তর দাও: NOTE, ROLEPLAY, SUGGESTION, অথবা QA।
অন্য কিছু লিখো না।"""


def _classify_with_llm(message: str) -> str:
    response = complete("intent", [
        {"role": "system", "content": _CLASSIFIER_SYSTEM},
        {"role": "user", "content": message},
    ])
    raw = (response.choices[0].message.content or "").strip().upper()
    for intent in _VALID_INTENTS:
        if intent in raw:
            return intent
    logger.info("intent_classifier_fallback_to_qa raw=%r", raw)
    return "QA"


_COMBINED_SYSTEM = """তুমি দুটি কাজ একসাথে করবে। ব্যবহারকারীর বার্তাটি পড়ে:

১) নিচের চারটি ক্যাটাগরির মধ্যে ঠিক একটি বেছে নাও:
NOTE - ব্যবহারকারী কোনো অধ্যায় বা বইয়ের নোট/সারাংশ চাইছে।
ROLEPLAY - ব্যবহারকারী তোমাকে কোনো চরিত্রে অভিনয় করতে বলছে।
SUGGESTION - ব্যবহারকারী পরামর্শ/মতামত/সুপারিশ চাইছে।
QA - ব্যবহারকারী সরাসরি কোনো তথ্যভিত্তিক প্রশ্ন জিজ্ঞাসা করছে।

২) বার্তাটি (বাংলা, Banglish, ইংরেজি বা আরবি যাই হোক) শুদ্ধ, সহজ, প্রমিত বাংলায় রূপান্তর করো।
বার্তার উত্তর দেবে না; অর্থ পরিবর্তন করবে না; মূল ভাবটি অক্ষুণ্ন রাখো।

""" + SEARCH_RULES + """

শুধুমাত্র একটি JSON অবজেক্ট দিয়ে উত্তর দাও, অন্য কিছু লিখো না:
{"intent": "NOTE|ROLEPLAY|SUGGESTION|QA", "rewritten_query": "<বাংলা রূপান্তর>"}"""


def _parse_json(raw: str) -> dict | None:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`").removeprefix("json").strip()
    for candidate in (raw, (re.search(r"\{.*\}", raw, re.DOTALL) or [None])[0]):
        if not candidate:
            continue
        try:
            data = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(data, dict):
            return data
    return None


def _classify_and_rewrite(message: str) -> str | None:
    """ONE LLM call for intent + Bengali rewrite. Returns the intent, or None if
    the reply is not valid JSON with a valid intent and a non-empty rewrite
    (the caller then falls back to the old two-call path). API errors are NOT
    caught here -- they must still surface as the router's 503."""
    response = complete(
        "intent",
        [{"role": "system", "content": _COMBINED_SYSTEM}, {"role": "user", "content": message}],
        token_budget=1500,
        response_format={"type": "json_object"},
    )
    data = _parse_json(response.choices[0].message.content or "")
    if not data:
        return None
    intent = str(data.get("intent", "")).strip().upper()
    rewritten = str(data.get("rewritten_query", "")).strip()
    if intent not in _VALID_INTENTS or not rewritten:
        return None
    if intent in ("QA", "SUGGESTION"):
        prime_rewrite(message, rewritten)  # retrieval's expand_query() reuses it
    return intent


def classify_intent(message: str, has_active_roleplay_session: bool) -> str:
    normalized = message.strip()

    if has_active_roleplay_session and not _ROLEPLAY_EXIT_RE.search(normalized):
        # An ongoing roleplay conversation shouldn't get reclassified as QA
        # (or anything else) on every follow-up turn.
        timing.mark("intent_classification", "session")
        tracing.record_intent("ROLEPLAY", method="session")
        return "ROLEPLAY"

    for intent, pattern in _INTENT_PATTERNS:
        if pattern.search(normalized):
            timing.mark("intent_classification", "regex")
            tracing.record_intent(intent, method="regex")
            return intent

    # A greeting needs no LLM, no retrieval: answer_question()/_qa_stream() turn
    # a QA-classified small-talk message into a canned reply.
    from app.services.qa_service import _is_conversational

    if _is_conversational(normalized):
        timing.mark("intent_classification", "greeting")
        tracing.record_intent("QA", method="conversational")
        return "QA"

    result = None
    if settings.combine_intent_rewrite and not mostly_bengali(normalized):
        timing.mark("intent_classification", "llm fallback (combined with rewrite)")
        result = _classify_and_rewrite(normalized)
        if result is None:
            logger.info("combined_intent_rewrite_invalid_json; falling back to two calls")
            timing.mark("intent_classification", "invalid JSON -> two-call fallback")
    else:
        timing.mark("intent_classification", "llm fallback")
    if result is None:
        result = _classify_with_llm(normalized)
    tracing.record_intent(result, method="llm")
    return result
