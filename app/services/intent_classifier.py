"""Intent classification for /chat.

Mirrors qa_service.py's _is_conversational() approach: cheap, deterministic
regex matching first (anchored, covers Bengali script + common Banglish
spellings), and only falls back to an LLM call when nothing matches.
"""

from __future__ import annotations

import logging
import re

from app.core.llm import get_client, get_model

logger = logging.getLogger(__name__)

_NOTE_RE = re.compile(
    r"(?:নোট\s*(?:বানা|তৈরি|লিখ|করে)|সারাংশ\s*(?:করো|কর|বানা)|সংক্ষেপ\s*(?:করো|কর)|"
    r"note\s*(?:banao|banan|likho|toiri|kore)|summary\s*(?:koro|kore|banao)|summarize)",
    re.IGNORECASE,
)

_ROLEPLAY_RE = re.compile(
    r"(?:রোল\s*প্লে|রোলপ্লে|তুমি\s+এখন\s+.+\s+হয়ে\s+যাও|তুমি\s+.+\s+হও|অভিনয়\s*করো|"
    r"চরিত্রে\s*অভিনয়|"
    r"role\s*play|roleplay|act\s*as|pretend\s*(?:to\s*be|you\s*are))",
    re.IGNORECASE,
)

_ROLEPLAY_EXIT_RE = re.compile(
    r"(?:রোল\s*প্লে\s*বন্ধ|রোলপ্লে\s*বন্ধ|রোল\s*প্লে\s*(?:থেকে\s*)?বের|অভিনয়\s*বন্ধ|"
    r"স্বাভাবিক\s*হয়ে\s*যাও|"
    r"stop\s*roleplay|exit\s*roleplay|end\s*roleplay|quit\s*roleplay)",
    re.IGNORECASE,
)

_SUGGESTION_RE = re.compile(
    r"(?:পরামর্শ\s*দাও|পরামর্শ\s*দিন|পরামর্শ\s*চাই|আমার\s*কি\s*করা\s*উচিত|কি\s*করা\s*উচিত|"
    r"কী\s*করা\s*উচিত|মতামত\s*(?:দাও|দিন)|"
    r"suggestion\s*(?:dao|den)?|suggest\s*(?:me)?|"
    r"ki\s*kora\s*uchit|advice\s*(?:dao|den)?)",
    re.IGNORECASE,
)

_INTENT_PATTERNS = (
    ("NOTE", _NOTE_RE),
    ("ROLEPLAY", _ROLEPLAY_RE),
    ("SUGGESTION", _SUGGESTION_RE),
)

_VALID_INTENTS = {"NOTE", "ROLEPLAY", "SUGGESTION", "QA"}

_CLASSIFIER_SYSTEM = """তুমি একজন ইনটেন্ট ক্লাসিফায়ার। ব্যবহারকারীর বার্তাটি পড়ে
নিচের চারটি ক্যাটাগরির মধ্যে ঠিক একটি বেছে নাও:

NOTE - ব্যবহারকারী কোনো অধ্যায় বা বইয়ের নোট/সারাংশ চাইছে।
ROLEPLAY - ব্যবহারকারী তোমাকে কোনো চরিত্রে অভিনয় করতে বলছে।
SUGGESTION - ব্যবহারকারী পরামর্শ/মতামত/সুপারিশ চাইছে।
QA - ব্যবহারকারী সরাসরি কোনো তথ্যভিত্তিক প্রশ্ন জিজ্ঞাসা করছে।

শুধুমাত্র একটি শব্দ দিয়ে উত্তর দাও: NOTE, ROLEPLAY, SUGGESTION, অথবা QA।
অন্য কিছু লিখো না।"""


def _classify_with_llm(message: str) -> str:
    response = get_client().chat.completions.create(
        model=get_model(),
        messages=[
            {"role": "system", "content": _CLASSIFIER_SYSTEM},
            {"role": "user", "content": message},
        ],
    )
    raw = (response.choices[0].message.content or "").strip().upper()
    for intent in _VALID_INTENTS:
        if intent in raw:
            return intent
    logger.info("intent_classifier_fallback_to_qa raw=%r", raw)
    return "QA"


def classify_intent(message: str, has_active_roleplay_session: bool) -> str:
    normalized = message.strip()

    if has_active_roleplay_session and not _ROLEPLAY_EXIT_RE.search(normalized):
        # An ongoing roleplay conversation shouldn't get reclassified as QA
        # (or anything else) on every follow-up turn.
        return "ROLEPLAY"

    for intent, pattern in _INTENT_PATTERNS:
        if pattern.search(normalized):
            return intent

    return _classify_with_llm(normalized)
