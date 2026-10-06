"""Small talk: greetings, salam, thanks, goodbyes and "who are you" questions.

These used to fall into book-QA, fail the relevance gate (a "hi" has nothing to
rank against) and get a canned or "not in the books" reply. This module
recognises them deterministically (no LLM) and answers in a way that fits
Islamic etiquette: a salam is answered with the full reply, "how are you" is
answered and returned.

Matching is against the WHOLE message. Leading pleasantries and address words
("ভাই", "bhai", "apni") are consumed until something else is left:
  - nothing left     -> pure small talk, classify_chitchat() returns remainder "".
  - a question left  -> "hi, নামাজের নিয়ম কী?" returns the greeting subtype plus
                        the question, which must never be dropped.

Why the `regex` package and explicit boundaries instead of `re` and `\\b`: the
stdlib counts Bengali vowel signs and Arabic harakat (combining marks) as
non-word characters, so `\\b` after "আছো" or "খাইরান" does not match and a
Bengali word is shredded at every vowel sign. Boundaries here are lookarounds on
letters, marks and digits. Pattern sources go through normalize_text() like the
message does, so composed and decomposed vowel signs (ো / ে+া) both match.
Because sources are lowercased, they must not contain upper-case escapes
(\\S, \\W, \\B, \\P): lowercasing would silently turn them into their opposite.

Not a pleasantry on its own: "শিবির কে?" / "শিবির কী?" are questions about the
organisation, and a pronoun is required before "কে" ("তুমি কে").

Known trade-off, by design: words that are also real topics ("ধন্যবাদ", "বিদায়",
bare "সালাম") only count as a greeting in front of a question when punctuation
separates them from it, so "বিদায় হজ্জের ভাষণ কী?" stays a question about the
Farewell Hajj instead of becoming a goodbye plus a mangled question.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass

import regex

from app.core.llm import complete
from app.rag.chunker import normalize

logger = logging.getLogger(__name__)

SALAM = "SALAM"
WELLBEING = "WELLBEING"
THANKS = "THANKS"
FAREWELL = "FAREWELL"
BOT_IDENTITY = "BOT_IDENTITY"
OTHER = "OTHER"

# Answered from templates (no LLM) unless the message is clearly English.
_TEMPLATE_SUBTYPES = (SALAM, WELLBEING, THANKS, FAREWELL)
# When several pleasantries are in one message, the first of these wins.
# WELLBEING outranks SALAM so "salam, kemon achen" answers both questions.
_PRIORITY = (BOT_IDENTITY, OTHER, WELLBEING, THANKS, FAREWELL, SALAM)

# How many past messages the LLM path sees (3 exchanges), each cut short: a
# greeting after a long book answer must not drag thousands of tokens along.
_HISTORY_MESSAGES = 6
_HISTORY_CHARS = 600

_PERSONA = """তুমি শিবির চ্যাট-এর বন্ধুসুলভ সহকারী। এটি সাধারণ কথোপকথন, বইয়ের প্রশ্নোত্তর নয়।
- সালামের জবাবে পূর্ণ সালাম দাও; কুশল জিজ্ঞাসার জবাব দিয়ে ফিরে কুশল জানতে চাও।
- ২-৩ বাক্যে, নরম ও শ্রদ্ধাশীল ভাষায় উত্তর দাও। ওয়াজ, তালিকা বা শিরোনাম নয়।
- ব্যবহারকারীর সম্বোধনের স্তর (আপনি/তুমি) অনুসরণ করো।
- ধর্মীয় বিধান বা ফতোয়া নিজে থেকে দেবে না; বই-সম্পর্কিত প্রশ্ন করলে সুন্দরভাবে প্রশ্নটি জিজ্ঞাসা করতে বলো।
- নিজের পরিচয় চাইলে: তুমি বইভিত্তিক একটি এআই সহকারী, মানুষ নও।
- প্রশ্ন ইংরেজিতে হলে উত্তরও ইংরেজিতে দাও (সালামের শব্দগুলো যেমন আছে তেমন রাখতে পারো)।"""

# Used only if the LLM returns nothing (a filtered or empty completion).
_FALLBACK_REPLY = "জি, বলুন — বইয়ের কোনো বিষয়ে জানতে চাইলে জিজ্ঞাসা করতে পারেন।"

_SALAM_REPLY = "ওয়া আলাইকুমুস সালাম ওয়া রাহমাতুল্লাহি ওয়া বারাকাতুহ।"

# Full replies to pure small talk. Every variant belongs to its own subtype, so
# a salam can never be answered with a goodbye.
_SALAM_REPLIES = (  # to an actual salam
    f"{_SALAM_REPLY} বইয়ের কোনো বিষয়ে জানতে চাইলে বলুন, আমি সাহায্য করার চেষ্টা করব।",
    f"{_SALAM_REPLY} বলুন, কীভাবে সাহায্য করতে পারি?",
    f"{_SALAM_REPLY} আশা করি ভালো আছেন। বইয়ের কোন বিষয়ে জানতে চান?",
)
# A plain "hello" is not a salam to answer, so it is met with one.
_HELLO_REPLIES = (
    "আসসালামু আলাইকুম। কেমন আছেন? বইয়ের কোনো বিষয়ে জানতে চাইলে বলুন।",
    "আসসালামু আলাইকুম, স্বাগতম! বলুন, কীভাবে সাহায্য করতে পারি?",
    "আসসালামু আলাইকুম। আপনাকে স্বাগতম, বইয়ের কোন বিষয়ে জানতে চান?",
)
_REPLIES: dict[str, tuple[str, ...]] = {
    WELLBEING: (
        "আলহামদুলিল্লাহ, ভালো আছি। আপনি কেমন আছেন?",
        "আলহামদুলিল্লাহ, ভালো আছি। আপনার খবর কী? বইয়ের কোনো বিষয়ে জানতে চাইলে বলুন।",
        "আল্লাহর রহমতে ভালো আছি, আলহামদুলিল্লাহ। আপনি কেমন আছেন?",
    ),
    THANKS: (
        "ওয়া ইয়্যাকুম, আল্লাহ আপনাকেও উত্তম প্রতিদান দিন। আর কিছু জানার থাকলে বলুন।",
        "ওয়া আইয়্যাকুম, আল্লাহ আপনাকেও উত্তম প্রতিদান দিন। বইয়ের আর কোনো বিষয়ে জানতে চাইলে বলুন।",
        "আপনাকেও ধন্যবাদ। আল্লাহ আপনাকে উত্তম প্রতিদান দিন। আর কিছু জানার থাকলে নির্দ্বিধায় বলুন।",
    ),
    FAREWELL: (
        "আল্লাহ হাফেজ। প্রয়োজনে আবার আসবেন।",
        "আল্লাহ হাফেজ, ভালো থাকবেন। বইয়ের কোনো প্রশ্ন থাকলে আবার জিজ্ঞাসা করবেন।",
        "ফি আমানিল্লাহ। আবার প্রশ্ন নিয়ে আসবেন, ইনশাআল্লাহ।",
    ),
}

# Short replies for when a real question follows ("hi, <question>"): the answer
# comes next, so a follow-up like "how can I help?" would only be noise. No
# FAREWELL entry: a goodbye in front of a question is not worth echoing.
_PREFIX_SALAM = (_SALAM_REPLY,)
_PREFIX_HELLO = ("আসসালামু আলাইকুম।",)
_PREFIXES: dict[str, tuple[str, ...]] = {
    WELLBEING: ("আলহামদুলিল্লাহ, ভালো আছি।",),
    THANKS: ("ওয়া আইয়্যাকুম, আল্লাহ আপনাকেও উত্তম প্রতিদান দিন।",),
}

_PUNCT = frozenset(",.;:!?।…-–—~\"'“”‘’()\n،؟")
_BENGALI = regex.compile(r"[ঀ-৿]")
_LEFT = r"(?<![\p{L}\p{M}\p{N}])"
_RIGHT = r"(?![\p{L}\p{M}\p{N}])"


def normalize_text(text: str) -> str:
    """NFC + whitespace/ZWJ/ZWNJ cleanup (chunker.normalize), then lowercase.

    Applied to the message AND to every pattern source, so a composed and a
    decomposed spelling of the same Bengali word (ো is ে+া, ৌ is ে+ৗ) cannot
    miss each other.
    """
    return normalize(text).lower()


@dataclass(frozen=True)
class _Pattern:
    subtype: str
    rx: regex.Pattern
    english: bool  # a clearly English phrase: replied to by the LLM, in English
    weak: bool  # also an ordinary topic word, see the module docstring
    pure_only: bool  # counts only when it is the whole message
    salam: bool  # an actual salam, so the reply is the salam reply
    hello: bool = False  # a plain hello, so the reply is a salam


def _p(
    subtype: str,
    *sources: str,
    english: bool = False,
    weak: bool = False,
    pure_only: bool = False,
    salam: bool = False,
    hello: bool = False,
) -> _Pattern:
    body = "|".join(normalize_text(s) for s in sources)
    return _Pattern(
        subtype,
        regex.compile(f"{_LEFT}(?:{body}){_RIGHT}"),
        english, weak, pure_only, salam, hello,
    )


_TAIL_BN = r"(?:\s*ওয়া\s*রাহমাতুল্লাহি?(?:\s*ওয়া)?\s*বারাকাতুহু?)?"
_TAIL_EN = r"(?:\s*wa\s*rahmat\w*(?:\s*wa)?\s*barakat\w*)?"
_ALAIKUM_EN = r"(?:alaikum|alaykum|aleikum|alikum|alekum|alaikom|laikum)"

# Order matters: the first pattern that matches at a position wins.
_PATTERNS: tuple[_Pattern, ...] = (
    # --- salam (full phrases; the bare word is below) -----------------------
    _p(SALAM,
       rf"আস্?\s*সালামু?\s*আলাইকুম{_TAIL_BN}",
       rf"(?:a|as)?[\s-]*sala+m+u?[\s'’-]*{_ALAIKUM_EN}{_TAIL_EN}",
       salam=True),
    _p(SALAM,  # replies to a salam
       rf"ওয়া?\s*আ?লাইকুমু?(?:স|ছ)?\s*(?:আস্?)?\s*সালাম{_TAIL_BN}",
       rf"(?:wa)?[\s'’-]*a?laikumu?s?[\s-]*(?:as)?[\s-]*salam{_TAIL_EN}",
       salam=True),
    # --- plain hello (answered with a salam) --------------------------------
    _p(SALAM, r"হ্যালো|হ্যাল্লো|হেলো|হাই+|হেই|হ্যাই|নমস্কার|সুপ্রভাত",
       "namaskar", hello=True),
    _p(SALAM, r"hi+|hel+o+|hey+|gm|good\s*(?:morning|afternoon|evening)",
       english=True, hello=True),
    # --- how are you --------------------------------------------------------
    _p(WELLBEING,
       r"(?:আপনি\s*|তুমি\s*)?কেমন\s*(?:আছেন|আছো|আছিস|আছ|আসেন|আসো)(?:\s*(?:আপনি|তুমি))?",
       r"(?:আপনার|তোমার)\s*(?:কী|কি)\s*খবর",
       r"(?:কী|কি)\s*খবর",
       r"কেমন\s*চলছে",
       r"ভালো\s*আছ(?:েন|ো|িস)(?:\s*(?:কি|কী))?",
       r"(?:apni\s+|tumi\s+)?kemon\s*(?:ach(?:en|o|is)|as(?:en|o|is)|asho)(?:\s+(?:apni|tumi))?",
       r"ki\s*kh(?:o|a)bor",
       r"(?:kmn|kmon)\s*(?:achen|acho|asen|aso)"),
    _p(WELLBEING,
       r"how\s*(?:are|r)\s*(?:you|u)(?:\s*doing)?",
       r"how['’]?s\s*(?:it\s*going|everything|life)",
       r"what['’]?s\s*up",
       r"wass?up",
       english=True),
    # --- thanks -------------------------------------------------------------
    _p(THANKS,
       r"ধন্যবাদ(?:\s*(?:আপনাকে|তোমাকে))?",
       r"থ্যাংক(?:স|\s*ইউ|িউ)?",
       r"থ্যাঙ্ক(?:স|\s*ইউ)?",
       r"শুকরিয়া",
       r"জা[যজ]াকাল্লাহু?(?:\s*খ(?:াই|ায়)রান)?",
       r"dh(?:o|a)nn?(?:o|y|yo)bad",
       r"shukr?iya",
       r"jaz+a?kall?ahu?(?:\s*kh(?:ai|ay|ey)r(?:an)?)?",
       r"jaz+a?k\s*all?ahu?(?:\s*kh(?:ai|ay|ey)r(?:an)?)?",
       weak=True),
    _p(THANKS,
       r"thanks?(?:\s*(?:a\s*lot|so\s*much|you|u))?",
       r"thank\s*(?:you|u)(?:\s*(?:so|very)\s*much)?",
       r"thx|ty",
       english=True, weak=True),
    # --- goodbye ------------------------------------------------------------
    _p(FAREWELL,
       r"বিদায়(?:\s*(?:বন্ধু|ভাই))?",
       r"আল্লাহ্?\s*হাফেজ",
       r"আল্লাহ্?\s*হাফিজ",
       r"খোদা\s*হাফেজ",
       r"ভালো\s*থাকবেন",
       r"টাটা",
       r"বাই(?:\s*বাই)?",
       r"ফি\s*আমানিল্লাহ",
       r"শুভ\s*রাত্রি",
       r"(?:allah|khod?a|khuda)\s*haf[ie]z",
       r"bid(?:ay|ai|ae)",
       r"tata",
       weak=True),
    _p(FAREWELL,
       r"bye(?:[\s-]*bye)?",
       r"good\s*(?:bye|night)",
       r"see\s*(?:you|ya)(?:\s*(?:later|soon))?",
       r"take\s*care",
       english=True, weak=True),
    # --- the bare word "salam" ----------------------------------------------
    # Pure only: "সালাম দেওয়ার নিয়ম কী?" is a question about etiquette.
    _p(SALAM, r"সালামু?", r"(?:a|as)?[\s-]*sala+m+u?|slm",
       pure_only=True, salam=True),
    # --- who are you (pure only; a pronoun is required: "শিবির কে?" is not) --
    _p(BOT_IDENTITY,
       r"(?:তুমি|আপনি|তুই)\s*কে(?:\s*(?:ভাই|বলো|বলুন))?",
       r"(?:তুমি|আপনি)\s*(?:কি|কী)\s*(?:মানুষ|রোবট|বট|এআই|এ\s*আই)",
       r"(?:তোমার|আপনার)\s*(?:নাম|পরিচয়)\s*(?:কি|কী)",
       r"(?:নিজের\s*)?পরিচয়\s*(?:দাও|দিন)",
       r"(?:tumi|apni|tui)\s*ke(?:\s*(?:bhai|vai|bolo|bolen))?",
       r"(?:tomar|apnar)\s*na(?:m|am)\s*ki",
       pure_only=True),
    _p(BOT_IDENTITY,
       r"who\s*(?:are|r)\s*(?:you|u)",
       r"what\s*(?:are|r)\s*(?:you|u)",
       r"what['’]?s\s*your\s*name|your\s*name",
       r"are\s*you\s*(?:a\s*)?(?:human|bot|robot|ai|real)",
       r"introduce\s*yourself",
       r"who\s*(?:made|created|built)\s*you",
       english=True, pure_only=True),
    # --- acknowledgements and dua (pure only) -------------------------------
    _p(OTHER,
       r"আলহামদুলিল্লাহ",
       r"সুবহানাল্লাহ",
       r"মাশা\s*আল্লাহ",
       r"ইনশা\s*আল্লাহ",
       r"আমী?ন",
       r"আচ্ছা|ঠিক\s*আছে|বুঝলাম|বুঝেছি|ওকে",
       r"ভালো\s*লাগলো|ভালো\s*লেগেছে|চমৎকার",
       r"দোয়া\s*করবেন",
       r"alhamdul(?:i)?llah|subhanallah|masha\s*allah|insha\s*allah|ameen|amin",
       r"accha|achha|thik\s*ach(?:e|he)|bujhlam|bujhechi",
       pure_only=True),
    _p(OTHER, r"ok(?:ay)?|alright|got\s*it|cool|nice|great|awesome",
       english=True, pure_only=True),
)

# Address words that may sit between or after pleasantries ("ভাই, আসসালামু
# আলাইকুম", "ধন্যবাদ ভাই"). Never a pleasantry on their own (scan() needs at
# least one real hit), and any that come AFTER the last pleasantry are handed
# back to the question: "আপনি" may be the first word of it.
_FILLER = regex.compile(
    _LEFT + "(?:" + "|".join(normalize_text(s) for s in (
        r"ভাই(?:য়া)?", r"আপু", r"আপা", r"স্যার", r"ম্যাম", r"জি",
        r"হুজুর", r"ওস্তাদ", r"আপনি", r"আপনাকে", r"তুমি", r"ও", r"এবং",
        r"bhai(?:ya)?", r"vai(?:ya)?", r"apu", r"apa", r"sir", r"madam", r"ji",
        r"huzur", r"ustad", r"apni", r"tumi", r"and",
    )) + ")" + _RIGHT
)


def _skip_sep(text: str, pos: int) -> int:
    while pos < len(text) and (text[pos].isspace() or text[pos] in _PUNCT):
        pos += 1
    return pos


@dataclass(frozen=True)
class _Scan:
    subtype: str
    remainder: str
    english: bool  # every pleasantry is English and the message has no Bengali script
    salam: bool  # contains an actual salam (not just "hello")
    hello: bool


def _scan(message: str) -> _Scan | None:
    base = normalize(message)
    text = base.lower()
    if not text:
        return None

    hits: list[_Pattern] = []
    pos = last_hit_end = 0
    while True:
        pos = _skip_sep(text, pos)
        for pattern in _PATTERNS:
            m = pattern.rx.match(text, pos)
            if m and m.end() > pos:
                hits.append(pattern)
                pos = last_hit_end = m.end()
                break
        else:
            filler = _FILLER.match(text, pos)
            if filler and filler.end() > pos:
                pos = filler.end()
                continue
            break
    if not hits:
        return None

    remainder = ""
    if pos < len(text):  # something real is left: a greeting plus a question
        if any(h.pure_only for h in hits):
            return None
        start = _skip_sep(text, last_hit_end)
        gap = text[last_hit_end:start]
        if hits[-1].weak and not any(c in _PUNCT for c in gap):
            return None
        # Keep the user's own casing when lowercasing did not shift offsets.
        remainder = (base if len(base) == len(text) else text)[start:].strip()

    subtype = next(s for s in _PRIORITY if any(h.subtype == s for h in hits))
    return _Scan(
        subtype=subtype,
        remainder=remainder,
        english=not _BENGALI.search(base) and all(h.english for h in hits),
        salam=any(h.salam for h in hits),
        hello=any(h.hello for h in hits),
    )


def classify_chitchat(message: str) -> tuple[str, str] | None:
    """(subtype, remainder) if the message opens with small talk, else None.

    subtype is SALAM, WELLBEING, THANKS, FAREWELL, BOT_IDENTITY or OTHER.
    remainder is the message with the leading pleasantries removed: "" for pure
    small talk, otherwise the real question that must still be answered.
    BOT_IDENTITY and OTHER only match as the whole message, never as a prefix.
    """
    scan = _scan(message)
    return None if scan is None else (scan.subtype, scan.remainder)


def _template(subtype: str, scan: _Scan | None, *, full: bool) -> str | None:
    """A template reply of the given subtype; `full=False` is the short form
    placed in front of a question. None when there is nothing worth saying."""
    if subtype == SALAM:
        is_salam = bool(scan and scan.salam)
        pool = (
            (_SALAM_REPLIES if full else _PREFIX_SALAM)
            if is_salam
            else (_HELLO_REPLIES if full else _PREFIX_HELLO)
        )
    else:
        pool = (_REPLIES if full else _PREFIXES).get(subtype)
    if not pool:
        return None
    reply = random.choice(pool)
    # "assalamualaikum, kemon achen": answer the salam first, then the question.
    if subtype == WELLBEING and scan and scan.salam:
        reply = f"{_SALAM_REPLY} {reply}"
    return reply


def _history_messages(history: list[dict] | None) -> list[dict]:
    out = []
    for turn in (history or [])[-_HISTORY_MESSAGES:]:
        role = turn.get("role")
        content = (turn.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            out.append({"role": role, "content": content[:_HISTORY_CHARS]})
    return out


def _llm_reply(message: str, history: list[dict] | None) -> str:
    """One short persona call. API errors are NOT caught here: the router maps
    them to its 503 exactly as for every other service.

    Task "roleplay": an existing key in app/core/llm.py (no new routing), the
    closest one to a persona conversation, and its token cap (1500) leaves a
    reasoning model room for hidden reasoning. A MODEL_BY_TASK override for
    roleplay therefore also moves small talk.
    """
    messages = [
        {"role": "system", "content": _PERSONA},
        *_history_messages(history),
        {"role": "user", "content": message},
    ]
    response = complete("roleplay", messages)
    return (response.choices[0].message.content or "").strip() or _FALLBACK_REPLY


def reply_for(subtype: str, message: str, history: list[dict] | None = None) -> str:
    """The reply to pure small talk.

    SALAM / WELLBEING / THANKS / FAREWELL get a template of the matching kind
    (no LLM), unless the message is clearly English: templates are Bengali only,
    so English small talk goes to the LLM, which answers in English. BOT_IDENTITY
    and OTHER always go to the LLM, with the last three exchanges for context.
    """
    scan = _scan(message)
    if subtype in _TEMPLATE_SUBTYPES and not (scan and scan.english):
        reply = _template(subtype, scan, full=True)
        if reply:
            return reply
    return _llm_reply(message, history)


def respond(message: str, history: list[dict] | None = None) -> str:
    """Reply to a message the intent classifier routed to CHITCHAT.

    A message that matched no pattern (the LLM classifier called it small talk)
    is OTHER. One that still carries a question goes to the LLM too, rather than
    a template that would ignore the question.
    """
    scan = _scan(message)
    if scan is None or scan.remainder:
        return reply_for(OTHER, message, history)
    return reply_for(scan.subtype, message, history)


def preface(message: str, history: list[dict] | None = None) -> tuple[str | None, str]:
    """Split a QA-path message into (reply to its small talk, the question).

    The one function behind both answer_question() and the streaming
    _qa_stream(), so the two cannot drift apart.

      no small talk        -> (None, message)       message untouched
      pure small talk      -> (reply, "")           caller answers with `reply`
      "hi, <question>"     -> (short greeting, question)
      same, in English     -> (None, question)      a Bengali greeting must not
                                                    open an English answer
    """
    scan = _scan(message)
    if scan is None:
        return None, message
    if not scan.remainder:
        return reply_for(scan.subtype, message, history), ""
    if scan.english:
        return None, scan.remainder
    return _template(scan.subtype, scan, full=False), scan.remainder
