"""Intent classification for /chat, and follow-up condensation.

Cheap, deterministic regex matching first (covers Bengali script + common
Banglish spellings), and only falls back to an LLM call when nothing matches.
Small talk is recognised by chitchat_service.classify_chitchat(): a message
that is only small talk is CHITCHAT; "hi, <question>" stays QA, and the QA path
puts the greeting in front of the answer.

Condensation (roadmap §1): when the caller passes the session history, the
same combined intent+rewrite LLM call also returns "standalone_query", the
message rewritten as a Bengali question that needs no history ("আরো বলো" ->
the previous question, asked for more detail). It is primed per message and
read once by condense(), which qa_service calls before retrieval. One LLM
call, as before; the two-call fallback path never condenses.

Mentor mode (docs/roleplay-v2.md): on a roleplay turn (the entry regex
matched, or the session is already in ROLEPLAY) the same call also returns
"role_id" (allow-list or null), "persona_ok" (false = the user asked for a
real person) and "level" (beginner | intermediate | advanced). They are
primed per message and read once by roleplay_meta() from roleplay_service.
The real-person refusal (roleplay_service.is_impersonation_request) runs
BEFORE this call and skips it; persona_ok is the second net.
"""

from __future__ import annotations

import logging
import re
import threading

import json

from app.core import timing, tracing
from app.core.config import settings
from app.core.llm import complete
from app.rag.chunker import normalize
from app.rag.query_rewriter import (
    SEARCH_RULES,
    _is_bengali_output,
    is_bengali_query,
    note_fallback,
    prime_rewrite,
)
from app.services.chitchat_service import classify_chitchat
from app.services.roleplay_service import is_impersonation_request, roles

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

# Mentor mode (docs/roleplay-v2.md) added the "<someone>-এর মতো (করে) বুঝিয়ে /
# কথা বলো" family: "একজন বড় ভাইয়ের মতো বুঝিয়ে বলো" is the canonical way to pick
# a role, and "X হয়ে যাও" alone missed it.
_ROLEPLAY_RE = re.compile(
    r"(?:রোল\s*প্লে|রোলপ্লে|তুমি\s+এখন\s+.+\s+হয়ে\s+যাও|তুমি\s+.+\s+হও|অভিনয়\s*(?:করো|করুন|কর)|"
    r"চরিত্রে\s*(?:অভিনয়|থেকে)|ভূমিকায়\s+(?:থেকে|আমার|আমাকে|কথা|অভিনয়)|ভান\s*করো|"
    r"(?:ধরো|ধরুন)\s+তুমি|"
    r"(?:মতো|মত)\s*(?:করে\s*)?(?:কথা\s*বল|বুঝিয়ে|বুঝাও|বোঝাও|উত্তর\s*দ|শেখাও|বল[ো]?\b)|"
    r"\bmoto\s*(?:kore\s*)?(?:kotha\s*bolo|bujhiye|bolo)\b|\bhoye\s*(?:jao|kotha\s*bolo)\b|"
    r"\b(?:talk|speak|explain|answer)\s+(?:to\s+me\s+)?(?:like|as)\s+(?:a|an|my)\b|"
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
_VALID_INTENTS = ("NOTE", "ROLEPLAY", "SUGGESTION", "CHITCHAT", "QA")

_CLASSIFIER_SYSTEM = """তুমি একজন ইনটেন্ট ক্লাসিফায়ার। ব্যবহারকারীর বার্তাটি পড়ে
নিচের পাঁচটি ক্যাটাগরির মধ্যে ঠিক একটি বেছে নাও:

NOTE - ব্যবহারকারী কোনো অধ্যায় বা বইয়ের নোট/সারাংশ চাইছে।
ROLEPLAY - ব্যবহারকারী তোমাকে কোনো চরিত্রে অভিনয় করতে বলছে।
SUGGESTION - ব্যবহারকারী পরামর্শ/মতামত/সুপারিশ চাইছে।
QA - ব্যবহারকারী সরাসরি কোনো তথ্যভিত্তিক প্রশ্ন জিজ্ঞাসা করছে।
CHITCHAT - সালাম, কুশল বিনিময়, ধন্যবাদ/দোয়া, বিদায়, বট সম্পর্কে প্রশ্ন, সাধারণ আলাপ।

শুধুমাত্র একটি শব্দ দিয়ে উত্তর দাও: NOTE, ROLEPLAY, SUGGESTION, QA, অথবা CHITCHAT।
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


# Follow-up detection (roadmap §1). is_bengali_query() alone used to decide
# that a message needs no LLM; with history that is not safe: "আরো বলো" is
# pure Bengali and still means nothing without the previous turn. A Bengali
# message skips condensation only when there is no history, or when it is
# long enough (settings.condense_min_words) AND contains none of these
# anaphors / continuation words. Non-Bengali messages always go to the LLM
# (they need the rewrite anyway). Substring match on purpose: Bengali has no
# reliable word boundary for suffixed forms ("এইটা", "সেটাই").
ANAPHORA = re.compile(
    r"এটা|এটি|সেটা|সেটি|ওটা|ওই|এই|সেই|আরো|আরও|বিস্তারিত|উদাহরণ|তাহলে|কেন|কীভাবে এটা|ওখানে|সেখানে"
)


def should_skip_condense(message: str, history: list[dict] | None) -> bool:
    """True when `message` can be retrieved as-is without an LLM call:
    all-Bengali script and (no history, or >= condense_min_words words with
    no anaphor)."""
    if not is_bengali_query(message):
        return False
    if not history:
        return True
    return (
        len(message.split()) >= settings.condense_min_words
        and not ANAPHORA.search(message)
    )


_COMBINED_BODY = """তুমি দুটি কাজ একসাথে করবে। ব্যবহারকারীর বার্তাটি পড়ে:

১) নিচের পাঁচটি ক্যাটাগরির মধ্যে ঠিক একটি বেছে নাও:
NOTE - ব্যবহারকারী কোনো অধ্যায় বা বইয়ের নোট/সারাংশ চাইছে।
ROLEPLAY - ব্যবহারকারী তোমাকে কোনো চরিত্রে অভিনয় করতে বলছে।
SUGGESTION - ব্যবহারকারী পরামর্শ/মতামত/সুপারিশ চাইছে।
QA - ব্যবহারকারী সরাসরি কোনো তথ্যভিত্তিক প্রশ্ন জিজ্ঞাসা করছে।
CHITCHAT - সালাম, কুশল বিনিময়, ধন্যবাদ/দোয়া, বিদায়, বট সম্পর্কে প্রশ্ন, সাধারণ আলাপ।

২) বার্তাটি (বাংলা, Banglish, ইংরেজি বা আরবি যাই হোক) শুদ্ধ, সহজ, প্রমিত বাংলায় রূপান্তর করো।
বার্তার উত্তর দেবে না; অর্থ পরিবর্তন করবে না; মূল ভাবটি অক্ষুণ্ন রাখো।

""" + SEARCH_RULES

# Without history: byte-identical to the prompt before condensation existed.
_COMBINED_SYSTEM = _COMBINED_BODY + """

শুধুমাত্র একটি JSON অবজেক্ট দিয়ে উত্তর দাও, অন্য কিছু লিখো না:
{"intent": "NOTE|ROLEPLAY|SUGGESTION|CHITCHAT|QA", "rewritten_query": "<বাংলা রূপান্তর>"}"""

# docs/roadmap-q4-2026.md §1 "Condensation prompt", verbatim.
_CONDENSE_RULES = """তোমাকে কথোপকথনের শেষ কয়েকটি পালা এবং ব্যবহারকারীর নতুন বার্তা দেওয়া হবে।
অতিরিক্ত একটি ক্ষেত্র দাও: "standalone_query" — নতুন বার্তাটিকে এমন একটি প্রমিত
বাংলা প্রশ্নে রূপান্তর করো যা পূর্বের কথোপকথন না দেখেও বোঝা যায়।

নিয়ম:
1. বার্তাটি নিজেই সম্পূর্ণ হলে (কোনো সর্বনাম/ইঙ্গিত আগের পালার দিকে না গেলে)
   শুধু বাংলায় পুনর্লিখন করো, অর্থ বদলাবে না, কিছু যোগ করবে না।
2. "এটা", "সেটা", "ওই বইটা", "আরো", "বিস্তারিত" ইত্যাদি আগের পালার দিকে নির্দেশ
   করলে, আগের **প্রশ্ন** এবং উদ্ধৃত **বই/অধ্যায়ের নাম** ব্যবহার করে ইঙ্গিতটি
   স্পষ্ট করো। আগের উত্তরের বিষয়বস্তু থেকে কোনো তথ্য প্রশ্নে ঢোকাবে না।
3. নতুন কোনো তথ্য, অনুমান বা উত্তর যোগ করবে না। এটি শুধু অনুসন্ধানের প্রশ্ন।
4. প্রশ্ন নয় এমন বার্তা (ধন্যবাদ, অভিবাদন) অপরিবর্তিত রাখো।"""

# With history: the same two tasks plus condensation, three JSON fields.
_COMBINED_SYSTEM_WITH_HISTORY = _COMBINED_BODY + "\n\n" + _CONDENSE_RULES + """

শুধুমাত্র একটি JSON অবজেক্ট দিয়ে উত্তর দাও, অন্য কিছু লিখো না:
{"intent": "NOTE|ROLEPLAY|SUGGESTION|CHITCHAT|QA", "rewritten_query": "<বাংলা রূপান্তর>", "standalone_query": "<স্বয়ংসম্পূর্ণ বাংলা প্রশ্ন>"}"""

_LEVELS = ("beginner", "intermediate", "advanced")

# Mentor-mode fields (docs/roleplay-v2.md §1, §3), added to the combined call
# on roleplay turns only. The roles are listed from settings so the model can
# only pick an allow-listed id.
_ROLEPLAY_META_RULES = """এটি একটি "ভূমিকা" (mentor mode) কথোপকথন। অতিরিক্ত তিনটি ক্ষেত্র দাও:
- "role_id": ব্যবহারকারী নিচের কোন ভূমিকায় কথা শুনতে চাইছে, তার id; কোনোটি না মিললে null।
  বার্তায় ভূমিকা বদলানোর কথা না থাকলে null দাও (বর্তমান ভূমিকা বজায় থাকবে)।
{roles}
- "persona_ok": ব্যবহারকারী যদি কোনো বাস্তব ব্যক্তির হয়ে কথা বলতে বলে -- নবী-রাসূল, সাহাবী,
  আলেম, নেতা, রাজনীতিবিদ বা অন্য কোনো সত্যিকারের/ঐতিহাসিক মানুষ -- তাহলে false; শুধু ওপরের
  ভূমিকাগুলোর মতো একটি সাধারণ ভূমিকা চাইলে বা কোনো ভূমিকার কথা না থাকলে true।
- "level": বার্তার শব্দচয়ন, প্রশ্নের জটিলতা ও স্পষ্ট ইঙ্গিত ("আমি নতুন", "আরো সহজ করে",
  "আরো গভীরে") দেখে ব্যবহারকারীর স্তর: "beginner", "intermediate" বা "advanced"।"""


def _roles_block() -> str:
    return "\n".join(f"  {r['id']}: {r['name_bn']} -- {r['description_bn']}" for r in roles())


def _combined_system(history: list[dict] | None, roleplay: bool) -> str:
    if not roleplay:
        return _COMBINED_SYSTEM_WITH_HISTORY if history else _COMBINED_SYSTEM
    fields = ['"intent": "NOTE|ROLEPLAY|SUGGESTION|CHITCHAT|QA"', '"rewritten_query": "<বাংলা রূপান্তর>"']
    parts = [_COMBINED_BODY]
    if history:
        parts.append(_CONDENSE_RULES)
        fields.append('"standalone_query": "<স্বয়ংসম্পূর্ণ বাংলা প্রশ্ন>"')
    parts.append(_ROLEPLAY_META_RULES.format(roles=_roles_block()))
    fields += ['"role_id": "<id>|null"', '"persona_ok": true|false', '"level": "beginner|intermediate|advanced"']
    return "\n\n".join(parts) + "\n\nশুধুমাত্র একটি JSON অবজেক্ট দিয়ে উত্তর দাও, অন্য কিছু লিখো না:\n{" + ", ".join(fields) + "}"

_ASSISTANT_EXCERPT_CHARS = 300


def _sources_label(turn: dict) -> str:
    """'(উৎস: book › chapter; book2 › chapter2) ' from the turn's stored
    citations -- what was cited, not what the answer said -- so the condenser
    resolves "ওই বইটা" to the right book."""
    seen: list[str] = []
    for s in turn.get("sources") or []:
        label = f"{s.get('book', '')} › {s.get('chapter', '')}".strip(" ›")
        if label and label not in seen:
            seen.append(label)
    return f"(উৎস: {'; '.join(seen)}) " if seen else ""


def _history_block(history: list[dict]) -> str:
    """The last settings.qa_history_turns turns, most recent first, in the
    layout the condensation prompt describes."""
    lines = []
    for turn in reversed(history[-settings.qa_history_turns:]):
        role, content = turn.get("role"), " ".join((turn.get("content") or "").split())
        if role == "user" and content:
            lines.append(f"[user]      {content}")
        elif role == "assistant" and content:
            lines.append(f"[assistant] {_sources_label(turn)}{content[:_ASSISTANT_EXCERPT_CHARS]}")
    return "\n".join(lines)


def _condense_user_message(message: str, history: list[dict], roleplay: bool = False) -> str:
    keys = '"intent": ..., "rewritten_query": ..., "standalone_query": ...'
    if roleplay:
        keys += ', "role_id": ..., "persona_ok": ..., "level": ...'
    return (
        "পূর্বের কথোপকথন (সর্বশেষ আগে):\n" + _history_block(history)
        + f"\n\nনতুন বার্তা: {message}\n\n"
        + "JSON: {" + keys + "}"
    )


# Standalone queries produced by the combined call, consumed once by
# condense(). Same bounded shape as query_rewriter._primed.
_primed_standalone: dict[str, str] = {}
_primed_lock = threading.Lock()
_PRIMED_MAX = 256


def _prime_standalone(message: str, standalone: str) -> None:
    key, value = normalize(message), normalize(standalone)
    if not key or not value:
        return
    with _primed_lock:
        _primed_standalone[key] = value
        while len(_primed_standalone) > _PRIMED_MAX:
            del _primed_standalone[next(iter(_primed_standalone))]


# Mentor-mode fields from the combined call, consumed once by roleplay_meta().
_primed_roleplay: dict[str, dict] = {}


def _prime_roleplay(message: str, meta: dict) -> None:
    key = normalize(message)
    if not key:
        return
    with _primed_lock:
        _primed_roleplay[key] = meta
        while len(_primed_roleplay) > _PRIMED_MAX:
            del _primed_roleplay[next(iter(_primed_roleplay))]


def roleplay_meta(message: str) -> dict | None:
    """{"role_id": str | None, "persona_ok": bool, "level": str | None} the
    intent call produced for this roleplay turn, read once; None when no call
    ran (the refusal regex short-circuited, the JSON was invalid, or the turn
    was not a roleplay turn)."""
    with _primed_lock:
        return _primed_roleplay.pop(normalize(message), None)


def condense(message: str, history: list[dict] | None, default: str | None = None) -> str:
    """The standalone question to retrieve `message` with: what the intent
    call produced for it (when history was passed and the skip rule did not
    apply), else `default` or the message itself, normalized. Reads the primed
    value once. Never calls the LLM: the one call happens in classify_intent()."""
    with _primed_lock:
        primed = _primed_standalone.pop(normalize(message), None)
    if primed:
        timing.mark("condense", "from intent call")
        return primed
    timing.mark("condense", "skipped" if not history or should_skip_condense(message, history) else "not primed")
    return normalize(default or message) or (default or message)


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


def _combined_call(message: str, history: list[dict] | None, roleplay: bool = False) -> dict | None:
    system = _combined_system(history, roleplay)
    if history:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": _condense_user_message(message, history, roleplay)},
        ]
    else:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": message}]
    response = complete(
        "intent", messages, token_budget=1500, response_format={"type": "json_object"}
    )
    content = response.choices[0].message.content
    data = _parse_json(content) if isinstance(content, str) else None
    if not data:
        return None
    intent = str(data.get("intent", "")).strip().upper()
    rewritten = str(data.get("rewritten_query", "")).strip()
    if intent not in _VALID_INTENTS or not rewritten:
        return None
    out = {
        "intent": intent,
        "rewritten_query": rewritten,
        "standalone_query": str(data.get("standalone_query", "")).strip(),
    }
    if roleplay:
        role_id = data.get("role_id")
        role_id = str(role_id).strip() if role_id else None
        level = str(data.get("level") or "").strip().lower()
        out["roleplay"] = {
            "role_id": role_id if any(r["id"] == role_id for r in roles()) else None,
            "persona_ok": data.get("persona_ok") is not False,  # missing -> true
            "level": level if level in _LEVELS else None,
        }
    return out


def _classify_and_rewrite(
    message: str, history: list[dict] | None = None, roleplay: bool = False
) -> str | None:
    """ONE LLM call for intent + Bengali rewrite (+ condensation when
    `history` is given). Returns the intent, or None if the reply is not valid
    JSON with a valid intent and a non-empty rewrite (the caller then falls
    back to the old two-call path). API errors are NOT caught here -- they
    must still surface as the router's 503.

    standalone_query is validated like the rewrite (>= 80 % Bengali); a bad
    one gets the whole call retried once, then condense() falls back to the
    raw message and the rewriter's fallback counter is incremented."""
    data = _combined_call(message, history, roleplay)
    if data is None:
        return None
    if history and not _is_bengali_output(data["standalone_query"]):
        retry = _combined_call(message, history, roleplay)
        if retry is not None and _is_bengali_output(retry["standalone_query"]):
            data = retry
        else:
            data["standalone_query"] = ""
            note_fallback("condensation did not produce Bengali; retrieving with the raw message")
    intent = data["intent"]
    if roleplay:
        # A roleplay turn's answer goes through qa_stream, so it needs the
        # rewrite/standalone too, whatever intent the model named.
        prime_rewrite(message, data["rewritten_query"])
        if data["standalone_query"]:
            _prime_standalone(message, data["standalone_query"])
        _prime_roleplay(message, data["roleplay"])
    elif intent in ("QA", "SUGGESTION"):
        prime_rewrite(message, data["rewritten_query"])  # retrieval's expand_query() reuses it
        if data["standalone_query"]:
            _prime_standalone(message, data["standalone_query"])  # qa_service.condense() reads it
    return intent


def _roleplay_turn(message: str, history: list[dict] | None, *, entering: bool, method: str) -> str:
    """A turn whose intent is already ROLEPLAY (entry regex or active
    session): run the combined call for role_id / persona_ok / level (and
    condensation), unless the real-person regex refuses first -- then no LLM
    is called and roleplay_service refuses from the same check."""
    chitchat = classify_chitchat(message)
    if is_impersonation_request(message, entering=entering):
        timing.mark("intent_classification", f"{method}; real-person request, no LLM")
    elif not entering and chitchat is not None and not chitchat[1]:
        # Pure small talk inside a mentor session ("hi", "ধন্যবাদ"): nothing
        # to infer a level from and no role to pick; the role and level stand.
        timing.mark("intent_classification", f"{method}; small talk, no LLM")
    elif settings.combine_intent_rewrite:
        timing.mark("intent_classification", f"{method}; roleplay meta via combined call")
        condense_now = bool(history) and not should_skip_condense(message, history)
        if _classify_and_rewrite(message, history if condense_now else None, roleplay=True) is None:
            logger.info("combined_intent_rewrite_invalid_json on a roleplay turn; no role/level this turn")
    else:
        timing.mark("intent_classification", method)
    tracing.record_intent("ROLEPLAY", method=method)
    return "ROLEPLAY"


def classify_intent(
    message: str, has_active_roleplay_session: bool, history: list[dict] | None = None
) -> str:
    """NOTE / ROLEPLAY / SUGGESTION / CHITCHAT / QA. `history` (the session's
    recent [{role, content, sources?}] turns) enables condensation: a message
    the skip rule does not clear gets its standalone_query from the same LLM
    call, for condense()."""
    normalized = message.strip()

    if has_active_roleplay_session and not _ROLEPLAY_EXIT_RE.search(normalized):
        # An ongoing roleplay conversation shouldn't get reclassified as QA
        # (or anything else) on every follow-up turn.
        return _roleplay_turn(normalized, history, entering=False, method="session")

    for intent, pattern in _INTENT_PATTERNS:
        if pattern.search(normalized):
            if intent == "ROLEPLAY":
                return _roleplay_turn(normalized, history, entering=True, method="regex")
            timing.mark("intent_classification", "regex")
            tracing.record_intent(intent, method="regex")
            return intent

    # Small talk needs no intent LLM call. Pure small talk is CHITCHAT; a
    # greeting in front of a real question stays QA, and the QA path answers
    # the greeting first and then the question (chitchat_service.preface).
    chitchat = classify_chitchat(normalized)
    if chitchat is not None:
        intent = "QA" if chitchat[1] else "CHITCHAT"
        timing.mark("intent_classification", "regex")
        tracing.record_intent(intent, method="regex")
        return intent

    result = None
    condense_now = bool(history) and not should_skip_condense(normalized, history)
    if settings.combine_intent_rewrite and (condense_now or not is_bengali_query(normalized)):
        timing.mark("intent_classification", "llm fallback (combined with rewrite)")
        result = _classify_and_rewrite(normalized, history if condense_now else None)
        if result is None:
            logger.info("combined_intent_rewrite_invalid_json; falling back to two calls")
            timing.mark("intent_classification", "invalid JSON -> two-call fallback")
    else:
        timing.mark("intent_classification", "llm fallback")
    if result is None:
        result = _classify_with_llm(normalized)
    if result == "ROLEPLAY":
        # The plain classifier named it (no regex hit): still a roleplay
        # entry, so get role_id / persona_ok / level the same way. One extra
        # call, on this rare path only.
        return _roleplay_turn(normalized, history, entering=True, method="llm")
    tracing.record_intent(result, method="llm")
    return result
