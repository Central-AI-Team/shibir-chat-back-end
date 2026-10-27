"""Mentor mode: a ROLE from an allow-list answers from the books.

Contract: docs/roleplay-v2.md. In one sentence, mentor mode is QA +
conversational memory + a tone layer. The user talks to a role (বন্ধু,
দায়িত্বশীল ভাই, শিক্ষক, অধ্যয়ন-সার্কেল পরিচালক; settings.roleplay_roles),
never a person: requests to speak as a real person -- the Prophet, a sahabi,
a scholar, a living leader -- are refused in code (REAL_PERSON_PATTERNS,
before any LLM call) and again on the intent call's persona_ok flag.

The reply body always comes from qa_service.qa_stream(message, history,
style_rules=...): retrieval, the per-citation gate, [১]-style citations and
the conversational-memory path are inherited, never reimplemented here, and
this module never calls the LLM for the answer. style_rules is the role's
tone (STYLE_RULES_BY_ROLE) + the reading level (STYLE_RULES_BY_LEVEL) + the
honesty rule + the next-steps rule. The level is inferred by the intent call
each turn, smoothed by majority over the last three turns, overridden at
once by explicit cues, stored on conversations.level and never announced.

After a grounded reply one more complete() call ("roleplay_meta") turns the
excerpts into `suggestions` (follow-up questions); parse failure -> [].

Flow per turn (handle_roleplay / handle_roleplay_stream):
    refusal check -> role (resolve_role) -> level (smooth_level)
    -> role-only turn? greet and ask for a question
    -> qa_stream(..., style_rules) -> suggestions

History is written by the router through append_history(), like every
other mode; this module writes nothing.
"""

from __future__ import annotations

import itertools
import json
import logging
import re
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

from app.core.config import settings
from app.core.llm import complete
from app.schemas.query import Citation

logger = logging.getLogger(__name__)


def qa_stream(*args, **kwargs):
    """qa_service.qa_stream, imported at call time: qa_service imports
    intent_classifier, which imports this module's refusal check."""
    from app.services import qa_service

    return qa_service.qa_stream(*args, **kwargs)


# ---------------------------------------------------------------------------
# Roles and the real-person refusal.
# ---------------------------------------------------------------------------

def roles() -> list[dict]:
    """The role allow-list (settings.roleplay_roles), in configured order."""
    return list(settings.roleplay_roles)


def role_by_id(role_id: str | None) -> dict | None:
    for role in roles():
        if role["id"] == role_id:
            return role
    return None


# Who the assistant must never speak AS. Matched case-insensitively against
# the message; a hit refuses when the message also frames an impersonation
# (IMPERSONATION_FRAMING below, or it is the message that enters roleplay).
# Without that framing, "নবীজি কীভাবে নামাজ পড়তেন?" inside a mentor session is
# an ordinary grounded question. Four groups: Bengali honorifics/titles,
# Bengali names, Banglish/English, Arabic. The named-people part WILL age --
# it is a floor, not a registry; the LLM's persona_ok check is the second net.
REAL_PERSON_PATTERNS: tuple[str, ...] = (
    # Bengali honorifics and titles
    r"রাস[ূু]ল", r"নবী", r"নবিজি|নবীজি", r"সাহাব[ীি]", r"সাহাবা", r"আল্লাহ", r"খলিফা", r"ইমাম",
    r"মাওলানা|মৌলানা", r"হুজুর", r"আম[ীি]র", r"\(\s*সা[.:ঃ]?\s*\)", r"\(\s*রা[.:ঃ]?\s*\)",
    r"\(\s*রহ[.:ঃ]?\s*\)", r"\(\s*দা[.:ঃ]?\s*বা[.:ঃ]?\s*\)",
    # Bengali names (prophets, companions, scholars, leaders, politicians)
    r"মু?হাম্মা?দ\s*\(?\s*সা", r"হযরত", r"আবু\s*বকর", r"উমর|ওমর", r"উসমান|ওসমান", r"আলী\b|আলি\b",
    r"আয়েশা|আয়িশা", r"ফাতিমা|ফাতেমা", r"মওদ[ূু]দ[ীি]", r"গোলাম\s*আ[যজ]ম", r"নিজামী|নিজামি",
    r"সাঈদী|সাইদী", r"মুজাহিদ", r"শেখ\s*হাসিনা|হাসিনা", r"খালেদা", r"তারেক\s*রহমান", r"ইউন[ূু]স",
    r"শেখ\s*মুজিব|মুজিব", r"জিয়াউর|জিয়া\b", r"এরশাদ", r"মোদি|মোদী", r"ট্রাম্প", r"এরদোয়ান|এরদোগান",
    # Banglish and English
    r"\bprophet\b", r"\bmessenger\b", r"\brasu+l\b|\brasool\b", r"\bnabi\b|\bnobi\b", r"\bsahab[ia]\b",
    r"\bcompanion(s)?\s+of\s+the\s+prophet\b", r"\ballah\b", r"\bkhalifa\b|\bcaliph\b", r"\bimam\b",
    r"\bmaulana\b|\bmawlana\b", r"\bma[wu]dudi\b", r"\bgolam\s*azam\b", r"\bnizami\b", r"\bsayeedi\b",
    r"\bhasina\b", r"\bkhaleda\b", r"\btarique\b", r"\byunus\b", r"\bmujib\b", r"\bzia\b", r"\bershad\b",
    r"\bmodi\b", r"\btrump\b", r"\berdogan\b", r"\bhazrat\b", r"\bhuzur\b",
    # Arabic honorifics and names
    r"ﷺ", r"صلى\s*الله\s*عليه\s*وسلم", r"رضي\s*الله\s*عنه", r"رحمه\s*الله", r"الله", r"رسول", r"نبي",
)
_REAL_PERSON_RE = re.compile("|".join(REAL_PERSON_PATTERNS), re.IGNORECASE)

# "speak AS someone": the roleplay entry phrases plus the usual ways of asking
# for a voice. Kept separate from intent_classifier._ROLEPLAY_RE on purpose:
# that one decides the intent, this one decides whether a real-person name is
# the persona or merely the topic.
IMPERSONATION_FRAMING: tuple[str, ...] = (
    r"হয়ে\s*(?:যাও|কথা|উত্তর|বলো|বল\b)", r"হও\b", r"মতো\s*(?:করে\s*)?(?:কথা|উত্তর|বলো|বল\b)",
    r"চরিত্রে", r"ভূমিকায়", r"অভিনয়", r"ভান\s*কর", r"সেজে", r"রোল\s*প্লে|রোলপ্লে",
    r"\b(?:act|speak|talk|answer|reply|respond)\s+(?:as|like)\b", r"\bpretend\b", r"\brole\s*play\b",
    r"\bbe\s+(?:the|a|an|my)\b", r"\bbe\s+\S+(?:\s+\S+){0,3}\s+and\s+(?:answer|reply|speak|talk|tell)\b",
    r"\bimpersonat", r"\bin\s+the\s+voice\s+of\b", r"\bhoye\s*(?:jao|kotha|bolo)\b",
    r"\bmoto\s*(?:kotha|bolo)\b",
    # Arabic: "speak as if you were ...", "be ...", "talk like ..."
    r"تكلم\s*كأنك", r"\bكن\s+", r"تحدث\s+(?:ك|مثل)", r"تظاهر",
)
_IMPERSONATION_RE = re.compile("|".join(IMPERSONATION_FRAMING), re.IGNORECASE)


def mentions_real_person(message: str) -> bool:
    return bool(_REAL_PERSON_RE.search(message))


def is_impersonation_request(message: str, *, entering: bool) -> bool:
    """True when the message asks the assistant to speak AS a real person:
    a REAL_PERSON_PATTERNS hit together with impersonation framing, or any hit
    on the message that enters roleplay (`entering`: the turn that matched the
    roleplay regex or chose a role, where the name can only be the persona)."""
    if not mentions_real_person(message):
        return False
    return entering or bool(_IMPERSONATION_RE.search(message))


def refusal_text() -> str:
    """The friendly refusal: what the assistant can do instead, and the roles.
    A normal answer (mode "roleplay", sources []), never a 4xx."""
    lines = [
        "দুঃখিত, আমি কোনো বাস্তব ব্যক্তির হয়ে কথা বলতে পারি না -- নবী-রাসূল, সাহাবী, আলেম বা "
        "জীবিত কোনো মানুষের হয়ে নয়। তবে আমি গ্রন্থাগারের বই থেকে একটি ভূমিকার ভঙ্গিতে কথা বলতে পারি:",
    ]
    for role in roles():
        lines.append(f"• {role['name_bn']} -- {role['description_bn']}")
    lines.append("কোন ভঙ্গিতে কথা বলব, বলুন।")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Role resolution and the reading level.
# ---------------------------------------------------------------------------

LEVELS = ("beginner", "intermediate", "advanced")
LEVEL_WINDOW = 3  # turns in the majority vote

# Role keywords resolve without the LLM; anything else goes through the
# intent call's role_id (nearest role) and, failing that, the "which role?"
# reply. Order = allow-list order; first match wins.
ROLE_PATTERNS: dict[str, str] = {
    "bondhu": r"বন্ধু|সহপাঠী|দোস্ত|\bbondhu\b|\bdost\b|\bfriend\b|\bclassmate\b|\bbuddy\b",
    "boro_bhai": r"বড়\s*ভাই|বড়\s*ভাই|দায়িত্বশীল|\bboro\s*bhai\b|\bbhaiya\b|\bmentor\b|\bbig\s*brother\b",
    "shikkhok": r"শিক্ষক|টিচার|উস্তাদ|ওস্তাদ|\bshikkhok\b|\bteacher\b|\bustad\b|\bustaz\b|\btutor\b",
    "circle_porichalok": r"সার্কেল|পরিচালক|হালকা|\bstudy\s*(?:circle|group)\b|\bcircle\b|\bhalaqa\b|\bporichalok\b",
}
_ROLE_RES = {rid: re.compile(p, re.IGNORECASE) for rid, p in ROLE_PATTERNS.items()}

# Explicit level cues override the inferred level for the turn and reset the
# smoothing window (docs/roleplay-v2.md §3).
BEGINNER_CUES = re.compile(
    r"সহজ\s*(?:করে|ভাবে|ভাষায়)|আরো\s*সহজ|আরও\s*সহজ|সহজ\s*করো|আমি\s*নতুন|নতুন\s*শিখছি|বুঝি\s*না|"
    r"বুঝতে\s*পারছি\s*না|বাচ্চাদের\s*মতো|\bsohoj\b|\bsimpl(?:e|er|y)\b|\beasier\b|\beasy\b|\bbasic\b|"
    r"\bbeginner\b|\bnotun\b|\bami\s*notun\b|\beli5\b",
    re.IGNORECASE,
)
ADVANCED_CUES = re.compile(
    r"গভীরে|আরো\s*গভীর|আরও\s*গভীর|গভীরভাবে|বিশদ|বিশদভাবে|বিস্তারিতভাবে|আরো\s*বিস্তারিত|আরও\s*বিস্তারিত|"
    r"তাত্ত্বিক|মূল\s*পাঠ|\bgobhir\b|\bbistarito\b|\bdeep(?:er)?\b|\bin\s*depth\b|\badvanced\b|"
    r"\bdetail(?:s|ed)?\b|\bscholarly\b",
    re.IGNORECASE,
)


def match_role(message: str) -> str | None:
    """The first allow-listed role whose keywords appear in the message."""
    for role_id, pattern in _ROLE_RES.items():
        if pattern.search(message):
            return role_id
    return None


def resolve_role(message: str, current: str | None, meta: dict | None) -> str | None:
    """Which role this turn speaks in.

    No current role (entering): a keyword match, else the intent call's
    role_id, else None (the caller asks which role). With a current role: it is
    kept unless the message asks for a different one (keyword match, or the
    intent call named one and the message carries impersonation framing).
    """
    wanted = match_role(message)
    if wanted is None and meta and meta.get("role_id"):
        if current is None or _IMPERSONATION_RE.search(message):
            wanted = meta["role_id"]
    if current is None:
        return wanted
    return wanted or current


def explicit_level(message: str) -> str | None:
    """"আরো সহজ করে বলো" -> beginner, "আরো গভীরে যাও" -> advanced, else None."""
    if BEGINNER_CUES.search(message):
        return "beginner"
    if ADVANCED_CUES.search(message):
        return "advanced"
    return None


def load_level_window(persona: str | None) -> list[str]:
    """The last inferred levels, kept as JSON in conversations.persona (free
    since free-form personas are gone; cleared on exit like before)."""
    try:
        data = json.loads(persona) if persona else {}
        window = data.get("levels", []) if isinstance(data, dict) else []
        return [lv for lv in window if lv in LEVELS][-LEVEL_WINDOW:]
    except (ValueError, TypeError):
        return []


def dump_level_window(window: list[str]) -> str:
    return json.dumps({"levels": window[-LEVEL_WINDOW:]})


def smooth_level(
    window: list[str], inferred: str | None, explicit: str | None, fallback: str
) -> tuple[str, list[str]]:
    """(level for this turn, new window). An explicit cue wins and resets the
    window to itself; otherwise the inferred level joins the window and the
    majority of the last LEVEL_WINDOW decides (ties: the most recent); with
    nothing inferred the previous majority, else `fallback`, stands."""
    if explicit in LEVELS:
        return explicit, [explicit]
    if inferred in LEVELS:
        window = (window + [inferred])[-LEVEL_WINDOW:]
    if not window:
        return fallback, []
    counts = Counter(window)
    best = max(counts.values())
    for lv in reversed(window):  # most recent among the tied
        if counts[lv] == best:
            return lv, window
    return fallback, window  # unreachable


# ---------------------------------------------------------------------------
# Style rules: what mentor mode adds to the QA system prompt.
# ---------------------------------------------------------------------------

STYLE_RULES_BY_LEVEL: dict[str, str] = {
    "beginner": (
        "পাঠকের স্তর: নতুন। ছোট ছোট বাক্যে লেখো। প্রতিটি পারিভাষিক শব্দ প্রথমবার ব্যবহারের সময় এক "
        "লাইনে সহজ ভাষায় বুঝিয়ে দাও। একটি সহজ উপমা বা দৈনন্দিন উদাহরণ দাও। এক উত্তরে সর্বোচ্চ দুটি "
        "মূল কথা বলো। শেষে একটি সহজ প্রশ্ন করে জানতে চাও আর কী জানতে চায়।"
    ),
    "intermediate": "পাঠকের স্তর: মাঝারি। স্বাভাবিক নিয়মে উত্তর দাও: স্পষ্ট, গোছানো, প্রয়োজনমতো বিস্তারিত।",
    "advanced": (
        "পাঠকের স্তর: অগ্রসর। বইয়ের পরিভাষা যেমন আছে তেমনই ব্যবহার করো, সরলীকরণ বা উপমা নয়। "
        "যতগুলো উদ্ধৃত অংশ প্রাসঙ্গিক, সবগুলোর উৎস দাও। একাধিক অংশ একই বিষয়ে ভিন্ন দিক দেখালে "
        "সেগুলো পাশাপাশি তুলনা করো।"
    ),
}

HONESTY_RULE = (
    "তুমি একজন সহকারী, যে গ্রন্থাগারের বই থেকে এই ভঙ্গিতে উত্তর দিচ্ছো; কেউ জিজ্ঞাসা করলে তা-ই বলো, "
    "নিজেকে অন্য কিছু দাবি কোরো না। \"বইয়ে বলা হয়েছে\" পর্যন্তই থাকো: নিজে কোনো ফতোয়া দিও না, "
    "সংগঠনের পক্ষে কথা বলো না। প্রতিটি দাবির উৎস [১], [২] আকারে দেওয়া এই ভঙ্গিতেও বাধ্যতামূলক।"
)

NEXT_STEPS_RULE = (
    "উদ্ধৃত অংশ থাকলে উত্তরের শেষে সেখান থেকে নেওয়া এক বা দুটি পরবর্তী ধাপ দাও: পড়ার মতো একটি "
    "অধ্যায়, করার মতো একটি আমল, বা ভাবার মতো একটি প্রশ্ন -- উদ্ধৃত অংশের বাইরে থেকে নয়। উদ্ধৃত অংশ "
    "না থাকলে পরবর্তী ধাপ দিও না; ভূমিকার ভঙ্গিতেই জানাও যে বইগুলোতে এটি পাওনি, এবং চাইলে স্পষ্টভাবে "
    "\"সাধারণ পরামর্শ, বই থেকে নয়\" বলে এক-দুই বাক্যে উৎসাহ দিতে পারো।"
)


def style_rules_by_role() -> dict[str, str]:
    """STYLE_RULES_BY_ROLE: each role's tone_bn, from settings."""
    return {role["id"]: role["tone_bn"] for role in roles()}


def compose_style_rules(role_id: str, level: str) -> str:
    """The system-prompt addition for one mentor turn: tone, level, honesty,
    next steps. Unknown level -> intermediate."""
    role = role_by_id(role_id) or {}
    tone = role.get("tone_bn", "")
    level_rule = STYLE_RULES_BY_LEVEL.get(level, STYLE_RULES_BY_LEVEL["intermediate"])
    return "\n".join(
        [f"ভূমিকার ভঙ্গি: {tone}", level_rule, HONESTY_RULE, NEXT_STEPS_RULE]
    )


# ---------------------------------------------------------------------------
# Turn handling.
# ---------------------------------------------------------------------------


@dataclass
class RoleplayResult:
    answer: str
    sources: list[Citation]
    suggestions: list[str]
    role_id: str | None
    level: str | None
    refused: bool
    grounded: bool = False
    # JSON for conversations.persona: the level window (see load_level_window).
    level_window: str | None = None
    standalone_query: str | None = None
    extra: dict = field(default_factory=dict)


# A message that only picks a role ("একজন বন্ধুর মতো বুঝিয়ে বলো", "act as a
# teacher") asks nothing yet: after removing the role words (with their
# Bengali suffixes), the entry framing and request filler, nothing is left.
_ROLE_WORDS_RE = re.compile("(?:" + "|".join(ROLE_PATTERNS.values()) + r")\S*", re.IGNORECASE)
_FRAMING_RE = re.compile("|".join(IMPERSONATION_FRAMING), re.IGNORECASE)
_REQUEST_FILLER_RE = re.compile(
    r"বুঝিয়ে|বোঝাও|বুঝাও|বলো|বল\b|বলুন|কথা|উত্তর|দাও|দিন|শেখাও|যাও|হও|হয়ে|সেজে|থেকে|হিসেবে|ভাষায়|"
    r"মতো|মত\b|করে|একজন|একটা|আমার|আমাকে|তুমি|তোমার|আপনি|এখন|কর[ো]?\b|প্লিজ|"
    r"\b(?:please|plz|now|me|my|a|an|like|as|the|to|be|act|speak|talk|explain|answer|and|"
    r"kotha|bolo|bol|hoye|jao|moto|kore|ekhon|tumi|amar|amake|ekjon|leader|moderator|guide|host)\b",
    re.IGNORECASE,
)


def _only_a_role_request(message: str) -> bool:
    if "?" in message or "？" in message:
        return False
    rest = _ROLE_WORDS_RE.sub(" ", message)
    rest = _FRAMING_RE.sub(" ", rest)
    rest = _REQUEST_FILLER_RE.sub(" ", rest)
    # Split on whitespace, not \w: Bengali vowel signs are combining marks
    # that \w does not match, so "যাকাত" would fall apart into consonants.
    rest = re.sub(r"[^\w\s\u0980-\u09ff]", " ", rest)
    return not rest.split()


def choose_role_text() -> str:
    lines = ["কোন ভঙ্গিতে কথা বলব? বেছে নিন:"]
    for role in roles():
        lines.append(f"• {role['name_bn']} -- {role['description_bn']}")
    lines.append("যেমন লিখতে পারেন: \"একজন বড় ভাইয়ের মতো বুঝিয়ে বলো\"।")
    return "\n".join(lines)


def greeting_text(role_id: str) -> str:
    role = role_by_id(role_id) or {"name_bn": role_id}
    return (
        f"ঠিক আছে, এখন থেকে আমি {role['name_bn']} হিসেবে কথা বলব -- গ্রন্থাগারের বই থেকে, উৎসসহ। "
        "কী জানতে চান? যেকোনো প্রশ্ন করুন।"
    )


def _meta_for(message: str) -> dict | None:
    from app.services import intent_classifier  # lazy: it imports this module

    return intent_classifier.roleplay_meta(message)


@dataclass
class _Turn:
    """Everything decided before generation."""

    role_id: str | None
    level: str | None
    level_window: str | None
    style_rules: str | None
    canned: str | None  # a reply that needs no retrieval (refusal / ask / greet)
    refused: bool


def _prepare_turn(message: str, session: dict) -> _Turn:
    current_role = session.get("role_id")
    current_level = session.get("level")
    window = load_level_window(session.get("persona"))
    meta = _meta_for(message)
    entering = current_role is None

    # 1. never a real person -- the regex first, the LLM's flag second
    if is_impersonation_request(message, entering=entering) or (meta is not None and not meta["persona_ok"]):
        return _Turn(current_role, current_level, dump_level_window(window) if window else None,
                     None, refusal_text(), True)

    # 2. which role
    role_id = resolve_role(message, current_role, meta)
    if role_id is None:
        return _Turn(None, current_level, None, None, choose_role_text(), False)
    role = role_by_id(role_id)

    # 3. which level (never announced)
    level, window = smooth_level(
        window,
        meta["level"] if meta else None,
        explicit_level(message),
        current_level or role["default_level"],
    )
    level_window = dump_level_window(window)

    # 4. a role pick with no question yet
    if (entering or role_id != current_role) and _only_a_role_request(message):
        return _Turn(role_id, level, level_window, None, greeting_text(role_id), False)

    return _Turn(role_id, level, level_window, compose_style_rules(role_id, level), None, False)


def _canned(turn: _Turn, message: str) -> RoleplayResult:
    return RoleplayResult(
        answer=turn.canned or "", sources=[], suggestions=[], role_id=turn.role_id, level=turn.level,
        refused=turn.refused, grounded=False, level_window=turn.level_window, standalone_query=None,
    )


def handle_roleplay(message: str, session: dict, history: list[dict] | None = None) -> RoleplayResult:
    """One mentor turn, non-streaming. `session` is the store snapshot (role_id,
    level, persona = level window); `history` the recent turns for
    qa_stream's conversational memory."""
    turn = _prepare_turn(message, session)
    if turn.canned is not None:
        return _canned(turn, message)
    sources, deltas, standalone = qa_stream(message, history, style_rules=turn.style_rules)
    answer = "".join(deltas)
    grounded = bool(sources)
    return RoleplayResult(
        answer=answer, sources=sources,
        suggestions=generate_suggestions(standalone, sources) if grounded else [],
        role_id=turn.role_id, level=turn.level, refused=False, grounded=grounded,
        level_window=turn.level_window, standalone_query=standalone,
    )


def handle_roleplay_stream(
    message: str,
    session: dict,
    history: list[dict] | None = None,
    *,
    trace_id: str | None = None,
    parent_observation_id: str | None = None,
) -> tuple[list[Citation], Iterator[str], Callable[[], RoleplayResult]]:
    """Streaming counterpart: (sources, deltas, finalize). finalize() is called
    after the deltas are exhausted and returns the RoleplayResult (answer ""),
    computing suggestions then so they ride in the `done` event."""
    turn = _prepare_turn(message, session)
    if turn.canned is not None:
        result = _canned(turn, message)
        return [], iter([result.answer]), lambda: result
    sources, deltas, standalone = qa_stream(
        message, history, style_rules=turn.style_rules,
        trace_id=trace_id, parent_observation_id=parent_observation_id,
    )
    grounded = bool(sources)

    def finalize() -> RoleplayResult:
        return RoleplayResult(
            answer="", sources=sources,
            suggestions=generate_suggestions(standalone, sources) if grounded else [],
            role_id=turn.role_id, level=turn.level, refused=False, grounded=grounded,
            level_window=turn.level_window, standalone_query=standalone,
        )

    return sources, itertools.chain(deltas), finalize


# ---------------------------------------------------------------------------
# Follow-up suggestions.
# ---------------------------------------------------------------------------

_SUGGEST_SYSTEM = (
    "তুমি একটি বাংলা ইসলামিক গ্রন্থাগার সহকারীর অংশ। নিচে একজন পাঠকের প্রশ্ন এবং বই থেকে পাওয়া "
    "উদ্ধৃত অংশ দেওয়া আছে। উদ্ধৃত অংশের ভিত্তিতে পাঠক এরপর যে প্রশ্নগুলো করতে পারে, এমন {n}টি "
    "ছোট, স্বয়ংসম্পূর্ণ, প্রমিত বাংলা প্রশ্ন লেখো -- প্রতিটি উদ্ধৃত অংশে উত্তরযোগ্য হতে হবে, নতুন "
    "তথ্য যোগ করবে না। শুধুমাত্র একটি JSON অবজেক্ট দাও: {{\"suggestions\": [\"...\", \"...\"]}}"
)


def generate_suggestions(question: str, sources: list[Citation]) -> list[str]:
    """2-3 follow-up questions from the excerpts, one complete() call (task
    roleplay_meta). Validated: a JSON object with a "suggestions" list of
    non-empty strings, trimmed to settings.roleplay_suggestion_count. Any
    failure -- API error, bad JSON, wrong shape -- returns [] and never
    raises: suggestions are an extra, not the answer."""
    n = settings.roleplay_suggestion_count
    if n <= 0 or not sources:
        return []
    excerpts = "\n\n".join(
        f"[{i}] বই: {c.book} | অধ্যায়: {c.chapter}\n{c.content[: settings.context_max_chars]}"
        for i, c in enumerate(sources[: settings.context_top_k], start=1)
    )
    try:
        response = complete(
            "roleplay_meta",
            [
                {"role": "system", "content": _SUGGEST_SYSTEM.format(n=n)},
                {"role": "user", "content": f"প্রশ্ন: {question}\n\nউদ্ধৃত অংশসমূহ:\n{excerpts}"},
            ],
            token_budget=1200,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        data = json.loads(content) if isinstance(content, str) else None
        items = data.get("suggestions") if isinstance(data, dict) else None
        if not isinstance(items, list):
            return []
        out = [s.strip() for s in items if isinstance(s, str) and s.strip()]
        return out[:n]
    except Exception:
        logger.warning("suggestions call failed; returning none", exc_info=True)
        return []
