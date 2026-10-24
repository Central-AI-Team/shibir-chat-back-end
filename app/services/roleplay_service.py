"""Persona roleplay chat.

No RAG retrieval here by design -- roleplay is a free-form persona
conversation, not a book lookup. If that changes later, wire in
retrieve_relevant_docs() the same way qa_service does.
"""

from __future__ import annotations

from app.core.llm import complete
from app.services.session_store import append_history, update_session

_PERSONA_EXTRACT_SYSTEM = """ব্যবহারকারীর নির্দেশনা থেকে শুধুমাত্র চরিত্র/ব্যক্তিত্বের
বর্ণনাটুকু বের করো (কে বা কী চরিত্রে অভিনয় করতে বলা হয়েছে)। শুধু চরিত্রের নাম ও
সংক্ষিপ্ত বর্ণনা লেখো, অন্য কোনো ব্যাখ্যা বা বাক্য যোগ কোরো না।"""

_ROLEPLAY_SYSTEM_TEMPLATE = """তুমি এখন {persona} চরিত্রে অভিনয় করছো। এই চরিত্রের ভাষা,
মনোভাব এবং দৃষ্টিভঙ্গি বজায় রেখে বাংলায় উত্তর দাও। বাস্তব দুনিয়ার AI/model হিসেবে
পরিচয় দিও না যতক্ষণ না ব্যবহারকারী explicitly জিজ্ঞাসা করে।"""


def _extract_persona(message: str) -> str:
    response = complete("persona", [
        {"role": "system", "content": _PERSONA_EXTRACT_SYSTEM},
        {"role": "user", "content": message},
    ])
    return (response.choices[0].message.content or "").strip()


def handle_roleplay(message: str, session: dict) -> str:
    if session.get("persona") is None:
        # `session` is a snapshot (session_store is Postgres-backed), so the
        # persona must be written through the store or the next turn
        # re-extracts it.
        session["persona"] = _extract_persona(message)
        update_session(session["id"], persona=session["persona"])

    system_prompt = _ROLEPLAY_SYSTEM_TEMPLATE.format(persona=session["persona"])

    messages = [{"role": "system", "content": system_prompt}]
    for entry in session["history"]:
        messages.append({"role": entry["role"], "content": entry["content"]})
    messages.append({"role": "user", "content": message})

    response = complete("roleplay", messages)
    # content is None on a safety-filtered/empty completion -- ChatResponse.answer
    # is typed `str`, and a bare None would also get stored in session history and
    # re-sent as {"role": "assistant", "content": None} on the next turn, which
    # most providers reject outright.
    reply = response.choices[0].message.content or ""

    # Both turns go through the store (router._record_turn skips ROLEPLAY so
    # they are not recorded twice). No-op if the session vanished mid-request.
    append_history(session["id"], "user", message)
    append_history(session["id"], "assistant", reply)

    return reply


# ---------------------------------------------------------------------------
# Mentor mode (docs/roleplay-v2.md), part 1: roles and the real-person refusal.
# The free-form persona code above is replaced by the mentor implementation
# in the next commits; these helpers are what it is built on.
# ---------------------------------------------------------------------------

import re  # noqa: E402

from app.core.config import settings  # noqa: E402


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
# Part 2: role resolution and the reading level.
# ---------------------------------------------------------------------------

import json  # noqa: E402
from collections import Counter  # noqa: E402

LEVELS = ("beginner", "intermediate", "advanced")
LEVEL_WINDOW = 3  # turns in the majority vote

# Role keywords resolve without the LLM; anything else goes through the
# intent call's role_id (nearest role) and, failing that, the "which role?"
# reply. Order = allow-list order; first match wins.
ROLE_PATTERNS: dict[str, str] = {
    "bondhu": r"বন্ধু|সহপাঠী|দোস্ত|\bbondhu\b|\bdost\b|\bfriend\b|\bclassmate\b|\bbuddy\b",
    "boro_bhai": r"বড়\s*ভাই|বড়\s*ভাই|দায়িত্বশীল|\bboro\s*bhai\b|\bbhaiya\b|\bmentor\b|\bbig\s*brother\b",
    "shikkhok": r"শিক্ষক|টিচার|উস্তাদ|ওস্তাদ|\bshikkhok\b|\bteacher\b|\bustad\b|\bustaz\b|\btutor\b",
    "circle_porichalok": r"সার্কেল|পরিচালক|হালকা|\bcircle\b|\bhalaqa\b|\bstudy\s*group\b|\bporichalok\b",
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
