"""Grounded Bengali answer generation.

CHANGES: generate_answer() gained three optional, keyword-only parameters
(model, client, extra_params) so scripts/eval_generation_ab.py can run the
SAME prompt/context-formatting through a different model/provider to compare
answer quality. qa_service.answer_question() (and any future plain caller)
still calls generate_answer(query, citations) exactly as before -- with
model=client=None, it now routes through complete("qa", ...) -- see
app/core/llm.py -- i.e. whichever model settings.model_by_task["qa"]
resolves to (today's single default until that's explicitly assigned).
Passing an explicit model/client (as an eval script does) bypasses that
routing and calls exactly what was asked for, extra_params passed straight
through -- unaffected by settings.model_by_task either way.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from app.core import timing
from app.core.config import settings
from app.core.llm import complete, get_client, get_model
from app.rag.script_check import count_stray_tokens, strip_stray
from app.schemas.query import Citation

logger = logging.getLogger(__name__)

# Incident 2026-10-05: English questions got Bengali answers. The answer now
# follows the question's language; the LLM decides English vs Banglish (a
# word-list heuristic would call Banglish without known particles "English").
# Excerpt labels stay [১], [২] either way (CLAUDE.md §4, rule 3).
ANSWER_LANGUAGE_RULE = (
    "উত্তরের ভাষা হবে প্রশ্নের ভাষা: প্রশ্ন ইংরেজিতে হলে সম্পূর্ণ উত্তর স্বাভাবিক ইংরেজিতে লেখো "
    "(ইসলামি পরিভাষা বইয়ে যেভাবে আছে সেভাবে রাখতে পারো); প্রশ্ন বাংলা, Banglish বা অন্য যেকোনো "
    "ভাষায় হলে সম্পূর্ণ উত্তর প্রমিত বাংলায় লেখো, রোমান হরফে বাংলা লিখবে না। "
    "উত্তরের প্রতিটি অংশ, এমনকি 'বইয়ে কী নেই' জানানোর বাক্যটিও, একই ভাষায় হবে — দুই ভাষা মেশাবে না। "
    "উৎস-নম্বর সবসময় [১], [২] ... আকারেই দেবে।"
)

_SYSTEM = f"""তুমি একজন জ্ঞানী, বিনয়ী ও আন্তরিক শিক্ষক। বাস্তব কথোপকথনের মতো স্বাভাবিক,
প্রবহমান ভাষায় (প্রমিত বাংলায়, তবে নিয়ম ৪ অনুযায়ী ইংরেজি প্রশ্নে ইংরেজিতে) উত্তর
দাও, এবং তোমার উত্তর শুধু নিচে দেওয়া বইয়ের অংশের ওপর ভিত্তি করে হবে।

নিয়মাবলি:
১। নিচের "উদ্ধৃত অংশ" প্রশ্নের সাথে প্রাসঙ্গিক হলে তা থেকেই উত্তর দাও, এবং
   প্রতিটি দাবির শেষে বর্গবন্ধনীতে উৎস দাও, যেমন [১] বা [২]। প্রশ্নের শব্দ হুবহু
   না মিললেও অর্থ বা প্রসঙ্গ মিললে সেটি প্রাসঙ্গিক ধরে নাও।
২। উদ্ধৃত অংশে **আংশিক তথ্য থাকলেও সেটুকু দিয়েই উত্তর দাও**, এবং শেষে এক বাক্যে
   লেখো কোন দিকটি বইগুলোতে পাওনি।
৩। উদ্ধৃত অংশ প্রশ্নের সাথে সম্পর্কহীন, অপ্রতুল, বা খালি হলে অনুমান করে বা
   তোমার নিজস্ব সাধারণ জ্ঞান থেকে উত্তর বানিও না -- তবে শুধু "তথ্য পাওয়া যায়নি"
   বলে থেমে যেও না। নম্র ও বন্ধুত্বপূর্ণ ভাষায় জানাও যে এই নির্দিষ্ট বিষয়ে
   বইগুলোতে তথ্য খুঁজে পাওনি, এবং প্রশ্নটি অন্যভাবে বা আরেকটু নির্দিষ্ট করে
   জিজ্ঞাসা করতে উৎসাহ দাও। "উদ্ধৃত অংশ"-এর বাইরের কোনো তথ্য উত্তরে যোগ কোরো না।
৪। {ANSWER_LANGUAGE_RULE}
৫। সরাসরি উত্তর দিয়ে শুরু করো। সাধারণভাবে স্বাভাবিক, প্রবহমান অনুচ্ছেদে লেখো; বুলেট
   তখনই ব্যবহার করো যখন বিষয়বস্তু সত্যিই তালিকাধর্মী। শিরোনাম বা পরীক্ষার খাতার মতো
   কাঠামো এড়িয়ে চলো। ছোট প্রশ্নে উত্তরও ছোট রাখো।
৬। আগের পালার উত্তর শুধু "কী বলা হয়েছে" বোঝার প্রসঙ্গ; সেগুলো উৎস নয়। প্রতিটি দাবি
   এখনকার "উদ্ধৃত অংশ" থেকেই আসবে এবং [১], [২] আকারে উৎস দেবে। আগের উত্তরে থাকা
   কোনো তথ্য এখনকার উদ্ধৃত অংশে না থাকলে সেটি পুনরাবৃত্তি করবে না।"""

# {original_line} is empty, or "ব্যবহারকারীর বার্তা: <what the user typed>\n\n"
# when the question below is a condensed standalone version of it (roadmap
# §1): the model answers the resolved question, but tone and the
# answer-language rule (৪) follow the actual message.
_USER = """উদ্ধৃত অংশসমূহ:
{context}

{original_line}প্রশ্ন: {query}

উপরের নিয়ম মেনে প্রশ্নের ভাষায় উত্তর দাও।

(Reply language: if the question above is written in English, write the ENTIRE reply in English, including any note about what the excerpts do not cover, with no Bengali sentences (Bengali excerpt labels like [১] and Islamic terms are fine); otherwise answer in Bengali.)"""


_BN_DIGITS = str.maketrans("0123456789", "০১২৩৪৫৬৭৮৯")


def format_context(citations: list[Citation]) -> str:
    """Numbered excerpt blocks for the prompt; shared with suggestion_service.

    Labels use Bengali digits ([১], [২], ...) because the model cites with
    whatever labels it sees: with ASCII labels it answered "[1][2]" despite the
    prompt asking for [১] (CLAUDE.md §4, rule 3).
    """
    blocks = []
    for i, c in enumerate(citations[: settings.context_top_k], start=1):
        label = str(i).translate(_BN_DIGITS)
        content = c.content[: settings.context_max_chars]
        blocks.append(f"[{label}] বই: {c.book} | অধ্যায়: {c.chapter}\n{content}")
    return "\n\n---\n\n".join(blocks)


def _history_messages(history: list[dict] | None) -> list[dict]:
    """The last settings.qa_history_turns turns as plain chat messages
    (role + content only; stored sources and anything else are dropped)."""
    out = []
    for turn in (history or [])[-settings.qa_history_turns:]:
        role, content = turn.get("role"), (turn.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            out.append({"role": role, "content": content})
    return out


def build_messages(
    query: str,
    citations: list[Citation],
    *,
    history: list[dict] | None = None,
    original_message: str | None = None,
    style_rules: str | None = None,
) -> list[dict]:
    """The chat messages for one QA generation, shared by the non-streaming
    and streaming paths: [system (+ style_rules)] + history + [user].
    With every keyword at its default this is exactly the pre-history prompt.
    """
    system = _SYSTEM if not style_rules else f"{_SYSTEM}\n\n{style_rules}"
    original_line = (
        f"ব্যবহারকারীর বার্তা: {original_message}\n\n"
        if original_message and original_message != query
        else ""
    )
    user = _USER.format(context=format_context(citations), query=query, original_line=original_line)
    return [
        {"role": "system", "content": system},
        *_history_messages(history),
        {"role": "user", "content": user},
    ]


def _generate_once(
    query: str,
    citations: list[Citation],
    *,
    history: list[dict] | None = None,
    original_message: str | None = None,
    style_rules: str | None = None,
    model: str | None = None,
    client=None,
    extra_params: dict | None = None,
) -> str:
    """Generate a grounded answer with the production prompt/context format.

    history / original_message / style_rules: see build_messages().

    model=client=None (the plain call every real caller makes) routes through
    complete("qa", ...) -- app/core/llm.py -- which resolves the "qa" task's
    model from settings.model_by_task and applies that model's own param
    quirks automatically. Passing an explicit model and/or client (as an
    eval script does, to run this exact prompt through an arbitrary model)
    bypasses that routing entirely and calls exactly what was asked for.
    """
    with timing.timed("prompt_build"):
        messages = build_messages(
            query, citations, history=history, original_message=original_message,
            style_rules=style_rules,
        )
    if model is None and client is None:
        response = complete("qa", messages, **(extra_params or {}))
    else:
        response = (client or get_client()).chat.completions.create(
            model=model or get_model("qa"),
            messages=messages,
            # gpt-5-mini only supports the default temperature (1) -- passing
            # any other value is a 400. A different model's own
            # temperature/token-budget knobs go through extra_params instead
            # of hardcoding another model's quirks in here.
            **(extra_params or {}),
        )
    # content is None on a safety-filtered/empty completion (finish_reason
    # e.g. "content_filter") -- QueryResponse.answer/ChatResponse.answer are
    # typed `str`, so returning None here would fail Pydantic validation with
    # an unhandled 500 instead of a plain (if unhelpful) empty-ish answer.
    return response.choices[0].message.content or ""


def with_script_check(produce: Callable[[], str]) -> str:
    """Run produce(); if the text has stray-script letters, regenerate up to
    settings.answer_script_retries times (keeping the cleanest), then strip
    whatever is left. Shared by QA and suggestion; see app/rag/script_check.py."""
    answer = produce()
    bad = count_stray_tokens(answer)
    for _ in range(settings.answer_script_retries if bad else 0):
        logger.warning("answer has %d stray-script token(s); regenerating", bad)
        retry = produce()
        retry_bad = count_stray_tokens(retry)
        if retry_bad < bad:
            answer, bad = retry, retry_bad
        if not bad:
            break
    if bad:
        logger.warning("answer still has %d stray-script token(s); stripping", bad)
        answer = strip_stray(answer)
    return answer


def generate_answer(
    query: str,
    citations: list[Citation],
    *,
    history: list[dict] | None = None,
    original_message: str | None = None,
    style_rules: str | None = None,
    model: str | None = None,
    client=None,
    extra_params: dict | None = None,
) -> str:
    """_generate_once() plus the stray-script check (app/rag/script_check.py).

    The check runs only on the production path (model=client=None). Eval
    scripts that pass an explicit model/client get that model's raw output, so
    comparisons are not flattered by the repair. Not applied to stream_answer():
    tokens are already sent by the time the answer could be checked.
    """
    prompt = dict(history=history, original_message=original_message, style_rules=style_rules)
    if model is not None or client is not None:
        return _generate_once(
            query, citations, model=model, client=client, extra_params=extra_params, **prompt
        )
    return with_script_check(
        lambda: _generate_once(query, citations, extra_params=extra_params, **prompt)
    )


def stream_answer(
    query: str,
    citations: list[Citation],
    *,
    history: list[dict] | None = None,
    original_message: str | None = None,
    style_rules: str | None = None,
    trace_id: str | None = None,
    parent_observation_id: str | None = None,
    timing_rec=None,
):
    """Streaming counterpart of generate_answer(): yields answer text deltas
    as they arrive from the model.

    Uses the exact same prompt, context formatting and model routing
    (complete("qa", ...)) as generate_answer -- the only difference is
    stream=True and yielding chunks instead of returning the whole string.
    Used by POST /chat/stream; the non-streaming /chat path is unchanged.

    trace_id/parent_observation_id, if given, are passed straight to
    complete() so the streamed generation nests under the request's Langfuse
    root span. The streaming path has to pass these explicitly because this
    generator is iterated by Starlette one `next()` at a time in a
    threadpool, each call in its own copied context, so the ambient trace
    context app/core/llm.py would otherwise read is not visible by this point
    (see app/core/tracing.py's module docstring). None -> unchanged.
    """
    with timing.timed("prompt_build", rec=timing_rec):
        messages = build_messages(
            query, citations, history=history, original_message=original_message,
            style_rules=style_rules,
        )
    extra = {}
    if trace_id:
        extra["trace_id"] = trace_id
    if parent_observation_id:
        extra["parent_observation_id"] = parent_observation_id
    with timing.timed("llm_generation", rec=timing_rec) as gen:
        first = True
        for chunk in complete("qa", messages, stream=True, **extra):
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                if first:
                    gen.lap("llm_first_token")
                    first = False
                yield delta