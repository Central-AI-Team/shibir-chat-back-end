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

from app.core.llm import complete, get_client, get_model
from app.schemas.query import Citation

_SYSTEM = """তুমি একজন বন্ধুত্বপূর্ণ বাংলা প্রশ্নোত্তর সহকারী। ব্যবহারকারীকে নিচে
দেওয়া বইয়ের অংশ থেকে সাহায্য করাই তোমার একমাত্র কাজ।

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
৪। সম্পূর্ণ উত্তর প্রমিত বাংলায় লেখো। ইংরেজি বাক্য বা রোমান হরফে বাংলা লিখবে না।
   পারিভাষিক শব্দ বইয়ে যেভাবে আছে সেভাবেই রাখো।
৫। উত্তর গুছিয়ে লেখো — প্রয়োজনে অনুচ্ছেদ বা বুলেট ব্যবহার করো, তবে ছোট প্রশ্নে
   অকারণে দীর্ঘ উত্তর দিও না।"""

_USER = """উদ্ধৃত অংশসমূহ:
{context}

প্রশ্ন: {query}

উপরের নিয়ম মেনে বাংলায় উত্তর দাও।"""


def _format_context(citations: list[Citation]) -> str:
    blocks = []
    for i, c in enumerate(citations, start=1):
        blocks.append(f"[{i}] বই: {c.book} | অধ্যায়: {c.chapter}\n{c.content}")
    return "\n\n---\n\n".join(blocks)


def generate_answer(
    query: str,
    citations: list[Citation],
    *,
    model: str | None = None,
    client=None,
    extra_params: dict | None = None,
) -> str:
    """Generate a grounded answer with the production prompt/context format.

    model=client=None (the plain call every real caller makes) routes through
    complete("qa", ...) -- app/core/llm.py -- which resolves the "qa" task's
    model from settings.model_by_task and applies that model's own param
    quirks automatically. Passing an explicit model and/or client (as an
    eval script does, to run this exact prompt through an arbitrary model)
    bypasses that routing entirely and calls exactly what was asked for.
    """
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": _USER.format(
            context=_format_context(citations), query=query
        )},
    ]
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
    return response.choices[0].message.content


def stream_answer(query: str, citations: list[Citation], *, trace_id: str | None = None):
    """Streaming counterpart of generate_answer(): yields answer text deltas
    as they arrive from the model.

    Uses the exact same prompt, context formatting and model routing
    (complete("qa", ...)) as generate_answer -- the only difference is
    stream=True and yielding chunks instead of returning the whole string.
    Used by POST /chat/stream; the non-streaming /chat path is unchanged.

    trace_id, if given, is passed straight to complete() so the streamed
    generation nests under the request's Langfuse trace. The streaming path
    has to pass it explicitly because it is iterated in a threadpool where
    the trace ContextVar app/core/llm.py would otherwise read is not visible
    (see tracing.finalize_request_trace's docstring). None -> unchanged.
    """
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": _USER.format(
            context=_format_context(citations), query=query
        )},
    ]
    extra = {"trace_id": trace_id} if trace_id else {}
    for chunk in complete("qa", messages, stream=True, **extra):
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta