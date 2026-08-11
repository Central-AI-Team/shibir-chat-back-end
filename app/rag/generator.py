"""Answer generation.

CHANGES vs the original:
  1. Prompt written in Bengali. Asking in English for a Bengali answer makes
     the model translate rather than compose -- it is the main reason the
     Bengali output reads stiff.
  2. Refusal rule softened. The old "if the Context does not contain enough
     information, say the database does not contain this" fires far too easily
     on partial matches. It now answers with whatever IS there and names the
     gap. Hard refusal is decided upstream in qa_service by score, not vibes.
  3. temperature=0.2. The old call sent no temperature, so Gemini used its
     default of 1.0 -- unnecessarily creative for grounded Q&A.
  4. Excerpts are numbered so the model can cite them.
"""

from __future__ import annotations

from openai import OpenAI

from app.core.config import settings
from app.schemas.query import Citation

_client = OpenAI(api_key=settings.gemini_api_key, base_url=settings.gemini_base_url)

_SYSTEM = """তুমি একটি বাংলা বই-ভিত্তিক প্রশ্নোত্তর সহকারী। তোমার কাজ হলো
নিচে দেওয়া বইয়ের অংশগুলো থেকে ব্যবহারকারীর প্রশ্নের উত্তর দেওয়া।

নিয়মাবলি:
১। শুধুমাত্র নিচের "উদ্ধৃত অংশ" থেকে উত্তর দাও। বাইরের কোনো জ্ঞান যোগ কোরো না।
২। প্রশ্নের শব্দ হুবহু না মিললেও অর্থ বা প্রসঙ্গ মিললে সেটি প্রাসঙ্গিক ধরে নাও।
৩। উদ্ধৃত অংশে **আংশিক তথ্য থাকলেও সেটুকু দিয়েই উত্তর দাও**, এবং শেষে এক বাক্যে
   লেখো কোন দিকটি বইগুলোতে পাওনি। শুধু বিষয়টির কোনো উল্লেখই না থাকলে তবেই বলবে
   যে এই বিষয়ে তথ্য পাওয়া যায়নি।
৪। প্রতিটি দাবির শেষে বর্গবন্ধনীতে উৎস দাও, যেমন [১] বা [২]।
৫। সম্পূর্ণ উত্তর প্রমিত বাংলায় লেখো। ইংরেজি বাক্য বা রোমান হরফে বাংলা লিখবে না।
   পারিভাষিক শব্দ বইয়ে যেভাবে আছে সেভাবেই রাখো।
৬। উত্তর গুছিয়ে লেখো — প্রয়োজনে অনুচ্ছেদ বা বুলেট ব্যবহার করো।"""

_USER = """উদ্ধৃত অংশসমূহ:
{context}

প্রশ্ন: {query}

উপরের নিয়ম মেনে বাংলায় উত্তর দাও।"""


def _format_context(citations: list[Citation]) -> str:
    blocks = []
    for i, c in enumerate(citations, start=1):
        blocks.append(f"[{i}] বই: {c.book} | অধ্যায়: {c.chapter}\n{c.content}")
    return "\n\n---\n\n".join(blocks)


def generate_answer(query: str, citations: list[Citation]) -> str:
    response = _client.chat.completions.create(
        model=settings.gemini_model,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": _USER.format(
                context=_format_context(citations), query=query
            )},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content