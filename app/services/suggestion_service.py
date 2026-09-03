"""Suggestion/advice generation.

Reuses retrieval + the same relevance gate as qa_service.answer_question(),
but with its own system prompt: unlike generator.py's _SYSTEM (strict
fact-only, refuses to go beyond the cited text), this one is explicitly
allowed to phrase a recommendation/opinion based on the retrieved context.
"""

from __future__ import annotations

from app.core.config import settings
from app.core.llm import get_client, get_model
from app.rag.retriever import retrieve_relevant_docs
from app.schemas.query import Citation

_SYSTEM_GROUNDED = """তুমি একজন বন্ধুত্বপূর্ণ বাংলা পরামর্শদাতা সহকারী। ব্যবহারকারীকে
নিচে দেওয়া বইয়ের অংশের ভিত্তিতে একটি পরামর্শ/সুপারিশ দেওয়াই তোমার কাজ।

নিয়মাবলি:
১। নিচের "উদ্ধৃত অংশ" পড়ে তার আলোকে ব্যবহারকারীকে একটি ব্যবহারিক পরামর্শ দাও।
   শুধু তথ্য তুলে ধরাই যথেষ্ট নয় -- সেই তথ্যের ভিত্তিতে কী করা উচিত তা পরামর্শ
   আকারে বলো।
২। প্রতিটি দাবির শেষে বর্গবন্ধনীতে উৎস দাও, যেমন [১] বা [২], যেখানে প্রাসঙ্গিক।
৩। উদ্ধৃত অংশের বাইরের কল্পিত তথ্য বানিও না -- পরামর্শ বইয়ের বক্তব্যের সাথে
   সঙ্গতিপূর্ণ হতে হবে।
৪। সম্পূর্ণ উত্তর প্রমিত বাংলায় লেখো। ইংরেজি বাক্য বা রোমান হরফে বাংলা লিখবে না।
৫। উত্তর গুছিয়ে লেখো, প্রয়োজনে বুলেট ব্যবহার করো, তবে অকারণে দীর্ঘ কোরো না।"""

_USER_GROUNDED = """উদ্ধৃত অংশসমূহ:
{context}

প্রশ্ন/পরিস্থিতি: {query}

উপরের নিয়ম মেনে বাংলায় একটি পরামর্শ দাও।"""

_SYSTEM_UNGROUNDED = """তুমি একজন বন্ধুত্বপূর্ণ বাংলা পরামর্শদাতা সহকারী।

নিয়মাবলি:
১। প্রথমে স্পষ্টভাবে জানাও যে এই নির্দিষ্ট বিষয়ে বইগুলোতে যথেষ্ট তথ্য পাওয়া
   যায়নি, তাই বইভিত্তিক গ্রাউন্ডেড পরামর্শ দেওয়া সম্ভব হচ্ছে না।
২। এরপর চাইলে একটি সাধারণ পরামর্শ দিতে পারো, কিন্তু সেটিকে স্পষ্টভাবে "সাধারণ
   পরামর্শ (বই-ভিত্তিক নয়)" বলে আলাদা করে উল্লেখ করো।
৩। সম্পূর্ণ উত্তর প্রমিত বাংলায় লেখো। ইংরেজি বাক্য বা রোমান হরফে বাংলা লিখবে না।
৪। উত্তর সংক্ষিপ্ত ও স্পষ্ট রাখো।"""

_USER_UNGROUNDED = """প্রশ্ন/পরিস্থিতি: {query}

উপরের নিয়ম মেনে বাংলায় উত্তর দাও।"""


def _format_context(citations: list[Citation]) -> str:
    blocks = []
    for i, c in enumerate(citations, start=1):
        blocks.append(f"[{i}] বই: {c.book} | অধ্যায়: {c.chapter}\n{c.content}")
    return "\n\n---\n\n".join(blocks)


def give_suggestion(query: str) -> tuple[str, list[Citation]]:
    citations = retrieve_relevant_docs(query)
    relevant = bool(citations) and citations[0].rerank_score >= settings.min_rerank_score
    grounding = citations if relevant else []

    if relevant:
        system, user = _SYSTEM_GROUNDED, _USER_GROUNDED.format(
            context=_format_context(grounding), query=query
        )
    else:
        system, user = _SYSTEM_UNGROUNDED, _USER_UNGROUNDED.format(query=query)

    response = get_client().chat.completions.create(
        model=get_model(),
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    answer = response.choices[0].message.content

    return answer, grounding
