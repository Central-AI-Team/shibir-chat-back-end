from __future__ import annotations

from app.core.llm import get_client, get_model
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


def generate_answer(query: str, citations: list[Citation]) -> str:
    response = get_client().chat.completions.create(
        model=get_model(),
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": _USER.format(
                context=_format_context(citations), query=query
            )},
        ],
        # gpt-5-mini only supports the default temperature (1) -- passing any
        # other value is a 400. Grounding/determinism now comes entirely from
        # the prompt, not sampling temperature.
    )
    return response.choices[0].message.content