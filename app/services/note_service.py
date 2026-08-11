"""Note / summary generation.  NEW FILE.

Fixes: "note banate bolle bole je enough information nai".

That was not a bug -- it was the wrong tool. "এই অধ্যায় থেকে নোট বানাও" is an
aggregation task, not a lookup. Similarity search hands the model 3-5 scattered
chunks, and the model correctly reports that it cannot build notes from them.

So notes do not go through Chroma at all. Your Postgres schema already has
books -> chapters -> pages with chapter.position, which is exactly the
structure needed: pull the WHOLE chapter in reading order, then map-reduce.
"""

from __future__ import annotations

from openai import OpenAI
from sqlalchemy.orm import joinedload

from app.core.config import settings
from app.db.models import Chapter, ContentStatus, Page
from app.db.session import SessionLocal
from app.rag.chunker import normalize

_client = OpenAI(api_key=settings.gemini_api_key, base_url=settings.gemini_base_url)

_MAP_PROMPT = """নিচে একটি বইয়ের অধ্যায়ের একটি অংশ দেওয়া হলো।
এই অংশের মূল বক্তব্যগুলো বুলেট আকারে সংক্ষেপে বাংলায় লেখো।
কোনো তথ্য নিজে থেকে যোগ কোরো না।

অংশ:
{chunk}"""

_REDUCE_PROMPT = """নিচে একটি অধ্যায়ের বিভিন্ন অংশের সারসংক্ষেপ দেওয়া হলো।
এগুলো একত্র করে একটি সুসংগঠিত, পড়ার উপযোগী নোট তৈরি করো।

কাঠামো:
- শিরোনাম
- মূল বিষয়বস্তু (উপশিরোনাম সহ)
- গুরুত্বপূর্ণ পয়েন্টসমূহ (বুলেট)
- সংক্ষিপ্ত উপসংহার

পুরোটাই প্রমিত বাংলায় লেখো। পুনরাবৃত্তি এড়াও।

বই: {book}
অধ্যায়: {chapter}

সারসংক্ষেপসমূহ:
{summaries}"""


def _llm(prompt: str, max_tokens: int = 2000) -> str:
    resp = _client.chat.completions.create(
        model=settings.gemini_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()


def _group(pages: list[str], budget: int = 12000) -> list[str]:
    """Pack pages into LLM-sized batches. Gemini Flash has a large context
    window, so batches can be generous -- fewer map calls, better coherence."""
    groups, buf = [], ""
    for text in pages:
        if buf and len(buf) + len(text) > budget:
            groups.append(buf)
            buf = text
        else:
            buf = f"{buf}\n\n{text}" if buf else text
    if buf:
        groups.append(buf)
    return groups


def generate_chapter_note(chapter_id: int) -> dict:
    session = SessionLocal()
    try:
        chapter = (
            session.query(Chapter)
            .options(joinedload(Chapter.book))
            .filter(Chapter.id == chapter_id)
            .one_or_none()
        )
        if chapter is None:
            return {"error": "অধ্যায়টি খুঁজে পাওয়া যায়নি।"}

        pages = (
            session.query(Page)
            .filter(Page.chapter_id == chapter_id)
            .filter(Page.status == ContentStatus.published)
            .order_by(Page.id)  # reading order
            .all()
        )
        if not pages:
            return {"error": "এই অধ্যায়ে প্রকাশিত কোনো পৃষ্ঠা নেই।"}

        book_name = chapter.book.name if chapter.book else "Unknown"
        texts = [normalize(p.content) for p in pages if p.content]
    finally:
        session.close()

    groups = _group(texts)
    summaries = [_llm(_MAP_PROMPT.format(chunk=g), max_tokens=1200) for g in groups]

    note = _llm(
        _REDUCE_PROMPT.format(
            book=book_name,
            chapter=chapter.name,
            summaries="\n\n---\n\n".join(summaries),
        ),
        max_tokens=3000,
    )
    return {
        "book": book_name,
        "chapter": chapter.name,
        "pages_used": len(texts),
        "note": note,
    }