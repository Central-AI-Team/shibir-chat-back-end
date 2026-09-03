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

from sqlalchemy.orm import joinedload

from app.core.llm import get_client, get_model
from app.db.models import Book, Chapter, ContentStatus, Page
from app.db.session import SessionLocal
from app.rag.chunker import normalize

# Purely a reverse-proxy/client timeout guard (nginx, browser) for a
# synchronous HTTP request that chains one generate_chapter_note() call per
# chapter -- not an API quota concern (paid OpenAI key). A book past this
# limit needs its chapters requested individually via generate_chapter_note()
# instead (no longer reachable over HTTP now that /note has been removed;
# call it in-process, e.g. from a script).
MAX_CHAPTERS_PER_BOOK_NOTE = 60

_MAP_PROMPT = """নিচে একটি বইয়ের অধ্যায়ের একটি অংশ দেওয়া হলো।
এই অংশের মূল বক্তব্যগুলো বুলেট আকারে সংক্ষেপে বাংলায় লেখো।

কঠোরভাবে মেনে চলো:
- শুধুমাত্র এই অংশে যা লেখা আছে তা সংক্ষেপ করো। তোমার নিজের জ্ঞান থেকে কোনো তথ্য, ব্যাখ্যা, উদাহরণ বা আধুনিক প্রাসঙ্গিকতা যোগ কোরো না।
- সংখ্যা, একক এবং পরিমাণ (যেমন তোলা, ভাগ, বছর) হুবহু রাখো — গ্রাম বা শতাংশে রূপান্তর কোরো না।
- কুরআনের আয়াত বা হাদীসের রেফারেন্স (সূরার নাম, আয়াত নম্বর) হুবহু উল্লেখ থাকলে তা অক্ষুণ্ন রাখো, বাদ দিও না বা সাধারণীকরণ কোরো না।
- নাম, ঘটনা এবং ঐতিহাসিক দৃষ্টান্ত উল্লেখ থাকলে তা রাখো।
- এই অংশে নেই এমন কোনো বিষয় (সেকশন, উদাহরণ, আধুনিক প্রসঙ্গ) নিজে থেকে তৈরি কোরো না।

অংশ:
{chunk}"""

_REDUCE_PROMPT = """নিচে একটি অধ্যায়ের বিভিন্ন অংশের সারসংক্ষেপ দেওয়া হলো।
এগুলো একত্র করে একটি সুসংগঠিত, পড়ার উপযোগী নোট তৈরি করো।

কঠোরভাবে মেনে চলো:
- শুধুমাত্র নিচের সারসংক্ষেপগুলোতে যা আছে তা পুনর্গঠন করো। নতুন কোনো তথ্য, সেকশন, উদাহরণ বা ব্যাখ্যা যোগ কোরো না যা সারসংক্ষেপে নেই।
- সারসংক্ষেপে থাকা সংখ্যা, একক এবং কুরআন/হাদীসের রেফারেন্স হুবহু রাখো — রূপান্তর, সরলীকরণ বা সাধারণীকরণ কোরো না।

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
    resp = get_client().chat.completions.create(
        model=get_model(),
        messages=[{"role": "user", "content": prompt}],
        # gpt-5-mini only supports the default temperature (1), and takes
        # max_completion_tokens instead of max_tokens.
        max_completion_tokens=max_tokens,
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
    # gpt-5-mini spends completion-token budget on hidden reasoning before it
    # writes any visible output -- 1200/3000 (sized for the old model) were
    # getting fully consumed by reasoning on anything but the shortest
    # chunks, so the call would return "" with finish_reason="length" and
    # the map/reduce step silently produced nothing.
    summaries = [_llm(_MAP_PROMPT.format(chunk=g), max_tokens=4000) for g in groups]

    note = _llm(
        _REDUCE_PROMPT.format(
            book=book_name,
            chapter=chapter.name,
            summaries="\n\n---\n\n".join(summaries),
        ),
        max_tokens=6000,
    )
    return {
        "book": book_name,
        "chapter": chapter.name,
        "pages_used": len(texts),
        "note": note,
    }


def generate_book_notes_from_text(text: str) -> dict:
    """Resolve a book from free-form Bengali text, then note every chapter.

    Book resolution is plain substring matching, deliberately not an LLM
    call -- it's cheap, deterministic, and good enough when the user is
    typing an actual book title (possibly inside a longer sentence).
    """
    normalized_text = normalize(text)

    session = SessionLocal()
    try:
        books = session.query(Book).all()
        matches = [b for b in books if normalize(b.name) and normalize(b.name) in normalized_text]
        if not matches:
            return {"error": "কোনো বই খুঁজে পাওয়া যায়নি। বইয়ের নাম আরেকটু স্পষ্ট করে লিখুন।"}

        # The Tarun_Associate / Nobin_Associate source SQLite DBs were migrated
        # without de-duplication, so the same title sometimes exists as two
        # separate book rows (e.g. "উলুমুল কুরআন ও উলুমুল হাদীস" -> ids 204 and
        # 219). Lowest id wins as a deterministic tie-break. Proper
        # de-duplication is a separate task (M0 in the project's milestone
        # tracker), not solved here.
        book = min(matches, key=lambda b: b.id)

        chapters = (
            session.query(Chapter)
            .filter(Chapter.book_id == book.id)
            .order_by(Chapter.position.asc().nullslast(), Chapter.id.asc())
            .all()
        )
        n_chapters = len(chapters)
        if n_chapters > MAX_CHAPTERS_PER_BOOK_NOTE:
            return {
                "too_many_chapters": True,
                "error": (
                    f"এই বইয়ে {MAX_CHAPTERS_PER_BOOK_NOTE}টির বেশি অধ্যায় আছে "
                    f"(মোট {n_chapters}টি), তাই পুরো বইয়ের নোট একসাথে বানানো সম্ভব "
                    "না। নির্দিষ্ট অধ্যায়ের নাম উল্লেখ করে আবার চেষ্টা করুন।"
                ),
            }

        book_name = book.name
        chapter_ids = [c.id for c in chapters]
    finally:
        session.close()

    chapters_out = []
    for chapter_id in chapter_ids:
        result = generate_chapter_note(chapter_id)
        if "error" in result:
            continue  # e.g. a chapter with no published pages -- skip it, don't fail the whole book
        chapters_out.append({
            "chapter": result["chapter"],
            "pages_used": result["pages_used"],
            "note": result["note"],
        })

    return {"book": book_name, "chapters": chapters_out}