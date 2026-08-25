from pydantic import BaseModel


class QueryRequest(BaseModel):
    query: str


class Citation(BaseModel):
    book: str
    chapter: str
    source_db: str
    content: str
    # NEW -- so you can see WHY something was retrieved. Essential for tuning
    # min_rerank_score, and for spotting bad retrieval from the API response
    # alone instead of guessing.
    similarity: float | None = None
    rerank_score: float | None = None


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: list[Citation]
    response_time_ms: float


class NoteRequest(BaseModel):
    chapter_id: int


class NoteByTextRequest(BaseModel):
    text: str


class ChapterNote(BaseModel):
    chapter: str
    pages_used: int
    note: str


class NoteByTextResponse(BaseModel):
    book: str
    chapters: list[ChapterNote]
