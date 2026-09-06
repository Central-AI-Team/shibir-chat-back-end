from datetime import datetime

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


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None
    # Optional caller-supplied user / anonymous id. Not used by any request
    # logic -- it is only forwarded to Langfuse (when tracing is enabled) so
    # traces can be grouped per end user. Safe to omit.
    user_id: str | None = None


class ChatResponse(BaseModel):
    mode: str  # "note" | "roleplay" | "suggestion" | "qa"
    answer: str
    sources: list[Citation] = []
    session_id: str
    response_time_ms: float


class ConversationSummary(BaseModel):
    """One row in GET /conversations -- enough for a sidebar list."""

    id: str
    title: str
    message_count: int
    created_at: datetime
    updated_at: datetime


class ConversationMessage(BaseModel):
    role: str  # "user" | "assistant"
    content: str
