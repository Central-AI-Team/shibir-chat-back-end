from datetime import datetime

from uuid import UUID

from pydantic import BaseModel, Field

from app.schemas.resources import Citation, ResourceFields


class QueryRequest(BaseModel):
    query: str


class QueryResponse(ResourceFields):
    query: str
    answer: str
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
    message: str = Field(min_length=1, max_length=12000)
    session_id: UUID | None = None
    request_id: UUID | None = None
    # Deprecated compatibility field. Ownership and trace user IDs come only
    # from the verified X-Chat-Identity credential; this value is ignored.
    user_id: str | None = None
    search_web: bool = False
    verify_claim: bool = False


class ChatResponse(ResourceFields):
    mode: str  # "note" | "roleplay" | "suggestion" | "qa" | "memory"
    answer: str
    session_id: str
    response_time_ms: float


class ConversationSummary(BaseModel):
    """One row in GET /conversations -- enough for a sidebar list."""

    id: str
    title: str
    message_count: int
    memory_enabled: bool = True
    created_at: datetime
    updated_at: datetime


class ConversationMessage(ResourceFields):
    id: str
    sequence: int
    status: str
    request_id: str
    mode: str | None = None
    options: dict[str, bool] = Field(default_factory=dict)
    role: str  # "user" | "assistant"
    content: str
