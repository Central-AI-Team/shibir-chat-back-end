from datetime import datetime

from pydantic import BaseModel

# 43 chars = base64url of 32 random bytes (secrets.token_urlsafe(32)). Shared by
# the body field and the path-parameter check in app/api/router.py.
SESSION_ID_PATTERN = r"^[A-Za-z0-9_-]{43}$"


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


class ChatRequest(BaseModel):
    message: str
    # Only ever continues a session the server issued to THIS browser (owner
    # cookie). Anything else -- unknown, someone else's, or malformed (e.g. a
    # uuid4 from before ids were tokens) -- starts a new session; see
    # router._resolve_session. Deliberately not validated here: a 422 would
    # break front-ends still holding an old-format id.
    session_id: str | None = None
    # Accepted for front-end compatibility but IGNORED: it is client-supplied
    # and spoofable. Langfuse gets a prefix of the server-side owner hash instead.
    user_id: str | None = None
    # true -> same dispatch as POST /chat/stream (Server-Sent Events). Default
    # false keeps the JSON response the existing front-end expects.
    stream: bool = False


class ChatResponse(BaseModel):
    mode: str  # "note" | "roleplay" | "suggestion" | "qa"
    answer: str
    sources: list[Citation] = []
    session_id: str
    response_time_ms: float
    # Guests only (null for logged-in users): responses left after this one,
    # and the per-guest total. See docs/auth.md.
    guest_remaining: int | None = None
    guest_limit: int | None = None


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
    # Assistant turns only: the citations that were returned with the answer,
    # so reopening a conversation restores them. [] for user turns, for modes
    # without retrieval (note, roleplay) and for answers that failed the gate.
    sources: list[Citation] = []
