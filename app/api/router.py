"""HTTP layer.

CHANGED: /ask is now async and pushes the blocking work (sentence-transformers
encode, cross-encoder predict, Gemini HTTP call) into a threadpool. The old
`def ask_question` also ran in a threadpool, but the pool is capped at 40
threads -- and bge-m3 + the reranker are heavy enough that you want that
concurrency limit to be explicit rather than accidental.

NEW: /note for chapter note generation, /health for uptime checks.
"""

from fastapi import APIRouter, HTTPException
from openai import APIError
from starlette.concurrency import run_in_threadpool

from app.schemas.query import NoteRequest, QueryRequest, QueryResponse
from app.services.note_service import generate_chapter_note
from app.services.qa_service import answer_question

router = APIRouter()


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.post("/ask", response_model=QueryResponse)
async def ask_question(body: QueryRequest) -> QueryResponse:
    query = body.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="query খালি রাখা যাবে না।")
    try:
        return await run_in_threadpool(answer_question, query)
    except APIError as e:
        # Retrieval succeeded but the LLM call failed (quota, bad key, upstream
        # outage, ...). Surface as a clean 503 instead of a bare 500 -- the
        # client should retry, not treat this like a malformed request.
        raise HTTPException(
            status_code=503,
            detail="উত্তর তৈরির সার্ভিস সাময়িকভাবে অনুপলব্ধ। কিছুক্ষণ পর আবার চেষ্টা করুন।",
        ) from e


@router.post("/note")
async def make_note(body: NoteRequest) -> dict:
    result = await run_in_threadpool(generate_chapter_note, body.chapter_id)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result
