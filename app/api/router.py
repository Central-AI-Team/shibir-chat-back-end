"""HTTP layer.

The public surface is deliberately small: GET /health and POST /chat. /chat
is the single entry point for every user-facing interaction -- it classifies
the free-text message into NOTE / ROLEPLAY / SUGGESTION / QA (see
app.services.intent_classifier) and dispatches internally to the matching
service. The previous /ask, /note, /note-by-text endpoints have been removed;
their underlying service functions (answer_question, generate_chapter_note,
generate_book_notes_from_text) are unchanged and are still called from here,
just not exposed as separate routes.
"""

from __future__ import annotations

import time

from fastapi import APIRouter, HTTPException
from openai import APIError
from starlette.concurrency import run_in_threadpool

from app.schemas.query import ChatRequest, ChatResponse, Citation
from app.services.intent_classifier import classify_intent
from app.services.note_service import generate_book_notes_from_text
from app.services.qa_service import answer_question
from app.services.roleplay_service import handle_roleplay
from app.services.session_store import get_or_create_session, update_session
from app.services.suggestion_service import give_suggestion

router = APIRouter()

_LLM_UNAVAILABLE_DETAIL = "উত্তর তৈরির সার্ভিস সাময়িকভাবে অনুপলব্ধ। কিছুক্ষণ পর আবার চেষ্টা করুন।"


def _format_note_result(result: dict) -> str:
    if "error" in result:
        return result["error"]
    parts = [f"বই: {result['book']}"]
    for chapter in result["chapters"]:
        parts.append(f"\n{chapter['chapter']}\n{chapter['note']}")
    return "\n".join(parts)


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest) -> ChatResponse:
    message = body.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="message খালি রাখা যাবে না।")

    session_id, session = get_or_create_session(body.session_id)
    was_roleplaying = session.get("mode") == "ROLEPLAY"

    start = time.perf_counter()
    try:
        # classify_intent can fall back to an LLM call, so it goes through
        # the threadpool the same as the dispatch branches below.
        intent = await run_in_threadpool(classify_intent, message, was_roleplaying)

        sources: list[Citation] = []
        if intent == "NOTE":
            result = await run_in_threadpool(generate_book_notes_from_text, message)
            answer = _format_note_result(result)
        elif intent == "ROLEPLAY":
            answer = await run_in_threadpool(handle_roleplay, message, session)
        elif intent == "SUGGESTION":
            answer, sources = await run_in_threadpool(give_suggestion, message)
        else:  # QA
            qa_response = await run_in_threadpool(answer_question, message)
            answer, sources = qa_response.answer, qa_response.sources
    except APIError as e:
        # Retrieval/classification succeeded but the LLM call failed (quota,
        # bad key, upstream outage, ...). Surface as a clean 503 instead of a
        # bare 500 -- the client should retry, not treat this like a
        # malformed request.
        raise HTTPException(status_code=503, detail=_LLM_UNAVAILABLE_DETAIL) from e

    if was_roleplaying and intent != "ROLEPLAY":
        # Explicit exit phrase was detected -- drop out of roleplay mode and
        # clear the persona so a future roleplay starts fresh instead of
        # picking the old character back up.
        update_session(session_id, mode=intent, persona=None)
    else:
        update_session(session_id, mode=intent)

    response_time_ms = round((time.perf_counter() - start) * 1000, 2)
    return ChatResponse(
        mode=intent.lower(),
        answer=answer,
        sources=sources,
        session_id=session_id,
        response_time_ms=response_time_ms,
    )
