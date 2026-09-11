"""HTTP layer.

Routes:
  GET    /health
  POST   /chat                            classify intent -> NOTE / ROLEPLAY /
                                          SUGGESTION / QA and dispatch
  POST   /chat/stream                     same dispatch, Server-Sent Events;
                                          the QA answer streams token-by-token
  GET    /conversations                   sidebar list (this worker's sessions)
  GET    /conversations/{id}/messages     the [{role, content}] turns
  DELETE /conversations/{id}              forget a session

/chat is the single entry point for every user-facing interaction -- it
classifies the free-text message into NOTE / ROLEPLAY / SUGGESTION / QA (see
app.services.intent_classifier) and dispatches internally to the matching
service. The previous /ask, /note, /note-by-text endpoints have been removed;
their underlying service functions (answer_question, generate_chapter_note,
generate_book_notes_from_text) are unchanged and are still called from here,
just not exposed as separate routes.

The /conversations endpoints read app.services.session_store, which is an
in-process dict -- see that module's docstring: they only see sessions this
worker handled and lose everything on restart. Good enough for the current
single-worker deployment; a shared store (Redis / DB) is the upgrade path.
"""

from __future__ import annotations

import json
import random
import time

from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import StreamingResponse
from openai import APIError
from starlette.concurrency import run_in_threadpool

from app.core import tracing
from app.core.config import settings
from app.rag.generator import stream_answer
from app.rag.retriever import retrieve_relevant_docs
from app.schemas.query import (
    ChatRequest,
    ChatResponse,
    Citation,
    ConversationMessage,
    ConversationSummary,
)
from app.services.intent_classifier import classify_intent
from app.services.note_service import generate_book_notes_from_text
from app.services.qa_service import (
    _CONVERSATIONAL_REPLIES,
    _is_conversational,
    answer_question,
)
from app.services.roleplay_service import handle_roleplay
from app.services.session_store import (
    append_history,
    delete_session,
    get_history,
    get_or_create_session,
    list_sessions,
    update_session,
)
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


def _record_turn(session_id: str, intent: str, message: str, answer: str) -> None:
    """Persist one user+assistant turn to the session history.

    Skipped for ROLEPLAY: roleplay_service already appends both turns to
    session["history"] itself, so recording again here would double them.
    """
    if intent != "ROLEPLAY":
        append_history(session_id, "user", message)
        append_history(session_id, "assistant", answer)


# ---------------------------------------------------------------------------
# health
# ---------------------------------------------------------------------------


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# chat (non-streaming)
# ---------------------------------------------------------------------------


@router.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest) -> ChatResponse:
    message = body.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="message খালি রাখা যাবে না।")

    session_id, session = get_or_create_session(body.session_id)
    was_roleplaying = session.get("mode") == "ROLEPLAY"

    # One Langfuse trace per request -- every nested LLM call (via
    # app/core/llm.py) and every pipeline-stage span groups under it. No-op
    # when tracing is disabled. Everything tracing-related below is inside the
    # try/finally so a tracing failure can never affect the response.
    tracing.start_request_trace(
        name="chat", query=message, session_id=session_id, user_id=body.user_id
    )
    start = time.perf_counter()
    intent = "QA"
    answer = ""
    sources: list[Citation] = []
    trace_error = False
    try:
        try:
            # classify_intent can fall back to an LLM call, so it goes through
            # the threadpool the same as the dispatch branches below.
            intent = await run_in_threadpool(classify_intent, message, was_roleplaying)

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
            # bad key, upstream outage, ...). Surface as a clean 503 instead of
            # a bare 500 -- the client should retry, not treat this like a
            # malformed request.
            trace_error = True
            raise HTTPException(status_code=503, detail=_LLM_UNAVAILABLE_DETAIL) from e

        if was_roleplaying and intent != "ROLEPLAY":
            # Explicit exit phrase was detected -- drop out of roleplay mode and
            # clear the persona so a future roleplay starts fresh instead of
            # picking the old character back up.
            update_session(session_id, mode=intent, persona=None)
        else:
            update_session(session_id, mode=intent)

        _record_turn(session_id, intent, message, answer)

        response_time_ms = round((time.perf_counter() - start) * 1000, 2)
        return ChatResponse(
            mode=intent.lower(),
            answer=answer,
            sources=sources,
            session_id=session_id,
            response_time_ms=response_time_ms,
        )
    finally:
        tracing.finalize_request_trace(
            answer=answer,
            sources=sources,
            mode=intent.lower(),
            response_time_ms=round((time.perf_counter() - start) * 1000, 2),
            error=trace_error,
        )
        tracing.clear_request_trace()


# ---------------------------------------------------------------------------
# chat (streaming, Server-Sent Events)
# ---------------------------------------------------------------------------


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _qa_stream(message: str, *, trace_id: str | None = None):
    """Return (sources, answer_delta_iterator) for a QA message, mirroring
    the relevance gate in qa_service.answer_question -- that function stays
    the source of truth for the non-streaming path; keep this in sync.

    trace_id is threaded through to stream_answer() so the streamed QA
    generation nests under the request trace -- the streaming path can't rely
    on the ContextVar for this (see below).
    """
    # Same cheap deterministic small-talk shortcut as answer_question: a
    # greeting has nothing to retrieve against, so answer it directly with
    # no retriever / LLM call.
    if _is_conversational(message):
        return [], iter([random.choice(_CONVERSATIONAL_REPLIES)])

    # The `retrieve` trace span is emitted inside retrieve_stages().
    citations = retrieve_relevant_docs(message)
    relevant = bool(citations) and citations[0].rerank_score >= settings.min_rerank_score
    tracing.record_gate(
        grounded=relevant,
        top_score=citations[0].rerank_score if citations else None,
        threshold=settings.min_rerank_score,
    )
    grounding = citations if relevant else []
    return grounding, stream_answer(message, grounding, trace_id=trace_id)


@router.post("/chat/stream")
def chat_stream(body: ChatRequest) -> StreamingResponse:
    message = body.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="message খালি রাখা যাবে না।")

    session_id, session = get_or_create_session(body.session_id)
    was_roleplaying = session.get("mode") == "ROLEPLAY"

    # A plain (sync) generator: Starlette iterates it in a threadpool, so the
    # blocking retriever / LLM-stream calls below don't stall the event loop.
    def gen():
        # One trace per streamed request. Starlette pulls this generator one
        # next() at a time via anyio.to_thread.run_sync, each call in its own
        # copied context, so the ContextVar set inside start_request_trace()
        # is only visible during the FIRST iteration (which is enough for the
        # intent/rewrite/retrieve/gate spans -- they all run before the first
        # yield). Anything that needs the trace after that -- the streamed QA
        # generation, and finalize_request_trace() in the finally -- gets it
        # from `trace` / `trace_id` captured here instead. No-op when disabled.
        trace = tracing.start_request_trace(
            name="chat_stream", query=message, session_id=session_id,
            user_id=body.user_id,
        )
        trace_id = tracing.trace_id_of(trace)
        start = time.perf_counter()
        intent = "QA"
        answer = ""
        sources = []
        trace_error = False
        try:
            try:
                intent = classify_intent(message, was_roleplaying)

                if intent == "QA":
                    sources, deltas = _qa_stream(message, trace_id=trace_id)
                    yield _sse("sources", {"sources": [c.model_dump() for c in sources]})
                    for delta in deltas:
                        answer += delta
                        yield _sse("token", {"text": delta})
                else:
                    if intent == "NOTE":
                        answer = _format_note_result(generate_book_notes_from_text(message))
                        sources = []
                    elif intent == "ROLEPLAY":
                        answer = handle_roleplay(message, session)
                        sources = []
                    else:  # SUGGESTION
                        answer, sources = give_suggestion(message)
                    # These paths produce a whole answer at once -- emit it as a
                    # single token event so the client renders them uniformly.
                    yield _sse("sources", {"sources": [c.model_dump() for c in sources]})
                    yield _sse("token", {"text": answer})
            except APIError:
                trace_error = True
                yield _sse("error", {"detail": _LLM_UNAVAILABLE_DETAIL})
                return

            if was_roleplaying and intent != "ROLEPLAY":
                update_session(session_id, mode=intent, persona=None)
            else:
                update_session(session_id, mode=intent)
            _record_turn(session_id, intent, message, answer)

            response_time_ms = round((time.perf_counter() - start) * 1000, 2)
            yield _sse(
                "done",
                {
                    "mode": intent.lower(),
                    "session_id": session_id,
                    "response_time_ms": response_time_ms,
                },
            )
        finally:
            # Pass `trace` explicitly: the ContextVar is gone by this
            # iteration (see the comment at the top of gen()).
            tracing.finalize_request_trace(
                trace=trace,
                answer=answer,
                sources=sources,
                mode=intent.lower(),
                response_time_ms=round((time.perf_counter() - start) * 1000, 2),
                error=trace_error,
            )
            tracing.clear_request_trace()

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# conversations
# ---------------------------------------------------------------------------


@router.get("/conversations", response_model=list[ConversationSummary])
def list_conversations() -> list[ConversationSummary]:
    return [ConversationSummary(**row) for row in list_sessions()]


@router.get(
    "/conversations/{session_id}/messages",
    response_model=list[ConversationMessage],
)
def conversation_messages(session_id: str) -> list[ConversationMessage]:
    history = get_history(session_id)
    if history is None:
        raise HTTPException(status_code=404, detail="conversation পাওয়া যায়নি।")
    return [ConversationMessage(**turn) for turn in history]


@router.delete("/conversations/{session_id}", status_code=204)
def delete_conversation(session_id: str) -> Response:
    if not delete_session(session_id):
        raise HTTPException(status_code=404, detail="conversation পাওয়া যায়নি।")
    return Response(status_code=204)
