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

The /conversations endpoints read app.services.session_store, which is backed
by the Postgres tables conversations / messages -- see that module's docstring
for the ownership rule and the read window. Sessions survive restarts and are
shared across workers.

Access control: every route declares a level in app/auth/policy.py. Chat and
conversation routes resolve the caller's principal (app/api/deps.py) -- a
logged-in user, or a guest identified by the `shibir_owner` cookie -- and only
ever see that principal's sessions (principal.owner_key). Guests get
settings.guest_message_limit responses in total (app/auth/guest_quota.py:
reserved before any LLM work, refunded if no response was produced) and cannot
request notes. An unknown session id and someone else's id get the
same 404 body on /conversations, and on /chat both silently start a new
session under a fresh id, so ownership cannot be probed. POST/DELETE also reject a
browser Origin that is not on the CORS allow-list (CSRF).
"""

from __future__ import annotations

import itertools
import json
import re
import time

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from openai import APIError
from sqlalchemy.exc import SQLAlchemyError
from starlette.concurrency import run_in_threadpool

from app.api.deps import (
    SERVICE_UNAVAILABLE_DETAIL,
    _client_ip,
    apply_owner_cookie,
    check_origin,
    get_principal,
    limit_chat,
    limit_reads,
    public_route,
    too_many_requests,
    trace_ids,
)
from app.auth import audit, guest_quota
from app.auth.principal import Principal
from app.core import timing, tracing
from app.core.network_errors import NETWORK_ERRORS, NETWORK_UNAVAILABLE_DETAIL, is_network_error
from app.core.config import settings
from app.rag.gpu_client import GPUServiceError
from app.rag.generator import stream_answer
from app.rag.script_check import strip_stray
from app.rag.retriever import retrieve_relevant_docs
from app.schemas.query import (
    ChatRequest,
    ChatResponse,
    Citation,
    ConversationMessage,
    ConversationSummary,
    SESSION_ID_PATTERN,
)
from app.services import rate_limiter
from app.services.chitchat_service import preface, respond
from app.services.intent_classifier import classify_intent
from app.services.note_service import generate_book_notes_from_text
from app.services.qa_service import answer_question
from app.services.roleplay_service import handle_roleplay
from app.services.session_store import (
    append_history,
    create_session,
    delete_session,
    get_history,
    get_owned_session,
    list_sessions,
    update_session,
)
from app.services.suggestion_service import give_suggestion

router = APIRouter()

_LLM_UNAVAILABLE_DETAIL = "উত্তর তৈরির সার্ভিস সাময়িকভাবে অনুপলব্ধ। কিছুক্ষণ পর আবার চেষ্টা করুন।"
# One body for "unknown" AND "not yours" -- they must be indistinguishable.
_NOT_FOUND_DETAIL = "conversation পাওয়া যায়নি।"
_SESSION_ID_RE = re.compile(SESSION_ID_PATTERN)
_LOGIN_FOR_NOTE_DETAIL = "নোট তৈরি করতে অনুগ্রহ করে লগইন বা নিবন্ধন করুন।"


def _not_found() -> HTTPException:
    return HTTPException(status_code=404, detail=_NOT_FOUND_DETAIL)


def _too_long_detail(limit: int) -> str:
    return f"message অনেক বড়। সর্বোচ্চ {limit} অক্ষর লেখা যাবে।"


def validate_chat(body: ChatRequest) -> None:
    """Runs before limit_chat and the guest quota, so an empty or oversized
    message consumes no rate-limit hit and no guest message."""
    message = body.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="message খালি রাখা যাবে না।")
    if len(message) > settings.max_message_chars:
        raise HTTPException(status_code=413, detail=_too_long_detail(settings.max_message_chars))


def _reserve_guest(
    principal: Principal, request: Request
) -> tuple[guest_quota.Reservation | None, Response | None]:
    """(reservation, None) for a guest with quota left; (None, None) for a
    user; (None, 401 response) for a guest who has run out. Blocking (DB):
    call from a threadpool. A database failure is a 503 (fail closed)."""
    if principal.is_user:
        return None, None
    ip = _client_ip(request)
    try:
        reservation = guest_quota.reserve(principal.owner_key, ip)
    except SQLAlchemyError as e:
        raise HTTPException(status_code=503, detail=SERVICE_UNAVAILABLE_DETAIL) from e
    if reservation is not None:
        return reservation, None
    audit.event("guest_limit_reached", ip=ip)
    body = {
        "detail": guest_quota.LIMIT_DETAIL,
        "code": guest_quota.LIMIT_CODE,
        "guest_limit": settings.guest_message_limit,
        "guest_remaining": 0,
    }
    return None, apply_owner_cookie(request, JSONResponse(status_code=401, content=body))


def _guest_fields(principal: Principal, reservation: guest_quota.Reservation | None) -> dict:
    if principal.is_user or reservation is None:
        return {"guest_remaining": None, "guest_limit": None}
    return {"guest_remaining": reservation.remaining, "guest_limit": settings.guest_message_limit}


def _resolve_session(session_id: str | None, principal: Principal) -> tuple[str, dict]:
    """The caller's session if the id exists AND is theirs; otherwise (no id,
    malformed, unknown, deleted, someone else's) a NEW session under a fresh
    server-issued id, which the response returns. Unknown and not-owned look
    identical, so ids cannot be probed, and the client's id is never reused.
    Why not 404: when the store was in-memory every restart made each open
    chat 404 until the front-end dropped its id; the silent fresh start stays
    because the front-end relies on it. A logged-in caller's user_id is stored
    on the row and selects the user cap."""
    if session_id is not None and _SESSION_ID_RE.fullmatch(session_id):
        session = get_owned_session(session_id, principal.owner_key)
        if session is not None:
            return session_id, session
    return create_session(principal.owner_key, user_id=principal.user_id)


def _format_note_result(result: dict) -> str:
    if "error" in result:
        return result["error"]
    parts = [f"বই: {result['book']}"]
    for chapter in result["chapters"]:
        parts.append(f"\n{chapter['chapter']}\n{chapter['note']}")
    return "\n".join(parts)


def _record_turn(
    session_id: str,
    intent: str,
    message: str,
    answer: str,
    sources: list[Citation] | None = None,
) -> None:
    """Persist one user+assistant turn to the session history. `sources` are
    the citations returned with the answer; they are stored on the assistant
    turn so GET /conversations/{id}/messages can hand them back.

    Skipped for ROLEPLAY: roleplay_service already appends both turns through
    append_history() itself, so recording again here would double them.
    """
    if intent != "ROLEPLAY":
        append_history(session_id, "user", message)
        append_history(session_id, "assistant", answer, sources=sources)


# ---------------------------------------------------------------------------
# health
# ---------------------------------------------------------------------------


@router.get("/health", dependencies=[Depends(public_route)])
def health() -> dict:
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# chat (non-streaming)
# ---------------------------------------------------------------------------


@router.post(
    "/chat",
    response_model=ChatResponse,
    dependencies=[Depends(check_origin), Depends(validate_chat), Depends(limit_chat)],
)
async def chat(body: ChatRequest, request: Request, principal: Principal = Depends(get_principal)):
    if body.stream:
        # chat_stream returns its own StreamingResponse and sets the cookie on it.
        # It reserves guest quota (a DB call), so it runs in the threadpool.
        return await run_in_threadpool(chat_stream, body, request, principal)
    reservation, rejected = await run_in_threadpool(_reserve_guest, principal, request)
    if rejected is not None:
        return rejected
    timing.start_request(body.message)
    try:
        return await _chat(body, principal, reservation, _client_ip(request))
    except Exception as e:
        timing.record_error("request_total", e)
        # No response was produced (upstream 503, guest NOTE 403, ...): give
        # the guest's message back. Never reached after a successful response.
        await run_in_threadpool(guest_quota.refund, reservation)
        raise
    finally:
        timing.finish_request()


async def _chat(
    body: ChatRequest, principal: Principal, reservation: guest_quota.Reservation | None, ip: str
) -> ChatResponse:
    owner = principal.owner_key
    message = body.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="message খালি রাখা যাবে না।")
    if len(message) > settings.max_message_chars:
        raise HTTPException(status_code=413, detail=_too_long_detail(settings.max_message_chars))

    session_id, session = _resolve_session(body.session_id, principal)
    was_roleplaying = session.get("mode") == "ROLEPLAY"
    trace_user, trace_session = trace_ids(owner, session_id)

    # One Langfuse trace per request -- every nested LLM call (via
    # app/core/llm.py) and every pipeline-stage span groups under it. No-op
    # when tracing is disabled. Everything tracing-related below is inside the
    # try/finally so a tracing failure can never affect the response.
    # body.user_id is client-supplied (spoofable) and ignored; neither the real
    # session id nor the owner hash leaves the process (see trace_ids()).
    tracing.start_request_trace(
        name="chat", query=message, session_id=trace_session, user_id=trace_user
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
            with timing.timed("intent_classification"):
                intent = await run_in_threadpool(classify_intent, message, was_roleplaying)
            timing.set_mode(intent)

            # NOTE/ROLEPLAY/CHITCHAT answers skip generator.py, so strip stray-script
            # letters here (no retry: a note is 28 s). QA/SUGGESTION retry in generator.
            if intent == "NOTE":
                if not principal.is_user:
                    audit.event("forbidden_access", ip=ip, path="/chat", reason="guest_note")
                    raise HTTPException(status_code=403, detail=_LOGIN_FOR_NOTE_DETAIL)
                if len(message) > settings.max_note_chars:
                    raise HTTPException(status_code=413, detail=_too_long_detail(settings.max_note_chars))
                retry = rate_limiter.note_per_owner.hit(owner, settings.note_limit_per_owner_per_hour)
                if retry is not None:
                    raise too_many_requests(rate_limiter.NOTE_LIMIT_DETAIL, retry)
                result = await run_in_threadpool(generate_book_notes_from_text, message)
                answer = strip_stray(_format_note_result(result))
            elif intent == "ROLEPLAY":
                answer = strip_stray(await run_in_threadpool(handle_roleplay, message, session))
            elif intent == "SUGGESTION":
                answer, sources = await run_in_threadpool(give_suggestion, message)
            elif intent == "CHITCHAT":
                # Greetings / thanks / "who are you": no retrieval, no sources.
                # The history is read before this turn is recorded below.
                answer = strip_stray(await run_in_threadpool(respond, message, list(session["history"])))
            else:  # QA
                qa_response = await run_in_threadpool(answer_question, message)
                answer, sources = qa_response.answer, qa_response.sources
        except (APIError, GPUServiceError, *NETWORK_ERRORS) as e:
            # An upstream dependency failed: the LLM call (quota, bad key,
            # outage, ...) or the external GPU embed/rerank service. Surface
            # as a clean 503 instead of a bare 500 -- the client should retry,
            # not treat this like a malformed request.
            trace_error = True
            detail = NETWORK_UNAVAILABLE_DETAIL if is_network_error(e) else _LLM_UNAVAILABLE_DETAIL
            raise HTTPException(status_code=503, detail=detail) from e

        if was_roleplaying and intent != "ROLEPLAY":
            # Explicit exit phrase was detected -- drop out of roleplay mode and
            # clear the persona so a future roleplay starts fresh instead of
            # picking the old character back up.
            update_session(session_id, mode=intent, persona=None)
        else:
            update_session(session_id, mode=intent)

        _record_turn(session_id, intent, message, answer, sources)

        response_time_ms = round((time.perf_counter() - start) * 1000, 2)
        return ChatResponse(
            mode=intent.lower(),
            answer=answer,
            sources=sources,
            session_id=session_id,
            response_time_ms=response_time_ms,
            **_guest_fields(principal, reservation),
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


def _qa_stream(
    message: str, *, trace_id: str | None = None, parent_observation_id: str | None = None
):
    rec = timing.current()
    """Return (sources, answer_delta_iterator) for a QA message, mirroring
    the relevance gate in qa_service.answer_question -- that function stays
    the source of truth for the non-streaming path; keep this in sync.

    trace_id/parent_observation_id are threaded through to stream_answer() so
    the streamed QA generation nests under the request's root span -- the
    streaming path can't rely on ambient trace context for this (see
    app/core/tracing.py's module docstring).
    """
    # Same small-talk split as answer_question (one shared function): pure
    # small talk is answered directly, with no retriever / book-QA LLM call;
    # "hi, <question>" yields the greeting as the first delta and then streams
    # the answer to the question alone.
    greeting, question = preface(message)
    if not question:
        return [], iter([greeting])

    # The `retrieve-context` span is emitted inside retrieve_stages().
    citations = retrieve_relevant_docs(question)
    relevant = bool(citations) and citations[0].rerank_score >= settings.min_rerank_score
    tracing.record_gate(
        grounded=relevant,
        top_score=citations[0].rerank_score if citations else None,
        threshold=settings.min_rerank_score,
    )
    grounding = citations if relevant else []
    deltas = stream_answer(
        question, grounding, trace_id=trace_id, parent_observation_id=parent_observation_id,
        **({"timing_rec": rec} if rec else {}),
    )
    if greeting:
        # Same text as the non-streaming "<greeting>\n\n<answer>".
        deltas = itertools.chain([f"{greeting}\n\n"], deltas)
    return grounding, deltas


@router.post(
    "/chat/stream", dependencies=[Depends(check_origin), Depends(validate_chat), Depends(limit_chat)]
)
def chat_stream(
    body: ChatRequest, request: Request, principal: Principal = Depends(get_principal)
) -> Response:
    owner = principal.owner_key
    message = body.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="message খালি রাখা যাবে না।")
    if len(message) > settings.max_message_chars:
        raise HTTPException(status_code=413, detail=_too_long_detail(settings.max_message_chars))
    # Plain HTTP 401 before the stream starts when a guest is out of messages.
    reservation, rejected = _reserve_guest(principal, request)
    if rejected is not None:
        return rejected
    ip = _client_ip(request)

    session_id, session = _resolve_session(body.session_id, principal)
    was_roleplaying = session.get("mode") == "ROLEPLAY"
    trace_user, trace_session = trace_ids(owner, session_id)

    # A plain (sync) generator: Starlette iterates it in a threadpool, so the
    # blocking retriever / LLM-stream calls below don't stall the event loop.
    def gen():
        # One trace per streamed request. Starlette pulls this generator one
        # next() at a time via anyio.to_thread.run_sync, each call in its own
        # copied context, so the ambient trace context set inside
        # start_request_trace() is only visible during the FIRST iteration
        # (which is enough for the intent/rewrite/retrieve/gate spans -- they
        # all run before the first yield). Anything that needs the trace
        # after that -- the streamed QA generation, and
        # finalize_request_trace() in the finally -- gets it from `trace` /
        # `trace_id` / `parent_observation_id` captured here instead. No-op
        # when disabled. See app/core/tracing.py's module docstring.
        trace = tracing.start_request_trace(
            name="chat_stream", query=message, session_id=trace_session,
            user_id=trace_user,
        )
        rec = timing.start_request(message, route="POST /chat/stream")
        trace_id = tracing.trace_id_of(trace)
        parent_observation_id = tracing.parent_observation_id_of(trace)
        start = time.perf_counter()
        intent = "QA"
        answer = ""
        sources = []
        trace_error = False
        try:
            try:
                with timing.timed("intent_classification"):
                    intent = classify_intent(message, was_roleplaying)
                timing.set_mode(intent)

                if intent == "QA":
                    sources, deltas = _qa_stream(
                        message, trace_id=trace_id, parent_observation_id=parent_observation_id
                    )
                    yield _sse("sources", {"sources": [c.model_dump() for c in sources]})
                    for delta in deltas:
                        answer += delta
                        yield _sse("token", {"text": delta})
                else:
                    if intent == "NOTE":
                        if not principal.is_user:
                            audit.event("forbidden_access", ip=ip, path="/chat/stream", reason="guest_note")
                            guest_quota.refund(reservation)
                            yield _sse("error", {"detail": _LOGIN_FOR_NOTE_DETAIL})
                            return
                        if len(message) > settings.max_note_chars:
                            yield _sse("error", {"detail": _too_long_detail(settings.max_note_chars)})
                            return
                        if rate_limiter.note_per_owner.hit(owner, settings.note_limit_per_owner_per_hour) is not None:
                            yield _sse("error", {"detail": rate_limiter.NOTE_LIMIT_DETAIL})
                            return
                        answer = strip_stray(_format_note_result(generate_book_notes_from_text(message)))
                        sources = []
                    elif intent == "ROLEPLAY":
                        answer = strip_stray(handle_roleplay(message, session))
                        sources = []
                    elif intent == "CHITCHAT":
                        answer = strip_stray(respond(message, list(session["history"])))
                        sources = []
                    else:  # SUGGESTION
                        answer, sources = give_suggestion(message)
                    # These paths produce a whole answer at once -- emit it as a
                    # single token event so the client renders them uniformly.
                    yield _sse("sources", {"sources": [c.model_dump() for c in sources]})
                    yield _sse("token", {"text": answer})
            except (APIError, GPUServiceError, *NETWORK_ERRORS) as e:
                trace_error = True
                timing.record_error("request_total", e, rec=rec)
                detail = NETWORK_UNAVAILABLE_DETAIL if is_network_error(e) else _LLM_UNAVAILABLE_DETAIL
                guest_quota.refund(reservation)
                yield _sse("error", {"detail": detail})
                return
            except Exception:
                guest_quota.refund(reservation)
                raise

            if was_roleplaying and intent != "ROLEPLAY":
                update_session(session_id, mode=intent, persona=None)
            else:
                update_session(session_id, mode=intent)
            _record_turn(session_id, intent, message, answer, sources)

            response_time_ms = round((time.perf_counter() - start) * 1000, 2)
            yield _sse(
                "done",
                {
                    "mode": intent.lower(),
                    "session_id": session_id,
                    "response_time_ms": response_time_ms,
                    **_guest_fields(principal, reservation),
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
            timing.finish_request(rec)

    # A returned Response drops headers set on the injected one, so the owner
    # cookie (if just minted) is attached here explicitly.
    return apply_owner_cookie(
        request,
        StreamingResponse(
            gen(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        ),
    )


# ---------------------------------------------------------------------------
# conversations
# ---------------------------------------------------------------------------


def _owned_id(session_id: str) -> str:
    """Path ids are checked for format first; a malformed one is the same 404
    as an unknown one (no separate 422 to probe with)."""
    if not _SESSION_ID_RE.fullmatch(session_id):
        raise _not_found()
    return session_id


@router.get(
    "/conversations", response_model=list[ConversationSummary], dependencies=[Depends(limit_reads)]
)
def list_conversations(principal: Principal = Depends(get_principal)) -> list[ConversationSummary]:
    return [ConversationSummary(**row) for row in list_sessions(principal.owner_key)]


@router.get(
    "/conversations/{session_id}/messages",
    response_model=list[ConversationMessage],
    dependencies=[Depends(limit_reads)],
)
def conversation_messages(
    session_id: str, principal: Principal = Depends(get_principal)
) -> list[ConversationMessage]:
    history = get_history(_owned_id(session_id), principal.owner_key)
    if history is None:
        raise _not_found()
    return [ConversationMessage(**turn) for turn in history]


@router.delete(
    "/conversations/{session_id}",
    status_code=204,
    dependencies=[Depends(check_origin), Depends(limit_reads)],
)
def delete_conversation(
    session_id: str, request: Request, principal: Principal = Depends(get_principal)
) -> Response:
    if not delete_session(_owned_id(session_id), principal.owner_key):
        raise _not_found()
    return apply_owner_cookie(request, Response(status_code=204))
