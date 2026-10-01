"""Owned, durable conversations. Streaming and normal chat share context preparation."""
from __future__ import annotations

import json
import logging
import random
import time
import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Response
from fastapi.responses import StreamingResponse
from openai import APIError
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from app.core import tracing
from app.core.config import settings
from app.rag.gpu_client import GPUServiceError
from app.rag.generator import stream_answer
from app.rag.retriever import retrieve_relevant_docs
from app.schemas.query import ChatRequest, ChatResponse, ConversationMessage, ConversationSummary
from app.schemas.resources import MessageResources, resource_snapshot
from app.services.context import build_context, resolve_followup, is_memory_question, answer_from_memory, refresh_memory
from app.services.identity import create_identity, require_identity
from app.services.intent_classifier import classify_intent
from app.services.note_service import generate_book_notes_from_text
from app.services.qa_service import _CONVERSATIONAL_REPLIES, _conversational_category, answer_question
from app.services.roleplay_service import handle_roleplay
from app.services.session_store import (begin_turn, finish_turn, checkpoint_resources, delete_session, get_history,
    get_or_create_session, list_sessions, list_memories, delete_memory, edit_session, create_conversation)
from app.services.suggestion_service import give_suggestion

logger = logging.getLogger(__name__)
router = APIRouter()


class DurableStreamingResponse(StreamingResponse):
    def __init__(self, iterator, *, cleanup, **kwargs):
        self.iterator = iterator
        self.cleanup = cleanup
        super().__init__(iterator, **kwargs)

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            # Starlette wraps synchronous generators; explicitly close on disconnect.
            await run_in_threadpool(self.iterator.close)
            await run_in_threadpool(self.cleanup)

_LLM_UNAVAILABLE_DETAIL = 'উত্তর তৈরির সার্ভিস সাময়িকভাবে অনুপলব্ধ। কিছুক্ষণ পর আবার চেষ্টা করুন।'
_UNEXPECTED_ERROR_DETAIL = 'একটি অপ্রত্যাশিত সমস্যা হয়েছে। কিছুক্ষণ পর আবার চেষ্টা করুন।'


def _format_note_result(result):
    if 'error' in result:
        return result['error']
    parts = [f"বই: {result['book']}"]
    for chapter in result['chapters']:
        parts.append(f"\n{chapter['chapter']}\n{chapter['note']}")
    return '\n'.join(parts)


def _sse(event, data):
    return f'event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n'


def _prepare(message, session, context):
    roleplaying = session.get('mode') == 'ROLEPLAY'
    category = None if roleplaying else _conversational_category(message)
    if category:
        tracing.record_intent('QA', method='conversational_shortcut')
        return 'QA', message, random.choice(_CONVERSATIONAL_REPLIES[category])
    if not roleplaying and is_memory_question(message):
        return 'MEMORY', message, answer_from_memory(message, context)
    query, clarification = (message, None) if roleplaying else resolve_followup(message, context)
    if clarification:
        return 'QA', query, clarification
    intent = classify_intent(query, roleplaying, context=context) if context.messages() else classify_intent(query, roleplaying)
    return intent, query, None


def _dispatch(intent, message, query, session, context):
    has_context = bool(context.messages())
    if intent == 'NOTE':
        return _format_note_result(generate_book_notes_from_text(query)), MessageResources()
    if intent == 'ROLEPLAY':
        answer = handle_roleplay(message, session, context=context) if has_context else handle_roleplay(message, session)
        return answer, MessageResources()
    if intent == 'SUGGESTION':
        answer, sources = give_suggestion(message, context=context, retrieval_query=query) if has_context else give_suggestion(message)
        return answer, _bundle(sources)
    qa = answer_question(message, context=context, retrieval_query=query) if has_context else answer_question(message)
    return qa.answer, qa.resources


def _bundle(value):
    return resource_snapshot(sources=value) if isinstance(value, list) else resource_snapshot(value)


def _resource_events(resources):
    data = resources.model_dump(mode='json')
    yield _sse('sources', {'sources': data['sources']})
    if data['web_results']:
        yield _sse('web_results', {'results': data['web_results']})
    if data['verification'] is not None:
        yield _sse('verification', {'verification': data['verification']})


def _qa_stream(message, *, query=None, context=None, trace_id=None, parent_observation_id=None):
    citations = retrieve_relevant_docs(query or message)
    relevant = bool(citations) and citations[0].rerank_score >= settings.min_rerank_score
    tracing.record_gate(grounded=relevant, top_score=citations[0].rerank_score if citations else None,
                        threshold=settings.min_rerank_score)
    grounding = citations if relevant else []
    extra = {'trace_id': trace_id, 'parent_observation_id': parent_observation_id}
    if context and context.messages():
        extra['context'] = context
    return grounding, stream_answer(message, grounding, **extra)


@router.get('/health')
def health():
    return {'status': 'ok'}


@router.post('/identity', status_code=201)
def identity():
    return create_identity()


@router.get('/identity')
def current_identity(owner_id: str = Depends(require_identity)):
    return {'user_id': owner_id}


def _start(body, owner_id):
    message = body.message.strip()
    if not message:
        raise HTTPException(400, 'message খালি রাখা যাবে না।')
    sid, session = get_or_create_session(str(body.session_id) if body.session_id else None, owner_id)
    rid = str(body.request_id or uuid.uuid4())
    claim = begin_turn(sid, owner_id, rid, message, options={
        key: True for key in ('search_web', 'verify_claim') if getattr(body, key)
    })
    replay = claim if isinstance(claim, dict) else None
    lease = claim if isinstance(claim, str) else None
    if lease:
        _, session = get_or_create_session(sid, owner_id)
        session['history'] = [m for m in session['history'] if m['request_id'] != rid]
    return message, sid, rid, session, replay, lease


@router.post('/chat', response_model=ChatResponse)
async def chat(body: ChatRequest, background_tasks: BackgroundTasks, owner_id: str = Depends(require_identity)):
    message, sid, rid, session, replay, lease = await run_in_threadpool(_start, body, owner_id)
    if replay:
        return ChatResponse(**replay)
    start = time.perf_counter()
    intent, answer, sources, success = 'QA', '', [], False
    resources = MessageResources()
    tracing.start_request_trace(name='chat', query=message, session_id=sid, user_id=owner_id)
    try:
        context = await run_in_threadpool(build_context, message, sid, owner_id, rid)
        intent, query, shortcut = await run_in_threadpool(_prepare, message, session, context)
        if shortcut is not None:
            answer = shortcut
        else:
            answer, resources = await run_in_threadpool(_dispatch, intent, message, query, session, context)
            resources = _bundle(resources)
            sources = resources.sources
        persona = session.get('persona') if intent == 'ROLEPLAY' else None
        saved = await run_in_threadpool(finish_turn, sid, owner_id, rid, answer,
            [c.model_dump() for c in sources], intent, persona, lease=lease, resources=resources)
        if not saved:
            raise HTTPException(409, 'Conversation request lease expired.')
        success = True
        background_tasks.add_task(refresh_memory, sid, owner_id)
        return ChatResponse(mode=intent.lower(), answer=answer, resources=resources, session_id=sid,
            response_time_ms=round((time.perf_counter() - start) * 1000, 2))
    except (APIError, GPUServiceError) as e:
        raise HTTPException(503, _LLM_UNAVAILABLE_DETAIL) from e
    finally:
        if not success:
            await run_in_threadpool(finish_turn, sid, owner_id, rid, answer, [], session.get('mode'), session.get('persona'), False, lease=lease)
        tracing.finalize_request_trace(answer=answer, sources=sources, mode=intent.lower(),
            response_time_ms=round((time.perf_counter() - start) * 1000, 2), error=not success)
        tracing.clear_request_trace()


@router.post('/chat/stream')
def chat_stream(body: ChatRequest, background_tasks: BackgroundTasks, owner_id: str = Depends(require_identity)):
    message, sid, rid, session, replay, lease = _start(body, owner_id)

    state = {'complete': bool(replay), 'answer': '', 'resources': None}

    def cleanup():
        if not state['complete']:
            finish_turn(sid, owner_id, rid, state['answer'], [], session.get('mode'),
                        session.get('persona'), False, lease=lease, resources=state['resources'])

    def publish(resources):
        # Commit before displaying cards: disconnects and worker crashes cannot lose them.
        if not checkpoint_resources(sid, owner_id, rid, resources, lease=lease):
            raise RuntimeError('Conversation request lease expired.')
        state['resources'] = resources
        yield from _resource_events(resources)

    def gen():
        yield _sse('session', {'session_id': sid, 'request_id': rid})
        if replay:
            yield from _resource_events(resource_snapshot(replay['resources']))
            yield _sse('token', {'text': replay['answer']})
            yield _sse('done', replay)
            return
        trace = tracing.start_request_trace(name='chat_stream', query=message, session_id=sid, user_id=owner_id)
        start = time.perf_counter()
        intent, answer, success = 'QA', '', False
        resources = MessageResources()
        try:
            context = build_context(message, sid, owner_id, rid)
            intent, query, shortcut = _prepare(message, session, context)
            if shortcut is not None:
                deltas = iter([shortcut])
            elif intent == 'QA':
                value, deltas = _qa_stream(message, query=query, context=context,
                    trace_id=tracing.trace_id_of(trace), parent_observation_id=tracing.parent_observation_id_of(trace))
                resources = _bundle(value)
            else:
                answer, value = _dispatch(intent, message, query, session, context)
                resources = _bundle(value)
                deltas = iter([answer])
                answer = ''
            yield from publish(resources)
            for delta in deltas:
                answer += delta
                state['answer'] = answer
                yield _sse('token', {'text': delta})
            saved = finish_turn(sid, owner_id, rid, answer, [], intent,
                        session.get('persona') if intent == 'ROLEPLAY' else None,
                        lease=lease, resources=resources)
            if not saved:
                raise RuntimeError('Conversation request lease expired.')
            success = True
            state['complete'] = True
            background_tasks.add_task(refresh_memory, sid, owner_id)
            yield _sse('done', ChatResponse(mode=intent.lower(), answer=answer, resources=resources,
                session_id=sid, response_time_ms=round((time.perf_counter() - start) * 1000, 2)).model_dump(mode='json'))
        except (APIError, GPUServiceError):
            yield _sse('error', {'detail': _LLM_UNAVAILABLE_DETAIL})
        except Exception:
            logger.exception('chat_stream_unexpected_error')
            yield _sse('error', {'detail': _UNEXPECTED_ERROR_DETAIL})
        finally:
            if not success:
                cleanup()
            tracing.finalize_request_trace(trace=trace, answer=answer, sources=resources.sources,
                mode=intent.lower(), response_time_ms=round((time.perf_counter() - start) * 1000, 2), error=not success)
            tracing.clear_request_trace()

    return DurableStreamingResponse(gen(), cleanup=cleanup, media_type='text/event-stream', background=background_tasks,
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


@router.get('/conversations', response_model=list[ConversationSummary])
def list_conversations(owner_id: str = Depends(require_identity)):
    return list_sessions(owner_id)


@router.get('/conversations/{session_id}/messages', response_model=list[ConversationMessage])
def conversation_messages(session_id: str, owner_id: str = Depends(require_identity)):
    return get_history(session_id, owner_id)


@router.delete('/conversations/{session_id}', status_code=204)
def delete_conversation(session_id: str, owner_id: str = Depends(require_identity)):
    delete_session(session_id, owner_id)
    return Response(status_code=204)


class ConversationEdit(BaseModel):
    title: str | None = Field(default=None, max_length=80)
    memory_enabled: bool | None = None


@router.patch('/conversations/{session_id}', status_code=204)
def patch_conversation(session_id: str, body: ConversationEdit, owner_id: str = Depends(require_identity)):
    edit_session(session_id, owner_id, body.title, body.memory_enabled)
    return Response(status_code=204)


@router.get('/memories')
def memories(owner_id: str = Depends(require_identity)):
    return list_memories(owner_id)


@router.delete('/memories/{memory_id}', status_code=204)
def forget_memory(memory_id: str, owner_id: str = Depends(require_identity)):
    delete_memory(memory_id, owner_id)
    return Response(status_code=204)


class ConversationCreate(BaseModel):
    session_id: uuid.UUID


@router.post('/conversations', status_code=201)
def new_conversation(body: ConversationCreate, owner_id: str = Depends(require_identity)):
    return create_conversation(owner_id, str(body.session_id))
