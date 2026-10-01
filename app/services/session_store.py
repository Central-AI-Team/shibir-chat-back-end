"""Database-backed conversation storage. Full history is retained; prompt windows are separate."""
from __future__ import annotations

import uuid
from datetime import timedelta

from fastapi import HTTPException
from sqlalchemy import delete, func, select, update

from app.db.chat_models import ChatMessage, Conversation, UserMemory, now
from app.db.session import SessionLocal
from app.schemas.resources import resource_snapshot

# Only the prompt window is bounded. No stored messages are discarded.
MAX_HISTORY = 20


def _owned(db, session_id, owner_id):
    row = db.scalar(select(Conversation).where(Conversation.id == session_id, Conversation.owner_id == owner_id))
    if row is None:
        raise HTTPException(404, 'conversation পাওয়া যায়নি।')
    return row


def _message(row):
    resources = resource_snapshot(row.resources, sources=row.sources).model_dump(mode='json')
    return {'id': row.id, 'sequence': row.sequence, 'role': row.role, 'content': row.content,
            'status': row.status, 'resources': resources, 'sources': resources['sources'],
            'web_results': resources['web_results'], 'verification': resources['verification'],
            'mode': row.mode, 'options': row.options or {}, 'request_id': row.request_id}


def get_or_create_session(session_id: str | None, owner_id: str) -> tuple[str, dict]:
    with SessionLocal.begin() as db:
        if session_id is None:
            row = Conversation(id=str(uuid.uuid4()), owner_id=owner_id)
            db.add(row)
            db.flush()
        else:
            row = _owned(db, session_id, owner_id)
        history = db.scalars(select(ChatMessage).where(ChatMessage.conversation_id == row.id,
            ChatMessage.status == 'complete').order_by(ChatMessage.sequence.desc()).limit(MAX_HISTORY)).all()
        return row.id, {'mode': row.mode, 'persona': row.persona, 'summary': row.summary,
                        'history': [_message(m) for m in reversed(history)]}


def begin_turn(session_id, owner_id, request_id, message, options=None):
    """Atomically claim a conversation; replay completed requests and reject concurrent writers.

    Claims expire after 15 minutes if a process dies. Conditional updates work on
    PostgreSQL and SQLite without holding a transaction open during generation.
    """
    with SessionLocal.begin() as db:
        row = _owned(db, session_id, owner_id)
        existing = db.scalar(select(ChatMessage).where(ChatMessage.conversation_id == session_id,
            ChatMessage.request_id == request_id, ChatMessage.role == 'assistant'))
        user = db.scalar(select(ChatMessage).where(ChatMessage.conversation_id == session_id,
            ChatMessage.request_id == request_id, ChatMessage.role == 'user'))
        options = dict(options or {})
        if user and (user.content != message or (user.options or {}) != options):
            raise HTTPException(409, 'This request_id already belongs to another message.')
        if existing and existing.status == 'complete':
            snapshot = resource_snapshot(existing.resources, sources=existing.sources).model_dump(mode='json')
            return {'answer': existing.content, 'resources': snapshot, 'sources': snapshot['sources'],
                    'web_results': snapshot['web_results'], 'verification': snapshot['verification'], 'mode': existing.mode,
                    'session_id': session_id, 'response_time_ms': 0}
        lease = str(uuid.uuid4())
        claimed = db.execute(update(Conversation).where(Conversation.id == session_id,
            (Conversation.active_request.is_(None)) | (Conversation.active_since < now() - timedelta(minutes=15)))
            .values(active_request=lease, active_since=now()).execution_options(synchronize_session=False)).rowcount
        if not claimed:
            raise HTTPException(409, 'এই conversation-এ একটি উত্তর তৈরি হচ্ছে। কিছুক্ষণ পর আবার চেষ্টা করুন।')
        db.refresh(row)
        # An expired worker's partial answer must never become completed context.
        db.execute(update(ChatMessage).where(ChatMessage.conversation_id == session_id,
            ChatMessage.role == 'assistant', ChatMessage.status == 'pending').values(status='interrupted'))
        if not existing:
            seq = row.next_sequence
            db.add(ChatMessage(id=str(uuid.uuid4()), conversation_id=session_id, sequence=seq,
                request_id=request_id, role='user', content=message, options=options))
            db.add(ChatMessage(id=str(uuid.uuid4()), conversation_id=session_id, sequence=seq + 1,
                request_id=request_id, role='assistant', content='', status='pending', options=options))
            db.execute(update(Conversation).where(Conversation.id == session_id).values(
                next_sequence=seq + 2, updated_at=now(),
                title=' '.join(message.split())[:60] if seq == 1 else row.title))
        else:
            existing.status = 'pending'
            existing.content = ''
            existing.sources = []
            existing.resources = resource_snapshot().model_dump(mode='json')
        return lease


def finish_turn(session_id, owner_id, request_id, answer, sources, mode, persona=None, complete=True, *, lease, resources=None):
    with SessionLocal.begin() as db:
        if db.scalar(select(Conversation.id).where(Conversation.id == session_id, Conversation.owner_id == owner_id)) is None:
            return False
        # A worker with an expired claim cannot overwrite a newer turn.
        claimed = db.execute(update(Conversation).where(Conversation.id == session_id,
            Conversation.active_request == lease).values(active_request=None, active_since=None,
                mode=mode, persona=persona, updated_at=now())).rowcount
        if not claimed:
            return False
        db.execute(update(ChatMessage).where(ChatMessage.conversation_id == session_id,
            ChatMessage.request_id == request_id, ChatMessage.role == 'user').values(mode=(mode or 'QA').lower()))
        values = {'content': answer, 'mode': (mode or 'QA').lower(),
                  'status': 'complete' if complete else 'interrupted'}
        if resources is not None or complete:
            snapshot = resource_snapshot(resources, sources=sources).model_dump(mode='json')
            values.update(resources=snapshot, sources=snapshot['sources'])
        elif sources:
            # Legacy callers can still finalize sources; an empty cleanup must not
            # discard a resource checkpoint already delivered to the browser.
            row = db.scalar(select(ChatMessage).where(ChatMessage.conversation_id == session_id,
                ChatMessage.request_id == request_id, ChatMessage.role == 'assistant'))
            snapshot = resource_snapshot(row.resources, sources=sources).model_dump(mode='json')
            values.update(resources=snapshot, sources=snapshot['sources'])
        db.execute(update(ChatMessage).where(ChatMessage.conversation_id == session_id,
            ChatMessage.request_id == request_id, ChatMessage.role == 'assistant').values(**values))
        return True


def list_sessions(owner_id):
    with SessionLocal() as db:
        counts = select(ChatMessage.conversation_id, func.count().label('n')).group_by(ChatMessage.conversation_id).subquery()
        rows = db.execute(select(Conversation, func.coalesce(counts.c.n, 0)).outerjoin(counts,
            counts.c.conversation_id == Conversation.id).where(Conversation.owner_id == owner_id)
            .order_by(Conversation.updated_at.desc())).all()
        return [{'id': c.id, 'title': c.title, 'message_count': n,
                 'created_at': c.created_at, 'updated_at': c.updated_at, 'memory_enabled': c.memory_enabled} for c, n in rows]


def get_history(session_id, owner_id):
    with SessionLocal() as db:
        _owned(db, session_id, owner_id)
        return [_message(m) for m in db.scalars(select(ChatMessage).where(
            ChatMessage.conversation_id == session_id).order_by(ChatMessage.sequence)).all()]


def delete_session(session_id, owner_id):
    with SessionLocal.begin() as db:
        _owned(db, session_id, owner_id)
        db.execute(delete(UserMemory).where(UserMemory.source_conversation_id == session_id))
        db.execute(delete(ChatMessage).where(ChatMessage.conversation_id == session_id))
        db.execute(delete(Conversation).where(Conversation.id == session_id, Conversation.owner_id == owner_id))


def list_memories(owner_id):
    with SessionLocal() as db:
        return [{'id': m.id, 'key': m.key, 'content': m.content, 'source_conversation_id': m.source_conversation_id,
                 'updated_at': m.updated_at} for m in db.scalars(select(UserMemory).where(
                     UserMemory.owner_id == owner_id).order_by(UserMemory.updated_at.desc())).all()]


def delete_memory(memory_id, owner_id):
    with SessionLocal.begin() as db:
        memory = db.scalar(select(UserMemory).where(UserMemory.id == memory_id, UserMemory.owner_id == owner_id))
        if memory is None:
            raise HTTPException(404, 'Memory not found.')
        source = memory.source_conversation_id
        db.execute(update(Conversation).where(Conversation.id == source, Conversation.owner_id == owner_id).values(memory_enabled=False))
        db.execute(delete(UserMemory).where(UserMemory.source_conversation_id == source, UserMemory.owner_id == owner_id))
        # Stop deleted facts being re-extracted or retrieved from their source chat.
        # Explicitly forgetting a memory excludes that source chat from cross-chat recall.
        # Its original transcript remains visible to its owner.
        return


def edit_session(session_id, owner_id, title=None, memory_enabled=None):
    with SessionLocal.begin() as db:
        c = _owned(db, session_id, owner_id)
        if title is not None:
            c.title = title.strip()[:80] or 'New chat'
        if memory_enabled is not None:
            c.memory_enabled = memory_enabled
            if not memory_enabled:
                db.execute(delete(UserMemory).where(UserMemory.source_conversation_id == session_id))
        c.updated_at = now()


def create_conversation(owner_id, session_id):
    """Explicit, idempotent creation; /chat never resurrects unknown or deleted ids."""
    from sqlalchemy.exc import IntegrityError
    try:
        with SessionLocal.begin() as db:
            existing = db.get(Conversation, session_id)
            if existing is not None:
                _owned(db, session_id, owner_id)
            else:
                db.add(Conversation(id=session_id, owner_id=owner_id))
    except IntegrityError:
        # Simultaneous retries can create the same id; ownership still checked.
        with SessionLocal() as db:
            _owned(db, session_id, owner_id)
    return {'session_id': session_id}


def checkpoint_resources(session_id, owner_id, request_id, resources, *, lease):
    """Commit a snapshot before emitting resource events; no stale worker may write it."""
    snapshot = resource_snapshot(resources).model_dump(mode='json')
    with SessionLocal.begin() as db:
        # Acquire the same conversation row lock/lease check as completion.
        claimed = db.execute(update(Conversation).where(Conversation.id == session_id,
            Conversation.owner_id == owner_id, Conversation.active_request == lease)
            .values(updated_at=now())).rowcount
        if not claimed:
            return False
        return bool(db.execute(update(ChatMessage).where(ChatMessage.conversation_id == session_id,
            ChatMessage.request_id == request_id, ChatMessage.role == 'assistant', ChatMessage.status == 'pending')
            .values(resources=snapshot, sources=snapshot['sources'])).rowcount)
