"""Chat session store on Postgres (tables `conversations` / `messages`,
app/db/models.py; schema docs/roadmap-q4-2026.md §2).

Sessions survive restarts and are shared by every worker/process, so the app
is no longer pinned to one uvicorn worker (rate_limiter and guest_quota are
already DB-backed).

Ownership: every conversation stores its owner's key (the sha256 of the guest
cookie token, app/api/deps.py, or principal.user_owner_key for a logged-in
user). Sessions are created ONLY by create_session(owner) with a
server-generated id; every lookup takes the owner and is
`WHERE id = :id AND owner_key = :owner AND deleted_at IS NULL`, so an unknown
id, someone else's id and a deleted id are indistinguishable to the caller
(None). Nothing here creates a session implicitly.

Read window: MAX_HISTORY is how many of the most recent turns `history` and
get_history() return. It is NOT a storage cap -- every turn is stored.

Caps: settings.guest_conversation_cap / user_conversation_cap live
conversations per owner. Creating one past the cap soft-deletes the owner's
least-recently-updated conversation. DELETE /conversations/{id} also only
sets deleted_at; scripts/purge_history.py hard-deletes later.

The dicts returned here are snapshots: mutating them does not persist.
Writers call update_session() / append_history(); the snapshot carries "id"
so a callee that only holds the session (roleplay_service) can do so.

Database access goes through _open() (SESSION_FACTORY), not SessionLocal
directly, so tests can point the store at SQLite like app.auth does. Every
call is its own short transaction, so no process-level lock is needed.
"""

from __future__ import annotations

import secrets
from datetime import datetime, timezone

from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.models import Conversation, Message

MAX_HISTORY = 20

# Tests set this to a sessionmaker bound to SQLite; None = the app's Postgres.
SESSION_FACTORY = None


def _open() -> Session:
    if SESSION_FACTORY is not None:
        return SESSION_FACTORY()
    from app.db.session import SessionLocal

    return SessionLocal()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _aware(value: datetime | None) -> datetime | None:
    """SQLite returns naive datetimes; everything stored here is UTC."""
    if value is not None and value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _live(stmt, owner: str | None = None):
    stmt = stmt.where(Conversation.deleted_at.is_(None))
    if owner is not None:
        stmt = stmt.where(Conversation.owner_key == owner)
    return stmt


def _owned(db: Session, session_id: str, owner: str) -> Conversation | None:
    """The ownership rule, in one place."""
    return db.execute(
        _live(select(Conversation).where(Conversation.id == session_id), owner)
    ).scalar_one_or_none()


def _get_live(db: Session, session_id: str) -> Conversation | None:
    """A live conversation by id only -- for writers that already resolved
    ownership on this request (update_session / append_history)."""
    return db.execute(
        _live(select(Conversation).where(Conversation.id == session_id))
    ).scalar_one_or_none()


def _turn(message: Message) -> dict:
    turn: dict = {"role": message.role, "content": message.content}
    if message.sources:
        turn["sources"] = message.sources
    return turn


def _history(db: Session, session_id: str, limit: int = MAX_HISTORY) -> list[dict]:
    """The last `limit` turns, oldest first."""
    rows = db.execute(
        select(Message)
        .where(Message.conversation_id == session_id)
        .order_by(Message.id.desc())
        .limit(limit)
    ).scalars().all()
    return [_turn(m) for m in reversed(rows)]


def _snapshot(db: Session, conv: Conversation) -> dict:
    return {
        "id": conv.id,
        "owner": conv.owner_key,
        "user_id": conv.user_id,
        "mode": conv.mode,
        "persona": conv.persona,
        "history": _history(db, conv.id),
        "created_at": _aware(conv.created_at),
        "updated_at": _aware(conv.updated_at),
    }


def _cap_for(user_id: str | None) -> int:
    return settings.user_conversation_cap if user_id else settings.guest_conversation_cap


def _enforce_cap(db: Session, owner: str, cap: int, *, room_for_new: bool) -> None:
    """Soft-delete the owner's least-recently-updated live conversations so
    that at most `cap` remain (`cap - 1` when a new one is about to be
    created)."""
    allowed = max(cap - 1, 0) if room_for_new else cap
    count = db.execute(_live(select(func.count(Conversation.id)), owner)).scalar_one()
    excess = count - allowed
    if excess <= 0:
        return
    victims = db.execute(
        _live(select(Conversation.id), owner).order_by(Conversation.updated_at.asc(), Conversation.id.asc()).limit(excess)
    ).scalars().all()
    db.execute(
        update(Conversation).where(Conversation.id.in_(victims)).values(deleted_at=_now())
    )


def create_session(owner: str, *, user_id: str | None = None) -> tuple[str, dict]:
    """A new empty session owned by `owner`, soft-deleting the owner's oldest
    if over the cap. `user_id` (a logged-in caller) selects the user cap and
    is stored on the row; guests pass None."""
    with _open() as db:
        _enforce_cap(db, owner, _cap_for(user_id), room_for_new=True)
        now = _now()
        while True:
            session_id = secrets.token_urlsafe(32)
            if db.get(Conversation, session_id) is None:  # astronomically unlikely; never overwrite
                break
        conv = Conversation(
            id=session_id,
            owner_key=owner,
            user_id=user_id,
            mode="QA",
            created_at=now,
            updated_at=now,
        )
        db.add(conv)
        db.flush()
        snapshot = _snapshot(db, conv)
        db.commit()
        return session_id, snapshot


def get_owned_session(session_id: str, owner: str) -> dict | None:
    """A snapshot of the session (mode, persona, the last MAX_HISTORY turns,
    ...), or None if the id is unknown, deleted OR belongs to someone else."""
    with _open() as db:
        conv = _owned(db, session_id, owner)
        return None if conv is None else _snapshot(db, conv)


_UPDATABLE = {"mode", "persona", "title", "role_id", "level"}


def update_session(session_id: str, **fields) -> dict | None:
    """Set columns (mode, persona, title, role_id, level) on an existing
    session and touch updated_at. Never creates one: if it vanished
    mid-request (concurrent DELETE or cap eviction) this does nothing."""
    unknown = set(fields) - _UPDATABLE
    if unknown:
        raise TypeError(f"update_session() cannot set {sorted(unknown)}")
    with _open() as db:
        conv = _get_live(db, session_id)
        if conv is None:
            return None
        for key, value in fields.items():
            setattr(conv, key, value)
        # Every request path ends with an update_session() call, so touching
        # the timestamp here keeps updated_at accurate for all modes.
        conv.updated_at = _now()
        db.flush()
        snapshot = _snapshot(db, conv)
        db.commit()
        return snapshot


def _derive_title(content: str) -> str:
    """The first user turn, trimmed to a sidebar-friendly length."""
    text = " ".join(content.split())
    return text[:60] + "…" if len(text) > 60 else text


def append_history(
    session_id: str, role: str, content: str, sources: list | None = None
) -> dict | None:
    """Append one turn. `sources` (assistant turns only) is the citation list
    returned with the answer, kept so GET /conversations/{id}/messages can
    restore them. Citation models are stored as plain dicts (JSON); a missing
    key means no sources. Readers that build LLM messages (chitchat_service,
    roleplay_service) only take role/content. The first user turn also sets
    the conversation title."""
    with _open() as db:
        conv = _get_live(db, session_id)
        if conv is None:
            return None
        now = _now()
        message = Message(conversation_id=session_id, role=role, content=content, created_at=now)
        if sources:
            message.sources = [
                s.model_dump() if hasattr(s, "model_dump") else dict(s) for s in sources
            ]
        db.add(message)
        if role == "user" and conv.title is None and content.strip():
            conv.title = _derive_title(content)
        conv.updated_at = now
        db.flush()
        snapshot = _snapshot(db, conv)
        db.commit()
        return snapshot


def list_sessions(owner: str) -> list[dict]:
    """Newest-activity-first summary of `owner`'s live sessions only.
    message_count is the stored total, not the read window."""
    count = (
        select(func.count(Message.id))
        .where(Message.conversation_id == Conversation.id)
        .correlate(Conversation)
        .scalar_subquery()
    )
    with _open() as db:
        rows = db.execute(
            _live(select(Conversation, count), owner).order_by(
                Conversation.updated_at.desc(), Conversation.id.asc()
            )
        ).all()
        return [
            {
                "id": conv.id,
                "title": conv.title or "New chat",
                "message_count": n,
                "mode": conv.mode,
                "created_at": _aware(conv.created_at),
                "updated_at": _aware(conv.updated_at),
            }
            for conv, n in rows
        ]


def get_history(session_id: str, owner: str) -> list[dict] | None:
    """The last MAX_HISTORY [{role, content, sources?}] turns of an owned
    session, or None (unknown, deleted or not yours). Does not create anything."""
    with _open() as db:
        conv = _owned(db, session_id, owner)
        return None if conv is None else _history(db, session_id)


def delete_session(session_id: str, owner: str) -> bool:
    """Soft delete (deleted_at). False for unknown, already deleted or not yours."""
    with _open() as db:
        result = db.execute(
            update(Conversation)
            .where(
                Conversation.id == session_id,
                Conversation.owner_key == owner,
                Conversation.deleted_at.is_(None),
            )
            .values(deleted_at=_now())
        )
        db.commit()
        return result.rowcount == 1


def reassign_owner(
    old_owner: str, new_owner: str, *, user_id: str | None = None, db: Session | None = None
) -> int:
    """Move every live conversation of `old_owner` to `new_owner` (a guest who
    logs in or registers keeps their chats), setting user_id when given.
    Callers pass only the guest owner from the cookie on THAT request. Past
    the new owner's cap their least-recently-updated conversations are
    soft-deleted. With `db` the UPDATE joins that session's transaction and
    is NOT committed here (auth_routes commits it with the login). Returns
    how many moved."""
    if old_owner == new_owner:
        return 0

    def _run(session: Session) -> int:
        values: dict = {"owner_key": new_owner}
        if user_id is not None:
            values["user_id"] = user_id
        result = session.execute(
            _live(update(Conversation), old_owner).values(**values)
        )
        _enforce_cap(session, new_owner, _cap_for(user_id), room_for_new=False)
        return result.rowcount

    if db is not None:
        return _run(db)
    with _open() as session:
        moved = _run(session)
        session.commit()
        return moved
