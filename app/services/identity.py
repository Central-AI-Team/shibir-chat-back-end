"""Opaque guest credentials. A supplied user_id is never an ownership credential."""
import hashlib
import secrets
import uuid

from fastapi import Header, HTTPException
from sqlalchemy import select

from app.db.chat_models import ChatIdentity
from app.db.session import SessionLocal


def create_identity() -> dict:
    token = secrets.token_urlsafe(32)
    owner_id = str(uuid.uuid4())
    with SessionLocal.begin() as db:
        db.add(ChatIdentity(id=owner_id, token_hash=hashlib.sha256(token.encode()).hexdigest()))
    return {'user_id': owner_id, 'token': token}


def require_identity(x_chat_identity: str | None = Header(default=None)) -> str:
    if not x_chat_identity or len(x_chat_identity) > 128:
        raise HTTPException(401, 'A private chat identity is required. Create one with POST /identity.')
    digest = hashlib.sha256(x_chat_identity.encode()).hexdigest()
    with SessionLocal() as db:
        owner = db.scalar(select(ChatIdentity.id).where(ChatIdentity.token_hash == digest))
    if not owner:
        raise HTTPException(401, 'Invalid chat identity.')
    return owner
