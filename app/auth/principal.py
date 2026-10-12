"""Who is calling: a guest (anonymous owner cookie) or a logged-in user."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass(frozen=True)
class Principal:
    kind: str  # "guest" | "user"
    role: str  # "guest" | "user" | "admin"
    user_id: str | None
    # What session_store scopes chat sessions by. A guest's is the existing
    # owner-cookie hash; a user's is user_owner_key(user_id). The two can
    # never collide in practice (different preimages, sha256).
    owner_key: str

    @property
    def is_user(self) -> bool:
        return self.kind == "user"


def user_owner_key(user_id: str) -> str:
    return hashlib.sha256(f"user:{user_id}".encode()).hexdigest()


def guest(owner_key: str) -> Principal:
    return Principal(kind="guest", role="guest", user_id=None, owner_key=owner_key)


def user(user_id: str, role: str) -> Principal:
    return Principal(kind="user", role=role, user_id=user_id, owner_key=user_owner_key(user_id))
