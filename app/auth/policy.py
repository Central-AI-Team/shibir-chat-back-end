"""The access level of every route, deny by default.

Each route is listed in ROUTE_POLICY by (METHOD, path) AND carries the
dependency for its level (LEVEL_DEPENDENCY, in app/api/deps.py).
tests/test_auth_policy.py walks app.routes and fails if a route is missing
from this map or lacks its level's dependency, so a new endpoint cannot ship
without someone deciding who may call it.

  public    anyone; the route resolves identity itself if it needs it
  guest_ok  guests (owner cookie) and users; depends on get_principal
  user      logged-in users only (401 otherwise); depends on require_user
  admin     role admin only (401 guest, 403 user); depends on require_role("admin")
"""

from __future__ import annotations

LEVELS = ("public", "guest_ok", "user", "admin")

ROUTE_POLICY: dict[tuple[str, str], str] = {
    ("GET", "/health"): "public",
    ("POST", "/chat"): "guest_ok",
    ("POST", "/chat/stream"): "guest_ok",
    ("GET", "/conversations"): "guest_ok",
    ("GET", "/conversations/{session_id}/messages"): "guest_ok",
    ("DELETE", "/conversations/{session_id}"): "guest_ok",
    ("POST", "/auth/register"): "public",
    ("POST", "/auth/login"): "public",
    ("GET", "/auth/me"): "public",
    ("POST", "/auth/logout"): "user",
    ("POST", "/auth/logout-all"): "user",
    ("POST", "/auth/change-password"): "user",
}
