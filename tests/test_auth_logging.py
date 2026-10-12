"""No password, token, hash, full email or full IP may reach any log line,
while the audit trail still records each event."""

from __future__ import annotations

import json
import logging
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.auth import audit, open_db, sessions
from app.core.config import settings
from app.db.models import AuthSession, User
from app.main import app

IP = "203.0.113.77"
EMAIL = "secret.person@example.com"
PASSWORD = "first password value"
NEW_PASSWORD = "second password value"


@pytest.fixture(autouse=True)
def _setup(auth_origin, auth_clock, monkeypatch):
    monkeypatch.setattr(settings, "guest_message_limit", 1)


def test_nothing_sensitive_is_logged(caplog, capsys):
    caplog.set_level(logging.DEBUG)
    client = TestClient(app, client=(IP, 50000), headers={"Origin": "http://localhost:5173"})
    tokens = []

    def remember():
        tokens.append(client.cookies.get(sessions.COOKIE_NAME_PLAIN))

    with patch("app.api.router.classify_intent", return_value="CHITCHAT"), \
            patch("app.api.router.respond", return_value="উত্তর"):
        client.post("/chat", json={"message": "প্রশ্ন"})
        client.post("/chat", json={"message": "প্রশ্ন"})  # guest_limit_reached
    owner_token = client.cookies.get("shibir_owner")
    client.post("/auth/register", json={"email": EMAIL, "password": PASSWORD})
    remember()
    client.post("/auth/login", json={"email": EMAIL, "password": "wrong password value"})
    client.post("/auth/login", json={"email": "missing@example.com", "password": PASSWORD})
    client.post("/auth/login", json={"email": EMAIL, "password": PASSWORD})
    remember()
    client.post("/auth/change-password", json={"current_password": PASSWORD, "new_password": NEW_PASSWORD})
    remember()
    stale = TestClient(app, client=(IP, 50000))
    stale.cookies.set(sessions.COOKIE_NAME_PLAIN, tokens[0], domain="testserver.local")
    stale.get("/auth/me")  # session_rejected
    client.post("/auth/logout-all")

    with open_db() as db:
        password_hash = db.execute(select(User.password_hash)).scalar_one()
        token_hashes = db.execute(select(AuthSession.token_hash)).scalars().all()

    logged = caplog.text + capsys.readouterr().err
    forbidden = [PASSWORD, NEW_PASSWORD, "wrong password value", password_hash, IP, EMAIL,
                 "missing@example.com", owner_token, *filter(None, tokens), *token_hashes]
    for secret in forbidden:
        assert secret not in logged, f"leaked: {secret[:12]}..."

    records = [json.loads(r.getMessage()) for r in caplog.records if r.name == "app.auth.audit"]
    events = [r["event"] for r in records]
    for expected in ("guest_limit_reached", "register", "login_failure", "login_success",
                     "password_changed", "session_rejected", "logout_all"):
        assert expected in events
    assert {r["ip_prefix"] for r in records} == {"203.0.113.0/24"}
    failure = next(r for r in records if r["event"] == "login_failure")
    assert failure["email_ref"] == audit.email_ref(EMAIL) and len(failure["email_ref"]) == 12


def test_ip_prefixes():
    assert audit.ip_prefix("198.51.100.200") == "198.51.100.0/24"
    assert audit.ip_prefix("2001:db8:abcd:1234::1") == "2001:db8:abcd::/48"
    assert audit.ip_prefix("testclient") == "unknown"
    assert audit.ip_prefix(None) == "unknown"


def test_unknown_event_names_are_refused():
    with pytest.raises(ValueError):
        audit.event("password_was", ip=IP)
