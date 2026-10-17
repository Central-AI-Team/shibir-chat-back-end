"""scripts/create_user.py: admin bootstrap."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.auth import open_db
from app.db.models import User
from app.main import app
from scripts import create_user


def test_creates_an_admin_from_the_env_password(monkeypatch, capsys, auth_origin):
    monkeypatch.setenv("CREATE_USER_PASSWORD", "a long enough secret")
    assert create_user.main(["--email", " Root@Example.com ", "--role", "admin"]) == 0
    with open_db() as db:
        user = db.execute(select(User)).scalar_one()
    assert (user.email, user.role, user.is_active) == ("root@example.com", "admin", True)
    assert "a long enough secret" not in capsys.readouterr().out
    client = TestClient(app, headers={"Origin": auth_origin})
    r = client.post("/auth/login", json={"email": "root@example.com", "password": "a long enough secret"})
    assert r.status_code == 200 and r.json()["role"] == "admin"


def test_prompts_when_no_env_password(monkeypatch):
    monkeypatch.delenv("CREATE_USER_PASSWORD", raising=False)
    answers = iter(["a long enough secret", "a long enough secret"])
    monkeypatch.setattr(create_user.getpass, "getpass", lambda prompt="": next(answers))
    assert create_user.main(["--email", "u@example.com"]) == 0


def test_mismatched_prompt_fails(monkeypatch):
    monkeypatch.delenv("CREATE_USER_PASSWORD", raising=False)
    answers = iter(["a long enough secret", "something else entirely"])
    monkeypatch.setattr(create_user.getpass, "getpass", lambda prompt="": next(answers))
    assert create_user.main(["--email", "u@example.com"]) == 1


def test_refuses_to_overwrite_and_applies_the_policy(monkeypatch):
    monkeypatch.setenv("CREATE_USER_PASSWORD", "a long enough secret")
    assert create_user.main(["--email", "u@example.com"]) == 0
    assert create_user.main(["--email", "U@example.com", "--role", "admin"]) == 1
    monkeypatch.setenv("CREATE_USER_PASSWORD", "short")
    assert create_user.main(["--email", "v@example.com"]) == 1
    with open_db() as db:
        assert db.execute(select(User.role)).scalars().all() == ["user"]


def test_password_is_not_an_argument():
    with pytest.raises(SystemExit):
        create_user.main(["--email", "u@example.com", "--password", "a long enough secret"])
