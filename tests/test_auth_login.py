"""Login, sessions, logout, change-password, throttling, lockout, registration
and the auth routes' Origin / content-type / caching rules."""

from __future__ import annotations

from datetime import timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.api import auth_routes
from app.auth import open_db, passwords, sessions
from app.core.config import settings
from app.db.models import AuthSession, User
from app.main import app
from app.services import rate_limiter
from tests.conftest import AUTH_TEST_PASSWORD as PASSWORD

COOKIE = sessions.COOKIE_NAME_PLAIN  # conftest turns cookie_secure off


@pytest.fixture(autouse=True)
def _setup(auth_origin, auth_clock):
    pass


@pytest.fixture
def mono(monkeypatch):
    """The in-process limiters' clock."""
    state = {"t": 1_000.0}
    monkeypatch.setattr(rate_limiter, "_monotonic", lambda: state["t"])
    return state


def _client(ip: str = "10.0.0.1", origin: str | None = "http://localhost:5173", **kwargs) -> TestClient:
    headers = {"Origin": origin} if origin else {}
    return TestClient(app, client=(ip, 50000), headers=headers, **kwargs)


def _login(client: TestClient, email="alice@example.com", password=PASSWORD):
    return client.post("/auth/login", json={"email": email, "password": password})


def _me(client: TestClient) -> dict:
    r = client.get("/auth/me")
    assert r.status_code == 200
    return r.json()


def _with_token(token: str, **kwargs) -> TestClient:
    client = _client(**kwargs)
    # Same domain the jar files server-set cookies under, so a later
    # Set-Cookie replaces this one instead of sitting beside it.
    client.cookies.set(COOKIE, token, domain="testserver.local")
    return client


# ---------------------------------------------------------------------------
# login + cookie
# ---------------------------------------------------------------------------


def test_login_success_sets_a_hardened_cookie_and_me_reports_the_user(make_user):
    user_id = make_user()
    client = _client()
    r = _login(client, email="  Alice@Example.COM ")  # normalised
    assert r.status_code == 200
    assert r.json() == {"id": user_id, "email": "alice@example.com", "role": "user"}
    cookie = r.headers["set-cookie"]
    assert cookie.startswith(f"{COOKIE}=")
    lowered = cookie.lower()
    assert "httponly" in lowered and "samesite=lax" in lowered and "path=/" in lowered
    assert "domain" not in lowered
    assert f"max-age={settings.auth_session_absolute_days * 86400}" in lowered
    assert r.headers["cache-control"] == "no-store"
    assert _me(client) == {"authenticated": True, "id": user_id, "email": "alice@example.com", "role": "user"}


def test_secure_cookie_uses_the_host_prefix(make_user, monkeypatch):
    monkeypatch.setattr(settings, "cookie_secure", True)
    make_user()
    client = _client(base_url="https://testserver")
    r = _login(client)
    cookie = r.headers["set-cookie"]
    assert cookie.startswith("__Host-shibir_session=")
    assert "secure" in cookie.lower() and "path=/" in cookie.lower() and "domain" not in cookie.lower()
    assert _me(client)["authenticated"] is True


def test_only_the_token_hash_is_stored(make_user):
    make_user()
    client = _client()
    _login(client)
    token = client.cookies[COOKIE]
    with open_db() as db:
        row = db.execute(select(AuthSession)).scalar_one()
    assert row.token_hash == sessions.hash_token(token) != token
    assert row.ip_prefix == "10.0.0.0/24"


def test_every_failure_kind_gets_the_identical_401(make_user, auth_clock):
    make_user()
    make_user(email="inactive@example.com", is_active=False)
    locked_id = make_user(email="locked@example.com")
    with open_db() as db:
        db.get(User, locked_id).locked_until = auth_clock["now"] + timedelta(minutes=5)
        db.commit()
    client = _client()
    responses = [
        _login(client, email="nobody@example.com"),
        _login(client, password="wrong password!!"),
        _login(client, email="inactive@example.com"),
        _login(client, email="locked@example.com"),  # correct password, but locked
        _login(client, email="not-an-email"),
    ]
    for r in responses:
        assert r.status_code == 401
        assert r.json() == {"detail": "invalid credentials"}
        assert r.headers["cache-control"] == "no-store"
        assert "set-cookie" not in r.headers


def test_unknown_email_still_spends_a_password_verify(make_user, monkeypatch):
    make_user(email="inactive@example.com", is_active=False)
    dummy, real = [], []
    original = passwords.verify_password
    monkeypatch.setattr(passwords, "dummy_verify", lambda p: dummy.append(1) or original(passwords._dummy_hash(), p))
    monkeypatch.setattr(passwords, "verify_password", lambda h, p: real.append(1) or original(h, p))
    client = _client()
    _login(client, email="nobody@example.com")
    assert dummy == [1]
    _login(client, email="inactive@example.com")
    assert dummy == [1] and len(real) == 1  # the inactive account's hash is verified too


def test_login_requires_json_and_rejects_extra_fields(make_user):
    make_user()
    client = _client()
    r = client.post("/auth/login", data={"email": "alice@example.com", "password": PASSWORD})
    assert r.status_code == 415
    r = client.post("/auth/login", json={"email": "alice@example.com", "password": PASSWORD, "role": "admin"})
    assert r.status_code == 422
    assert r.headers["cache-control"] == "no-store"
    r = client.post("/auth/login", json={"email": "alice@example.com", "password": "x" * 129})
    assert r.status_code == 422
    r = client.post("/auth/login", json={"email": "a" * 250 + "@x.io", "password": PASSWORD})
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# session fixation
# ---------------------------------------------------------------------------


def test_attacker_supplied_cookie_is_never_accepted_or_reused(make_user):
    make_user()
    planted = "A" * 43  # well-formed, but never issued
    client = _with_token(planted)
    assert _me(client)["authenticated"] is False
    r = _login(client)
    assert r.status_code == 200
    issued = client.cookies[COOKIE]
    assert issued != planted
    assert _me(_with_token(planted))["authenticated"] is False


def test_login_replaces_and_revokes_a_real_session_planted_in_the_browser(make_user):
    make_user()
    make_user(email="mallory@example.com")
    attacker = _client()
    _login(attacker, email="mallory@example.com")
    planted = attacker.cookies[COOKIE]

    victim = _with_token(planted, ip="10.0.0.2")
    _login(victim)
    assert victim.cookies[COOKIE] != planted
    assert _me(victim)["email"] == "alice@example.com"
    # the planted (attacker's) session no longer works for anyone
    assert _me(_with_token(planted))["authenticated"] is False


# ---------------------------------------------------------------------------
# expiry, revocation, cap
# ---------------------------------------------------------------------------


def test_idle_expiry(make_user, auth_clock):
    make_user()
    client = _client()
    _login(client)
    auth_clock["now"] += timedelta(days=settings.auth_session_idle_days, seconds=1)
    assert _me(client)["authenticated"] is False


def test_activity_slides_idle_expiry_but_never_the_absolute_one(make_user, auth_clock):
    make_user()
    client = _client()
    _login(client)
    for _ in range(4):  # 24 days, active every 6 days
        auth_clock["now"] += timedelta(days=6)
        assert _me(client)["authenticated"] is True
    auth_clock["now"] += timedelta(days=5, hours=23)  # 29 d 23 h
    assert _me(client)["authenticated"] is True
    auth_clock["now"] += timedelta(hours=2)  # past 30 days
    assert _me(client)["authenticated"] is False


def test_sliding_writes_at_most_every_five_minutes(make_user, auth_clock):
    make_user()
    client = _client()
    _login(client)

    def last_seen():
        with open_db() as db:
            return db.execute(select(AuthSession.last_seen_at)).scalar_one()

    first = last_seen()
    auth_clock["now"] += timedelta(minutes=4)
    _me(client)
    assert last_seen() == first
    auth_clock["now"] += timedelta(minutes=2)
    _me(client)
    assert last_seen() != first


def test_inactive_user_sessions_are_rejected(make_user):
    user_id = make_user()
    client = _client()
    _login(client)
    with open_db() as db:
        db.get(User, user_id).is_active = False
        db.commit()
    assert _me(client)["authenticated"] is False


def test_logout_revokes_and_clears_the_cookie(make_user):
    make_user()
    client = _client()
    _login(client)
    token = client.cookies[COOKIE]
    r = client.post("/auth/logout")
    assert r.status_code == 204
    assert r.headers["cache-control"] == "no-store"
    assert COOKIE not in client.cookies
    assert _me(_with_token(token))["authenticated"] is False
    assert _with_token(token).post("/auth/logout").status_code == 401


def test_logout_all_revokes_every_session(make_user):
    make_user()
    a, b = _client(), _client(ip="10.0.0.2")
    _login(a)
    _login(b)
    assert a.post("/auth/logout-all").status_code == 204
    assert _me(a)["authenticated"] is False
    assert _me(b)["authenticated"] is False


def test_change_password_revokes_others_and_reissues_this_one(make_user):
    make_user()
    a, b = _client(), _client(ip="10.0.0.2")
    _login(a)
    _login(b)
    old_a = a.cookies[COOKIE]
    r = a.post("/auth/change-password", json={"current_password": PASSWORD, "new_password": "a brand new secret"})
    assert r.status_code == 200
    assert a.cookies[COOKIE] != old_a
    assert _me(a)["authenticated"] is True
    assert _me(b)["authenticated"] is False
    assert _me(_with_token(old_a))["authenticated"] is False
    fresh = _client(ip="10.0.0.3")
    assert _login(fresh).status_code == 401
    assert _login(fresh, password="a brand new secret").status_code == 200


def test_change_password_checks_current_and_policy(make_user):
    make_user()
    client = _client()
    _login(client)
    r = client.post("/auth/change-password", json={"current_password": "wrong one!!", "new_password": "a brand new secret"})
    assert r.status_code == 400
    r = client.post("/auth/change-password", json={"current_password": PASSWORD, "new_password": "short"})
    assert r.status_code == 400
    assert _me(client)["authenticated"] is True


def test_user_routes_need_a_login():
    client = _client()
    assert client.post("/auth/logout").status_code == 401
    assert client.post("/auth/logout-all").status_code == 401
    r = client.post("/auth/change-password", json={"current_password": PASSWORD, "new_password": "a brand new secret"})
    assert r.status_code == 401
    assert r.json() == {"detail": "authentication required"}


def test_at_most_ten_sessions_per_user_oldest_revoked(make_user, monkeypatch, auth_clock):
    monkeypatch.setattr(auth_routes, "LOGIN_PER_EMAIL_PER_15_MIN", 100)
    make_user()
    clients = []
    for i in range(sessions.MAX_SESSIONS_PER_USER + 1):
        auth_clock["now"] += timedelta(seconds=1)
        client = _client(ip=f"10.0.1.{i}")
        assert _login(client).status_code == 200
        clients.append(client)
    assert _me(clients[0])["authenticated"] is False
    assert all(_me(c)["authenticated"] for c in clients[1:])


def test_expired_rows_are_purged_on_login(make_user, auth_clock):
    make_user()
    _login(_client())
    auth_clock["now"] += timedelta(days=31)
    _login(_client(ip="10.0.0.2"))
    with open_db() as db:
        assert len(db.execute(select(AuthSession)).scalars().all()) == 1


# ---------------------------------------------------------------------------
# throttling and lockout
# ---------------------------------------------------------------------------


def test_per_email_login_throttle(make_user, mono):
    make_user()
    for i in range(5):
        assert _login(_client(ip=f"10.0.2.{i}"), password="wrong password!!").status_code == 401
    r = _login(_client(ip="10.0.2.99"))  # even the right password, from a new IP
    assert r.status_code == 429
    assert r.headers["Retry-After"] == "900"
    assert r.headers["cache-control"] == "no-store"
    # the limit is per normalised email
    assert _login(_client(ip="10.0.2.98"), email=" ALICE@example.com").status_code == 429
    mono["t"] += 901
    assert _login(_client(ip="10.0.2.97")).status_code == 200


def test_per_ip_login_throttle(make_user, mono):
    make_user()
    client = _client()
    for i in range(20):
        assert _login(client, email=f"user{i}@example.com").status_code == 401
    r = _login(client)
    assert r.status_code == 429
    assert r.headers["Retry-After"] == "3600"
    assert _login(_client(ip="10.0.0.2")).status_code == 200


def test_lockout_backoff_and_reset(make_user, monkeypatch, auth_clock):
    monkeypatch.setattr(auth_routes, "LOGIN_PER_EMAIL_PER_15_MIN", 1000)
    monkeypatch.setattr(auth_routes, "LOGIN_PER_IP_PER_HOUR", 1000)
    user_id = make_user()
    client = _client()

    def state():
        with open_db() as db:
            u = db.get(User, user_id)
            return u.failed_login_count, u.locked_until

    for _ in range(9):
        _login(client, password="wrong password!!")
    assert state() == (9, None)
    _login(client, password="wrong password!!")  # 10th: locked 30 s
    count, until = state()
    assert count == 10
    assert _login(client).status_code == 401  # right password, but locked
    assert state()[0] == 10  # guesses while locked are not counted
    auth_clock["now"] += timedelta(seconds=31)
    _login(client, password="wrong password!!")  # 11th: locked 60 s
    assert state()[0] == 11
    auth_clock["now"] += timedelta(seconds=59)
    assert _login(client).status_code == 401
    auth_clock["now"] += timedelta(seconds=2)
    assert _login(client).status_code == 200
    assert state() == (0, None)


def test_lock_duration_is_capped():
    assert auth_routes._lock_seconds(10) == 30
    assert auth_routes._lock_seconds(11) == 60
    assert auth_routes._lock_seconds(15) == 900
    assert auth_routes._lock_seconds(10_000) == 900


# ---------------------------------------------------------------------------
# registration
# ---------------------------------------------------------------------------


def test_register_creates_a_user_and_logs_in():
    client = _client()
    r = client.post("/auth/register", json={"email": " New@Example.com ", "password": "a long enough secret"})
    assert r.status_code == 201
    body = r.json()
    assert body["email"] == "new@example.com" and body["role"] == "user"
    assert r.headers["cache-control"] == "no-store"
    assert _me(client)["authenticated"] is True


def test_register_ignores_a_client_supplied_role():
    r = _client().post("/auth/register", json={"email": "x@example.com", "password": "a long enough secret", "role": "admin"})
    assert r.status_code == 422


def test_register_duplicate_is_409(make_user):
    make_user()
    r = _client().post("/auth/register", json={"email": "ALICE@example.com", "password": "a long enough secret"})
    assert r.status_code == 409


@pytest.mark.parametrize(
    "email, password",
    [
        ("no-at-sign", "a long enough secret"),
        ("two@@example.com", "a long enough secret"),
        ("a@nodot", "a long enough secret"),
        ("alice.smith@example.com", "alice.smith"),
        ("bob@example.com", "short"),
        ("bob@example.com", " " * 12),
    ],
)
def test_register_validates(email, password):
    assert _client().post("/auth/register", json={"email": email, "password": password}).status_code == 400


def test_register_disabled_is_404(monkeypatch):
    monkeypatch.setattr(settings, "auth_registration_enabled", False)
    r = _client().post("/auth/register", json={"email": "x@example.com", "password": "a long enough secret"})
    assert r.status_code == 404


def test_register_per_ip_limit(mono):
    client = _client()
    for i in range(3):
        assert client.post("/auth/register", json={"email": f"u{i}@example.com", "password": "a long enough secret"}).status_code == 201
    r = _client().post("/auth/register", json={"email": "u9@example.com", "password": "a long enough secret"})
    assert r.status_code == 429
    assert r.headers["Retry-After"] == "3600"


def test_register_global_daily_cap(monkeypatch):
    monkeypatch.setattr(settings, "auth_registrations_per_day", 2)
    for i in range(2):
        r = _client(ip=f"10.0.3.{i}").post("/auth/register", json={"email": f"g{i}@example.com", "password": "a long enough secret"})
        assert r.status_code == 201
    r = _client(ip="10.0.3.9").post("/auth/register", json={"email": "g9@example.com", "password": "a long enough secret"})
    assert r.status_code == 429
    assert int(r.headers["Retry-After"]) >= 1


# ---------------------------------------------------------------------------
# Origin (CSRF)
# ---------------------------------------------------------------------------


def test_session_cookie_without_origin_is_rejected_on_post(make_user):
    make_user()
    client = _client()
    _login(client)
    token = client.cookies[COOKIE]
    bare = _with_token(token, origin=None)
    assert bare.post("/auth/logout").status_code == 403
    assert bare.post("/auth/login", json={"email": "alice@example.com", "password": PASSWORD}).status_code == 403
    assert _me(_with_token(token))["authenticated"] is True  # nothing was revoked


def test_disallowed_origin_is_rejected(make_user):
    make_user()
    evil = _client(origin="https://evil.example")
    assert _login(evil).status_code == 403
    assert evil.post("/auth/register", json={"email": "x@example.com", "password": "a long enough secret"}).status_code == 403


def test_no_origin_and_no_session_cookie_may_log_in(make_user):
    make_user()
    assert _login(_client(origin=None)).status_code == 200
