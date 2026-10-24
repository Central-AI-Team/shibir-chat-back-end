"""Account routes under /auth. Contract for the front-end: docs/auth.md.

  POST /auth/register         open unless AUTH_REGISTRATION_ENABLED=false (then 404)
  POST /auth/login            generic 401 for every failure kind
  GET  /auth/me               always 200: the user, or the guest's remaining quota
  POST /auth/logout           revoke this session
  POST /auth/logout-all       revoke every session of this user
  POST /auth/change-password  revoke every session, issue a new one here

Every response carries Cache-Control: no-store (NoStoreRoute). State-changing
routes run check_origin AND check_session_origin. Bodies are JSON only and
strict (extra fields are a 422). Passwords, tokens and hashes are never logged;
see app/auth/audit.py for what is.
"""

from __future__ import annotations

import uuid
from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.exception_handlers import http_exception_handler, request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from pydantic import BaseModel, ConfigDict
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from app.api.deps import (
    AUTH_SESSION_ATTR,
    SERVICE_UNAVAILABLE_DETAIL,
    _client_ip,
    check_origin,
    check_session_origin,
    current_user,
    peek_owner,
    public_route,
    require_json,
    require_user,
    too_many_requests,
)
from app.auth import aware, audit, guest_quota, now, open_db, passwords
from app.auth import sessions as auth_sessions
from app.auth.principal import Principal, user_owner_key
from app.core.config import settings
from app.db.models import User
from app.services import rate_limiter
from app.services.session_store import reassign_owner

LOGIN_PER_EMAIL_PER_15_MIN = 5
LOGIN_PER_IP_PER_HOUR = 20
REGISTER_PER_IP_PER_HOUR = 3
LOCK_AFTER_FAILURES = 10
LOCK_BASE_SECONDS = 30
LOCK_MAX_SECONDS = 15 * 60  # never permanent: a permanent lock is a lock-out DoS

INVALID_CREDENTIALS = "invalid credentials"


class NoStoreRoute(APIRoute):
    """Adds Cache-Control: no-store to every response of these routes,
    including HTTPException and validation-error responses."""

    def get_route_handler(self):
        handler = super().get_route_handler()

        async def no_store_handler(request: Request) -> Response:
            try:
                response = await handler(request)
            except HTTPException as exc:
                response = await http_exception_handler(request, exc)
            except RequestValidationError as exc:
                response = await request_validation_exception_handler(request, exc)
            response.headers["Cache-Control"] = "no-store"
            return response

        return no_store_handler


router = APIRouter(prefix="/auth", route_class=NoStoreRoute)

_STATE_CHANGING = [Depends(check_origin), Depends(check_session_origin)]
_JSON_BODY = [Depends(require_json)]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# No Field(max_length=...) here: lengths are checked in the routes
# (email_shape_ok, passwords.policy_error) so that an over-long email or
# password is a 400 with a readable detail, like every other policy error,
# instead of a 422 that only the length violations would produce. The proxy's
# body cap (64 KB) bounds what can arrive at all.
class Credentials(_Strict):
    email: str
    password: str


class PasswordChange(_Strict):
    current_password: str
    new_password: str


def normalise_email(email: str) -> str:
    return email.strip().lower()


def email_shape_ok(email: str) -> bool:
    """Deliberately simple (no email-validator dependency): one '@', a
    non-empty local part of at most 64, a dotted domain, no whitespace."""
    if not 3 <= len(email) <= 254 or email.count("@") != 1 or any(c.isspace() for c in email):
        return False
    local, domain = email.split("@")
    return 0 < len(local) <= 64 and "." in domain.strip(".") and not domain.startswith(".") and not domain.endswith(".")


def _user_body(user: User) -> dict:
    return {"id": user.id, "email": user.email, "role": user.role}


def _unavailable(_: SQLAlchemyError) -> HTTPException:
    return HTTPException(status_code=503, detail=SERVICE_UNAVAILABLE_DETAIL)


def _start_session(db, user: User, request: Request) -> tuple[str, int]:
    """Revoke any session the browser brought along (never reuse it), create a
    fresh one, move the guest's chats to the account and commit it all
    together. Returns (token, moved guest chat sessions)."""
    incoming = auth_sessions.request_token(request)
    if incoming is not None:
        auth_sessions.revoke_token(db, incoming)
    token = auth_sessions.create(db, user, request)
    # Only the guest identified by the owner cookie on THIS request is migrated:
    # UPDATE conversations SET owner_key = <user key>, user_id = <id> WHERE
    # owner_key = <guest key>, in the login transaction, so a failed commit
    # moves nothing.
    guest_owner = peek_owner(request)
    moved = (
        reassign_owner(guest_owner, user_owner_key(user.id), user_id=user.id, db=db)
        if guest_owner
        else 0
    )
    db.commit()
    return token, moved


def _lock_seconds(failures: int) -> int:
    exponent = min(failures - LOCK_AFTER_FAILURES, 10)
    return min(LOCK_MAX_SECONDS, LOCK_BASE_SECONDS * 2**exponent)


# ---------------------------------------------------------------------------
# register / login
# ---------------------------------------------------------------------------


@router.post("/register", status_code=201, dependencies=[Depends(public_route), *_STATE_CHANGING, *_JSON_BODY])
def register(body: Credentials, request: Request) -> JSONResponse:
    if not settings.auth_registration_enabled:
        raise HTTPException(status_code=404, detail="Not Found")
    ip = _client_ip(request)
    email = normalise_email(body.email)
    if not email_shape_ok(email):
        raise HTTPException(status_code=400, detail="invalid email address")
    problem = passwords.policy_error(body.password, email)
    if problem is not None:
        raise HTTPException(status_code=400, detail=problem)
    # Both limiters count only well-formed attempts: a person who trips the
    # password policy a few times must not burn the 3-per-hour allowance.
    retry = rate_limiter.register_per_ip.hit(ip, REGISTER_PER_IP_PER_HOUR)
    if retry is not None:
        raise too_many_requests(rate_limiter.AUTH_RATE_LIMIT_DETAIL, retry)
    retry = rate_limiter.registrations_daily.hit(settings.auth_registrations_per_day)
    if retry is not None:
        raise too_many_requests(rate_limiter.AUTH_RATE_LIMIT_DETAIL, retry)

    at = now()
    user = User(
        id=str(uuid.uuid4()),
        email=email,
        password_hash=passwords.hash_password(body.password),
        role="user",
        is_active=True,
        failed_login_count=0,
        created_at=at,
        password_changed_at=at,
        last_login_at=at,
    )
    try:
        with open_db() as db:
            db.add(user)
            try:
                db.flush()
            except IntegrityError:
                db.rollback()
                raise HTTPException(status_code=409, detail="email already registered") from None
            token, moved = _start_session(db, user, request)
            body_user = _user_body(user)
    except SQLAlchemyError as e:
        raise _unavailable(e) from e
    audit.event("register", ip=ip, user_id=body_user["id"], migrated_chats=moved)
    response = JSONResponse(status_code=201, content=body_user)
    auth_sessions.set_cookie(response, token)
    return response


@router.post("/login", dependencies=[Depends(public_route), *_STATE_CHANGING, *_JSON_BODY])
def login(body: Credentials, request: Request) -> JSONResponse:
    ip = _client_ip(request)
    email = normalise_email(body.email)
    email_key = audit.email_ref(email)
    # The per-email limiter counts failures only (recorded in fail()), so a
    # person who logs in from several devices never throttles themself; the
    # per-IP limiter counts every attempt.
    retry = rate_limiter.login_per_ip.hit(ip, LOGIN_PER_IP_PER_HOUR)
    if retry is None:
        retry = rate_limiter.login_per_email.blocked(email_key, LOGIN_PER_EMAIL_PER_15_MIN)
    if retry is not None:
        audit.event("login_failure", ip=ip, email=email, reason="throttled")
        raise too_many_requests(rate_limiter.AUTH_RATE_LIMIT_DETAIL, retry)

    def fail(reason: str, user_id: str | None = None) -> HTTPException:
        rate_limiter.login_per_email.hit(email_key, LOGIN_PER_EMAIL_PER_15_MIN)
        audit.event("login_failure", ip=ip, email=email, user_id=user_id, reason=reason)
        return HTTPException(status_code=401, detail=INVALID_CREDENTIALS)

    if len(body.password) > passwords.MAX_LENGTH:
        # No stored password is longer (policy), so this cannot be right; it
        # is a failure like any other, without spending an argon2 verify.
        raise fail("password_too_long")

    try:
        with open_db() as db:
            user = db.execute(select(User).where(User.email == email)).scalar_one_or_none()
            if user is None:
                passwords.dummy_verify(body.password)  # same cost as a real account
                raise fail("unknown_email")
            # Always verify, even for a locked or inactive account, so every
            # failure kind takes the same time.
            ok = passwords.verify_password(user.password_hash, body.password)
            at = now()
            if not user.is_active:
                raise fail("inactive", user.id)
            locked_until = aware(user.locked_until)
            if locked_until is not None and locked_until > at:
                raise fail("locked", user.id)  # not counted: a lock cannot be extended by guessing
            if not ok:
                user.failed_login_count = (user.failed_login_count or 0) + 1
                if user.failed_login_count >= LOCK_AFTER_FAILURES:
                    seconds = _lock_seconds(user.failed_login_count)
                    user.locked_until = at + timedelta(seconds=seconds)
                    audit.event("account_locked", ip=ip, user_id=user.id, seconds=seconds)
                db.commit()
                raise fail("wrong_password", user.id)
            user.failed_login_count = 0
            user.locked_until = None
            user.last_login_at = at
            if passwords.needs_rehash(user.password_hash):
                user.password_hash = passwords.hash_password(body.password)
            token, moved = _start_session(db, user, request)
            body_user = _user_body(user)
    except SQLAlchemyError as e:
        raise _unavailable(e) from e
    audit.event("login_success", ip=ip, user_id=body_user["id"], migrated_chats=moved)
    response = JSONResponse(content=body_user)
    auth_sessions.set_cookie(response, token)
    return response


# ---------------------------------------------------------------------------
# me / logout / change password
# ---------------------------------------------------------------------------


@router.get("/me", dependencies=[Depends(public_route)])
def me(request: Request, user: Principal | None = Depends(current_user)) -> dict:
    """Always 200. Never mints a guest cookie."""
    try:
        if user is not None:
            with open_db() as db:
                row = db.get(User, user.user_id)
                return {"authenticated": True, **_user_body(row)}
        left = guest_quota.remaining(peek_owner(request), _client_ip(request))
    except SQLAlchemyError as e:
        raise _unavailable(e) from e
    return {"authenticated": False, "guest_limit": settings.guest_message_limit, "guest_remaining": left}


@router.post("/logout", dependencies=_STATE_CHANGING)
def logout(request: Request, user: Principal = Depends(require_user)) -> Response:
    try:
        with open_db() as db:
            auth_sessions.revoke(db, getattr(request.state, AUTH_SESSION_ATTR))
            db.commit()
    except SQLAlchemyError as e:
        raise _unavailable(e) from e
    audit.event("logout", ip=_client_ip(request), user_id=user.user_id)
    response = Response(status_code=204)
    auth_sessions.clear_cookie(response)
    return response


@router.post("/logout-all", dependencies=_STATE_CHANGING)
def logout_all(request: Request, user: Principal = Depends(require_user)) -> Response:
    try:
        with open_db() as db:
            auth_sessions.revoke_all(db, user.user_id)
            db.commit()
    except SQLAlchemyError as e:
        raise _unavailable(e) from e
    audit.event("logout_all", ip=_client_ip(request), user_id=user.user_id)
    response = Response(status_code=204)
    auth_sessions.clear_cookie(response)
    return response


@router.post("/change-password", dependencies=[*_STATE_CHANGING, *_JSON_BODY])
def change_password(body: PasswordChange, request: Request, user: Principal = Depends(require_user)) -> JSONResponse:
    ip = _client_ip(request)
    # A stolen session must not be able to brute-force the current password.
    email_key = f"user:{user.user_id}"
    retry = rate_limiter.login_per_ip.hit(ip, LOGIN_PER_IP_PER_HOUR)
    if retry is None:
        retry = rate_limiter.login_per_email.blocked(email_key, LOGIN_PER_EMAIL_PER_15_MIN)
    if retry is not None:
        raise too_many_requests(rate_limiter.AUTH_RATE_LIMIT_DETAIL, retry)
    try:
        with open_db() as db:
            row = db.get(User, user.user_id)
            if not passwords.verify_password(row.password_hash, body.current_password):
                rate_limiter.login_per_email.hit(email_key, LOGIN_PER_EMAIL_PER_15_MIN)
                audit.event("login_failure", ip=ip, user_id=row.id, reason="change_password_wrong_current")
                raise HTTPException(status_code=400, detail=INVALID_CREDENTIALS)
            problem = passwords.policy_error(body.new_password, row.email)
            if problem is not None:
                raise HTTPException(status_code=400, detail=problem)
            row.password_hash = passwords.hash_password(body.new_password)
            row.password_changed_at = now()
            auth_sessions.revoke_all(db, row.id)
            token = auth_sessions.create(db, row, request)
            db.commit()
            body_user = _user_body(row)
    except SQLAlchemyError as e:
        raise _unavailable(e) from e
    audit.event("password_changed", ip=ip, user_id=body_user["id"])
    response = JSONResponse(content=body_user)
    auth_sessions.set_cookie(response, token)
    return response
