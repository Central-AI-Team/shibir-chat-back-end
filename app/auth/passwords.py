"""argon2id password hashing (argon2-cffi, library-default parameters) and the
password policy. Never log a password or a hash.

Policy (NIST SP 800-63B style): 10-128 characters, not whitespace-only, not the
email's local part. No composition rules. The 128 cap bounds hashing cost.
"""

from __future__ import annotations

import secrets
from functools import lru_cache

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError

MIN_LENGTH = 10
MAX_LENGTH = 128

# Library defaults (RFC 9106 low-memory profile: argon2id, t=3, m=64 MiB, p=4).
# Do not lower them; tests swap in a cheap hasher only for speed.
_hasher = PasswordHasher()


def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify_password(password_hash: str, password: str) -> bool:
    """Constant-time (argon2 compares in constant time). False for a wrong
    password or an unparseable hash; never raises."""
    try:
        return _hasher.verify(password_hash, password)
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return False


def needs_rehash(password_hash: str) -> bool:
    try:
        return _hasher.check_needs_rehash(password_hash)
    except InvalidHashError:
        return True


@lru_cache(maxsize=1)
def _dummy_hash() -> str:
    # A hash of a random password nobody knows, made with the current hasher so
    # verifying against it costs the same as verifying a real account.
    return _hasher.hash(secrets.token_urlsafe(32))


def dummy_verify(password: str) -> None:
    """Spend the same time as a real verify, for an unknown email, so response
    time does not reveal whether an account exists."""
    verify_password(_dummy_hash(), password)


def policy_error(password: str, email: str) -> str | None:
    """Why `password` is not allowed for `email` (normalised), or None."""
    if len(password) < MIN_LENGTH:
        return f"password must be at least {MIN_LENGTH} characters"
    if len(password) > MAX_LENGTH:
        return f"password must be at most {MAX_LENGTH} characters"
    if not password.strip():
        return "password must not be only whitespace"
    local_part = email.split("@", 1)[0]
    if password.strip().lower() == local_part.lower():
        return "password must not be the same as your email name"
    return None
