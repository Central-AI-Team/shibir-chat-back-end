"""Create a user account, e.g. the first admin.

Usage:
    uv run python -m scripts.create_user --email admin@example.com --role admin

The password is read from the CREATE_USER_PASSWORD environment variable if set,
otherwise prompted for twice with getpass. It is never taken from the command
line (argv shows up in shell history and `ps`). The same password policy as
registration applies. An existing account is never overwritten.

The tables are created when the app starts; before the first start, apply
docs/sql/auth_tables.sql instead.
"""

from __future__ import annotations

import argparse
import getpass
import os
import sys
import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.auth_routes import email_shape_ok, normalise_email
from app.auth import now, passwords
from app.db.models import User

ROLES = ("user", "admin")


class CreateUserError(ValueError):
    pass


def create_user(db: Session, email: str, password: str, role: str) -> str:
    """Insert the user and return its id. Raises CreateUserError on a bad
    email, role or password, or if the email is taken. The caller commits."""
    email = normalise_email(email)
    if not email_shape_ok(email):
        raise CreateUserError("invalid email address")
    if role not in ROLES:
        raise CreateUserError(f"role must be one of {', '.join(ROLES)}")
    problem = passwords.policy_error(password, email)
    if problem is not None:
        raise CreateUserError(problem)
    if db.execute(select(User.id).where(User.email == email)).first() is not None:
        raise CreateUserError("a user with this email already exists; refusing to overwrite it")
    at = now()
    user = User(
        id=str(uuid.uuid4()),
        email=email,
        password_hash=passwords.hash_password(password),
        role=role,
        is_active=True,
        failed_login_count=0,
        created_at=at,
        password_changed_at=at,
    )
    db.add(user)
    db.flush()
    return user.id


def _read_password() -> str:
    from_env = os.environ.get("CREATE_USER_PASSWORD")
    if from_env:
        return from_env
    first = getpass.getpass("Password: ")
    if getpass.getpass("Repeat password: ") != first:
        raise CreateUserError("passwords do not match")
    return first


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--email", required=True)
    parser.add_argument("--role", choices=ROLES, default="user")
    args = parser.parse_args(argv)

    from app.auth import open_db

    try:
        password = _read_password()
        with open_db() as db:
            user_id = create_user(db, args.email, password, args.role)
            db.commit()
    except CreateUserError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"created {args.role} {normalise_email(args.email)} (id {user_id})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
