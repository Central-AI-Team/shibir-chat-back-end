"""Guest message quota, stored in Postgres (guest_usage) so restarts and
deploys do not reset it.

Two counters per guest request; running out of EITHER means "log in or register":
  owner:<owner hash>   settings.guest_message_limit responses, lifetime of the cookie
  ip:<IPv4 | IPv6 /64> settings.guest_ip_limit_per_day responses per 24 h window
                       (the window starts at the first message after the last one
                       expired), so clearing the cookie does not reset the quota
The IP is request.client.host, which uvicorn --proxy-headers already resolved
from X-Forwarded-For only for FORWARDED_ALLOW_IPS; never read that header here.

Reserve-then-refund: the router reserves one unit on both counters BEFORE any
LLM work and refunds it if no response was produced (upstream 503, a guest
NOTE). Each reservation is one conditional UPDATE (count < limit) checked by
the row it returns, so concurrent requests can never exceed the limit; the
first use of a key INSERTs, and a lost insert race retries the UPDATE.

guest_usage holds IP addresses: rows idle for PURGE_AFTER are deleted, a
bounded batch at a time, whenever a new key is inserted.
"""

from __future__ import annotations

import ipaddress
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

from sqlalchemy import case, delete, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.auth import aware, now, open_db
from app.core.config import settings
from app.db.models import GuestUsage

logger = logging.getLogger(__name__)

IP_WINDOW = timedelta(hours=24)
PURGE_AFTER = timedelta(days=30)
PURGE_BATCH = 500
_INSERT_RETRIES = 3

LIMIT_DETAIL = (
    "অতিথি হিসেবে বার্তা পাঠানোর সীমা শেষ হয়েছে। চালিয়ে যেতে অনুগ্রহ করে লগইন বা নিবন্ধন করুন।"
)
LIMIT_CODE = "guest_limit_reached"


@dataclass(frozen=True)
class Reservation:
    keys: tuple[str, ...]
    remaining: int  # responses left after this one (the smaller of the two counters)


def owner_key(owner: str) -> str:
    return f"owner:{owner}"


def ip_key(ip: str | None) -> str:
    """The full IPv4 address, or the IPv6 /64 (one subscriber's usual
    allocation; a full IPv6 address is trivially rotated)."""
    try:
        address = ipaddress.ip_address(ip or "")
    except ValueError:
        return f"ip:{ip or 'unknown'}"
    if address.version == 6:
        return f"ip:{ipaddress.ip_network(f'{address}/64', strict=False)}"
    return f"ip:{address}"


def _counters(owner: str, ip: str | None, at: datetime) -> list[tuple[str, int, datetime | None]]:
    """(key, limit, window cutoff or None for a lifetime counter)."""
    return [
        (owner_key(owner), settings.guest_message_limit, None),
        (ip_key(ip), settings.guest_ip_limit_per_day, at - IP_WINDOW),
    ]


def _reserve_one(db: Session, key: str, limit: int, cutoff: datetime | None, at: datetime) -> int | None:
    """Take one unit of `key`; returns its new count, or None at the limit.
    Commits on success."""
    if limit <= 0:
        return None
    if cutoff is None:
        stmt = update(GuestUsage).where(GuestUsage.key == key, GuestUsage.count < limit).values(
            count=GuestUsage.count + 1, last_seen=at
        )
    else:
        rolled = GuestUsage.window_start <= cutoff
        stmt = (
            update(GuestUsage)
            .where(GuestUsage.key == key, rolled | (GuestUsage.count < limit))
            .values(
                count=case((rolled, 1), else_=GuestUsage.count + 1),
                window_start=case((rolled, at), else_=GuestUsage.window_start),
                last_seen=at,
            )
        )
    for _ in range(_INSERT_RETRIES):
        count = db.execute(stmt.returning(GuestUsage.count)).scalar_one_or_none()
        if count is not None:
            db.commit()
            return count
        db.rollback()
        if db.execute(select(GuestUsage.key).where(GuestUsage.key == key)).first() is not None:
            return None  # the row exists and is at its limit
        try:
            db.add(GuestUsage(key=key, count=1, first_seen=at, last_seen=at, window_start=at))
            _purge(db, at)
            db.commit()
            return 1
        except IntegrityError:
            db.rollback()  # another request inserted it first: retry the UPDATE
    return None


def _release(db: Session, key: str) -> None:
    db.execute(
        update(GuestUsage).where(GuestUsage.key == key, GuestUsage.count > 0).values(count=GuestUsage.count - 1)
    )
    db.commit()


def _purge(db: Session, at: datetime) -> None:
    old = select(GuestUsage.key).where(GuestUsage.last_seen < at - PURGE_AFTER).limit(PURGE_BATCH)
    db.execute(delete(GuestUsage).where(GuestUsage.key.in_(old.scalar_subquery())))


def reserve(owner: str, ip: str | None) -> Reservation | None:
    """One unit on both counters, or None (nothing taken) when either is
    exhausted. Raises SQLAlchemyError if the database is unreachable (the
    router answers 503: fail closed)."""
    at = now()
    counters = _counters(owner, ip, at)
    taken: list[tuple[str, int, int]] = []
    with open_db() as db:
        for key, limit, cutoff in counters:
            count = _reserve_one(db, key, limit, cutoff, at)
            if count is None:
                for done_key, _, _ in taken:
                    _release(db, done_key)
                return None
            taken.append((key, limit, count))
    remaining = min(limit - count for _, limit, count in taken)
    return Reservation(keys=tuple(key for key, _, _ in taken), remaining=max(0, remaining))


def refund(reservation: Reservation | None) -> None:
    """Give the unit back (no response was produced). Never raises: a failed
    refund only costs the guest one message, and must not mask the real error."""
    if reservation is None:
        return
    try:
        with open_db() as db:
            for key in reservation.keys:
                _release(db, key)
    except Exception:
        logger.warning("guest quota refund failed", exc_info=True)


def remaining(owner: str | None, ip: str | None) -> int:
    """Responses left without reserving anything (GET /auth/me). A guest with
    no owner cookie yet is limited by the IP counter alone."""
    at = now()
    counters = _counters(owner or "", ip, at)
    if owner is None:
        counters = counters[1:]
    left = settings.guest_message_limit
    with open_db() as db:
        for key, limit, cutoff in counters:
            row = db.get(GuestUsage, key)
            used = 0
            if row is not None and not (cutoff is not None and aware(row.window_start) <= cutoff):
                used = row.count
            left = min(left, limit - used)
    return max(0, left)
