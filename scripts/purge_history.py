"""Hard-delete expired chat history (the retention job for conversations / messages).

Usage:
    uv run python -m scripts.purge_history            # DRY RUN (default): counts only
    uv run python -m scripts.purge_history --apply    # delete
    uv run python -m scripts.purge_history --days 7   # override GUEST_HISTORY_RETENTION_DAYS

Deletes, in one transaction, conversations (and their messages, ON DELETE
CASCADE) that are either

  1. guest conversations (user_id IS NULL) not updated for
     settings.guest_history_retention_days -- a guest who lost the cookie can
     never reach them again, and the owner key is a cookie hash, not an account; or
  2. soft-deleted by anyone (deleted_at set: DELETE /conversations/{id} or the
     per-owner cap) at least that long ago.

A logged-in user's live conversations are never touched. Run it from cron on
the host (`docker compose exec -T app uv run python -m scripts.purge_history
--apply`) or by hand; it is idempotent.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone

from sqlalchemy import delete, func, or_, select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.models import Conversation, Message


def _expired(cutoff: datetime):
    """The WHERE clause shared by the count and the delete."""
    return or_(
        (Conversation.user_id.is_(None)) & (Conversation.updated_at < cutoff),
        Conversation.deleted_at < cutoff,
    )


def counts(s: Session, cutoff: datetime) -> dict[str, int]:
    expired_ids = select(Conversation.id).where(_expired(cutoff))
    return {
        "conversations_total": s.execute(select(func.count(Conversation.id))).scalar_one(),
        "conversations_expired": s.execute(
            select(func.count(Conversation.id)).where(_expired(cutoff))
        ).scalar_one(),
        "messages_expired": s.execute(
            select(func.count(Message.id)).where(Message.conversation_id.in_(expired_ids))
        ).scalar_one(),
    }


def purge(s: Session, cutoff: datetime) -> int:
    """Delete expired conversations (messages cascade). Does not commit.
    Messages are deleted explicitly too, for databases whose FK was created
    without ON DELETE CASCADE (SQLite in tests)."""
    expired_ids = select(Conversation.id).where(_expired(cutoff))
    s.execute(delete(Message).where(Message.conversation_id.in_(expired_ids)))
    result = s.execute(delete(Conversation).where(_expired(cutoff)))
    return result.rowcount


def cutoff_for(days: int, now: datetime | None = None) -> datetime:
    return (now or datetime.now(timezone.utc)) - timedelta(days=days)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--apply", action="store_true", help="delete (default: dry run)")
    parser.add_argument(
        "--days", type=int, default=settings.guest_history_retention_days,
        help=f"retention in days (default: GUEST_HISTORY_RETENTION_DAYS={settings.guest_history_retention_days})",
    )
    args = parser.parse_args(argv)
    if args.days < 1:
        print("--days must be at least 1", file=sys.stderr)
        return 2

    from app.db.session import SessionLocal

    cutoff = cutoff_for(args.days)
    s = SessionLocal()
    try:
        before = counts(s, cutoff)
        print(f"cutoff: {cutoff.isoformat(timespec='seconds')} ({args.days} days)")
        print("before:", before)
        if not args.apply:
            print(
                f"\nDRY RUN would delete {before['conversations_expired']} conversations "
                f"and {before['messages_expired']} messages. Re-run with --apply."
            )
            return 0
        deleted = purge(s, cutoff)
        s.commit()
        print(f"deleted {deleted} conversations ({before['messages_expired']} messages)")
        print("after:", counts(s, cutoff))
    finally:
        s.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
