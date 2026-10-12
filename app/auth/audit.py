"""Security audit log: one JSON line per event on the `app.auth.audit` logger.

Never pass a password, token, hash, full email or full IP: emails go in as a
short sha256 prefix (email_ref) and IPs as a /24 (IPv4) or /48 (IPv6) prefix.
Nothing here is sent to Langfuse. There is no request id in this app yet, so
lines carry none.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import logging

from app.auth import now

logger = logging.getLogger("app.auth.audit")
# Nothing configures app logging under uvicorn (the root logger has no handler
# and drops INFO), so the audit trail gets its own stderr handler and does not
# depend on that. propagate stays on so pytest's caplog still sees the records.
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)

EVENTS = frozenset(
    {
        "login_success",
        "login_failure",
        "register",
        "logout",
        "logout_all",
        "password_changed",
        "account_locked",
        "session_rejected",
        "guest_limit_reached",
        "forbidden_access",
    }
)


def ip_prefix(ip: str | None) -> str:
    """IPv4 -> its /24, IPv6 -> its /48, anything else -> "unknown"."""
    try:
        address = ipaddress.ip_address(ip or "")
    except ValueError:
        return "unknown"
    bits = 24 if address.version == 4 else 48
    return str(ipaddress.ip_network(f"{address}/{bits}", strict=False))


def email_ref(normalised_email: str) -> str:
    """A short, non-reversible-in-practice reference to an email for failures."""
    return hashlib.sha256(normalised_email.encode()).hexdigest()[:12]


def event(
    name: str,
    *,
    ip: str | None = None,
    user_id: str | None = None,
    email: str | None = None,
    **fields: object,
) -> None:
    """Log one event. `ip` is the client address (truncated here); `email` is
    the normalised email (hashed here). Extra fields must be non-sensitive."""
    if name not in EVENTS:
        raise ValueError(f"unknown audit event {name!r}")
    record: dict[str, object] = {"ts": now().isoformat(), "event": name}
    if user_id is not None:
        record["user_id"] = user_id
    record["ip_prefix"] = ip_prefix(ip)
    if email is not None:
        record["email_ref"] = email_ref(email)
    record.update(fields)
    logger.info(json.dumps(record, ensure_ascii=False, sort_keys=True))
