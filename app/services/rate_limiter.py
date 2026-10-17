"""In-process rate limits and the daily LLM request budget.

The proxy's limit (nginx: 30/min per IP on /chat, /chat/stream) covers only
nginx deployments, only per IP, and nothing per owner, so a client rotating
cookies -- or a Caddy deployment without its plugin -- could drain the OpenAI
budget. These limits are enforced by the app itself (app/api/deps.py
dependencies, plus the NOTE check in app/api/router.py).

NOTE: like session_store, this is module-level state, so it is correct only
for a single worker, and everything resets on restart (a restart hands every
client a fresh window and the daily budget starts again from zero).

Memory is bounded: each key keeps at most `limit` timestamps, expired keys are
swept once per window, and past max_keys the least-recently-hit key is evicted
(an evicted key just starts a fresh window). Requests run in a threadpool, so
every mutation holds the limiter's Lock.
"""

from __future__ import annotations

import math
import threading
import time
from collections import OrderedDict, deque

# Patched by tests to control time (as gpu_client._monotonic is).
# _monotonic drives the sliding windows; _time (epoch seconds) decides the UTC day.
_monotonic = time.monotonic
_time = time.time

MAX_TRACKED_KEYS = 10_000
_DAY_SECONDS = 86_400

RATE_LIMIT_DETAIL = "অনেক বেশি অনুরোধ পাঠানো হয়েছে। কিছুক্ষণ পর আবার চেষ্টা করুন।"
NOTE_LIMIT_DETAIL = "নোট তৈরির সীমা পূর্ণ হয়েছে। কিছুক্ষণ পর আবার চেষ্টা করুন।"
DAILY_BUDGET_DETAIL = "আজকের জন্য সার্ভিসের সীমা পূর্ণ হয়ে গেছে। আগামীকাল আবার চেষ্টা করুন।"


class SlidingWindowLimiter:
    """At most `limit` hits per key in any `window_seconds` (a sliding log)."""

    def __init__(self, window_seconds: float, max_keys: int = MAX_TRACKED_KEYS) -> None:
        self.window = window_seconds
        self.max_keys = max_keys
        self._hits: OrderedDict[str, deque[float]] = OrderedDict()
        self._last_sweep = 0.0
        self._lock = threading.Lock()

    def hit(self, key: str, limit: int) -> int | None:
        """Record a hit and return None, or return the seconds until one is
        allowed again (Retry-After) without recording anything."""
        return self._check(key, limit, record=True)

    def blocked(self, key: str, limit: int) -> int | None:
        """Like hit() but never records: the Retry-After seconds if `key` is
        over the limit, else None. For limiters that count only failures
        (login per email): check with blocked() first, hit() on failure."""
        return self._check(key, limit, record=False)

    def _check(self, key: str, limit: int, record: bool) -> int | None:
        now = _monotonic()
        with self._lock:
            if now - self._last_sweep >= self.window:
                self._sweep(now)
            hits = self._hits.get(key)
            if hits is not None:
                while hits and hits[0] <= now - self.window:
                    hits.popleft()
                # Rejected attempts count as recent too, so a client that keeps
                # hammering while blocked is the last to be evicted (eviction
                # would hand it a fresh window).
                self._hits.move_to_end(key)
                if len(hits) >= limit:
                    return max(1, math.ceil(hits[0] + self.window - now))
            if not record:
                return None
            if hits is None:
                if len(self._hits) >= self.max_keys:
                    self._sweep(now)
                    while len(self._hits) >= self.max_keys:
                        self._hits.popitem(last=False)  # least recently hit
                hits = self._hits[key] = deque()
            hits.append(now)
            return None

    def _sweep(self, now: float) -> None:
        cutoff = now - self.window
        for key in [k for k, hits in self._hits.items() if not hits or hits[-1] <= cutoff]:
            del self._hits[key]
        self._last_sweep = now

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()
            self._last_sweep = 0.0


class DailyBudget:
    """A global request counter per UTC day. limit <= 0 means unlimited."""

    def __init__(self) -> None:
        self._day: int | None = None
        self._count = 0
        self._lock = threading.Lock()

    def hit(self, limit: int) -> int | None:
        """Count a request and return None, or return the seconds until the
        next UTC midnight when today's budget is spent."""
        if limit <= 0:
            return None
        now = _time()
        day = int(now // _DAY_SECONDS)
        with self._lock:
            if day != self._day:
                self._day, self._count = day, 0
            if self._count >= limit:
                return max(1, math.ceil((day + 1) * _DAY_SECONDS - now))
            self._count += 1
            return None

    def reset(self) -> None:
        with self._lock:
            self._day, self._count = None, 0


class KeyedDailyLimiter:
    """At most `limit` hits per key per UTC day. limit <= 0 means unlimited.
    The whole table is dropped when the day changes, so memory is one int per
    key seen today."""

    def __init__(self) -> None:
        self._day: int | None = None
        self._counts: dict[str, int] = {}
        self._lock = threading.Lock()

    def hit(self, key: str, limit: int) -> int | None:
        """Count a hit and return None, or return the seconds until the next
        UTC midnight when this key's budget is spent."""
        if limit <= 0:
            return None
        now = _time()
        day = int(now // _DAY_SECONDS)
        with self._lock:
            if day != self._day:
                self._day, self._counts = day, {}
            count = self._counts.get(key, 0)
            if count >= limit:
                return max(1, math.ceil((day + 1) * _DAY_SECONDS - now))
            self._counts[key] = count + 1
            return None

    def reset(self) -> None:
        with self._lock:
            self._day, self._counts = None, {}


USER_DAILY_LIMIT_DETAIL = "আজকের বার্তার সীমা পূর্ণ হয়েছে। আগামীকাল আবার চেষ্টা করুন।"
AUTH_RATE_LIMIT_DETAIL = "too many attempts; try again later"

chat_per_ip = SlidingWindowLimiter(60)
chat_per_owner = SlidingWindowLimiter(60)
read_per_ip = SlidingWindowLimiter(60)
note_per_owner = SlidingWindowLimiter(3600)
daily_budget = DailyBudget()
# Accounts (app/api/auth_routes.py, app/api/deps.py).
login_per_email = SlidingWindowLimiter(15 * 60)
login_per_ip = SlidingWindowLimiter(3600)
register_per_ip = SlidingWindowLimiter(3600)
registrations_daily = DailyBudget()
user_daily = KeyedDailyLimiter()


def reset() -> None:
    """Clear every limiter (tests)."""
    for limiter in (
        chat_per_ip, chat_per_owner, read_per_ip, note_per_owner, daily_budget,
        login_per_email, login_per_ip, register_per_ip, registrations_daily, user_daily,
    ):
        limiter.reset()
