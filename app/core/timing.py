"""Tracked shim for the local-only latency recorder (app/core/timing_local.py).

App code imports ONLY from here. timing_local.py is git-ignored, so on a fresh
clone / in production the import fails and every helper below is a no-op.
"""

from __future__ import annotations

try:
    from app.core.timing_local import (  # noqa: F401
        current, finish_request, mark, record_error, set_mode, start_request, timed, use,
    )
except ImportError:
    from contextlib import nullcontext

    class _NoHandle:
        def lap(self, name):
            pass

    _NO_HANDLE = _NoHandle()

    def start_request(query, mode=None, route="POST /chat"):
        return None

    def current():
        return None

    def set_mode(mode, rec=None):
        pass

    def timed(step, rec=None):
        return nullcontext(_NO_HANDLE)

    def use(rec):
        return nullcontext(rec)

    def mark(step, note, rec=None):
        pass

    def record_error(step, exc, rec=None):
        pass

    def finish_request(rec=None):
        pass
