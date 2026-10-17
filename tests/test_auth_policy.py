"""Deny-by-default guard: every route must be in app/auth/policy.py's
ROUTE_POLICY and carry its level's dependency. A new endpoint fails this
test until someone decides who may call it."""

from __future__ import annotations

import importlib

import pytest
from fastapi.routing import APIRoute

from app.api import deps
from app.auth.policy import LEVELS, ROUTE_POLICY

LEVEL_DEPENDENCY = {
    "public": deps.public_route,
    "guest_ok": deps.get_principal,
    "user": deps.require_user,
    "admin": deps.require_role("admin"),
}
# Dependencies that grant or check identity; a route must carry exactly the one
# for its level (a 'public' route must not quietly require a user, and so on).
_ACCESS_DEPENDENCIES = set(LEVEL_DEPENDENCY.values())
_DOCS_PATHS = {"/docs", "/docs/oauth2-redirect", "/redoc", "/openapi.json"}


def _calls(dependant) -> set:
    found = set()
    for sub in dependant.dependencies:
        found.add(sub.call)
        found |= _calls(sub)
    return found


def _flatten(routes):
    """(full path, route) for every route. FastAPI 0.141 keeps included
    routers as _IncludedRouter entries in app.routes; their routes (with the
    prefix applied) come from effective_route_contexts()."""
    for route in routes:
        if hasattr(route, "effective_route_contexts"):
            for context in route.effective_route_contexts():
                yield context.path, context.original_route
        else:
            yield route.path, route


def _top_level_access(route: APIRoute) -> set:
    """Access dependencies declared directly on the route (not nested inside
    another one, e.g. limit_chat -> get_principal)."""
    return {sub.call for sub in route.dependant.dependencies if sub.call in _ACCESS_DEPENDENCIES}


@pytest.fixture(params=[False, True], ids=["docs-off", "docs-on"])
def app_routes(request, monkeypatch):
    from app.core.config import settings
    import app.main as main_module

    monkeypatch.setattr(settings, "enable_docs", request.param)
    yield importlib.reload(main_module).app.routes
    monkeypatch.undo()
    importlib.reload(main_module)


def test_every_route_declares_a_level_and_carries_its_dependency(app_routes):
    seen = set()
    for path, route in _flatten(app_routes):
        if not isinstance(route, APIRoute):
            assert path in _DOCS_PATHS, f"unexpected non-API route {path}"
            continue
        for method in route.methods:
            key = (method, path)
            assert key in ROUTE_POLICY, f"{key} has no access level in app/auth/policy.py"
            level = ROUTE_POLICY[key]
            assert level in LEVELS
            declared = _top_level_access(route)
            assert declared == {LEVEL_DEPENDENCY[level]}, (
                f"{key} is '{level}' but declares {sorted(d.__name__ for d in declared)}"
            )
            assert LEVEL_DEPENDENCY[level] in _calls(route.dependant)
            seen.add(key)
    assert len(seen) == len(ROUTE_POLICY) > 0
    assert set(ROUTE_POLICY) == seen, f"stale policy entries: {set(ROUTE_POLICY) - seen}"


def test_state_changing_routes_check_origin(app_routes):
    for path, route in _flatten(app_routes):
        if isinstance(route, APIRoute) and route.methods & {"POST", "DELETE"}:
            assert deps.check_origin in _calls(route.dependant), path
            if path.startswith("/auth/"):
                assert deps.check_session_origin in _calls(route.dependant), path


def test_the_guard_catches_an_undeclared_route():
    from fastapi import FastAPI

    rogue = FastAPI()

    @rogue.get("/rogue")
    def handler():
        return {}

    route = next(r for r in rogue.routes if isinstance(r, APIRoute))
    assert ("GET", "/rogue") not in ROUTE_POLICY
    assert _top_level_access(route) == set()
