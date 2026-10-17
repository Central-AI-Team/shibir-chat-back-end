"""The interactive docs are opt-in (settings.enable_docs) and CORS only
allows the Content-Type request header.

app/main.py reads the setting at import time, so each case reloads the module
with the setting patched and reloads it again on teardown.
"""

from __future__ import annotations

import importlib

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings, settings
import app.main as main_module

DOC_PATHS = ["/docs", "/redoc", "/openapi.json"]


@pytest.fixture
def make_client(monkeypatch):
    def _make(enable_docs: bool) -> TestClient:
        monkeypatch.setattr(settings, "enable_docs", enable_docs)
        return TestClient(importlib.reload(main_module).app)

    yield _make
    monkeypatch.undo()
    importlib.reload(main_module)


def test_enable_docs_defaults_to_false():
    assert type(settings).model_fields["enable_docs"].default is False


def test_host_defaults_to_loopback_when_unset(monkeypatch):
    monkeypatch.delenv("HOST", raising=False)
    assert Settings(_env_file=None, openai_api_key="x").host == "127.0.0.1"


@pytest.mark.parametrize("path", DOC_PATHS)
def test_docs_return_404_by_default(make_client, path):
    assert make_client(False).get(path).status_code == 404


@pytest.mark.parametrize("path", DOC_PATHS)
def test_docs_return_200_when_enabled(make_client, path):
    assert make_client(True).get(path).status_code == 200


def test_cors_allows_only_content_type_header(make_client):
    origin = settings.cors_allow_origins[0]
    client = make_client(False)
    headers = {
        "Origin": origin,
        "Access-Control-Request-Method": "POST",
    }
    ok = client.options("/chat", headers={**headers, "Access-Control-Request-Headers": "Content-Type"})
    assert ok.status_code == 200
    # Starlette always appends the CORS-safelisted headers; "*" is what is gone.
    assert "Content-Type" in ok.headers["access-control-allow-headers"]
    assert "*" not in ok.headers["access-control-allow-headers"]

    bad = client.options("/chat", headers={**headers, "Access-Control-Request-Headers": "X-Custom"})
    assert bad.status_code == 400


def test_cors_exposes_retry_after(make_client):
    """Retry-After is not CORS-safelisted: without expose_headers a browser
    front-end cannot read it from a 429."""
    client = make_client(False)
    r = client.get("/health", headers={"Origin": settings.cors_allow_origins[0]})
    assert r.status_code == 200
    assert r.headers["access-control-expose-headers"] == "Retry-After"
    # Starlette sends the simple headers to every origin; what keeps a browser
    # on a disallowed origin from reading anything is the missing allow-origin.
    r = client.get("/health", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in r.headers
