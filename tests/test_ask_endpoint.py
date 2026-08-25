"""Regression test for response_time_ms on /ask.

Mocks retrieval and generation so this exercises only the HTTP layer and
qa_service's timing logic -- no embedding model, reranker, or LLM call.
"""

from __future__ import annotations

from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app
from app.schemas.query import Citation

client = TestClient(app)

_CITATIONS = [
    Citation(
        book="Test Book",
        chapter="Chapter 1",
        source_db="chroma",
        content="প্রাসঙ্গিক অংশ।",
        similarity=0.9,
        rerank_score=0.95,
    )
]


def test_ask_returns_response_time_ms():
    with (
        patch("app.services.qa_service.retrieve_relevant_docs", return_value=_CITATIONS),
        patch("app.services.qa_service.generate_answer", return_value="উত্তর।"),
    ):
        response = client.post("/ask", json={"query": "প্রশ্ন?"})

    assert response.status_code == 200
    body = response.json()
    assert "response_time_ms" in body
    assert isinstance(body["response_time_ms"], (int, float))
    assert body["response_time_ms"] >= 0


def test_ask_below_threshold_still_returns_response_time_ms_and_no_sources():
    # No relevant citations -> generate_answer is still called (general-
    # knowledge/conversational fallback), but with empty grounding, so
    # sources stay empty.
    with (
        patch("app.services.qa_service.retrieve_relevant_docs", return_value=[]),
        patch("app.services.qa_service.generate_answer", return_value="উত্তর।") as mock_generate,
    ):
        response = client.post("/ask", json={"query": "প্রশ্ন?"})

    mock_generate.assert_called_once_with("প্রশ্ন?", [])
    assert response.status_code == 200
    body = response.json()
    assert "response_time_ms" in body
    assert body["response_time_ms"] >= 0
    assert body["sources"] == []
