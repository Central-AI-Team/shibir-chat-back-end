"""Regression tests for /chat -- the single user-facing endpoint (besides
/health) that classifies intent and dispatches to NOTE / ROLEPLAY /
SUGGESTION / QA internally. Replaces test_ask_endpoint.py now that /ask,
/note, /note-by-text have been removed; the QA-mode tests here cover the
same grounding/no-hallucination behavior the old /ask tests checked, just
reached through /chat.

Mocks service-level functions (and classify_intent, for deterministic
dispatch) so this exercises only the HTTP layer + router dispatch logic --
no embedding model, reranker, or real LLM call.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

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


def _completion(content: str) -> MagicMock:
    message = MagicMock()
    message.content = content
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


def test_chat_note_intent_routes_to_note_generation():
    note_result = {
        "book": "Test Book",
        "chapters": [
            {"chapter": "Chapter 1", "pages_used": 3, "note": "নোট বিষয়বস্তু।"},
        ],
    }
    with (
        patch("app.api.router.classify_intent", return_value="NOTE"),
        patch("app.api.router.generate_book_notes_from_text", return_value=note_result) as mock_note,
    ):
        response = client.post("/chat", json={"message": "নোট বানাও পরীক্ষা বই থেকে"})

    assert response.status_code == 200
    body = response.json()
    mock_note.assert_called_once_with("নোট বানাও পরীক্ষা বই থেকে")
    assert body["mode"] == "note"
    assert "Test Book" in body["answer"]
    assert "নোট বিষয়বস্তু।" in body["answer"]
    assert body["sources"] == []
    assert body["session_id"]


def test_chat_roleplay_intent_continues_persona_across_turns():
    with (
        patch("app.api.router.classify_intent", return_value="ROLEPLAY"),
        patch("app.services.roleplay_service.get_client") as mock_get_client,
    ):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # Turn 1: persona extraction call, then the in-character reply call.
        mock_client.chat.completions.create.side_effect = [
            _completion("শার্লক হোমস"),
            _completion("আমি শার্লক হোমস, বলুন কী জানতে চান।"),
        ]
        response1 = client.post(
            "/chat", json={"message": "তুমি এখন শার্লক হোমস চরিত্রে অভিনয় করো"}
        )
        assert response1.status_code == 200
        body1 = response1.json()
        assert body1["mode"] == "roleplay"
        assert body1["answer"] == "আমি শার্লক হোমস, বলুন কী জানতে চান।"
        session_id = body1["session_id"]
        assert session_id

        # Turn 2, same session: only the reply call should happen -- persona
        # should not be re-extracted.
        mock_client.chat.completions.create.side_effect = [
            _completion("দ্বিতীয় উত্তর, একই চরিত্রে।"),
        ]
        response2 = client.post(
            "/chat", json={"message": "তোমার পরিচয় কী?", "session_id": session_id}
        )
        assert response2.status_code == 200
        body2 = response2.json()
        assert body2["mode"] == "roleplay"
        assert body2["answer"] == "দ্বিতীয় উত্তর, একই চরিত্রে।"
        assert body2["session_id"] == session_id

    # 1 persona-extraction call + 2 reply calls = 3 total LLM calls.
    assert mock_client.chat.completions.create.call_count == 3


def test_chat_suggestion_intent_returns_sources_when_grounded():
    with (
        patch("app.api.router.classify_intent", return_value="SUGGESTION"),
        patch("app.services.suggestion_service.retrieve_relevant_docs", return_value=_CITATIONS),
        patch("app.services.suggestion_service.get_client") as mock_get_client,
    ):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.chat.completions.create.return_value = _completion("আমার পরামর্শ হলো...")

        response = client.post("/chat", json={"message": "আমার কি করা উচিত?"})

    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "suggestion"
    assert body["answer"] == "আমার পরামর্শ হলো..."
    assert len(body["sources"]) == 1
    assert body["sources"][0]["book"] == "Test Book"


def test_chat_plain_question_falls_through_to_qa_mode():
    with (
        patch("app.api.router.classify_intent", return_value="QA"),
        patch("app.services.qa_service.retrieve_relevant_docs", return_value=_CITATIONS),
        patch("app.services.qa_service.generate_answer", return_value="উত্তর।"),
    ):
        response = client.post("/chat", json={"message": "তৃতীয় অধ্যায়ে কী লেখা আছে?"})

    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "qa"
    assert body["answer"] == "উত্তর।"
    assert len(body["sources"]) == 1
    assert "response_time_ms" in body
    assert body["response_time_ms"] >= 0
    assert body["session_id"]


def test_chat_qa_below_threshold_returns_no_sources():
    # No relevant citations -> generate_answer is still called (general-
    # knowledge/conversational fallback), but with empty grounding, so
    # sources stay empty. Same behavior the old /ask endpoint had.
    with (
        patch("app.api.router.classify_intent", return_value="QA"),
        patch("app.services.qa_service.retrieve_relevant_docs", return_value=[]),
        patch("app.services.qa_service.generate_answer", return_value="উত্তর।") as mock_generate,
    ):
        response = client.post("/chat", json={"message": "প্রশ্ন?"})

    mock_generate.assert_called_once_with("প্রশ্ন?", [])
    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "qa"
    assert "response_time_ms" in body
    assert body["response_time_ms"] >= 0
    assert body["sources"] == []


def test_chat_rejects_empty_message():
    response = client.post("/chat", json={"message": "   "})
    assert response.status_code == 400
