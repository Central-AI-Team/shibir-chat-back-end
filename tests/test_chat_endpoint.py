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

import json
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


def test_chat_note_intent_routes_to_note_generation(logged_in_user):
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


def test_chat_roleplay_picks_a_role_then_answers_from_the_books_in_that_style():
    """Mentor mode (docs/roleplay-v2.md): the entry turn stores the role and
    greets without retrieval; the next turn is QA through qa_stream with the
    role's tone as style_rules, so sources come back and no persona LLM call
    is made. The level is never announced."""
    meta = {"intent": "ROLEPLAY", "rewritten_query": "বন্ধুর মতো বুঝিয়ে বলো",
            "role_id": "bondhu", "persona_ok": True, "level": "beginner"}
    seen = {}

    def fake_stream(question, citations, *, style_rules=None, **kw):
        seen["style_rules"] = style_rules
        seen["citations"] = citations
        return iter(["উত্তর", "।"])

    with (
        patch("app.services.intent_classifier.complete", return_value=_completion(json.dumps(meta))),
        patch("app.services.qa_service.retrieve_relevant_docs", return_value=_CITATIONS) as retrieve,
        patch("app.services.qa_service.stream_answer", side_effect=fake_stream),
        patch("app.services.roleplay_service.complete",
              return_value=_completion('{"suggestions": ["যাকাত কাদের উপর ফরজ?"]}')),
    ):
        r1 = client.post("/chat", json={"message": "একজন বন্ধুর মতো বুঝিয়ে বলো"})
        assert r1.status_code == 200
        body1 = r1.json()
        assert body1["mode"] == "roleplay" and body1["sources"] == []
        assert "বন্ধু" in body1["answer"]  # greets in the role, asks for a question
        retrieve.assert_not_called()
        sid = body1["session_id"]

        r2 = client.post("/chat", json={"message": "যাকাত কী?", "session_id": sid})
    assert r2.status_code == 200
    body2 = r2.json()
    assert body2["mode"] == "roleplay"
    assert body2["answer"] == "উত্তর।"
    assert [s["book"] for s in body2["sources"]] == ["Test Book"]
    assert body2["session_id"] == sid
    retrieve.assert_called_once()
    assert "বন্ধু" in seen["style_rules"] and "পাঠকের স্তর: নতুন" in seen["style_rules"]
    assert "beginner" not in body2["answer"] and "নতুন" not in body2["answer"]


def test_chat_suggestion_intent_returns_sources_when_grounded():
    with (
        patch("app.api.router.classify_intent", return_value="SUGGESTION"),
        patch("app.services.suggestion_service.retrieve_relevant_docs", return_value=_CITATIONS),
        # suggestion_service now calls llm.complete(); patch the client there.
        patch("app.core.llm.get_client") as mock_get_client,
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


def test_chat_gpu_service_failure_returns_503():
    from app.api.router import _LLM_UNAVAILABLE_DETAIL
    from app.rag.gpu_client import GPUServiceError

    with (
        patch("app.api.router.classify_intent", return_value="QA"),
        patch(
            "app.services.qa_service.retrieve_relevant_docs",
            side_effect=GPUServiceError("GPU service /embed: HTTP 503", 503),
        ),
    ):
        response = client.post("/chat", json={"message": "প্রশ্ন?"})

    assert response.status_code == 503
    assert response.json()["detail"] == _LLM_UNAVAILABLE_DETAIL


def test_chat_message_over_max_message_chars_returns_413():
    from app.core.config import settings

    with patch("app.api.router.classify_intent") as mock_classify:
        response = client.post("/chat", json={"message": "ক" * (settings.max_message_chars + 1)})

    assert response.status_code == 413
    assert str(settings.max_message_chars) in response.json()["detail"]
    mock_classify.assert_not_called()


def test_chat_message_exactly_at_max_message_chars_works():
    from app.core.config import settings

    with (
        patch("app.api.router.classify_intent", return_value="QA"),
        patch("app.services.qa_service.retrieve_relevant_docs", return_value=_CITATIONS),
        patch("app.services.qa_service.generate_answer", return_value="উত্তর।"),
    ):
        response = client.post("/chat", json={"message": "ক" * settings.max_message_chars})

    assert response.status_code == 200
    assert response.json()["mode"] == "qa"


def test_chat_note_over_max_note_chars_returns_413(logged_in_user):
    from app.core.config import settings

    with (
        patch("app.api.router.classify_intent", return_value="NOTE"),
        patch("app.api.router.generate_book_notes_from_text") as mock_note,
    ):
        response = client.post("/chat", json={"message": "ক" * (settings.max_note_chars + 1)})

    assert response.status_code == 413
    assert str(settings.max_note_chars) in response.json()["detail"]
    mock_note.assert_not_called()
