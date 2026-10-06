"""Small talk: classification, replies, intent routing and the QA-path split.

No network and no real models: the chitchat LLM call, retrieval and generation
are mocked at the module boundary.
"""

from __future__ import annotations

import json
import unicodedata
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from openai import APIError

from app.main import app
from app.schemas.query import Citation
from app.services import chitchat_service as cs
from app.services import intent_classifier as ic
from app.services import qa_service
from app.services.chitchat_service import (
    BOT_IDENTITY,
    FAREWELL,
    OTHER,
    SALAM,
    THANKS,
    WELLBEING,
    classify_chitchat,
    normalize_text,
    preface,
    reply_for,
    respond,
)

client = TestClient(app)

_CITATIONS = [
    Citation(
        book="Test Book", chapter="Chapter 1", source_db="chroma",
        content="প্রাসঙ্গিক অংশ।", similarity=0.9, rerank_score=0.95,
    )
]


def _completion(text: str) -> MagicMock:
    message = MagicMock()
    message.content = text
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


def _events(raw: str) -> list[tuple[str, dict]]:
    out = []
    for block in raw.strip().split("\n\n"):
        event = data = None
        for line in block.splitlines():
            if line.startswith("event: "):
                event = line[len("event: "):]
            elif line.startswith("data: "):
                data = json.loads(line[len("data: "):])
        out.append((event, data))
    return out


# --------------------------------------------------------------------------
# classification
# --------------------------------------------------------------------------

_PURE = [
    # salam, with and without the long form
    ("assalamualaikum", SALAM),
    ("assalamualikum", SALAM),
    ("assalamu alaikum", SALAM),
    ("Assalamu Alaikum wa rahmatullahi wa barakatuhu", SALAM),
    ("আসসালামু আলাইকুম", SALAM),
    ("আসসালামু আলাইকুম ওয়া রাহমাতুল্লাহি ওয়া বারাকাতুহ", SALAM),
    ("walaikum assalam", SALAM),
    ("ওয়ালাইকুম আসসালাম", SALAM),
    ("ওয়া আলাইকুমুস সালাম", SALAM),
    ("hi", SALAM),
    ("Hello!", SALAM),
    ("hey", SALAM),
    # how are you
    ("কেমন আছেন", WELLBEING),
    ("কেমন আছো", WELLBEING),
    ("কেমন আছিস", WELLBEING),
    ("apni kemon achen", WELLBEING),
    ("kemon acho", WELLBEING),
    ("kemon asen", WELLBEING),
    ("ki khobor", WELLBEING),
    # thanks and dua
    ("jazakallah", THANKS),
    ("জাযাকাল্লাহু খাইরান", THANKS),
    ("ধন্যবাদ", THANKS),
    ("thanks", THANKS),
    ("আলহামদুলিল্লাহ", OTHER),
    # goodbye
    ("allah hafez", FAREWELL),
    ("khoda hafez", FAREWELL),
    ("বিদায়", FAREWELL),
    # who are you
    ("তুমি কে", BOT_IDENTITY),
    ("আপনি কে", BOT_IDENTITY),
    ("who are you", BOT_IDENTITY),
    ("tumi ke", BOT_IDENTITY),
    # address words and compounds
    ("ভাই আসসালামু আলাইকুম", SALAM),
    ("ধন্যবাদ ভাই", THANKS),
    ("assalamualaikum bhai, kemon achen?", WELLBEING),
]


@pytest.mark.parametrize("text,subtype", _PURE)
def test_pure_small_talk_is_classified(text, subtype):
    assert classify_chitchat(text) == (subtype, "")


@pytest.mark.parametrize("text", [
    "শিবির কে?", "শিবির কী?", "শিবির কি?",  # questions about the organisation
    "নামাজের নিয়ম কী?",
    "জি, নামাজের নিয়ম কী?",  # an address word alone is not small talk
    "আপনি নামাজ কীভাবে পড়েন?",
    # topics that merely START with a pleasantry word
    "সালাম দেওয়ার নিয়ম কী?",
    "বিদায় হজ্জের ভাষণ কী?",
    "ধন্যবাদ দেওয়ার ফজিলত কী?",
    # identity words only count as the whole message
    "who are you and what is zakat",
])
def test_questions_are_not_chitchat(text):
    assert classify_chitchat(text) is None


@pytest.mark.parametrize("text,subtype,remainder", [
    ("hi, নামাজের নিয়ম কী?", SALAM, "নামাজের নিয়ম কী?"),
    ("assalamualaikum, zakat koto taka?", SALAM, "zakat koto taka?"),
    ("আসসালামু আলাইকুম, যাকাতের নিসাব কত?", SALAM, "যাকাতের নিসাব কত?"),
    ("kemon achen? roja kivabe rakhte hoy", WELLBEING, "roja kivabe rakhte hoy"),
    ("ধন্যবাদ, এবার রোজার নিয়ম বলুন", THANKS, "এবার রোজার নিয়ম বলুন"),
    ("hello, what is the rule of zakat?", SALAM, "what is the rule of zakat?"),
])
def test_greeting_plus_question_keeps_the_whole_question(text, subtype, remainder):
    assert classify_chitchat(text) == (subtype, remainder)


def test_composed_and_decomposed_unicode_both_match():
    composed = unicodedata.normalize("NFC", "কেমন আছো")
    decomposed = unicodedata.normalize("NFD", "কেমন আছো")
    assert composed != decomposed  # ো really is ে + া in the second form
    assert "ো" in decomposed
    assert classify_chitchat(composed) == (WELLBEING, "")
    assert classify_chitchat(decomposed) == (WELLBEING, "")
    assert normalize_text(composed) == normalize_text(decomposed)


def test_zero_width_joiners_do_not_break_matching():
    assert classify_chitchat("কে‌মন আ‍ছেন") == (WELLBEING, "")


# --------------------------------------------------------------------------
# template replies
# --------------------------------------------------------------------------

_ALL_POOLS = {
    "salam": cs._SALAM_REPLIES,
    "hello": cs._HELLO_REPLIES,
    **{k: v for k, v in cs._REPLIES.items()},
}


@pytest.mark.parametrize("text,pool", [
    ("assalamualaikum", "salam"),
    ("আসসালামু আলাইকুম", "salam"),
    ("হ্যালো", "hello"),
    ("কেমন আছেন", WELLBEING),
    ("ধন্যবাদ", THANKS),
    ("jazakallah", THANKS),
    ("আল্লাহ হাফেজ", FAREWELL),
])
def test_template_reply_matches_its_subtype_and_never_another(text, pool):
    subtype = classify_chitchat(text)[0]
    with patch.object(cs, "complete") as complete:
        replies = {reply_for(subtype, text, []) for _ in range(60)}
    complete.assert_not_called()  # templates make no LLM call
    own = set(_ALL_POOLS[pool])
    others = {r for name, p in _ALL_POOLS.items() if name != pool for r in p}
    assert replies <= own
    assert not replies & others


def test_a_salam_is_answered_with_the_full_salam_and_never_a_goodbye():
    with patch.object(cs, "complete"):
        for _ in range(30):
            reply = reply_for(SALAM, "assalamualaikum", [])
            assert reply.startswith("ওয়া আলাইকুমুস সালাম ওয়া রাহমাতুল্লাহি ওয়া বারাকাতুহ।")
            assert "হাফেজ" not in reply and "আমানিল্লাহ" not in reply


def test_salam_plus_how_are_you_answers_both():
    with patch.object(cs, "complete"):
        reply = reply_for(WELLBEING, "assalamualaikum, kemon achen?", [])
    assert reply.startswith("ওয়া আলাইকুমুস সালাম")
    assert "ভালো আছি" in reply


def test_every_template_pool_has_two_or_three_variants():
    for name, pool in _ALL_POOLS.items():
        assert 2 <= len(pool) <= 3, name


# --------------------------------------------------------------------------
# LLM path
# --------------------------------------------------------------------------


def test_identity_goes_to_the_llm_with_the_persona_and_recent_history():
    history = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}" + "x" * 900}
               for i in range(10)]
    with patch.object(cs, "complete", return_value=_completion("আমি একটি এআই সহকারী।")) as complete:
        reply = reply_for(BOT_IDENTITY, "তুমি কে?", history)

    assert reply == "আমি একটি এআই সহকারী।"
    task, messages = complete.call_args.args
    assert task == "roleplay"  # an existing task key; none was added
    assert messages[0] == {"role": "system", "content": cs._PERSONA}
    assert messages[-1] == {"role": "user", "content": "তুমি কে?"}
    past = messages[1:-1]
    assert [m["content"][:2] for m in past] == ["m4", "m5", "m6", "m7", "m8", "m9"]
    assert all(len(m["content"]) <= 600 for m in past)


def test_persona_prompt_is_the_specified_text():
    assert cs._PERSONA.startswith("তুমি শিবির চ্যাট-এর বন্ধুসুলভ সহকারী।")
    assert "নিজের পরিচয় চাইলে: তুমি বইভিত্তিক একটি এআই সহকারী, মানুষ নও।" in cs._PERSONA
    assert "প্রশ্ন ইংরেজিতে হলে উত্তরও ইংরেজিতে দাও" in cs._PERSONA


@pytest.mark.parametrize("text", ["hi", "hello", "thanks", "thank you", "bye", "how are you"])
def test_clearly_english_small_talk_goes_to_the_llm_not_a_bengali_template(text):
    subtype = classify_chitchat(text)[0]
    with patch.object(cs, "complete", return_value=_completion("Hi there!")) as complete:
        assert reply_for(subtype, text, []) == "Hi there!"
    complete.assert_called_once()


def test_banglish_is_not_english():
    with patch.object(cs, "complete") as complete:
        reply_for(SALAM, "assalamualaikum", [])
        reply_for(WELLBEING, "kemon achen", [])
        reply_for(THANKS, "dhonnobad", [])
    complete.assert_not_called()


def test_empty_llm_reply_falls_back_instead_of_returning_nothing():
    with patch.object(cs, "complete", return_value=_completion("")):
        assert reply_for(OTHER, "আলহামদুলিল্লাহ", []) == cs._FALLBACK_REPLY


def test_respond_treats_unmatched_text_as_other():
    with patch.object(cs, "complete", return_value=_completion("জি।")) as complete:
        assert respond("আজকের আবহাওয়াটা সুন্দর", []) == "জি।"
    assert complete.call_args.args[1][-1]["content"] == "আজকের আবহাওয়াটা সুন্দর"


# --------------------------------------------------------------------------
# intent classification
# --------------------------------------------------------------------------


def test_chitchat_is_a_valid_intent_without_substring_clashes():
    assert "CHITCHAT" in ic._VALID_INTENTS
    for a in ic._VALID_INTENTS:
        for b in ic._VALID_INTENTS:
            if a != b:
                assert a not in b, f"{a!r} is a substring of {b!r}"


@pytest.mark.parametrize("text", [t for t, _ in _PURE])
def test_pure_small_talk_routes_to_chitchat_without_an_llm_call(text):
    with patch.object(ic, "complete") as complete:
        assert ic.classify_intent(text, False) == "CHITCHAT"
    complete.assert_not_called()


@pytest.mark.parametrize("text", ["hi, নামাজের নিয়ম কী?", "assalamualaikum, zakat koto taka?"])
def test_greeting_plus_question_routes_to_qa_without_an_llm_call(text):
    with patch.object(ic, "complete") as complete:
        assert ic.classify_intent(text, False) == "QA"
    complete.assert_not_called()


def test_organisation_question_falls_through_to_the_normal_classifier():
    with patch.object(ic, "complete", return_value=_completion("QA")) as complete:
        assert ic.classify_intent("শিবির কে?", False) == "QA"
    complete.assert_called()  # not short-circuited as small talk


def test_llm_classifier_can_return_chitchat():
    with patch.object(ic, "complete", return_value=_completion("CHITCHAT")):
        assert ic._classify_with_llm("আজ খুব গরম") == "CHITCHAT"
    with patch.object(ic, "complete", return_value=_completion("QA")):
        assert ic._classify_with_llm("যাকাতের নিসাব কত?") == "QA"


def test_classifier_prompts_list_chitchat_and_keep_the_json_format():
    assert "CHITCHAT -" in ic._CLASSIFIER_SYSTEM
    assert "CHITCHAT -" in ic._COMBINED_SYSTEM
    assert '"intent": "NOTE|ROLEPLAY|SUGGESTION|CHITCHAT|QA", "rewritten_query"' in ic._COMBINED_SYSTEM


def test_active_roleplay_session_is_not_interrupted_by_a_greeting():
    assert ic.classify_intent("hi", True) == "ROLEPLAY"


def test_existing_intents_still_win_over_small_talk():
    with patch.object(ic, "complete"):
        assert ic.classify_intent("যাকাতের নোট দাও", False) == "NOTE"


# --------------------------------------------------------------------------
# QA path (shared by /chat and /chat/stream)
# --------------------------------------------------------------------------


def test_is_conversational_is_still_importable_and_means_pure_small_talk():
    from app.services.qa_service import _is_conversational

    assert _is_conversational("assalamualaikum")
    assert not _is_conversational("hi, নামাজের নিয়ম কী?")
    assert not _is_conversational("নামাজের নিয়ম কী?")


def test_preface_contract():
    with patch.object(cs, "complete", return_value=_completion("Hi!")):
        assert preface("নামাজের নিয়ম কী?") == (None, "নামাজের নিয়ম কী?")
        greeting, question = preface("hi, নামাজের নিয়ম কী?")
        assert question == "নামাজের নিয়ম কী?"
        assert greeting == "আসসালামু আলাইকুম।"
        pure, empty = preface("assalamualaikum")
        assert empty == "" and pure.startswith("ওয়া আলাইকুমুস সালাম")
        # An English question gets no Bengali greeting in front of it.
        assert preface("hi, what is zakat?") == (None, "what is zakat?")


def test_mixed_message_retrieves_once_on_the_question_and_prefixes_the_answer():
    with (
        patch.object(qa_service, "retrieve_relevant_docs", return_value=_CITATIONS) as retrieve,
        patch.object(qa_service, "generate_answer", return_value="উত্তর।") as generate,
    ):
        response = qa_service.answer_question("assalamualaikum, zakat koto taka?")

    retrieve.assert_called_once_with("zakat koto taka?")  # the greeting is not searched
    generate.assert_called_once_with("zakat koto taka?", _CITATIONS)
    assert response.answer == f"{cs._SALAM_REPLY}\n\nউত্তর।"
    assert response.sources == _CITATIONS  # sources come only from the RAG part


def test_mixed_message_below_the_gate_has_no_sources():
    weak = [_CITATIONS[0].model_copy(update={"rerank_score": 0.01})]
    with (
        patch.object(qa_service, "retrieve_relevant_docs", return_value=weak),
        patch.object(qa_service, "generate_answer", return_value="বইয়ে নেই।") as generate,
    ):
        response = qa_service.answer_question("hi, নামাজের নিয়ম কী?")
    generate.assert_called_once_with("নামাজের নিয়ম কী?", [])
    assert response.sources == []
    assert response.answer.endswith("\n\nবইয়ে নেই।")


@pytest.mark.parametrize("text", ["assalamualaikum", "ধন্যবাদ", "kemon achen", "allah hafez"])
def test_pure_small_talk_makes_no_retrieval_or_rag_call(text):
    with (
        patch.object(qa_service, "retrieve_relevant_docs") as retrieve,
        patch.object(qa_service, "generate_answer") as generate,
        patch.object(cs, "complete") as complete,
    ):
        response = qa_service.answer_question(text)
    retrieve.assert_not_called()
    generate.assert_not_called()
    complete.assert_not_called()
    assert response.sources == [] and response.answer


def test_plain_question_is_untouched_by_the_split():
    with (
        patch.object(qa_service, "retrieve_relevant_docs", return_value=_CITATIONS) as retrieve,
        patch.object(qa_service, "generate_answer", return_value="উত্তর।") as generate,
    ):
        response = qa_service.answer_question("নামাজের নিয়ম কী?")
    retrieve.assert_called_once_with("নামাজের নিয়ম কী?")
    generate.assert_called_once_with("নামাজের নিয়ম কী?", _CITATIONS)
    assert response.answer == "উত্তর।"


# --------------------------------------------------------------------------
# HTTP: /chat and /chat/stream
# --------------------------------------------------------------------------


def test_chat_pure_small_talk_has_mode_chitchat_and_no_retrieval():
    with (
        patch("app.services.qa_service.retrieve_relevant_docs") as retrieve,
        patch("app.api.router.retrieve_relevant_docs") as retrieve_router,
        patch.object(cs, "complete") as complete,
    ):
        response = client.post("/chat", json={"message": "আসসালামু আলাইকুম"})

    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "chitchat"
    assert body["sources"] == []
    assert body["answer"].startswith("ওয়া আলাইকুমুস সালাম")
    retrieve.assert_not_called()
    retrieve_router.assert_not_called()
    complete.assert_not_called()


def test_chat_identity_question_uses_the_llm_and_sees_the_session_history():
    with patch.object(cs, "complete", return_value=_completion("আমি একটি এআই সহকারী।")) as complete:
        first = client.post("/chat", json={"message": "আসসালামু আলাইকুম"}).json()
        second = client.post(
            "/chat", json={"message": "তুমি কে?", "session_id": first["session_id"]}
        ).json()

    assert second["mode"] == "chitchat" and second["answer"] == "আমি একটি এআই সহকারী।"
    messages = complete.call_args.args[1]
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
    assert messages[1]["content"] == "আসসালামু আলাইকুম"


def test_chat_mixed_message_is_qa_with_greeting_and_rag_sources():
    with (
        patch("app.services.qa_service.retrieve_relevant_docs", return_value=_CITATIONS) as retrieve,
        patch("app.services.qa_service.generate_answer", return_value="উত্তর।"),
    ):
        response = client.post("/chat", json={"message": "hi, নামাজের নিয়ম কী?"})

    body = response.json()
    assert body["mode"] == "qa"
    assert body["answer"] == "আসসালামু আলাইকুম।\n\nউত্তর।"
    assert len(body["sources"]) == 1
    retrieve.assert_called_once_with("নামাজের নিয়ম কী?")


def test_chat_chitchat_llm_failure_is_a_503_like_other_modes():
    from app.api.router import _LLM_UNAVAILABLE_DETAIL

    with patch.object(cs, "complete", side_effect=APIError("boom", request=MagicMock(), body=None)):
        response = client.post("/chat", json={"message": "who are you"})
    assert response.status_code == 503
    assert response.json()["detail"] == _LLM_UNAVAILABLE_DETAIL


def test_stream_pure_small_talk_is_one_token_and_no_retrieval():
    with (
        patch("app.api.router.retrieve_relevant_docs") as retrieve,
        patch("app.api.router.stream_answer") as stream,
    ):
        r = client.post("/chat/stream", json={"message": "assalamualaikum"})

    events = _events(r.text)
    assert [e for e, _ in events] == ["sources", "token", "done"]
    assert events[0][1]["sources"] == []
    assert events[1][1]["text"].startswith("ওয়া আলাইকুমুস সালাম")
    assert events[-1][1]["mode"] == "chitchat"
    retrieve.assert_not_called()
    stream.assert_not_called()


def test_stream_qa_path_answers_pure_small_talk_without_retrieval():
    # classify_intent forced to QA: _qa_stream's own split must still catch it.
    with (
        patch("app.api.router.classify_intent", return_value="QA"),
        patch("app.api.router.retrieve_relevant_docs") as retrieve,
        patch("app.api.router.stream_answer") as stream,
    ):
        r = client.post("/chat/stream", json={"message": "ধন্যবাদ"})

    events = _events(r.text)
    assert [e for e, _ in events] == ["sources", "token", "done"]
    assert events[1][1]["text"] in cs._REPLIES[THANKS]
    assert events[-1][1]["mode"] == "qa"
    retrieve.assert_not_called()
    stream.assert_not_called()


def test_stream_chitchat_intent_emits_a_single_token():
    with patch.object(cs, "complete", return_value=_completion("আমি একটি এআই সহকারী।")):
        r = client.post("/chat/stream", json={"message": "তুমি কে?"})
    events = _events(r.text)
    assert [e for e, _ in events] == ["sources", "token", "done"]
    assert events[1][1]["text"] == "আমি একটি এআই সহকারী।"
    assert events[-1][1]["mode"] == "chitchat"


def test_stream_mixed_message_yields_the_greeting_first_then_the_answer():
    seen = {}

    def fake_stream(question, citations, **kwargs):
        seen["question"] = question
        return iter(["উ", "ত্ত", "র।"])

    with (
        patch("app.api.router.retrieve_relevant_docs", return_value=_CITATIONS) as retrieve,
        patch("app.api.router.stream_answer", side_effect=fake_stream),
    ):
        r = client.post("/chat/stream", json={"message": "hi, নামাজের নিয়ম কী?"})

    events = _events(r.text)
    kinds = [e for e, _ in events]
    assert kinds[0] == "sources" and kinds[-1] == "done"
    tokens = [d["text"] for e, d in events if e == "token"]
    assert tokens[0] == "আসসালামু আলাইকুম।\n\n"  # the greeting is the first delta
    assert "".join(tokens) == "আসসালামু আলাইকুম।\n\nউত্তর।"  # same text as /chat
    assert len(events[0][1]["sources"]) == 1  # sources come from the RAG part only
    retrieve.assert_called_once_with("নামাজের নিয়ম কী?")
    assert seen["question"] == "নামাজের নিয়ম কী?"


def test_stream_chitchat_llm_failure_emits_an_error_event():
    from app.api.router import _LLM_UNAVAILABLE_DETAIL

    with patch.object(cs, "complete", side_effect=APIError("boom", request=MagicMock(), body=None)):
        r = client.post("/chat/stream", json={"message": "who are you"})
    events = _events(r.text)
    assert [e for e, _ in events] == ["error"]
    assert events[0][1]["detail"] == _LLM_UNAVAILABLE_DETAIL


# --------------------------------------------------------------------------
# generator prompt
# --------------------------------------------------------------------------


def test_generator_prompt_is_conversational_but_keeps_the_grounding_rules():
    from app.rag.generator import _SYSTEM, ANSWER_LANGUAGE_RULE

    assert "শিক্ষক" in _SYSTEM
    assert "৫। সরাসরি উত্তর দিয়ে শুরু করো।" in _SYSTEM
    assert f"৪। {ANSWER_LANGUAGE_RULE}" in _SYSTEM  # rule 4 verbatim
    assert "আংশিক তথ্য থাকলেও সেটুকু দিয়েই উত্তর দাও" in _SYSTEM  # rule 2
    assert "বন্ধুত্বপূর্ণ বাংলা প্রশ্নোত্তর সহকারী" not in _SYSTEM
