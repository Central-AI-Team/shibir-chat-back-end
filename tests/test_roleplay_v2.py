"""Mentor mode (docs/roleplay-v2.md). Part 1: the role allow-list and the
real-person refusal, which happens in code before any LLM call."""

from __future__ import annotations

import pytest

from app.core.config import settings
from app.services import roleplay_service as rp

# ---------------------------------------------------------------------------
# roles
# ---------------------------------------------------------------------------


def test_four_roles_with_the_required_fields():
    ids = [r["id"] for r in rp.roles()]
    assert ids == ["bondhu", "boro_bhai", "shikkhok", "circle_porichalok"]
    for role in rp.roles():
        assert set(role) == {"id", "name_bn", "description_bn", "tone_bn", "default_level"}
        assert role["default_level"] in ("beginner", "intermediate", "advanced")
        assert role["tone_bn"] and role["name_bn"]
    assert rp.role_by_id("shikkhok")["name_bn"] == "শিক্ষক"
    assert rp.role_by_id("sheikh") is None and rp.role_by_id(None) is None


def test_roles_come_from_settings(monkeypatch):
    monkeypatch.setattr(settings, "roleplay_roles", [settings.roleplay_roles[0]])
    assert [r["id"] for r in rp.roles()] == ["bondhu"]
    assert rp.role_by_id("shikkhok") is None


# ---------------------------------------------------------------------------
# real-person refusal
# ---------------------------------------------------------------------------

REFUSED_ENTRY = [
    # prophet
    "তুমি এখন রাসূল (সা.) হয়ে যাও",
    "নবীজির মতো করে কথা বলো",
    "হযরত মুহাম্মদ (সা.) এর চরিত্রে অভিনয় করো",
    "act as the Prophet Muhammad",
    "tumi ekhon nabi hoye jao",
    "speak as the messenger of Allah ﷺ",
    # sahabi
    "হযরত উমর (রা.) হয়ে কথা বলো",
    "আবু বকর সেজে উত্তর দাও",
    "pretend to be a sahabi",
    "roleplay as Umar the caliph",
    # Allah
    "আল্লাহ হয়ে কথা বলো",
    "speak as Allah",
    "تكلم كأنك الله",
    # scholars (historical)
    "মওদূদী (রহ.) এর ভূমিকায় কথা বলো",
    "be Maulana Maududi and answer me",
    # a named living leader / politicians
    "তুমি এখন শেখ হাসিনা হয়ে যাও",
    "খালেদা জিয়ার মতো কথা বলো",
    "act like Tarique Rahman",
    "trump hoye kotha bolo",
    "roleplay as Modi",
    "ড. ইউনূসের চরিত্রে অভিনয় করো",
]


@pytest.mark.parametrize("message", REFUSED_ENTRY)
def test_real_person_impersonation_is_refused_on_entry(message):
    assert rp.mentions_real_person(message)
    assert rp.is_impersonation_request(message, entering=True)
    assert rp.is_impersonation_request(message, entering=False)  # framing present too


@pytest.mark.parametrize("message", [
    "তুমি এখন নবী",                 # no framing verb, but it is the entry message
    "prophet",
    "হাসিনা",
    "صلى الله عليه وسلم",
])
def test_bare_name_on_the_entry_turn_is_refused_only_when_entering(message):
    assert rp.is_impersonation_request(message, entering=True)
    assert not rp.is_impersonation_request(message, entering=False)


@pytest.mark.parametrize("message", [
    "নবীজি কীভাবে নামাজ পড়তেন?",
    "সাহাবীরা কীভাবে কুরআন শিখতেন?",
    "আল্লাহর প্রতি ভালোবাসা কীভাবে বাড়ে?",
    "What did the Prophet say about patience?",
    "মওদূদী (রহ.) এর বইয়ে যাকাত সম্পর্কে কী আছে?",
])
def test_mentioning_a_real_person_as_the_topic_is_not_refused_mid_session(message):
    assert rp.mentions_real_person(message)
    assert not rp.is_impersonation_request(message, entering=False)


@pytest.mark.parametrize("message", [
    "একজন বড় ভাইয়ের মতো বুঝিয়ে বলো",
    "তুমি এখন আমার শিক্ষক হয়ে যাও",
    "bondhu hoye kotha bolo",
    "act as a study circle leader",
    "যাকাত কাদের উপর ফরজ?",
])
def test_roles_and_plain_questions_are_not_real_people(message):
    assert not rp.mentions_real_person(message)
    assert not rp.is_impersonation_request(message, entering=True)


def test_refusal_text_is_bengali_and_lists_every_role():
    text = rp.refusal_text()
    assert text.startswith("দুঃখিত")
    for role in rp.roles():
        assert role["name_bn"] in text and role["description_bn"] in text
    assert "AI" not in text  # no "I am an AI / not an AI" framing either way


# ---------------------------------------------------------------------------
# Part 2: intent-call contract on roleplay turns, role resolution, level
# ---------------------------------------------------------------------------

import json  # noqa: E402
from types import SimpleNamespace  # noqa: E402
from unittest.mock import patch  # noqa: E402

from app.services import intent_classifier as ic  # noqa: E402


def _resp(payload) -> SimpleNamespace:
    text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


def _meta(**extra) -> dict:
    return {"intent": "ROLEPLAY", "rewritten_query": "বন্ধুর মতো বুঝিয়ে বলো",
            "role_id": "bondhu", "persona_ok": True, "level": "beginner", **extra}


@pytest.fixture(autouse=True)
def _clean_primed():
    ic._primed_roleplay.clear()
    ic._primed_standalone.clear()
    yield
    ic._primed_roleplay.clear()
    ic._primed_standalone.clear()


def test_entry_turn_runs_one_combined_call_with_the_roleplay_fields():
    with patch.object(ic, "complete", return_value=_resp(_meta())) as complete:
        assert ic.classify_intent("একজন বন্ধুর মতো বুঝিয়ে বলো", False) == "ROLEPLAY"
    assert complete.call_count == 1
    system = complete.call_args.args[1][0]["content"]
    assert '"role_id"' in system and '"persona_ok"' in system and '"level"' in system
    for role in rp.roles():  # the allow-list is in the prompt
        assert f"{role['id']}: {role['name_bn']}" in system
    assert "standalone_query" not in system  # no history -> no condensation
    assert ic.roleplay_meta("একজন বন্ধুর মতো বুঝিয়ে বলো") == {
        "role_id": "bondhu", "persona_ok": True, "level": "beginner"}
    assert ic.roleplay_meta("একজন বন্ধুর মতো বুঝিয়ে বলো") is None  # consumed


def test_active_session_turn_also_gets_meta_and_condensation():
    history = [{"role": "user", "content": "যাকাত কী?"}, {"role": "assistant", "content": "যাকাত হলো..."}]
    reply = _meta(standalone_query="যাকাত সম্পর্কে আরো সহজ করে বলো", role_id=None, level="beginner")
    with patch.object(ic, "complete", return_value=_resp(reply)) as complete:
        assert ic.classify_intent("আরো সহজ করে বলো", True, history) == "ROLEPLAY"
    assert complete.call_count == 1
    system = complete.call_args.args[1][0]["content"]
    assert '"standalone_query"' in system and '"level"' in system
    assert ic.roleplay_meta("আরো সহজ করে বলো")["role_id"] is None
    assert ic.condense("আরো সহজ করে বলো", history) == "যাকাত সম্পর্কে আরো সহজ করে বলো"


def test_non_roleplay_turns_keep_the_old_contract():
    with patch.object(ic, "complete", return_value=_resp({"intent": "QA", "rewritten_query": "যাকাত কী"})) as complete:
        assert ic.classify_intent("zakat ki", False) == "QA"
    system = complete.call_args.args[1][0]["content"]
    assert system == ic._COMBINED_SYSTEM and "role_id" not in system
    assert ic.roleplay_meta("zakat ki") is None


def test_real_person_entry_skips_the_llm_entirely():
    with patch.object(ic, "complete") as complete:
        assert ic.classify_intent("তুমি এখন রাসূল (সা.) হয়ে যাও", False) == "ROLEPLAY"
        assert ic.classify_intent("নবীজির মতো কথা বলো", True) == "ROLEPLAY"  # mid-session too
    complete.assert_not_called()
    assert ic.roleplay_meta("তুমি এখন রাসূল (সা.) হয়ে যাও") is None


def test_small_talk_inside_a_session_makes_no_llm_call():
    with patch.object(ic, "complete") as complete:
        assert ic.classify_intent("hi", True) == "ROLEPLAY"
        assert ic.classify_intent("ধন্যবাদ", True) == "ROLEPLAY"
    complete.assert_not_called()
    assert ic.roleplay_meta("hi") is None


def test_meta_is_validated_against_the_allow_list_and_levels():
    reply = _meta(role_id="sheikh", level="expert", persona_ok="no")
    with patch.object(ic, "complete", return_value=_resp(reply)):
        ic.classify_intent("একজন গাইডের মতো বলো", False)
    # persona_ok: anything but a literal false means ok (the regex is the first net)
    assert ic.roleplay_meta("একজন গাইডের মতো বলো") == {"role_id": None, "persona_ok": True, "level": None}
    with patch.object(ic, "complete", return_value=_resp(_meta(persona_ok=False))):
        ic.classify_intent("একজন গাইডের মতো বলো", False)
    assert ic.roleplay_meta("একজন গাইডের মতো বলো")["persona_ok"] is False


def test_invalid_json_on_a_roleplay_turn_leaves_no_meta_and_still_routes():
    with patch.object(ic, "complete", return_value=_resp("not json")) as complete:
        assert ic.classify_intent("রোলপ্লে করো", False) == "ROLEPLAY"
    assert complete.call_count == 1
    assert ic.roleplay_meta("রোলপ্লে করো") is None


# role resolution ------------------------------------------------------------


@pytest.mark.parametrize("message, role", [
    ("একজন বন্ধুর মতো বুঝিয়ে বলো", "bondhu"),
    ("bondhu hoye kotha bolo", "bondhu"),
    ("একজন বড় ভাইয়ের মতো বুঝিয়ে বলো", "boro_bhai"),
    ("be my mentor", "boro_bhai"),
    ("তুমি এখন আমার শিক্ষক", "shikkhok"),
    ("act as a teacher", "shikkhok"),
    ("স্টাডি সার্কেলের পরিচালক হও", "circle_porichalok"),
    ("run this like a halaqa", "circle_porichalok"),
    ("একজন গাইডের মতো বলো", None),
])
def test_role_keywords(message, role):
    assert rp.match_role(message) == role


def test_unknown_free_text_takes_the_llms_nearest_role_or_asks():
    assert rp.resolve_role("একজন গাইডের মতো বলো", None, {"role_id": "boro_bhai"}) == "boro_bhai"
    assert rp.resolve_role("একজন গাইডের মতো বলো", None, {"role_id": None}) is None  # caller asks
    assert rp.resolve_role("একজন গাইডের মতো বলো", None, None) is None
    assert rp.resolve_role("একজন বন্ধুর মতো বলো", None, {"role_id": "shikkhok"}) == "bondhu"  # keyword wins


def test_current_role_is_kept_unless_the_user_asks_for_another():
    # a question that merely mentions nothing role-like keeps the role
    assert rp.resolve_role("যাকাত কী?", "bondhu", {"role_id": None}) == "bondhu"
    # the LLM guessing a role from a plain question does not switch it
    assert rp.resolve_role("নামাজের নিয়ম শেখাও", "bondhu", {"role_id": "shikkhok"}) == "bondhu"
    # an explicit switch does
    assert rp.resolve_role("এখন শিক্ষকের মতো বলো", "bondhu", None) == "shikkhok"
    assert rp.resolve_role("এখন অন্য কারো মতো কথা বলো", "bondhu", {"role_id": "boro_bhai"}) == "boro_bhai"


# level ---------------------------------------------------------------------


@pytest.mark.parametrize("message, level", [
    ("আরো সহজ করে বলো", "beginner"),
    ("আমি নতুন, বুঝি না", "beginner"),
    ("explain it simpler please", "beginner"),
    ("আরো গভীরে যাও", "advanced"),
    ("bistarito bolo", "advanced"),
    ("go deeper into the texts", "advanced"),
    ("যাকাত কী?", None),
])
def test_explicit_level_cues(message, level):
    assert rp.explicit_level(message) == level


def test_level_majority_over_the_last_three_turns():
    level, window = rp.smooth_level([], "beginner", None, "intermediate")
    assert (level, window) == ("beginner", ["beginner"])
    level, window = rp.smooth_level(window, "advanced", None, "intermediate")
    assert level == "advanced" and window == ["beginner", "advanced"]  # tie -> most recent
    level, window = rp.smooth_level(window, "beginner", None, "intermediate")
    assert level == "beginner" and window == ["beginner", "advanced", "beginner"]
    level, window = rp.smooth_level(window, "advanced", None, "intermediate")
    assert level == "advanced" and window == ["advanced", "beginner", "advanced"]  # oldest dropped
    # nothing inferred this turn: the standing majority holds
    assert rp.smooth_level(window, None, None, "intermediate") == ("advanced", window)
    assert rp.smooth_level([], None, None, "intermediate") == ("intermediate", [])


def test_explicit_cue_overrides_immediately_and_resets_the_window():
    window = ["advanced", "advanced", "advanced"]
    assert rp.smooth_level(window, "advanced", "beginner", "intermediate") == ("beginner", ["beginner"])
    assert rp.smooth_level(["beginner"], "beginner", "advanced", "intermediate") == ("advanced", ["advanced"])


def test_level_window_round_trips_through_the_persona_column():
    assert rp.load_level_window(None) == []
    assert rp.load_level_window("শার্লক হোমস") == []  # a v1 free-form persona is ignored
    assert rp.load_level_window('{"levels": ["beginner", "nope", "advanced"]}') == ["beginner", "advanced"]
    dumped = rp.dump_level_window(["a", "beginner", "advanced", "beginner"][1:])
    assert rp.load_level_window(dumped) == ["beginner", "advanced", "beginner"]


# ---------------------------------------------------------------------------
# Part 3: turns end to end (service and HTTP), style rules, suggestions, exit
# ---------------------------------------------------------------------------

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402
from app.schemas.query import Citation  # noqa: E402
from app.services import session_store  # noqa: E402

_CIT = [Citation(book="বই", chapter="অধ্যায়", source_db="tarun", content="অংশ", similarity=0.7,
                 rerank_score=0.9, row_key="page_1")]


def _session(**over) -> dict:
    base = {"id": "s" * 43, "owner": "o", "mode": "ROLEPLAY", "persona": None,
            "role_id": "bondhu", "level": "beginner", "history": []}
    base.update(over)
    return base


def _qa(sources=_CIT, text=("উত্তর", "।"), standalone="প্রশ্ন"):
    """qa_stream stand-in recording style_rules."""
    seen = {}

    def fake(message, history=None, *, style_rules=None, **kw):
        seen["style_rules"] = style_rules
        seen["message"] = message
        return qa_service_result(sources, text, standalone)

    return fake, seen


def qa_service_result(sources, text, standalone):
    from app.services.qa_service import QAStream

    return QAStream(list(sources), iter(text), standalone)


def test_style_rules_differ_by_level_and_role():
    rules = {(r, lv): rp.compose_style_rules(r, lv) for r in ("bondhu", "shikkhok") for lv in rp.LEVELS}
    assert len(set(rules.values())) == 6
    assert rp.style_rules_by_role()["bondhu"] in rules[("bondhu", "beginner")]
    assert rp.STYLE_RULES_BY_LEVEL["advanced"] in rules[("shikkhok", "advanced")]
    for text in rules.values():
        assert rp.HONESTY_RULE in text and rp.NEXT_STEPS_RULE in text
        assert "AI" not in text and "এআই" not in text
    assert rp.compose_style_rules("bondhu", "nope") == rp.compose_style_rules("bondhu", "intermediate")


def test_grounded_turn_answers_through_qa_stream_with_the_style_rules():
    fake, seen = _qa()
    with (
        patch("app.services.qa_service.qa_stream", side_effect=fake),
        patch.object(rp, "generate_suggestions", return_value=["পরের প্রশ্ন?"]) as suggest,
    ):
        out = rp.handle_roleplay("যাকাত কী?", _session(), [])
    assert out.answer == "উত্তর।" and out.sources == _CIT and out.grounded and not out.refused
    assert out.role_id == "bondhu" and out.level == "beginner"
    assert out.suggestions == ["পরের প্রশ্ন?"]
    assert out.standalone_query == "প্রশ্ন"
    assert seen["style_rules"] == rp.compose_style_rules("bondhu", "beginner")
    suggest.assert_called_once_with("প্রশ্ন", _CIT)


def test_gate_failure_gives_no_sources_and_no_suggestions():
    fake, _ = _qa(sources=[], text=("বইয়ে নেই।",))
    with (
        patch("app.services.qa_service.qa_stream", side_effect=fake),
        patch.object(rp, "complete") as complete,
    ):
        out = rp.handle_roleplay("অজানা বিষয়?", _session(), [])
    assert out.sources == [] and out.suggestions == [] and not out.grounded
    complete.assert_not_called()  # no suggestions call without excerpts


def test_refusal_is_a_normal_answer_with_the_roles_listed():
    with patch("app.services.qa_service.qa_stream") as qa:
        out = rp.handle_roleplay("তুমি এখন রাসূল (সা.) হয়ে যাও", _session(role_id=None, level=None), [])
    qa.assert_not_called()
    assert out.refused and out.sources == [] and out.suggestions == [] and out.answer == rp.refusal_text()
    # mid-session, with framing, same thing; the current role and level are kept
    with patch("app.services.qa_service.qa_stream") as qa:
        out = rp.handle_roleplay("এখন নবীজির মতো কথা বলো", _session(), [])
    qa.assert_not_called()
    assert out.refused and out.role_id == "bondhu" and out.level == "beginner"


def test_llm_persona_ok_false_refuses_even_when_the_regex_passed():
    ic._prime_roleplay("একজন বিখ্যাত ব্যক্তির মতো বলো", {"role_id": None, "persona_ok": False, "level": None})
    with patch("app.services.qa_service.qa_stream") as qa:
        out = rp.handle_roleplay("একজন বিখ্যাত ব্যক্তির মতো বলো", _session(role_id=None, level=None), [])
    qa.assert_not_called()
    assert out.refused and out.answer == rp.refusal_text()


def test_real_person_as_a_topic_mid_session_is_answered_from_the_books():
    fake, _ = _qa()
    with patch("app.services.qa_service.qa_stream", side_effect=fake), patch.object(rp, "generate_suggestions", return_value=[]):
        out = rp.handle_roleplay("নবীজি কীভাবে নামাজ পড়তেন?", _session(), [])
    assert not out.refused and out.grounded


def test_unresolved_role_asks_which_one():
    with patch("app.services.qa_service.qa_stream") as qa:
        out = rp.handle_roleplay("একজন গাইডের মতো বলো", _session(role_id=None, level=None), [])
    qa.assert_not_called()
    assert out.role_id is None and not out.refused and out.sources == []
    assert out.answer == rp.choose_role_text()
    for role in rp.roles():
        assert role["name_bn"] in out.answer


def test_role_only_entry_greets_and_sets_role_and_default_level():
    with patch("app.services.qa_service.qa_stream") as qa:
        out = rp.handle_roleplay("একজন বড় ভাইয়ের মতো বুঝিয়ে বলো", _session(role_id=None, level=None), [])
    qa.assert_not_called()
    assert out.role_id == "boro_bhai" and out.level == "intermediate"  # the role's default
    assert "দায়িত্বশীল ভাই" in out.answer and out.sources == []
    assert rp.load_level_window(out.level_window) == []


def test_level_follows_cues_and_majority_and_is_never_announced():
    fake, seen = _qa()
    with patch("app.services.qa_service.qa_stream", side_effect=fake), patch.object(rp, "generate_suggestions", return_value=[]):
        # explicit cue overrides the stored level at once
        out = rp.handle_roleplay("আরো সহজ করে বলো", _session(level="advanced", persona='{"levels": ["advanced", "advanced"]}'), [])
        assert out.level == "beginner" and rp.load_level_window(out.level_window) == ["beginner"]
        assert "পাঠকের স্তর: নতুন" in seen["style_rules"]
        # inferred levels accumulate; the majority decides
        ic._prime_roleplay("যাকাতের নিসাব কত?", {"role_id": None, "persona_ok": True, "level": "advanced"})
        out = rp.handle_roleplay("যাকাতের নিসাব কত?", _session(level="beginner", persona=out.level_window), [])
        assert out.level == "advanced"  # tie -> most recent
        ic._prime_roleplay("যাকাত কাকে দেওয়া যায়?", {"role_id": None, "persona_ok": True, "level": "beginner"})
        out = rp.handle_roleplay("যাকাত কাকে দেওয়া যায়?", _session(level="advanced", persona=out.level_window), [])
        assert out.level == "beginner" and rp.load_level_window(out.level_window) == ["beginner", "advanced", "beginner"]
    assert "beginner" not in out.answer and "advanced" not in out.answer


def test_suggestions_parse_failure_never_raises():
    bad = [
        SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="not json"))]),
        SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"suggestions": "x"}'))]),
        SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=None))]),
        RuntimeError("api down"),
    ]
    with patch.object(rp, "complete", side_effect=bad):
        for _ in bad:
            assert rp.generate_suggestions("প্রশ্ন", _CIT) == []
    good = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
        content='{"suggestions": [" এক? ", "", 3, "দুই?", "তিন?", "চার?"]}'))])
    with patch.object(rp, "complete", return_value=good) as complete:
        assert rp.generate_suggestions("প্রশ্ন", _CIT) == ["এক?", "দুই?", "তিন?"]
    assert complete.call_args.args[0] == "roleplay_meta"
    assert rp.generate_suggestions("প্রশ্ন", []) == []


def test_suggestion_count_setting(monkeypatch):
    monkeypatch.setattr(settings, "roleplay_suggestion_count", 0)
    with patch.object(rp, "complete") as complete:
        assert rp.generate_suggestions("প্রশ্ন", _CIT) == []
    complete.assert_not_called()


# HTTP -------------------------------------------------------------------------

_META = {"intent": "ROLEPLAY", "rewritten_query": "যাকাত কী", "role_id": None, "persona_ok": True, "level": "beginner"}


def _owner(client):
    from app.api.deps import COOKIE_NAME, hash_token

    return hash_token(client.cookies[COOKIE_NAME])


def _mocked_pipeline():
    def fake_complete(task, messages, stream=False, **kw):
        if task == "qa":
            if stream:
                return iter(SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=t))]) for t in ["উত্ত", "র।"])
            return _resp("উত্তর।")
        if task == "roleplay_meta":
            return _resp({"suggestions": ["পরের প্রশ্ন?", "আরেকটি?"]})
        raise AssertionError(task)

    return (
        patch("app.services.intent_classifier.complete", return_value=_resp(_META)),
        patch("app.services.qa_service.retrieve_relevant_docs", return_value=_CIT),
        patch("app.rag.generator.complete", side_effect=fake_complete),
        patch("app.services.roleplay_service.complete", side_effect=fake_complete),
    )


def _events(raw):
    return [(b.split("\n")[0][7:], json.loads(b.split("\ndata: ", 1)[1])) for b in raw.strip().split("\n\n")]


def test_chat_and_stream_return_identical_answer_sources_and_suggestions():
    a, b = TestClient(app), TestClient(app)
    p1, p2, p3, p4 = _mocked_pipeline()
    with p1, p2, p3, p4:
        for c in (a, b):
            c.post("/chat", json={"message": "একজন বন্ধুর মতো বুঝিয়ে বলো"})
        plain = a.post("/chat", json={"message": "যাকাত কী?", "session_id": a.cookies and _sid(a)}).json()
        raw = b.post("/chat/stream", json={"message": "যাকাত কী?", "session_id": _sid(b)}).text
    events = _events(raw)
    done = events[-1][1]
    assert plain["mode"] == done["mode"] == "roleplay"
    assert plain["answer"] == "".join(d["text"] for e, d in events if e == "token") == "উত্তর।"
    assert plain["sources"] == events[0][1]["sources"] and plain["sources"][0]["book"] == "বই"
    assert plain["suggestions"] == done["suggestions"] == ["পরের প্রশ্ন?", "আরেকটি?"]


def _sid(client) -> str:
    return client.get("/conversations").json()[0]["id"]


def test_role_and_level_are_stored_and_exit_clears_them():
    client = TestClient(app)
    p1, p2, p3, p4 = _mocked_pipeline()
    with p1, p2, p3, p4:
        sid = client.post("/chat", json={"message": "একজন বন্ধুর মতো বুঝিয়ে বলো"}).json()["session_id"]
        session = session_store.get_owned_session(sid, _owner(client))
        assert (session["mode"], session["role_id"], session["level"]) == ("ROLEPLAY", "bondhu", "beginner")
        client.post("/chat", json={"message": "আরো গভীরে যাও", "session_id": sid})
        session = session_store.get_owned_session(sid, _owner(client))
        assert session["level"] == "advanced" and rp.load_level_window(session["persona"]) == ["advanced"]
        # history recorded by the router, with the sources on the assistant turn
        assert [t["role"] for t in session["history"]] == ["user", "assistant", "user", "assistant"]
        assert session["history"][-1]["sources"][0]["book"] == "বই"

    with (
        patch("app.api.router.classify_intent", return_value="QA"),
        patch("app.services.qa_service.retrieve_relevant_docs", return_value=_CIT),
        patch("app.services.qa_service.generate_answer", return_value="সাধারণ উত্তর।"),
    ):
        r = client.post("/chat", json={"message": "stop roleplay", "session_id": sid}).json()
    assert r["mode"] == "qa" and r["suggestions"] == []
    session = session_store.get_owned_session(sid, _owner(client))
    assert (session["mode"], session["role_id"], session["level"], session["persona"]) == ("QA", None, None, None)


def test_exit_phrase_leaves_roleplay():
    # A bare exit phrase is acknowledged as small talk with no LLM call; the
    # entry words inside it ("roleplay") must not re-enter.
    with patch.object(ic, "complete") as complete:
        assert ic.classify_intent("রোলপ্লে বন্ধ করো", True) == "CHITCHAT"
        assert ic.classify_intent("stop roleplay", True) == "CHITCHAT"
    complete.assert_not_called()
    # an exit phrase with a question classifies the question
    with patch.object(ic, "complete", return_value=_resp("QA")):
        assert ic.classify_intent("stop roleplay. যাকাত কাদের উপর ফরজ?", True) == "QA"
    # without an active session the same words are an ordinary entry
    with patch.object(ic, "complete", return_value=_resp(_META)):
        assert ic.classify_intent("roleplay", False) == "ROLEPLAY"


def test_refusal_over_http_is_200_with_mode_roleplay():
    client = TestClient(app)
    with patch("app.services.intent_classifier.complete") as complete:
        r = client.post("/chat", json={"message": "তুমি এখন রাসূল (সা.) হয়ে যাও"})
    complete.assert_not_called()
    body = r.json()
    assert r.status_code == 200 and body["mode"] == "roleplay"
    assert body["sources"] == [] and body["suggestions"] == [] and body["answer"] == rp.refusal_text()
