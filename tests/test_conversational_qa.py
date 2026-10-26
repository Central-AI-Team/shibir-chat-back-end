"""Conversational QA (docs/roadmap-q4-2026.md §1): the skip rule, condensation
in the intent call and its fallback, carry-over of cited pages into the
rerank pool, the per-citation gate, the single QA path behind /chat and
/chat/stream, and the generator's history / original-message / style_rules
keywords. Every model and network boundary is mocked."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.core.config import settings
from app.db.models import Message
from app.main import app
from app.rag import generator, query_rewriter as qr, retriever
from app.schemas.query import Citation
from app.services import intent_classifier as ic, qa_service, session_store

HISTORY = [
    {"role": "user", "content": "যাকাত কী?"},
    {
        "role": "assistant",
        "content": "যাকাত হলো সম্পদের একটি নির্দিষ্ট অংশ... " * 20,
        "sources": [
            {"book": "ইসলামের অর্থনীতি", "chapter": "যাকাত", "row_key": "page_9", "content": "x"},
            {"book": "ইসলামের অর্থনীতি", "chapter": "যাকাত", "row_key": "page_9", "content": "y"},
            {"book": "তাফহীমুল কুরআন", "chapter": "সূরা বাকারা", "row_key": "page_42", "content": "z"},
        ],
    },
]


def _resp(text: str):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


def _json(intent="QA", rewritten="আরো বিস্তারিত বলো", standalone="যাকাত সম্পর্কে আরো বিস্তারিত বলো"):
    return _resp(json.dumps(
        {"intent": intent, "rewritten_query": rewritten, "standalone_query": standalone},
        ensure_ascii=False,
    ))


def _cit(score, row_key="page_1", content="অংশ"):
    return Citation(
        book="বই", chapter="অধ্যায়", source_db="tarun", content=content,
        similarity=0.6, rerank_score=score, row_key=row_key,
    )


@pytest.fixture(autouse=True)
def _clean_primed():
    ic._primed_standalone.clear()
    qr._reset_fallback_count()
    yield
    ic._primed_standalone.clear()


# ---------------------------------------------------------------------------
# (b) skip rule
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("message, history, expected", [
    ("আরো বলো", None, True),                                   # no history: nothing to condense
    ("আরো বলো", HISTORY, False),                               # short + anaphor
    ("কেন", HISTORY, False),                                   # anaphor-only Bengali: must NOT skip
    ("নামাজের ফরজ সমূহ কয়টি এবং কী কী বলুন", HISTORY, True),  # long, self-contained
    ("নামাজের ফরজ সমূহ কয়টি এবং এটা কোথায় আছে", HISTORY, False),  # long but anaphoric
    ("namaz er niyom ki", HISTORY, False),                     # Banglish always goes to the LLM
    ("namaz er niyom ki", None, False),
])
def test_skip_rule(message, history, expected):
    assert ic.should_skip_condense(message, history) is expected


def test_min_words_comes_from_settings(monkeypatch):
    msg = "নামাজের ফরজ কয়টি"  # 3 words, no anaphor
    assert ic.should_skip_condense(msg, HISTORY) is False
    monkeypatch.setattr(settings, "condense_min_words", 3)
    assert ic.should_skip_condense(msg, HISTORY) is True


def test_skippable_bengali_follow_up_makes_no_llm_call():
    with patch.object(ic, "complete") as complete:
        with patch.object(ic, "_classify_with_llm", return_value="QA"):
            assert ic.classify_intent("নামাজের ফরজ সমূহ কয়টি এবং কী কী বলুন", False, HISTORY) == "QA"
    complete.assert_not_called()
    assert ic.condense("নামাজের ফরজ সমূহ কয়টি এবং কী কী বলুন", HISTORY) == "নামাজের ফরজ সমূহ কয়টি এবং কী কী বলুন"


def test_anaphoric_follow_up_is_condensed_in_one_call_with_history_and_sources():
    with patch.object(ic, "complete", return_value=_json()) as complete:
        assert ic.classify_intent("আরো বিস্তারিত বলো", False, HISTORY) == "QA"
    assert complete.call_count == 1
    system, user = complete.call_args.args[1]
    assert "standalone_query" in system["content"] and ic._CONDENSE_RULES in system["content"]
    body = user["content"]
    assert body.startswith("পূর্বের কথোপকথন (সর্বশেষ আগে):\n[assistant] (উৎস: ইসলামের অর্থনীতি › যাকাত; তাফহীমুল কুরআন › সূরা বাকারা) ")
    assert "[user]      যাকাত কী?" in body
    assert body.rstrip().endswith('JSON: {"intent": ..., "rewritten_query": ..., "standalone_query": ...}')
    assert "নতুন বার্তা: আরো বিস্তারিত বলো" in body
    # the assistant excerpt is cut to 300 chars
    first_line = body.splitlines()[1]
    assert len(first_line) < 300 + 120
    assert ic.condense("আরো বিস্তারিত বলো", HISTORY) == "যাকাত সম্পর্কে আরো বিস্তারিত বলো"
    assert ic.condense("আরো বিস্তারিত বলো", HISTORY) == "আরো বিস্তারিত বলো"  # consumed


def test_prompt_without_history_is_unchanged():
    with patch.object(ic, "complete", return_value=_json()) as complete:
        ic.classify_intent("zakat ki", False, None)
    system, user = complete.call_args.args[1]
    assert system["content"] == ic._COMBINED_SYSTEM
    assert "standalone_query" not in system["content"]
    assert user["content"] == "zakat ki"


# ---------------------------------------------------------------------------
# (a) condensation fallback
# ---------------------------------------------------------------------------


def test_non_bengali_standalone_is_retried_then_falls_back(caplog):
    bad = _json(standalone="tell me more about zakat")
    with patch.object(ic, "complete", side_effect=[bad, bad]) as complete:
        with caplog.at_level("WARNING"):
            assert ic.classify_intent("আরো বলো", False, HISTORY) == "QA"
    assert complete.call_count == 2  # one retry
    assert qr.get_fallback_count() == 1
    assert any("condensation did not produce Bengali" in r.message for r in caplog.records)
    assert ic.condense("আরো বলো", HISTORY) == "আরো বলো"  # raw message


def test_retry_that_produces_bengali_is_used():
    with patch.object(ic, "complete", side_effect=[_json(standalone="more please"), _json()]) as complete:
        ic.classify_intent("আরো বলো", False, HISTORY)
    assert complete.call_count == 2
    assert qr.get_fallback_count() == 0
    assert ic.condense("আরো বলো", HISTORY) == "যাকাত সম্পর্কে আরো বিস্তারিত বলো"


# ---------------------------------------------------------------------------
# (c) carry-over keys reach the rerank pool
# ---------------------------------------------------------------------------


class _Collection:
    """rows: (id, doc, meta, vec). Supports query() and get(ids= / where=)."""

    def __init__(self, rows):
        self.rows = rows

    def query(self, query_embeddings, n_results, include=None):
        docs, metas, dists = [], [], []
        for q in query_embeddings:
            q = np.asarray(q, dtype=float)
            scored = sorted(
                ((1.0 - float(q @ v / (np.linalg.norm(q) * np.linalg.norm(v))), d, m) for _, d, m, v in self.rows),
                key=lambda t: t[0],
            )[:n_results]
            dists.append([s[0] for s in scored]); docs.append([s[1] for s in scored]); metas.append([s[2] for s in scored])
        return {"documents": docs, "metadatas": metas, "distances": dists}

    def get(self, ids=None, where=None, include=None):
        if where is not None:
            want = where["row_key"]
            want = want["$in"] if isinstance(want, dict) else [want]
            rows = [r for r in self.rows if r[2]["row_key"] in want]
        else:
            rows = [r for r in self.rows if r[0] in ids]
        return {
            "ids": [r[0] for r in rows], "documents": [r[1] for r in rows],
            "metadatas": [r[2] for r in rows], "embeddings": np.array([r[3] for r in rows]),
        }


def _row(row_key, chunk, doc, vec):
    return (f"{row_key}_c{chunk}", doc, {"row_key": row_key, "chunk_index": chunk, "book": "বই",
                                          "chapter": "অ", "source_db": "tarun"}, np.array(vec, dtype=float))


@pytest.fixture
def pool(monkeypatch):
    rows = [
        _row("page_1", 0, "near", [1.0, 0.0]),
        _row("page_2", 0, "also near", [0.9, 0.1]),
        _row("page_9", 0, "far chunk a", [0.0, 1.0]),   # orthogonal: never a dense hit
        _row("page_9", 1, "far chunk b", [0.0, 1.0]),
    ]
    collection = _Collection(rows)
    seen = {}
    monkeypatch.setattr(retriever, "get_collection", lambda: collection)
    monkeypatch.setattr(retriever, "expand_query", lambda q, max_variants=None: (q,))
    monkeypatch.setattr(retriever, "embed_queries", lambda qs: [[1.0, 0.0] for _ in qs])
    monkeypatch.setattr(settings, "hybrid_enabled", False)

    def fake_rerank(query, docs, top_n):
        seen["docs"] = list(docs)
        return [(i, 0.9) for i in range(len(docs))][:top_n]

    monkeypatch.setattr(retriever, "rerank", fake_rerank)
    return seen


def test_without_extra_keys_the_pool_is_unchanged(pool):
    retriever.retrieve_stages("q")
    assert pool["docs"] == ["near", "also near"]


def test_carry_over_pages_are_added_to_the_rerank_pool_and_deduped(pool):
    candidates, final = retriever.retrieve_stages("q", extra_row_keys=["page_9", "page_1", "page_9"])
    assert pool["docs"] == ["near", "also near", "far chunk a", "far chunk b"]  # appended, page_1 not doubled
    assert [c.row_key for c in candidates] == ["page_1", "page_2", "page_9", "page_9"]
    cits = retriever.retrieve_relevant_docs("q", extra_row_keys=["page_9"])
    assert {c.row_key for c in cits} >= {"page_9"}
    assert all(c.row_key for c in cits)  # Citation.row_key is filled


def test_unknown_carry_over_key_is_ignored(pool):
    retriever.retrieve_stages("q", extra_row_keys=["page_404"])
    assert pool["docs"] == ["near", "also near"]


def test_previous_row_keys_come_from_the_last_assistant_turn():
    assert qa_service.previous_row_keys(HISTORY) == ["page_9", "page_42"]
    assert qa_service.previous_row_keys(HISTORY[:1]) == []  # last turn is the user's
    assert qa_service.previous_row_keys(None) == []
    assert qa_service.previous_row_keys([{"role": "assistant", "content": "x", "sources": [{"book": "b"}]}]) == []


def test_qa_passes_carry_over_and_standalone_to_the_retriever():
    ic._prime_standalone("এটা কোন বইয়ে আছে?", "যাকাত কোন বইয়ে আলোচিত হয়েছে?")
    with (
        patch("app.services.qa_service.retrieve_relevant_docs", return_value=[_cit(0.9)]) as retrieve,
        patch("app.services.qa_service.generate_answer", return_value="উত্তর।") as generate,
    ):
        out = qa_service.answer_question("এটা কোন বইয়ে আছে?", HISTORY)
    retrieve.assert_called_once_with(
        "যাকাত কোন বইয়ে আলোচিত হয়েছে?", extra_row_keys=["page_9", "page_42"],
        standalone_query="যাকাত কোন বইয়ে আলোচিত হয়েছে?",
    )
    assert out.query == "যাকাত কোন বইয়ে আলোচিত হয়েছে?"
    kwargs = generate.call_args.kwargs
    assert generate.call_args.args[0] == "যাকাত কোন বইয়ে আলোচিত হয়েছে?"
    assert kwargs["original_message"] == "এটা কোন বইয়ে আছে?"
    assert kwargs["history"] == HISTORY[-settings.qa_history_turns:]
    assert "style_rules" not in kwargs


# ---------------------------------------------------------------------------
# (e) per-citation gate and the single QA path
# ---------------------------------------------------------------------------


def test_gate_admits_each_citation_individually():
    cits = [_cit(0.9, "a"), _cit(0.3, "b"), _cit(0.7, "c"), _cit(None, "d"), _cit(0.5, "e")]
    assert [c.row_key for c in qa_service.gate(cits)] == ["a", "c", "e"]  # order kept, 0.5 inclusive
    assert qa_service.gate([_cit(0.49)]) == []


def test_answer_question_grounds_on_the_passing_citations_only():
    cits = [_cit(0.9, "a"), _cit(0.3, "b"), _cit(0.7, "c")]
    with (
        patch("app.services.qa_service.retrieve_relevant_docs", return_value=cits),
        patch("app.services.qa_service.generate_answer", return_value="উত্তর।") as generate,
    ):
        out = qa_service.answer_question("যাকাত কী?")
    assert [c.row_key for c in out.sources] == ["a", "c"]
    assert [c.row_key for c in generate.call_args.args[1]] == ["a", "c"]
    with (
        patch("app.services.qa_service.retrieve_relevant_docs", return_value=[_cit(0.2), _cit(0.1)]),
        patch("app.services.qa_service.generate_answer", return_value="নেই।") as generate,
    ):
        out = qa_service.answer_question("যাকাত কী?")
    assert out.sources == [] and generate.call_args.args[1] == []


def _fake_complete(task, messages, stream=False, **kw):
    assert task == "qa"
    if stream:
        return iter(
            SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=t))])
            for t in ["উত্ত", "র।"]
        )
    return _resp("উত্তর।")


def test_chat_and_stream_give_identical_sources_and_answer():
    cits = [_cit(0.9, "a", "ক"), _cit(0.2, "b", "খ"), _cit(0.6, "c", "গ")]
    prompts = []

    def complete(task, messages, stream=False, **kw):
        prompts.append(messages)
        return _fake_complete(task, messages, stream=stream, **kw)

    with (
        patch("app.api.router.classify_intent", return_value="QA"),
        patch("app.services.qa_service.retrieve_relevant_docs", return_value=cits),
        patch("app.rag.generator.complete", side_effect=complete),
    ):
        plain = TestClient(app).post("/chat", json={"message": "যাকাত কী?"}).json()
        raw = TestClient(app).post("/chat/stream", json={"message": "যাকাত কী?"}).text

    events = [(b.split("\n")[0][7:], json.loads(b.split("\ndata: ", 1)[1])) for b in raw.strip().split("\n\n")]
    streamed_sources = events[0][1]["sources"]
    streamed_answer = "".join(d["text"] for e, d in events if e == "token")
    assert plain["sources"] == streamed_sources
    assert [s["row_key"] for s in streamed_sources] == ["a", "c"]
    assert plain["answer"] == streamed_answer == "উত্তর।"
    assert prompts[0] == prompts[1]  # same messages to the model


# ---------------------------------------------------------------------------
# (d) generator prompt
# ---------------------------------------------------------------------------


def test_system_prompt_is_byte_identical_without_style_rules():
    msgs = generator.build_messages("প্রশ্ন", [])
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert msgs[0]["content"] == generator._SYSTEM
    assert "৬।" in generator._SYSTEM  # rule 6 is part of the base prompt
    assert "ব্যবহারকারীর বার্তা" not in msgs[1]["content"]


def test_style_rules_are_appended_only_when_given():
    msgs = generator.build_messages("প্রশ্ন", [], style_rules="সংক্ষেপে লেখো।")
    assert msgs[0]["content"] == generator._SYSTEM + "\n\nসংক্ষেপে লেখো।"
    assert generator.build_messages("প্রশ্ন", [], style_rules="")[0]["content"] == generator._SYSTEM


def test_history_is_inserted_between_system_and_user_content_only(monkeypatch):
    monkeypatch.setattr(settings, "qa_history_turns", 2)
    history = [{"role": "user", "content": "১"}, {"role": "assistant", "content": "২", "sources": [{"book": "b"}]},
               {"role": "user", "content": "৩"}, {"role": "assistant", "content": "৪"}, {"role": "system", "content": "x"}]
    msgs = generator.build_messages("প্রশ্ন", [], history=history, original_message="এটা?")
    assert msgs[1:-1] == [{"role": "user", "content": "৩"}, {"role": "assistant", "content": "৪"}]
    assert msgs[-1]["content"].count("ব্যবহারকারীর বার্তা: এটা?\n\nপ্রশ্ন: প্রশ্ন") == 1
    # same wording as the question -> no extra line
    assert "ব্যবহারকারীর বার্তা" not in generator.build_messages("প্রশ্ন", [], original_message="প্রশ্ন")[-1]["content"]


def test_stream_answer_uses_the_same_messages_as_generate_answer():
    seen = []

    def complete(task, messages, stream=False, **kw):
        seen.append(messages)
        return _fake_complete(task, messages, stream=stream)

    with patch("app.rag.generator.complete", side_effect=complete):
        generator.generate_answer("প্রশ্ন", [_cit(0.9)], history=HISTORY, original_message="এটা?")
        list(generator.stream_answer("প্রশ্ন", [_cit(0.9)], history=HISTORY, original_message="এটা?"))
    assert seen[0] == seen[1]


# ---------------------------------------------------------------------------
# store: standalone_query on the user turn
# ---------------------------------------------------------------------------


def test_standalone_query_is_stored_on_the_user_turn():
    client = TestClient(app)
    with (
        patch("app.api.router.classify_intent", return_value="QA"),
        patch("app.services.qa_service.condense", return_value="যাকাত কোন বইয়ে আলোচিত হয়েছে?"),
        patch("app.services.qa_service.retrieve_relevant_docs", return_value=[_cit(0.9)]),
        patch("app.services.qa_service.generate_answer", return_value="উত্তর।"),
    ):
        sid = client.post("/chat", json={"message": "এটা কোন বইয়ে আছে?"}).json()["session_id"]
    with session_store._open() as db:
        rows = db.execute(
            select(Message.role, Message.standalone_query).where(Message.conversation_id == sid).order_by(Message.id)
        ).all()
    assert rows == [("user", "যাকাত কোন বইয়ে আলোচিত হয়েছে?"), ("assistant", None)]
    # the standalone is not part of the history handed back to callers
    assert all("standalone_query" not in t for t in session_store.get_history(sid, _owner(client)))


def _owner(client):
    from app.api.deps import COOKIE_NAME, hash_token

    return hash_token(client.cookies[COOKIE_NAME])
