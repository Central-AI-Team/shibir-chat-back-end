"""The user's Bengali -> simplify -> English failure, including evidence and safety gates.

Tests use real storage/API code but explicit model and reranker fixtures.
"""
import json
import uuid
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas.query import Citation
from app.services import context, grounding, identity, session_store as store

client = TestClient(app)
SEARCH = 'যাকাত কাদের দিতে হয়?'


def completion(content):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def citations():
    return [Citation(book='Fixture book', chapter='যাকাত', source_db='test',
        content='Fixture recipient excerpt.', rerank_score=.99, page='12', url='https://example.com/book#page=12'),
        Citation(book='Fixture book', chapter='যাকাত', source_db='test', content='Second fixture excerpt.', rerank_score=.98)]


def save_turn(owner, sid, question, answer, sources=None, *, complete=True, mode='QA'):
    rid = str(uuid.uuid4())
    lease = store.begin_turn(sid, owner, rid, question)
    store.finish_turn(sid, owner, rid, answer, sources or [], mode, complete=complete, lease=lease)
    return rid


def seeded_chat(owner):
    sid, _ = store.get_or_create_session(None, owner)
    save_turn(owner, sid, 'যাকাত কী?', 'যাকাতের পরিচয়। ' * 120, citations())
    save_turn(owner, sid, 'এটা কাদের দিতে হয়?', 'প্রাপকদের দীর্ঘ আলোচনা। ' * 180, citations())
    return sid


def parse_done(response, endpoint):
    if endpoint == '/chat':
        return response.json()
    assert 'event: error' not in response.text
    return next(json.loads(block.split('data: ', 1)[1]) for block in response.text.strip().split('\n\n')
                if block.startswith('event: done\n'))


@pytest.mark.parametrize('endpoint', ['/chat', '/chat/stream'])
@pytest.mark.parametrize('miss', ['empty', 'below_gate'])
def test_simplify_then_english_keep_real_book_sources_when_fresh_search_misses(endpoint, miss, isolated_chat_store):
    owner = isolated_chat_store['user_id']
    sid = seeded_chat(owner)
    fresh = [] if miss == 'empty' else [citations()[0].model_copy(update={'rerank_score': .01})]
    for message, resolved in [('আগের উত্তরটি সহজ ভাষায় বলো।', 'যাকাতের প্রাপকদের উত্তর সহজ ভাষায় বলো।'),
                              ('Who can receive it?', 'Who can receive zakat?')]:
        resolution = completion(json.dumps({'query': resolved, 'retrieval_query': SEARCH, 'clarification': None}, ensure_ascii=False))
        with patch.object(context, 'complete', return_value=resolution), \
             patch('app.api.router.classify_intent', return_value='QA') as classify, \
             patch('app.services.qa_service.retrieve_relevant_docs', return_value=fresh) as normal_retrieve, \
             patch('app.api.router.retrieve_relevant_docs', return_value=fresh) as stream_retrieve, \
             patch.object(grounding, 'rerank', return_value=[(0, .97), (1, .91)]) as rerank, \
             patch('app.rag.generator.complete', return_value=completion('Fixture grounded answer [1].')) as normal_model, \
             patch('app.api.router.stream_answer', return_value=iter(['Fixture grounded answer [1].'])) as stream_model:
            response = client.post(endpoint, json={'session_id': sid, 'message': message})
        assert response.status_code == 200
        output = parse_done(response, endpoint)
        assert len(output['sources']) == 2
        assert output['sources'][0]['rerank_score'] == .97  # Fresh score, not old .99.
        assert output['sources'][0]['page'] == '12'
        assert output['sources'][0]['url'].endswith('#page=12')
        classify.assert_called_once()
        assert classify.call_args.args[0] == resolved
        retrieve = normal_retrieve if endpoint == '/chat' else stream_retrieve
        retrieve.assert_called_once_with(SEARCH)
        rerank.assert_called_once_with(SEARCH, [c.content for c in citations()], top_n=2)
        if endpoint == '/chat':
            prompt = normal_model.call_args.args[1]
            assert message in prompt[-1]['content']
            assert resolved in prompt[-1]['content']
            assert 'Fixture recipient excerpt.' in prompt[-1]['content']
        else:
            assert stream_model.call_args.kwargs['resolved_query'] == resolved
            assert len(stream_model.call_args.args[1]) == 2
        isolated_chat_store['engine'].dispose()
        restored = client.get(f'/conversations/{sid}/messages').json()
        assert restored[-1]['resources'] == output['resources']
    ctx = context.build_context('Who can receive it?', sid, owner)
    user_texts = [m['content'] for m in ctx.recent if m['role'] == 'user']
    assert 'যাকাত কী?' in user_texts
    assert 'এটা কাদের দিতে হয়?' in user_texts
    assert 'আগের উত্তরটি সহজ ভাষায় বলো।' in user_texts
    assert 'Who can receive it?' in user_texts
    assert len(ctx.text().encode('utf-8')) <= context.settings.chat_context_token_budget
    assert len(store.get_history(sid, owner)) == 8


def test_book_evidence_is_separate_from_memory_and_scoped_to_this_conversation(isolated_chat_store):
    owner = isolated_chat_store['user_id']
    sid = seeded_chat(owner)
    # A source-less simplification must not erase the preceding actual excerpts.
    save_turn(owner, sid, 'আগের উত্তরটি সহজ ভাষায় বলো।', 'সংক্ষিপ্ত উত্তর।')
    ctx = context.build_context('Who can receive it?', sid, owner)
    assert len(ctx.prior_sources) == 2
    assert 'Fixture recipient excerpt.' not in ctx.text()
    other_sid, _ = store.get_or_create_session(None, owner)
    assert context.build_context('আগের চ্যাটে যাকাত', other_sid, owner).prior_sources == []
    other = identity.create_identity()['user_id']
    assert context.build_context('যাকাত', sid, other).prior_sources == []


@pytest.mark.parametrize('complete,mode', [(False, 'QA'), (True, 'ROLEPLAY')])
def test_interrupted_or_roleplay_sources_cannot_become_grounding(complete, mode, isolated_chat_store):
    owner = isolated_chat_store['user_id']
    sid, _ = store.get_or_create_session(None, owner)
    save_turn(owner, sid, 'Topic', 'Answer', citations(), complete=complete, mode=mode)
    assert context.build_context('Who can receive it?', sid, owner).prior_sources == []


@pytest.mark.parametrize('score', [0.0, .49])
def test_saved_sources_must_clear_current_question_relevance_gate(score):
    ctx = context.ChatContext(prior_sources=citations(), followup=True)
    with patch.object(grounding, 'rerank', return_value=[(0, score), (1, score)]):
        assert grounding.select_grounding([], SEARCH, ctx) == []


def test_standalone_topic_change_cannot_silently_reuse_old_book_sources():
    ctx = context.ChatContext(prior_sources=citations(), followup=False)
    with patch.object(grounding, 'rerank') as rerank:
        assert grounding.select_grounding([], 'Explain astronomy', ctx) == []
    rerank.assert_not_called()


def test_fresh_relevant_sources_do_not_trigger_an_extra_rerank():
    ctx = context.ChatContext(prior_sources=citations(), followup=True)
    with patch.object(grounding, 'rerank') as rerank:
        assert grounding.select_grounding(citations(), SEARCH, ctx) == citations()
    rerank.assert_not_called()


def test_mixed_language_resolution_separates_action_from_bengali_search():
    ctx = context.ChatContext(recent=[{'role': 'user', 'content': 'যাকাত কী?'}], prior_sources=citations())
    with patch.object(context, 'complete', return_value=completion(json.dumps({
            'query': 'Who can receive zakat?', 'retrieval_query': SEARCH, 'clarification': None}))) as model:
        assert context.resolve_followup('Who can receive it?', ctx) == ('Who can receive zakat?', None)
    assert ctx.retrieval_query == SEARCH and ctx.resolved_query == 'Who can receive zakat?' and ctx.followup
    payload = json.loads(model.call_args.args[1][-1]['content'])
    assert payload['book_topics'][0]['book'] == 'Fixture book'
    assert 'retrieval_query' in model.call_args.args[1][0]['content']


@pytest.mark.parametrize('bad_search', ['', None, 123, 'x' * 4001])
def test_bad_search_query_requests_clarification(bad_search):
    ctx = context.ChatContext(recent=[{'role': 'user', 'content': 'যাকাত কী?'}])
    with patch.object(context, 'complete', return_value=completion(json.dumps({
            'query': 'Who can receive zakat?', 'retrieval_query': bad_search, 'clarification': None}))):
        assert context.resolve_followup('Who can receive it?', ctx)[1]
    assert not ctx.followup and ctx.resolved_query is None


def test_resolver_cannot_claim_success_with_an_unchanged_ambiguous_question():
    ctx = context.ChatContext(recent=[{'role': 'user', 'content': 'যাকাত কী?'}])
    with patch.object(context, 'complete', return_value=completion(json.dumps({
            'query': 'Who can receive it?', 'retrieval_query': SEARCH, 'clarification': None}))):
        assert context.resolve_followup('Who can receive it?', ctx)[1]


def test_qa_prompt_names_subject_and_does_not_blame_user_for_missing_excerpts():
    from app.rag.generator import _messages
    messages = _messages('Who can receive it?', [], resolved_query='Who can receive zakat?')
    assert 'Who can receive zakat?' in messages[-1]['content']
    assert 'খালি উদ্ধৃতি পাঠানোর জন্য দোষ দেবে না' in messages[0]['content']
    assert 'আগের উত্তরের উদ্ধৃতি নম্বর কপি করবে না' in messages[0]['content']
