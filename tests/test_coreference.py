"""Coreference gate and endpoint regressions; model/provider replies are explicit fixtures."""
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas.query import Citation
from app.services import context, session_store as store

client = TestClient(app)


def completion(query, clarification=None):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
        content=json.dumps({'query': query, 'clarification': clarification}, ensure_ascii=False)))])


def discussion():
    return context.ChatContext(recent=[
        {'role': 'user', 'content': 'যাকাত নিয়ে আলোচনা করো।'},
        {'role': 'assistant', 'content': 'যাকাতের প্রাপকদের বিস্তারিত দেব?'},
    ])


@pytest.mark.parametrize('message', ['Who receives it?', 'Explain that', 'dao', 'DAO!', 'দাও',
                                    'দিন', 'daw', 'aro dao', 'নোট দাও', 'eta ki?', 'সেটা কী?'])
def test_reference_detector_invokes_resolver_with_prior_turns(message):
    query = 'যাকাতের প্রাপকদের বিস্তারিত দাও।'
    with patch.object(context, 'complete', return_value=completion(query)) as resolver:
        resolved, clarification = context.resolve_followup(message, discussion())
    assert resolved == query and clarification is None
    resolver.assert_called_once()
    payload = json.loads(resolver.call_args.args[1][-1]['content'])
    assert payload['message'] == message
    assert 'যাকাতের প্রাপকদের বিস্তারিত দেব?' in payload['context']


@pytest.mark.parametrize('message', ['What is a habit?', 'Explain credit', 'What is faith?',
                                    'যাকাতের নোট দাও', 'Give a summary of Chapter 1'])
def test_standalone_requests_do_not_match_substrings_or_elliptical_commands(message):
    with patch.object(context, 'complete') as resolver:
        assert context.resolve_followup(message, context.ChatContext()) == (message, None)
    resolver.assert_not_called()


@pytest.mark.parametrize('message', ['dao', 'দাও', 'Explain that', 'Who receives it?'])
def test_missing_antecedent_clarifies_without_model_or_retrieval(message):
    with patch.object(context, 'complete') as resolver, \
         patch('app.api.router.classify_intent') as classify, \
         patch('app.services.qa_service.retrieve_relevant_docs') as retrieve:
        response = client.post('/chat', json={'message': message})
    assert response.status_code == 200
    assert 'স্পষ্ট করবেন' in response.json()['answer']
    assert response.json()['sources'] == []
    resolver.assert_not_called()
    classify.assert_not_called()
    retrieve.assert_not_called()


def test_new_chat_dao_does_not_guess_from_cross_chat_profile_or_archive():
    ctx = context.ChatContext(memories=[
        {'kind': 'user_fact', 'content': 'বাংলায় উত্তর পছন্দ করেন।'},
        {'kind': 'past_discussion', 'content': 'অন্য চ্যাটে যাকাতের আলোচনা'},
    ])
    with patch.object(context, 'complete') as resolver:
        query, clarification = context.resolve_followup('dao', ctx)
    assert query == 'dao' and clarification
    resolver.assert_not_called()


@pytest.mark.parametrize('endpoint', ['/chat', '/chat/stream'])
@pytest.mark.parametrize('message', ['Who receives it?', 'Explain that', 'dao', 'দাও'])
def test_both_endpoints_resolve_before_classification_and_retrieval_after_reopen(endpoint, message, isolated_chat_store):
    owner = isolated_chat_store['user_id']
    sid, _ = store.get_or_create_session(None, owner)
    lease = store.begin_turn(sid, owner, 'prior-request', 'যাকাত নিয়ে আলোচনা করো।')
    store.finish_turn(sid, owner, 'prior-request', 'যাকাতের প্রাপকদের বিস্তারিত দেব?', [], 'QA', lease=lease)
    isolated_chat_store['engine'].dispose()
    resolved = 'যাকাতের প্রাপকদের বিস্তারিত দাও।'
    sources = [Citation(book='Fixture book', chapter='যাকাত', source_db='test', content='Fixture evidence',
                        rerank_score=0.95, page='12', url='https://example.com/book#page=12')]
    with patch.object(context, 'complete', return_value=completion(resolved)), \
         patch('app.api.router.classify_intent', return_value='QA') as classify, \
         patch('app.services.qa_service.retrieve_relevant_docs', return_value=sources) as normal_retrieve, \
         patch('app.api.router.retrieve_relevant_docs', return_value=sources) as stream_retrieve, \
         patch('app.rag.generator.complete', return_value=SimpleNamespace(choices=[SimpleNamespace(
             message=SimpleNamespace(content='Fixture answer [1].'))])) as normal_generate, \
         patch('app.api.router.stream_answer', return_value=iter(['Fixture answer [1].'])) as stream_generate:
        response = client.post(endpoint, json={'message': message, 'session_id': sid})
    assert response.status_code == 200
    assert classify.call_args.args[0] == resolved
    retrieve = normal_retrieve if endpoint == '/chat' else stream_retrieve
    retrieve.assert_called_once_with(resolved)
    if endpoint == '/chat':
        prompt = normal_generate.call_args.args[1]
        assert any('যাকাতের প্রাপকদের বিস্তারিত দেব?' == row['content'] for row in prompt)
        assert message in prompt[-1]['content']
    else:
        assert 'যাকাতের প্রাপকদের বিস্তারিত দেব?' in stream_generate.call_args.kwargs['context'].text()
        assert 'event: done' in response.text and 'event: error' not in response.text
    restored = client.get(f'/conversations/{sid}/messages').json()
    assert len(restored) == 4
    assert restored[-1]['sources'][0]['page'] == '12'
    assert restored[-1]['sources'][0]['url'].endswith('#page=12')


@pytest.mark.parametrize('endpoint', ['/chat', '/chat/stream'])
def test_dao_preserves_an_offered_note_action(endpoint, isolated_chat_store):
    sid, _ = store.get_or_create_session(None, isolated_chat_store['user_id'])
    lease = store.begin_turn(sid, isolated_chat_store['user_id'], 'prior-request', 'Sample book নিয়ে বলো।')
    store.finish_turn(sid, isolated_chat_store['user_id'], 'prior-request', 'Sample book-এর নোট তৈরি করব?', [], 'QA', lease=lease)
    resolved = 'Sample book-এর নোট তৈরি করো।'
    with patch.object(context, 'complete', return_value=completion(resolved)), \
         patch('app.api.router.generate_book_notes_from_text', return_value={'book': 'Sample book',
             'chapters': [{'chapter': 'Chapter 1', 'note': 'Fixture note.'}]}) as note, \
         patch('app.services.qa_service.retrieve_relevant_docs') as retrieve:
        response = client.post(endpoint, json={'message': 'dao', 'session_id': sid})
    assert response.status_code == 200
    note.assert_called_once_with(resolved)
    retrieve.assert_not_called()
    assert 'Fixture note.' in response.text


@pytest.mark.parametrize('raw', ['not JSON', '{}', '[]', 'null', '1', '{"query":null}', '{"query":"x","clarification":true}'])
def test_invalid_resolver_output_asks_for_clarification(raw):
    reply = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=raw))])
    with patch.object(context, 'complete', return_value=reply):
        assert context.resolve_followup('dao', discussion())[1]


@pytest.mark.parametrize('endpoint', ['/chat', '/chat/stream'])
def test_ambiguous_dao_clarifies_instead_of_retrieving_wrong_topic(endpoint, isolated_chat_store):
    sid, _ = store.get_or_create_session(None, isolated_chat_store['user_id'])
    lease = store.begin_turn(sid, isolated_chat_store['user_id'], 'prior-request', 'যাকাত ও নামাজ নিয়ে আলোচনা করি।')
    store.finish_turn(sid, isolated_chat_store['user_id'], 'prior-request', 'যাকাত নাকি নামাজের নোট চান?', [], 'QA', lease=lease)
    clarification = 'যাকাত নাকি নামাজের নোট চান?'
    with patch.object(context, 'complete', return_value=completion('dao', clarification)), \
         patch('app.api.router.classify_intent') as classify, \
         patch('app.services.qa_service.retrieve_relevant_docs') as retrieve, \
         patch('app.api.router.retrieve_relevant_docs') as stream_retrieve:
        response = client.post(endpoint, json={'message': 'dao', 'session_id': sid})
    assert response.status_code == 200
    classify.assert_not_called()
    retrieve.assert_not_called()
    stream_retrieve.assert_not_called()
    assert clarification in response.text
