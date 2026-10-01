"""Context/persistence behavior, with deterministic model boundaries and isolated databases."""
import json
import uuid
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi import BackgroundTasks, HTTPException
from fastapi.testclient import TestClient
from sqlalchemy import select, update

from app.main import app
from app.db.chat_models import Conversation, ChatMessage
from app.schemas.query import ChatRequest, Citation
from app.services import context, identity, session_store as store

client = TestClient(app)


def completion(text):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


def save_turn(owner, sid, question, answer='উত্তর।', mode='QA', rid=None):
    rid = rid or str(uuid.uuid4())
    lease = store.begin_turn(sid, owner, rid, question)
    store.finish_turn(sid, owner, rid, answer, [], mode, lease=lease)
    return rid


def new_chat(owner):
    return store.get_or_create_session(None, owner)[0]


def test_same_chat_history_reaches_qa_and_retrieval_resolves_followup(isolated_chat_store):
    owner = isolated_chat_store['user_id']
    sid = new_chat(owner)
    save_turn(owner, sid, 'আমরা যাকাত নিয়ে আলোচনা করছি।', 'যাকাতের যোগ্যতা ও নিয়ম নিয়ে আলোচনা করা যাক।')
    resolved = completion(json.dumps({'query': 'যাকাত কাদের দিতে হবে?', 'clarification': None}, ensure_ascii=False))
    citations = [Citation(book='বই', chapter='যাকাত', source_db='test', content='যাকাতের বিধান', rerank_score=0.95)]
    with (patch('app.services.context.complete', return_value=resolved),
          patch('app.api.router.classify_intent', return_value='QA'),
          patch('app.services.qa_service.retrieve_relevant_docs', return_value=citations) as retrieve,
          patch('app.rag.generator.complete', return_value=completion('যাকাতের প্রাপকদের আলোচনা। [১]')) as llm):
        response = client.post('/chat', json={'session_id': sid, 'message': 'এটা কাদের দিতে হবে?'})
    assert response.status_code == 200
    retrieve.assert_called_once_with('যাকাত কাদের দিতে হবে?')
    prompt = llm.call_args.args[1]
    assert any('আমরা যাকাত নিয়ে আলোচনা করছি।' in m['content'] for m in prompt)
    assert prompt[-1]['content'].endswith('উপরের নিয়ম মেনে বাংলায় উত্তর দাও।')
    assert len(store.get_history(sid, owner)) == 4


def test_streaming_uses_the_same_history_and_resolved_query(isolated_chat_store):
    owner = isolated_chat_store['user_id']
    sid = new_chat(owner)
    save_turn(owner, sid, 'যাকাত নিয়ে আলোচনা করো।')
    with (patch('app.services.context.complete', return_value=completion(json.dumps({'query': 'যাকাত কাদের দিতে হবে?', 'clarification': None}))),
          patch('app.api.router.classify_intent', return_value='QA'),
          patch('app.api.router.retrieve_relevant_docs', return_value=[]) as retrieve,
          patch('app.api.router.stream_answer', return_value=iter(['উত্তর।'])) as generate):
        response = client.post('/chat/stream', json={'session_id': sid, 'message': 'এটা কাদের দিতে হবে?'})
    assert response.status_code == 200
    assert response.text.startswith('event: session\n')
    retrieve.assert_called_once_with('যাকাত কাদের দিতে হবে?')
    assert 'যাকাত' in generate.call_args.kwargs['context'].text()


def test_recall_crosses_conversations_but_not_users(isolated_chat_store):
    owner = isolated_chat_store['user_id']
    old = new_chat(owner)
    save_turn(owner, old, 'যাকাত নিয়ে আলোচনা করো।', 'আগের আলোচনায় যাকাতের বিষয় ছিল।')
    other = identity.create_identity()
    hidden = new_chat(other['user_id'])
    save_turn(other['user_id'], hidden, 'যাকাত নিয়ে গোপন আলোচনা SECRET')
    current = new_chat(owner)
    ctx = context.build_context('যাকাত সম্পর্কে বলুন', current, owner)
    assert any(m['source_conversation_id'] == old for m in ctx.memories)
    assert 'SECRET' not in ctx.text()
    assert 'আগের আলোচনায়' in ctx.text()
    with patch('app.services.context.complete', return_value=completion('আমরা যাকাত নিয়ে আলোচনা করেছিলাম।')) as llm:
        response = client.post('/chat', json={'session_id': current, 'message': 'আগের চ্যাটে আমরা কী আলোচনা করেছি?'})
    assert response.status_code == 200
    assert response.json()['mode'] == 'memory'
    assert response.json()['sources'] == []
    assert old in str(llm.call_args.args[1])


def test_ownership_enforced_for_chat_list_read_delete_and_edit(isolated_chat_store):
    sid = new_chat(isolated_chat_store['user_id'])
    other = identity.create_identity()
    headers = {'X-Chat-Identity': other['token']}
    assert client.get('/conversations', headers=headers).json() == []
    assert client.get(f'/conversations/{sid}/messages', headers=headers).status_code == 404
    assert client.delete(f'/conversations/{sid}', headers=headers).status_code == 404
    assert client.patch(f'/conversations/{sid}', headers=headers, json={'title': 'stolen'}).status_code == 404
    assert client.post('/chat', headers=headers, json={'message': 'hi', 'session_id': sid}).status_code == 404
    assert client.post('/chat', headers={'X-Chat-Identity': 'invalid'}, json={'message': 'hi'}).status_code == 401
    # The tracing field never authorizes another owner's data.
    assert client.post('/chat', headers=headers, json={'message': 'hi', 'session_id': sid,
        'user_id': isolated_chat_store['user_id']}).status_code == 404


def test_summary_and_preference_corrections_are_durable(isolated_chat_store):
    owner = isolated_chat_store['user_id']
    first = new_chat(owner)
    save_turn(owner, first, 'আমাকে বিস্তারিত উত্তর দাও।')
    first_user = store.get_history(first, owner)[0]
    with patch('app.services.context.complete', return_value=completion(json.dumps({
        'summary': 'ব্যবহারকারী বিস্তারিত উত্তর চান।', 'facts': [{'key': 'response_length',
        'content': 'বিস্তারিত উত্তর পছন্দ করেন।', 'source_message_id': first_user['id']}]}))):
        context.refresh_memory(first, owner)
    second = new_chat(owner)
    save_turn(owner, second, 'এখন থেকে আমাকে ছোট উত্তর দাও।')
    second_user = store.get_history(second, owner)[0]
    with patch('app.services.context.complete', return_value=completion(json.dumps({
        'summary': 'ব্যবহারকারী এখন ছোট উত্তর চান।', 'facts': [{'key': 'response_length',
        'content': 'ছোট উত্তর পছন্দ করেন।', 'source_message_id': second_user['id']}]}))):
        context.refresh_memory(second, owner)
    memories = store.list_memories(owner)
    assert len(memories) == 1
    assert memories[0]['content'] == 'ছোট উত্তর পছন্দ করেন।'
    third = new_chat(owner)
    assert 'ছোট উত্তর পছন্দ করেন।' in context.build_context('নামাজ কী?', third, owner).text()
    with isolated_chat_store['factory']() as db:
        c = db.get(Conversation, second)
        assert c.summarized_through == 2
        assert c.summary == 'ব্যবহারকারী এখন ছোট উত্তর চান।'


def test_forgetting_memory_excludes_source_and_deleted_chat_stays_deleted(isolated_chat_store):
    owner = isolated_chat_store['user_id']
    sid = new_chat(owner)
    save_turn(owner, sid, 'ছোট উত্তর চাই।')
    user = store.get_history(sid, owner)[0]
    with patch('app.services.context.complete', return_value=completion(json.dumps({
        'summary': 'ছোট উত্তর', 'facts': [{'key': 'response_length', 'content': 'ছোট উত্তর', 'source_message_id': user['id']}]}))):
        context.refresh_memory(sid, owner)
    memory = store.list_memories(owner)[0]
    assert client.delete('/memories/' + memory['id']).status_code == 204
    assert store.list_memories(owner) == []
    assert context.build_context('ছোট উত্তর', new_chat(owner), owner).memories == []
    assert len(store.get_history(sid, owner)) == 2
    assert client.delete('/conversations/' + sid).status_code == 204
    assert client.post('/chat', json={'message': 'hi', 'session_id': sid}).status_code == 404


def test_long_chat_keeps_full_history_and_recalls_older_details_with_bounded_context(isolated_chat_store, monkeypatch):
    owner = isolated_chat_store['user_id']
    sid = new_chat(owner)
    save_turn(owner, sid, 'পড়ার পরিকল্পনার নাম অনন্যপরিকল্পনা।', 'এই নামটি মনে রাখলাম।')
    for i in range(15):
        save_turn(owner, sid, f'অন্য প্রশ্ন {i}', 'অন্য উত্তর ' * 200)
    assert len(store.get_history(sid, owner)) == 32
    monkeypatch.setattr(context.settings, 'chat_context_token_budget', 6000)
    ctx = context.build_context('অনন্যপরিকল্পনা সম্পর্কে বলো', sid, owner)
    assert 'অনন্যপরিকল্পনা' in ctx.text()
    assert len(ctx.text().encode('utf-8')) <= 6000


def test_roleplay_is_not_promoted_to_user_memory(isolated_chat_store):
    owner = isolated_chat_store['user_id']
    sid = new_chat(owner)
    save_turn(owner, sid, 'আমি কাল্পনিক চরিত্র SHERLOCK_SECRET', mode='ROLEPLAY')
    save_turn(owner, sid, 'রোলপ্লে বন্ধ', mode='QA')
    new = new_chat(owner)
    assert 'SHERLOCK_SECRET' not in context.build_context('আগের চ্যাটের আলোচনা', new, owner).text()


def test_retries_replay_completed_turn_and_reject_changed_payload(isolated_chat_store):
    owner = isolated_chat_store['user_id']
    sid = new_chat(owner)
    rid = str(uuid.uuid4())
    first = client.post('/chat', json={'message': 'hi', 'session_id': sid, 'request_id': rid})
    second = client.post('/chat', json={'message': 'hi', 'session_id': sid, 'request_id': rid})
    assert first.status_code == second.status_code == 200
    assert first.json()['answer'] == second.json()['answer']
    assert len(store.get_history(sid, owner)) == 2
    assert client.post('/chat', json={'message': 'different', 'session_id': sid, 'request_id': rid}).status_code == 409


def test_concurrent_turn_and_expired_worker_cannot_overwrite(isolated_chat_store):
    owner = isolated_chat_store['user_id']
    sid = new_chat(owner)
    rid = str(uuid.uuid4())
    old_lease = store.begin_turn(sid, owner, rid, 'প্রশ্ন')
    with pytest.raises(HTTPException) as exc:
        store.begin_turn(sid, owner, str(uuid.uuid4()), 'অন্য প্রশ্ন')
    assert exc.value.status_code == 409
    from app.db.chat_models import now
    with isolated_chat_store['factory'].begin() as db:
        db.execute(update(Conversation).where(Conversation.id == sid).values(active_since=now() - timedelta(minutes=20)))
    new_lease = store.begin_turn(sid, owner, rid, 'প্রশ্ন')
    store.finish_turn(sid, owner, rid, 'stale answer', [], 'QA', lease=old_lease)
    assert store.get_history(sid, owner)[-1]['status'] == 'pending'
    store.finish_turn(sid, owner, rid, 'new answer', [], 'QA', lease=new_lease)
    assert store.get_history(sid, owner)[-1]['content'] == 'new answer'
    assert len(store.get_history(sid, owner)) == 2


def test_interrupted_stream_records_partial_answer_and_can_retry(isolated_chat_store):
    owner = isolated_chat_store['user_id']
    sid = new_chat(owner)
    rid = str(uuid.uuid4())
    from app.api.router import chat_stream
    with (patch('app.api.router.classify_intent', return_value='QA'),
          patch('app.api.router.retrieve_relevant_docs', return_value=[]),
          patch('app.api.router.stream_answer', return_value=iter(['partial', 'rest']))):
        response = chat_stream(ChatRequest(message='প্রশ্ন', session_id=sid, request_id=rid), BackgroundTasks(), owner)
        iterator = response.iterator
        next(iterator)  # early session id
        next(iterator)  # sources
        next(iterator)  # partial answer
        iterator.close()
    history = store.get_history(sid, owner)
    assert history[-1]['status'] == 'interrupted'
    assert history[-1]['content'] == 'partial'
    ctx = context.build_context('প্রশ্ন', sid, owner)
    assert 'partial' not in ctx.text()
    lease = store.begin_turn(sid, owner, rid, 'প্রশ্ন')
    store.finish_turn(sid, owner, rid, 'finished', [], 'QA', lease=lease)
    assert len(store.get_history(sid, owner)) == 2


def test_ambiguous_followup_requests_clarification_without_retrieval(isolated_chat_store):
    owner = isolated_chat_store['user_id']
    sid = new_chat(owner)
    save_turn(owner, sid, 'যাকাত ও নামাজ নিয়ে আলোচনা করছি।')
    with (patch('app.services.context.complete', return_value=completion(json.dumps({'query': 'এটা কী?', 'clarification': 'যাকাত নাকি নামাজ বোঝাচ্ছেন?'}))),
          patch('app.services.qa_service.retrieve_relevant_docs') as retrieve):
        response = client.post('/chat', json={'message': 'এটা কী?', 'session_id': sid})
    assert response.json()['answer'] == 'যাকাত নাকি নামাজ বোঝাচ্ছেন?'
    retrieve.assert_not_called()


def test_failed_summary_keeps_raw_recall_and_does_not_advance_cursor(isolated_chat_store):
    owner = isolated_chat_store['user_id']
    sid = new_chat(owner)
    save_turn(owner, sid, 'যাকাতের আলোচনা')
    with patch('app.services.context.complete', side_effect=RuntimeError('offline')):
        context.refresh_memory(sid, owner)
    with isolated_chat_store['factory']() as db:
        assert db.get(Conversation, sid).summarized_through == 0
    assert 'যাকাত' in context.build_context('যাকাত', new_chat(owner), owner).text()


def test_explicit_remember_is_saved_and_background_refresh_is_scheduled(isolated_chat_store):
    with patch('app.api.router.refresh_memory') as refresh:
        response = client.post('/chat', json={'message': 'মনে রেখ আমি ছোট উত্তর পছন্দ করি।'})
    assert response.status_code == 200
    assert response.json()['mode'] == 'memory'
    assert 'রাখা হয়েছে' in response.json()['answer']
    refresh.assert_called_once_with(response.json()['session_id'], isolated_chat_store['user_id'])
    history = store.get_history(response.json()['session_id'], isolated_chat_store['user_id'])
    assert history[0]['content'] == 'মনে রেখ আমি ছোট উত্তর পছন্দ করি।'


def test_empty_context_followup_does_not_guess_topic():
    query, clarification = context.resolve_followup('এটা কাদের দিতে হবে?', context.ChatContext())
    assert clarification
    assert query == 'এটা কাদের দিতে হবে?'


def test_explicit_creation_is_idempotent_and_requires_ownership(isolated_chat_store):
    sid = str(uuid.uuid4())
    assert client.post('/conversations', json={'session_id': sid}).status_code == 201
    assert client.post('/conversations', json={'session_id': sid}).status_code == 201
    assert len(client.get('/conversations').json()) == 1
    other = identity.create_identity()
    assert client.post('/conversations', json={'session_id': sid},
        headers={'X-Chat-Identity': other['token']}).status_code == 404


def test_late_old_summary_cannot_override_a_newer_preference(isolated_chat_store):
    owner = isolated_chat_store['user_id']
    old = new_chat(owner)
    save_turn(owner, old, 'বিস্তারিত উত্তর পছন্দ করি।')
    recent = new_chat(owner)
    save_turn(owner, recent, 'এখন ছোট উত্তর চাই।')
    for sid, fact in [(recent, 'ছোট উত্তর'), (old, 'বিস্তারিত উত্তর')]:
        source = store.get_history(sid, owner)[0]
        with patch('app.services.context.complete', return_value=completion(json.dumps({
            'summary': fact, 'facts': [{'key': 'response_length', 'content': fact, 'source_message_id': source['id']}]}))):
            context.refresh_memory(sid, owner)
    assert store.list_memories(owner)[0]['content'] == 'ছোট উত্তর'


def test_unknown_fact_source_is_rejected(isolated_chat_store):
    owner = isolated_chat_store['user_id']
    sid = new_chat(owner)
    save_turn(owner, sid, 'নামাজ নিয়ে আলোচনা করো।')
    assistant = store.get_history(sid, owner)[1]
    with patch('app.services.context.complete', return_value=completion(json.dumps({
        'summary': 'নামাজের আলোচনা', 'facts': [{'key': 'response_length', 'content': 'made up', 'source_message_id': assistant['id']}]}))):
        context.refresh_memory(sid, owner)
    assert store.list_memories(owner) == []


def test_starting_a_topic_is_not_mistaken_for_a_memory_question():
    assert not context.is_memory_question('আমরা যাকাত নিয়ে আলোচনা করছি।')
    assert context.is_memory_question('আগের চ্যাটে আমরা কী আলোচনা করেছি?')
