"""Lossless resource snapshots, crash recovery, replay, and additive upgrades.

All providers are mocked; the autouse fixture uses an isolated database/schema.
"""
import json
from datetime import timedelta
import uuid
from unittest.mock import patch

import pytest
from fastapi import BackgroundTasks
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.api import router
from app.db.chat_models import ChatMessage, Conversation, initialize_chat_schema, now
from app.main import app
from app.schemas.query import ChatRequest, QueryResponse
from app.schemas.resources import MessageResources, resource_snapshot
from app.services import session_store as store

client = TestClient(app)


def bundle():
    return MessageResources.model_validate({
        'version': 1,
        'sources': [
            {'book': 'Sample book', 'chapter': 'Chapter 1', 'source_db': 'test',
             'content': 'Sample excerpt.', 'page': '12', 'url': 'https://example.com/book#page=12',
             'provenance': {'record_id': 'page-12', 'tags': ['first']}},
            {'book': 'Second book', 'chapter': 'Chapter 2', 'source_db': 'test', 'content': 'Second excerpt.'},
        ],
        'web_results': [{'title': 'Sample web result', 'url': 'https://example.com/article',
                         'snippet': 'A fixture, not a live search.', 'provider': {'id': 'web-1'}}],
        'verification': {'claim': 'Sample claim', 'verdict': 'unverified',
                         'sources': [{'url': 'https://example.com/article'}], 'confidence': None},
        'provider_metadata': {'request': 'fixture-1'},
    })


def parse_events(raw):
    events = []
    for block in raw.strip().split('\n\n'):
        event, data = block.split('\n', 1)
        events.append((event.removeprefix('event: '), json.loads(data.removeprefix('data: '))))
    return events


@pytest.mark.parametrize('stream', [False, True])
def test_resources_survive_history_reload_and_replay(stream, isolated_chat_store):
    resources = bundle()
    rid = str(uuid.uuid4())
    body = {'message': 'Explain this book chapter', 'request_id': rid, 'search_web': True, 'verify_claim': True}
    with patch.object(router, 'classify_intent', return_value='QA'), \
         patch.object(router, '_qa_stream', return_value=(resources, iter(['Saved answer.']))), \
         patch.object(router, 'answer_question', return_value=QueryResponse(
             query=body['message'], answer='Saved answer.', resources=resources, response_time_ms=1)):
        response = client.post('/chat/stream' if stream else '/chat', json=body)
    assert response.status_code == 200
    output = parse_events(response.text)[-1][1] if stream else response.json()
    sid = output['session_id']
    expected = resources.model_dump(mode='json')
    assert output['resources'] == expected
    if stream:
        kinds = [kind for kind, _ in parse_events(response.text)]
        assert kinds == ['session', 'sources', 'web_results', 'verification', 'token', 'done']
    # History opens a fresh connection after completing the original request.
    isolated_chat_store['engine'].dispose()
    history = client.get(f'/conversations/{sid}/messages').json()
    assert history[-1]['resources'] == expected
    assert history[-1]['sources'][0]['page'] == '12'
    assert history[-1]['sources'][0]['url'].endswith('#page=12')
    assert history[-1]['web_results'] == expected['web_results']
    assert history[-1]['verification'] == expected['verification']
    assert history[0]['options'] == {'search_web': True, 'verify_claim': True}
    # No provider should run for a completed retry.
    retry = client.post('/chat/stream', json={**body, 'session_id': sid})
    events = parse_events(retry.text)
    assert events[-1][0] == 'done'
    assert events[-1][1]['resources'] == expected
    assert len(store.get_history(sid, isolated_chat_store['user_id'])) == 2
    assert client.post('/chat', json={**body, 'session_id': sid, 'verify_claim': False}).status_code == 409
    other = client.post('/identity').json()['token']
    assert client.get(f'/conversations/{sid}/messages', headers={'X-Chat-Identity': other}).status_code == 404


def test_stream_checkpoints_before_first_resource_event_and_disconnect(isolated_chat_store):
    owner = isolated_chat_store['user_id']
    rid = str(uuid.uuid4())
    with patch.object(router, 'classify_intent', return_value='QA'), \
         patch.object(router, '_qa_stream', return_value=(bundle(), iter(['Never consumed']))):
        response = router.chat_stream(ChatRequest(message='Explain book', request_id=rid), BackgroundTasks(), owner)
        iterator = response.iterator
        sid = parse_events(next(iterator))[0][1]['session_id']
        assert parse_events(next(iterator))[0][0] == 'sources'
        row = store.get_history(sid, owner)[-1]
        assert row['status'] == 'pending'
        assert row['resources'] == bundle().model_dump(mode='json')
        # Stop even before web/verification/token events are pulled.
        iterator.close()
        response.cleanup()
    row = store.get_history(sid, owner)[-1]
    assert row['status'] == 'interrupted'
    assert row['content'] == ''
    assert row['resources'] == bundle().model_dump(mode='json')


def test_checkpoint_rejects_stale_worker_and_empty_cleanup_preserves_resources(isolated_chat_store):
    owner = isolated_chat_store['user_id']
    sid, _ = store.get_or_create_session(None, owner)
    rid = str(uuid.uuid4())
    lease = store.begin_turn(sid, owner, rid, 'Question')
    assert not store.checkpoint_resources(sid, owner, rid, bundle(), lease='expired-worker')
    assert store.checkpoint_resources(sid, owner, rid, bundle(), lease=lease)
    assert store.finish_turn(sid, owner, rid, '', [], 'QA', complete=False, lease=lease)
    assert store.get_history(sid, owner)[-1]['resources'] == bundle().model_dump(mode='json')
    assert not store.finish_turn(sid, owner, rid, 'stale', [], 'QA', lease=lease)
    # Interrupted retry gets a new lease and an empty snapshot, not stale cards.
    new_lease = store.begin_turn(sid, owner, rid, 'Question')
    assert isinstance(new_lease, str) and new_lease != lease
    assert not store.checkpoint_resources(sid, owner, rid, bundle(), lease=lease)
    assert store.get_history(sid, owner)[-1]['resources']['sources'] == []


def test_upgrade_existing_database_retains_ordered_legacy_sources(isolated_chat_store):
    engine = isolated_chat_store['engine']
    owner = isolated_chat_store['user_id']
    sid, _ = store.get_or_create_session(None, owner)
    rid = str(uuid.uuid4())
    lease = store.begin_turn(sid, owner, rid, 'Question')
    store.finish_turn(sid, owner, rid, 'Legacy answer', bundle().sources, 'QA', lease=lease)
    schema = engine.get_execution_options().get('schema_translate_map', {}).get(None)
    quote = engine.dialect.identifier_preparer.quote
    table = (quote(schema) + '.' if schema else '') + quote('chat_messages')
    with engine.begin() as connection:
        connection.exec_driver_sql(f'ALTER TABLE {table} DROP COLUMN resources')
        connection.exec_driver_sql(f'ALTER TABLE {table} DROP COLUMN options')
    initialize_chat_schema(engine)
    expected = resource_snapshot(sources=bundle().sources).model_dump(mode='json')
    assert store.get_history(sid, owner)[-1]['resources'] == expected
    assert store.get_history(sid, owner)[0]['options'] == {}
    # Repeated startup must not backfill over newer snapshots.
    with isolated_chat_store['factory'].begin() as db:
        row = db.scalar(select(ChatMessage).where(ChatMessage.conversation_id == sid, ChatMessage.role == 'assistant'))
        row.resources = bundle().model_dump(mode='json')
    initialize_chat_schema(engine)
    assert store.get_history(sid, owner)[-1]['resources'] == bundle().model_dump(mode='json')


def test_canonical_empty_snapshot_does_not_resurrect_legacy_sources():
    assert resource_snapshot({'version': 1, 'sources': []}, sources=bundle().sources).sources == []
    response = QueryResponse(query='x', answer='x', response_time_ms=1,
                             resources=MessageResources(), sources=bundle().sources)
    assert response.sources == []
    assert response.resources.sources == []


def test_worker_crash_and_later_turn_retain_each_messages_own_resources(isolated_chat_store):
    owner = isolated_chat_store['user_id']
    sid, _ = store.get_or_create_session(None, owner)
    rid = str(uuid.uuid4())
    lease = store.begin_turn(sid, owner, rid, 'First question')
    store.checkpoint_resources(sid, owner, rid, bundle(), lease=lease)
    # Simulate a worker dying after checkpoint; no generator cleanup executes.
    with isolated_chat_store['factory'].begin() as db:
        db.get(Conversation, sid).active_since = now() - timedelta(minutes=16)
    isolated_chat_store['engine'].dispose()
    rid2 = str(uuid.uuid4())
    lease2 = store.begin_turn(sid, owner, rid2, 'Second question')
    different = MessageResources(web_results=[{'title': 'Second result', 'url': 'https://example.com/second'}])
    store.finish_turn(sid, owner, rid2, 'Second answer', [], 'QA', lease=lease2, resources=different)
    assert not store.checkpoint_resources(sid, owner, rid, different, lease=lease)
    assert not store.finish_turn(sid, owner, rid, 'Old answer', [], 'QA', lease=lease, resources=different)
    history = store.get_history(sid, owner)
    assert [row['sequence'] for row in history] == [1, 2, 3, 4]
    assert history[1]['status'] == 'interrupted'
    assert history[1]['resources'] == bundle().model_dump(mode='json')
    assert history[3]['resources'] == different.model_dump(mode='json')
    assert history[0]['resources'] == history[2]['resources'] == MessageResources().model_dump(mode='json')
