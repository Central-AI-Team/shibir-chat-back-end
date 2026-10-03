"""Reproduce follow-up recovery locally with real storage and explicit model/reranker fixtures.

Run: uv run python -m scripts.followup_grounding_demo
No production database or live model calls.
"""
import json
import tempfile
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.api import router
from app.db.chat_models import initialize_chat_schema
from app.main import app
from app.schemas.query import Citation
from app.services import context, grounding, identity, session_store as store


def completion(value):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=value))])


def main():
    with tempfile.TemporaryDirectory() as directory:
        engine = create_engine(f'sqlite:///{Path(directory) / "chat.sqlite"}', connect_args={'check_same_thread': False})
        initialize_chat_schema(engine)
        factory = sessionmaker(bind=engine, expire_on_commit=False)
        with patch.object(identity, 'SessionLocal', factory), patch.object(store, 'SessionLocal', factory), \
             patch.object(context, 'SessionLocal', factory), patch.object(router, 'refresh_memory', lambda *args: None):
            credential = identity.create_identity()
            sid, _ = store.get_or_create_session(None, credential['user_id'])
            sources = [Citation(book='Fixture book', chapter='যাকাত', source_db='test',
                content='Test-only excerpt about recipients.', page='12', url='https://example.com/book#page=12', rerank_score=.99)]
            for question, answer, evidence in [('যাকাত কী?', 'যাকাতের পরিচয়। ' * 120, sources),
                    ('এটা কাদের দিতে হয়?', 'প্রাপকদের দীর্ঘ আলোচনা। ' * 180, sources),
                    ('আগের উত্তরটি সহজ ভাষায় বলো।', 'সংক্ষিপ্ত উত্তর।', [])]:
                rid = str(uuid.uuid4())
                lease = store.begin_turn(sid, credential['user_id'], rid, question)
                store.finish_turn(sid, credential['user_id'], rid, answer, evidence, 'QA', lease=lease)
            body = {'message': 'Who can receive it?', 'session_id': sid, 'request_id': str(uuid.uuid4())}
            resolution = {'query': 'Who can receive zakat?', 'retrieval_query': 'যাকাত কাদের দিতে হয়?', 'clarification': None}
            with patch.object(context, 'complete', return_value=completion(json.dumps(resolution))), \
                 patch.object(router, 'classify_intent', return_value='QA'), \
                 patch('app.services.qa_service.retrieve_relevant_docs', return_value=[]), \
                 patch.object(grounding, 'rerank', return_value=[(0, .97)]), \
                 patch('app.rag.generator.complete', return_value=completion('প্রাসঙ্গিক বইয়ের অংশসহ সংক্ষিপ্ত উত্তর। [1]')) as model:
                client = TestClient(app, headers={'X-Chat-Identity': credential['token']})
                response = client.post('/chat', json=body)
                response.raise_for_status()
                result = response.json()
            engine.dispose()
            history = client.get(f'/conversations/{sid}/messages').json()
            replay = client.post('/chat', json=body).json()
            prompt = model.call_args.args[1][-1]['content']
            print(json.dumps({
                'input': body['message'], 'resolved_request': resolution['query'],
                'retrieval_query': resolution['retrieval_query'], 'fresh_retrieval_sources': 0,
                'rechecked_saved_sources': len(result['sources']),
                'new_relevance_score': result['sources'][0]['rerank_score'],
                'subject_present_in_qa_prompt': 'Who can receive zakat?' in prompt,
                'book_excerpt_present_in_qa_prompt': sources[0].content in prompt,
                'after_reload_page': history[-1]['sources'][0]['page'],
                'after_reload_sources_identical': history[-1]['resources'] == result['resources'],
                'replay_resources_identical': replay['resources'] == result['resources'],
                'answer_mocked': result['answer'],
            }, ensure_ascii=False, indent=2))
            client.close()
        engine.dispose()


if __name__ == '__main__':
    main()
