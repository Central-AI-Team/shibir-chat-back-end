"""Resource reload demonstration using an isolated DB and explicit fixture cards.

Run: uv run python -m scripts.message_resources_demo
No provider/model calls or production database access.
"""
import json
import tempfile
import uuid
from pathlib import Path
from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.chat_models import initialize_chat_schema
from app.schemas.query import ConversationMessage
from app.schemas.resources import MessageResources
from app.services import identity, session_store as store


def main():
    resources = MessageResources.model_validate({
        'sources': [{'book': 'Sample book', 'chapter': 'Chapter 1', 'source_db': 'test',
                     'content': 'Sample excerpt.', 'page': '12', 'url': 'https://example.com/book#page=12'}],
        'web_results': [{'title': 'Fixture web result', 'url': 'https://example.com/article'}],
        'verification': {'claim': 'Fixture claim', 'verdict': 'unverified',
                         'sources': [{'url': 'https://example.com/article'}]},
    })
    with tempfile.TemporaryDirectory() as directory:
        url = f'sqlite:///{Path(directory) / "chat.sqlite"}'
        engine = create_engine(url)
        initialize_chat_schema(engine)
        factory = sessionmaker(bind=engine, expire_on_commit=False)
        with patch.object(identity, 'SessionLocal', factory), patch.object(store, 'SessionLocal', factory):
            owner = identity.create_identity()['user_id']
            sid, _ = store.get_or_create_session(None, owner)
            rid = str(uuid.uuid4())
            lease = store.begin_turn(sid, owner, rid, 'Show fixture resources')
            store.finish_turn(sid, owner, rid, 'Sample answer [1].', [], 'QA', lease=lease, resources=resources)
        engine.dispose()
        # Reopen the database through a new engine/factory, as after process restart.
        reopened = create_engine(url)
        initialize_chat_schema(reopened)
        with patch.object(store, 'SessionLocal', sessionmaker(bind=reopened, expire_on_commit=False)):
            row = ConversationMessage(**store.get_history(sid, owner)[-1]).model_dump(mode='json')
            replay = store.begin_turn(sid, owner, rid, 'Show fixture resources')
            rid2 = str(uuid.uuid4())
            lease2 = store.begin_turn(sid, owner, rid2, 'Interrupt this fixture')
            store.checkpoint_resources(sid, owner, rid2, resources, lease=lease2)
            store.finish_turn(sid, owner, rid2, '', [], 'QA', complete=False, lease=lease2)
            interrupted = store.get_history(sid, owner)[-1]
            print(json.dumps({
                'history_sources': len(row['sources']),
                'history_source_page': row['sources'][0]['page'],
                'history_source_url': row['sources'][0]['url'],
                'history_web_results': row['web_results'],
                'history_verification': row['verification'],
                'history_resources': row['resources'],
                'citation_model_preserves_page': row['resources']['sources'][0]['page'] == '12',
                'replay_resources_identical': replay['resources'] == row['resources'],
                'interrupted_resources_identical': interrupted['resources'] == row['resources'],
                'interrupted_status': interrupted['status'],
            }, indent=2))
        reopened.dispose()


if __name__ == '__main__':
    main()
