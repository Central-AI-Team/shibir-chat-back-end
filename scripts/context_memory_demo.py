"""Reproducible before/after prompt inspection. Isolated SQLite and mocked LLMs.

Usage: uv run python -m scripts.context_memory_demo
No network calls or production database reads/writes are made.
"""
import json
import tempfile
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.chat_models import ChatBase
from app.services import context, identity, session_store
from app.rag.generator import generate_answer


def completion(content):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def main():
    with tempfile.TemporaryDirectory() as directory:
        engine = create_engine(f'sqlite:///{Path(directory) / "chat.sqlite"}')
        ChatBase.metadata.create_all(engine)
        factory = sessionmaker(bind=engine, expire_on_commit=False)
        with (patch.object(identity, 'SessionLocal', factory),
              patch.object(session_store, 'SessionLocal', factory),
              patch.object(context, 'SessionLocal', factory)):
            owner = identity.create_identity()['user_id']
            sid, _ = session_store.get_or_create_session(None, owner)
            question = 'আমরা যাকাত নিয়ে আলোচনা করছি।'
            reply = 'যাকাতের যোগ্যতা ও নিয়ম নিয়ে আলোচনা করা যাক।'
            rid = str(uuid.uuid4())
            lease = session_store.begin_turn(sid, owner, rid, question)
            session_store.finish_turn(sid, owner, rid, reply, [], 'QA', lease=lease)
            followup = 'এটা কাদের দিতে হবে?'
            ctx = context.build_context(followup, sid, owner)
            with patch.object(context, 'complete', return_value=completion(json.dumps({
                'query': 'যাকাত কাদের দিতে হবে?', 'clarification': None}, ensure_ascii=False))):
                resolved, _ = context.resolve_followup(followup, ctx)
            with patch('app.rag.generator.complete', return_value=completion('mock reply')) as llm:
                generate_answer(followup, [], context=ctx)
                prompt = llm.call_args.args[1]
            print('Stored history messages:', len(session_store.get_history(sid, owner)))
            print('Messages sent to QA model:', len(prompt))
            print('Prior topic sent to QA model:', any(question in m['content'] for m in prompt))
            print('Resolved retrieval question:', resolved)
            for i in range(15):
                rid = str(uuid.uuid4())
                lease = session_store.begin_turn(sid, owner, rid, f'question {i}')
                session_store.finish_turn(sid, owner, rid, f'answer {i}', [], 'QA', lease=lease)
            print('Stored messages after 32 appends:', len(session_store.get_history(sid, owner)))
            print('Original topic still stored:', session_store.get_history(sid, owner)[0]['content'] == question)
            other, _ = session_store.get_or_create_session(None, owner)
            memory = context.build_context('আগের চ্যাটে যাকাত নিয়ে কী আলোচনা হয়েছিল?', other, owner)
            with patch.object(context, 'complete', return_value=completion('আমরা যাকাতের যোগ্যতা ও নিয়ম নিয়ে আলোচনা করেছিলাম।')):
                answer = context.answer_from_memory('আগের চ্যাটে আমরা কী আলোচনা করেছি?', memory)
            print('Prior topic available in a new conversation:', 'যাকাত' in memory.text())
            print('New-conversation answer (mocked):', answer)
            outsider = identity.create_identity()['user_id']
            outsider_chat, _ = session_store.get_or_create_session(None, outsider)
            print('Another user receives this memory:', 'যাকাত' in context.build_context('যাকাত', outsider_chat, outsider).text())
        engine.dispose()


if __name__ == '__main__':
    main()
