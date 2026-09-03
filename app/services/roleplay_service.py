"""Persona roleplay chat.

No RAG retrieval here by design -- roleplay is a free-form persona
conversation, not a book lookup. If that changes later, wire in
retrieve_relevant_docs() the same way qa_service does.
"""

from __future__ import annotations

from app.core.llm import get_client, get_model
from app.services.session_store import MAX_HISTORY

_PERSONA_EXTRACT_SYSTEM = """ব্যবহারকারীর নির্দেশনা থেকে শুধুমাত্র চরিত্র/ব্যক্তিত্বের
বর্ণনাটুকু বের করো (কে বা কী চরিত্রে অভিনয় করতে বলা হয়েছে)। শুধু চরিত্রের নাম ও
সংক্ষিপ্ত বর্ণনা লেখো, অন্য কোনো ব্যাখ্যা বা বাক্য যোগ কোরো না।"""

_ROLEPLAY_SYSTEM_TEMPLATE = """তুমি এখন {persona} চরিত্রে অভিনয় করছো। এই চরিত্রের ভাষা,
মনোভাব এবং দৃষ্টিভঙ্গি বজায় রেখে বাংলায় উত্তর দাও। বাস্তব দুনিয়ার AI/model হিসেবে
পরিচয় দিও না যতক্ষণ না ব্যবহারকারী explicitly জিজ্ঞাসা করে।"""


def _extract_persona(message: str) -> str:
    response = get_client().chat.completions.create(
        model=get_model(),
        messages=[
            {"role": "system", "content": _PERSONA_EXTRACT_SYSTEM},
            {"role": "user", "content": message},
        ],
    )
    return (response.choices[0].message.content or "").strip()


def handle_roleplay(message: str, session: dict) -> str:
    if session.get("persona") is None:
        session["persona"] = _extract_persona(message)

    system_prompt = _ROLEPLAY_SYSTEM_TEMPLATE.format(persona=session["persona"])

    messages = [{"role": "system", "content": system_prompt}]
    for entry in session["history"]:
        messages.append({"role": entry["role"], "content": entry["content"]})
    messages.append({"role": "user", "content": message})

    response = get_client().chat.completions.create(
        model=get_model(),
        messages=messages,
    )
    reply = response.choices[0].message.content

    # `session` is the same dict object held in session_store's module-level
    # store (get_or_create_session doesn't copy), so mutating it in place
    # here is equivalent to routing through append_history -- it's just done
    # as one capped update instead of two separate calls.
    session["history"].append({"role": "user", "content": message})
    session["history"].append({"role": "assistant", "content": reply})
    session["history"] = session["history"][-MAX_HISTORY:]

    return reply
