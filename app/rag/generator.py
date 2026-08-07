from openai import OpenAI

from app.core.config import settings
from app.schemas.query import Citation

_client = OpenAI(api_key=settings.gemini_api_key, base_url=settings.gemini_base_url)


def generate_answer(query: str, citations: list[Citation]) -> str:
    context = "\n\n".join(c.content for c in citations)
    prompt = f"""
You are an assistant that answers strictly using the provided book excerpts (the database).
Do not use any outside knowledge and do not guess.

- Base your answer only on the excerpts in the Context section below.
- If the Context does not contain enough information to answer, say clearly in Bengali
  that the database does not contain this information — do not make anything up.
- Answer entirely in Bengali.

Context:
{context}

Question:
{query}
"""

    response = _client.chat.completions.create(
        model=settings.gemini_model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content
