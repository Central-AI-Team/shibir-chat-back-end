import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Use Groq API via OpenAI client compatibility
client = OpenAI(
    # api_key=os.getenv("GROQ_API_KEY"),
    api_key=os.getenv("OPENAI_API_KEY")
    # base_url="https://api.groq.com/openai/v1"
)

def generate_answer(query, docs):
    context = "\n\n".join(docs)
    prompt = f"""
Context:
{context}

Question:
{query}

give all answer in bengali
"""

    response = client.chat.completions.create(
        model="gpt-4o", # Using a supported Groq model
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
