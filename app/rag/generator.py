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
You are an assistant that answers using ONLY the context below.

Context:
{context}

Question:
{query}

Answer only based on the context:
give all answer in bengali
"""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo", # Using a supported Groq model
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
