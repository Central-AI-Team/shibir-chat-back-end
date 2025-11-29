from fastapi import FastAPI
from dotenv import load_dotenv
from app.schemas.query import QueryRequest
from app.rag.retriever import retrieve_relevant_docs
from app.rag.generator import generate_answer

load_dotenv()

app = FastAPI()

@app.post("/ask")
def ask_question(body: QueryRequest):
    query = body.query
    # 1. Retrieve relevant documents
    docs = retrieve_relevant_docs(query)
    # 2. Generate answer using the docs
    answer = generate_answer(query, docs)
    return {"query": query, "answer": answer, "sources": docs}
