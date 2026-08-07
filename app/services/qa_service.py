from app.rag.generator import generate_answer
from app.rag.retriever import retrieve_relevant_docs
from app.schemas.query import QueryResponse


def answer_question(query: str) -> QueryResponse:
    citations = retrieve_relevant_docs(query)
    answer = generate_answer(query, citations)
    return QueryResponse(query=query, answer=answer, sources=citations)
