from fastapi import APIRouter

from app.schemas.query import QueryRequest, QueryResponse
from app.services.qa_service import answer_question

router = APIRouter()


@router.post("/ask", response_model=QueryResponse)
def ask_question(body: QueryRequest) -> QueryResponse:
    return answer_question(body.query)
