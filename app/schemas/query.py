from pydantic import BaseModel


class QueryRequest(BaseModel):
    query: str


class Citation(BaseModel):
    book: str
    chapter: str
    source_db: str
    content: str


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: list[Citation]
