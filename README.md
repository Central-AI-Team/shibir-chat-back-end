# Shibir Chat Back-End

This is the back-end service for Shibir Chat built with FastAPI. You ask a question, it retrieves
relevant excerpts from the book library and returns a Bengali answer grounded in those excerpts.

## Features

- FastAPI web service with a single `POST /ask` endpoint
- Retrieval-augmented generation over a Chroma vector store, backed by two source libraries
  (`data/Tarun_Associate.db` and `data/Nobin_Associate.db`)
- Answers generated via Groq (OpenAI-compatible API)

## Installation

### Prerequisites

- Python 3.8 or higher
- pip

### Steps

1. Clone the repository:

```bash
git clone <repository-url>
cd shibir-chat-back-end
```

2. (Optional but recommended) Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Copy `.env.example` to `.env` and set `GROQ_API_KEY` (all other settings have defaults).

5. Build the vector store from both source databases (run once, and again whenever the source
   `.db` files change):

```bash
python -m app.rag.ingest
```

6. Run the FastAPI server:

```bash
uvicorn app.main:app --reload
```

The API will be available at `http://127.0.0.1:8000`.

## API Usage

- POST `/ask`
  - Request body: `{"query": "your question here"}`
  - Response:
    ```json
    {
      "query": "your question here",
      "answer": "...",
      "sources": [
        {"book": "...", "chapter": "...", "source_db": "tarun", "content": "..."}
      ]
    }
    ```

## Project Structure

- `app/` - Main application code
  - `main.py` - FastAPI app factory
  - `core/config.py` - Centralized settings (reads `.env`)
  - `api/router.py` - HTTP routes (`POST /ask`)
  - `services/qa_service.py` - Retrieve-then-generate orchestration
  - `schemas/` - Pydantic request/response models
  - `rag/` - Embedding, Chroma client, retriever, generator, and ingestion modules
- `data/` - Source SQLite databases (`Tarun_Associate.db`, `Nobin_Associate.db`)
- `chroma_db/` - Generated vector store (gitignored; rebuild with `python -m app.rag.ingest`)
- `requirements.txt` - Python dependencies

## License

Specify your license here.

