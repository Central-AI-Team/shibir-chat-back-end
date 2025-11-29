# Shibir Chat Back-End

This is the back-end service for Shibir Chat built with FastAPI. It provides an endpoint to ask questions and get answers generated based on relevant document retrieval.

## Features

- FastAPI web service
- Query processing with document retrieval and answer generation

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

4. Create a `.env` file in the root directory for environment variables if needed.

5. Run the FastAPI server:

```bash
uvicorn app.main:app --reload
```

The API will be available at `http://127.0.0.1:8000`.

## API Usage

- POST `/ask`
  - Request body: `{"query": "your question here"}`
  - Response: JSON with the query, generated answer, and source documents.

## Project Structure

- `app/` - Main application code
  - `main.py` - FastAPI app entrypoint
  - `schemas/` - Pydantic models
  - `rag/` - Document retriever and answer generator modules
- `requirements.txt` - Python dependencies

## License

Specify your license here.
