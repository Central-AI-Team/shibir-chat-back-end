# Shibir Chat Back-End

This is the back-end service for Shibir Chat, built with FastAPI. You ask a question (in
Bengali script, Banglish, or English), it retrieves relevant excerpts from a Bengali book
library and returns an answer grounded strictly in those excerpts — it refuses rather than
guesses when the library doesn't cover the question.

For a deeper architecture walkthrough (retrieval pipeline, prompt contract, reindexing
gotchas), see [`PROJECT.md`](./PROJECT.md).

## Features

- `POST /ask` — retrieval-augmented Q&A: bi-encoder search (`BAAI/bge-m3`) over a chunked
  Chroma vector store, cross-encoder reranking (`BAAI/bge-reranker-v2-m3`), then a grounded
  Gemini answer. Works across Bengali script, Banglish, and English queries via automatic
  query rewriting.
- `POST /note` — generates a structured Bengali summary of an entire book chapter
  (map-reduce over every published page, not similarity search).
- `GET /health` — plain liveness check.
- Content is stored in PostgreSQL (`categories` → `books` → `chapters` → `pages`, plus a
  standalone `articles` table).

## Prerequisites

- **Python 3.10 or higher** (the code uses `X | None` union type hints, which need 3.10+
  even without a virtual environment quirk — tested on 3.12).
- **PostgreSQL** (any recent version), with the content already loaded into it. This service
  reads from Postgres; it does not seed it.
- A **Gemini API key** (https://aistudio.google.com/apikey). Free tier is capped at 20
  requests/day **per Google Cloud project** — fine for light testing, not for sustained use.
- ~3 GB free disk for the embedding + reranker models (downloaded once, cached locally).
- git, pip.

## Installation

The steps are the same on every OS; only the shell commands differ. Ubuntu/Debian, macOS,
and Windows instructions are given separately below — pick yours.

### 1. Clone the repository

```bash
git clone <repository-url>
cd shibir-chat-back-end
```

### 2. Create and activate a virtual environment

**Ubuntu / Debian:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```
(If `python3` isn't found, install it first: `brew install python`.)

**Windows (PowerShell):**
```powershell
py -m venv venv
venv\Scripts\Activate.ps1
```
If PowerShell blocks the activation script with an execution-policy error, run PowerShell as
Administrator once and allow local scripts:
`Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

**Windows (Command Prompt):**
```bat
py -m venv venv
venv\Scripts\activate.bat
```

### 3. Install PostgreSQL client libraries (if needed)

`psycopg2-binary` in `requirements.txt` ships a self-contained wheel with `libpq` bundled in
— on all three platforms this normally installs with **no separate PostgreSQL client install
and no C compiler needed**. You only need a full PostgreSQL install if you're also running
the database server itself on this machine:

- **Ubuntu/Debian**: `sudo apt install postgresql`
- **macOS**: `brew install postgresql@16 && brew services start postgresql@16`
- **Windows**: https://www.postgresql.org/download/windows/ (installer includes the server)

### 4. Install Python dependencies

Same command on every OS:
```bash
pip install -r requirements.txt
```
`requirements.txt` is pinned from a Linux test environment, but GPU/CUDA packages
(`nvidia-*`, `triton`) and `uvloop` (no Windows support) carry environment markers so `pip`
only installs what's actually usable on your platform — this file installs cleanly on
Windows and macOS as well as Linux, CPU-only or with an NVIDIA GPU.

### 5. Configure environment variables

Copy `.env.example` to `.env`:

- **Ubuntu/macOS**: `cp .env.example .env`
- **Windows**: `copy .env.example .env`

Then edit `.env` and set at minimum:
- `GEMINI_API_KEY` — your key from https://aistudio.google.com/apikey
- `DATABASE_URL` — SQLAlchemy DSN, e.g.
  `postgresql+psycopg2://user:password@localhost:5432/shibir_chat`

Everything else in `.env.example` has a working default — see the comments in that file, or
the configuration table in [`PROJECT.md`](./PROJECT.md).

### 6. Download the embedding and reranker models (first run only)

The models (`BAAI/bge-m3`, `BAAI/bge-reranker-v2-m3`) are downloaded from Hugging Face and
cached locally the first time they're used. `run.sh` sets `HF_HUB_OFFLINE=1`, which blocks
that first download, so do one of these once before running the server or the ingest script:

```bash
HF_HUB_OFFLINE=0 python -c "
from sentence_transformers import SentenceTransformer, CrossEncoder
SentenceTransformer('BAAI/bge-m3')
CrossEncoder('BAAI/bge-reranker-v2-m3')
print('models cached')
"
```
On Windows PowerShell, set the variable first instead of inlining it:
```powershell
$env:HF_HUB_OFFLINE = "0"
python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; SentenceTransformer('BAAI/bge-m3'); CrossEncoder('BAAI/bge-reranker-v2-m3'); print('models cached')"
```

Needs ~2.8 GB of disk and a working internet connection. Subsequent runs work fully offline.

### 7. Build the vector store

Reads published pages/articles from Postgres, chunks them, and embeds them into Chroma. Run
once, and again whenever the source content changes (see the "Reindexing" section of
[`PROJECT.md`](./PROJECT.md) for the full re-embed procedure — deleting `chroma_db/` alone is
**not** enough):

```bash
python -m app.rag.ingest
```
This is CPU-bound and can take a while on a large corpus without a GPU — expect anywhere from
a few minutes to a few hours depending on corpus size and available RAM.

### 8. Run the server

**Ubuntu/macOS** — either use the helper script (creates the venv and installs deps too, if
you skipped the manual steps above):
```bash
./run.sh
```
or run uvicorn directly:
```bash
HF_HUB_OFFLINE=1 uvicorn app.main:app --host 0.0.0.0 --port 9200
```

**Windows** — `run.sh` is a bash script and won't run natively in PowerShell/cmd. Either use
**WSL** (Windows Subsystem for Linux) and follow the Ubuntu instructions inside it, or **Git
Bash**, or run uvicorn directly:
```powershell
$env:HF_HUB_OFFLINE = "1"
uvicorn app.main:app --host 0.0.0.0 --port 9200
```

The API is available at `http://127.0.0.1:9200`.

## API Usage

- `GET /health` → `{"status": "ok"}`

- `POST /ask`
  - Request: `{"query": "your question in Bengali, Banglish, or English"}`
  - Response:
    ```json
    {
      "query": "...",
      "answer": "... (always Bengali)",
      "sources": [
        {
          "book": "...", "chapter": "...", "source_db": "tarun",
          "content": "...", "similarity": 0.65, "rerank_score": 0.98
        }
      ]
    }
    ```
    `sources` is an empty array whenever the answer is a refusal.

- `POST /note`
  - Request: `{"chapter_id": 2509}`
  - Response: `{"book": "...", "chapter": "...", "pages_used": 139, "note": "..."}`

Example:
```bash
curl -s http://127.0.0.1:9200/ask -H "Content-Type: application/json" \
  -d '{"query": "যাকাতের অর্থ কোন কোন খাতে ব্যয় করা যায়?"}'
```

## Project Structure

- `app/` — application code
  - `main.py` — FastAPI app factory
  - `core/config.py` — centralized settings (reads `.env`)
  - `api/router.py` — HTTP routes (`/health`, `/ask`, `/note`)
  - `db/` — SQLAlchemy models and session (Postgres)
  - `services/` — orchestration layer (`qa_service.py`, `note_service.py`)
  - `schemas/` — Pydantic request/response models
  - `rag/` — chunking, embedding, reranking, query rewriting, Chroma client, retriever,
    generator, and the ingest script
- `scripts/` — one-time migration script and the `tune_threshold.py` refusal-threshold tuner
- `tests/` — regression tests (Bengali tokenization/embedding correctness)
- `chroma_db/` — generated vector store (gitignored; see "Reindexing" in `PROJECT.md`)
- `requirements.txt` — Python dependencies (cross-platform: Linux, macOS, Windows)

See [`PROJECT.md`](./PROJECT.md) for the full architecture, prompt contract, and
configuration reference.

## License

Specify your license here.
