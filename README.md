# Shibir Chat Back-End

This is the back-end service for Shibir Chat, built with FastAPI. You ask a question (in
Bengali script, Banglish, or English), it retrieves relevant excerpts from a Bengali book
library and returns an answer grounded strictly in those excerpts — it refuses rather than
guesses when the library doesn't cover the question.

For a deeper architecture walkthrough (retrieval pipeline, prompt contract, reindexing
gotchas), see [`CLAUDE.md`](./CLAUDE.md).

## Features

- `POST /chat` — the single entry point. Classifies each message into one of four modes and
  dispatches internally:
  - **qa** — retrieval-augmented Q&A: bi-encoder search (`BAAI/bge-m3`) over a chunked
    Chroma vector store, cross-encoder reranking (`BAAI/bge-reranker-v2-m3`), then a grounded
    answer from OpenAI `gpt-5-mini`. Works across Bengali script, Banglish, and English
    queries via automatic query rewriting.
  - **note** — a structured Bengali summary of a whole book (map-reduce over every published
    page of each chapter, not similarity search).
  - **roleplay** — multi-turn persona conversation (no retrieval).
  - **suggestion** — book-grounded recommendation, with the same relevance gate as qa.
- `POST /chat/stream` — same as `/chat`, as Server-Sent Events.
- `GET /conversations`, `GET /conversations/{id}/messages`, `DELETE /conversations/{id}` —
  in-process chat history (single worker only).
- `GET /health` — plain liveness check.
- Embedding and reranking run either locally (CPU, sentence-transformers) or on the external
  GPU service `shibir-chat-gpu-service` (Modal) — see "GPU service" below.
- Content is stored in PostgreSQL (`categories` → `books` → `chapters` → `pages`, plus a
  standalone `articles` table).

## Prerequisites

- **Python 3.11 or higher** (`networkx`, a transitive dependency, requires 3.11+; the
  app code itself only needs the 3.10+ `X | None` union type hints — tested on 3.12).
- **[uv](https://docs.astral.sh/uv/getting-started/installation/)** — manages the virtual
  environment and dependencies (replaces pip/venv). Install it once, globally; it will
  download the right Python for you if needed.
- **PostgreSQL** (any recent version), with the content already loaded into it. This service
  reads from Postgres; it does not seed it.
- An **OpenAI API key** (https://platform.openai.com/api-keys).
- Either ~3 GB free disk (+ ~6 GB RAM) for the local embedding + reranker models, **or** a
  running `shibir-chat-gpu-service` deployment and its API key (see "GPU service" below).
- git.

## Installation

The steps are the same on every OS; only the shell commands differ. Ubuntu/Debian, macOS,
and Windows instructions are given separately below — pick yours.

### 1. Clone the repository

```bash
git clone <repository-url>
cd shibir-chat-back-end
```

### 2. Install uv

**Ubuntu/Debian/macOS:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

See [uv's install docs](https://docs.astral.sh/uv/getting-started/installation/) for other
methods (pipx, Homebrew, etc).

### 3. Install PostgreSQL client libraries (if needed)

`psycopg2-binary` in `pyproject.toml` ships a self-contained wheel with `libpq` bundled in
— on all three platforms this normally installs with **no separate PostgreSQL client install
and no C compiler needed**. You only need a full PostgreSQL install if you're also running
the database server itself on this machine:

- **Ubuntu/Debian**: `sudo apt install postgresql`
- **macOS**: `brew install postgresql@16 && brew services start postgresql@16`
- **Windows**: https://www.postgresql.org/download/windows/ (installer includes the server)

### 4. Install Python dependencies

Same command on every OS — creates `.venv` (downloading a matching Python automatically if
you don't already have one) and installs the exact versions pinned in `uv.lock`:
```bash
uv sync --locked
```
GPU/CUDA packages (`nvidia-*`, `triton`) and `uvloop` (no Windows support) carry environment
markers so `uv` only installs what's actually usable on your platform — this installs cleanly
on Windows and macOS as well as Linux, CPU-only or with an NVIDIA GPU. There's no separate
"activate" step — prefix commands with `uv run` (e.g. `uv run uvicorn ...`), as shown below.

### 5. Configure environment variables

Copy `.env.example` to `.env`:

- **Ubuntu/macOS**: `cp .env.example .env`
- **Windows**: `copy .env.example .env`

Then edit `.env` and set at minimum:
- `OPENAI_API_KEY` — your OpenAI key
- `DATABASE_URL` — SQLAlchemy DSN, e.g.
  `postgresql+psycopg2://user:password@localhost:5432/shibir_chat`

Everything else in `.env.example` has a working default — see the comments in that file, or
the configuration table in [`CLAUDE.md`](./CLAUDE.md).

### 6. Download the embedding and reranker models (first run only)

Skip this step if you use the GPU service (`GPU_SERVICE_URL` set) — nothing is loaded locally
then.

The models (`BAAI/bge-m3`, `BAAI/bge-reranker-v2-m3`) are downloaded from Hugging Face and
cached locally the first time they're used. `run.sh` sets `HF_HUB_OFFLINE=1`, which blocks
that first download, so do one of these once before running the server or the ingest script:

```bash
HF_HUB_OFFLINE=0 uv run python -c "
from sentence_transformers import SentenceTransformer, CrossEncoder
SentenceTransformer('BAAI/bge-m3')
CrossEncoder('BAAI/bge-reranker-v2-m3')
print('models cached')
"
```
On Windows PowerShell, set the variable first instead of inlining it:
```powershell
$env:HF_HUB_OFFLINE = "0"
uv run python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; SentenceTransformer('BAAI/bge-m3'); CrossEncoder('BAAI/bge-reranker-v2-m3'); print('models cached')"
```

Needs ~2.8 GB of disk and a working internet connection. Subsequent runs work fully offline.

### 7. Build the vector store

Reads published pages/articles from Postgres, chunks them, and embeds them into Chroma. Run
once, and again whenever the source content changes (see the "Reindexing" section of
[`CLAUDE.md`](./CLAUDE.md) for the full re-embed procedure — deleting `chroma_db/` alone is
**not** enough):

```bash
uv run python -m app.rag.ingest
```
Locally this is CPU-bound and can take a while on a large corpus — expect anywhere from a few
minutes to a few hours depending on corpus size and available RAM. With `GPU_SERVICE_URL` set,
embedding runs on the GPU service in batches of 256 chunks and is much faster.

### 8. Run the server

**Ubuntu/macOS** — either use the helper script (syncs dependencies too, if you skipped the
manual steps above):
```bash
./run.sh
```
or run uvicorn directly:
```bash
HF_HUB_OFFLINE=1 uv run uvicorn app.main:app --host 0.0.0.0 --port 9200
```

**Windows** — `run.sh` is a bash script and won't run natively in PowerShell/cmd. Either use
**WSL** (Windows Subsystem for Linux) and follow the Ubuntu instructions inside it, or **Git
Bash**, or run uvicorn directly:
```powershell
$env:HF_HUB_OFFLINE = "1"
uv run uvicorn app.main:app --host 0.0.0.0 --port 9200
```

The API is available at `http://127.0.0.1:9200`.

## API Usage

- `GET /health` → `{"status": "ok"}`

- `POST /chat`
  - Request: `{"message": "your question in Bengali, Banglish, or English", "session_id": null, "user_id": null}`
    (omit `session_id` to start a new session; reuse the one returned to continue it)
  - Response:
    ```json
    {
      "mode": "qa",
      "answer": "... (always Bengali)",
      "sources": [
        {
          "book": "...", "chapter": "...", "source_db": "tarun",
          "content": "...", "similarity": 0.65, "rerank_score": 0.98
        }
      ],
      "session_id": "...",
      "response_time_ms": 842.17
    }
    ```
    `mode` is `qa` | `note` | `roleplay` | `suggestion`. `sources` is an empty array whenever
    nothing grounded the answer. If the LLM or the GPU service is unavailable, the response is
    a `503` with a Bengali "temporarily unavailable" detail.

- `POST /chat/stream` — same request; Server-Sent Events `sources`, then `token` (one or
  more), then `done` (`mode`/`session_id`/`response_time_ms`), or a single `error` event.

- `GET /conversations`, `GET /conversations/{id}/messages`, `DELETE /conversations/{id}`.

Example:
```bash
curl -s http://127.0.0.1:9200/chat -H "Content-Type: application/json" \
  -d '{"message": "যাকাতের অর্থ কোন কোন খাতে ব্যয় করা যায়?"}'
```

## GPU service (optional)

Embedding (`BAAI/bge-m3`) and reranking (`BAAI/bge-reranker-v2-m3`) can run on the separate
`shibir-chat-gpu-service` repository, deployed on Modal, instead of on this box's CPU. Set in
`.env`:

```
GPU_SERVICE_URL=https://<workspace>--shibir-chat-gpu-service-gpuservice-web.modal.run
GPU_API_KEY=...
```

With `GPU_SERVICE_URL` set, `app/rag/embedder.py` and `app/rag/reranker.py` call the service's
`/embed` and `/rerank` endpoints (`app/rag/gpu_client.py`) and never import torch. A call gets
a longer timeout (`GPU_COLD_TIMEOUT_SECONDS`, default 120) to cover a Modal cold start when it is
the first in the process or comes more than `GPU_WARM_WINDOW_SECONDS` (default 240, below
Modal's 300 s scaledown window) after the last successful one; otherwise `GPU_TIMEOUT_SECONDS`
(default 30). Connection errors (including a connection dropped mid-request), 5xx and 429 are
retried with backoff (`GPU_MAX_RETRIES`, default 2); read timeouts and other 4xx fail at once.

The service must serve the same embedding model as the existing Chroma index — every `/embed`
response's `model` and `dim` (1024) are checked, and a mismatch raises instead of writing
incompatible vectors. `/rerank` responses' `model` is checked against `RERANKER_MODEL_NAME` the
same way. Leave `GPU_SERVICE_URL` empty to use the local models.

## LLM tracing with Langfuse (self-hosted)

End-to-end tracing of live `/chat` requests: one trace per request, with every
LLM call **and** every RAG pipeline stage (intent → rewrite → retrieve → rerank
→ gate → generate) grouped under it. Use it to debug real misses ("the right
page *was* retrieved but scored 0.42, below the 0.5 gate") and to track
cost/latency/quality over time.

**It is opt-in and cannot affect the app.** With no keys set, the langfuse SDK
is never imported, `get_client()` returns the plain OpenAI client, and `/chat`
behaves byte-identically — no added latency, no background thread. When enabled,
event delivery is fire-and-forget on a background thread (nothing is flushed on
the request path), and every tracing call is wrapped so a slow/down/erroring
Langfuse can never break or delay a response.

### 1. Run a self-hosted Langfuse

```bash
git clone https://github.com/langfuse/langfuse
cd langfuse
docker compose up -d          # brings up Langfuse + its Postgres/ClickHouse
# open http://localhost:3000 , create an account + a project,
# then Project Settings → API Keys → create → copy the public & secret keys
```

Everything stays on your infra — trace data (user queries, book excerpts,
answers) never leaves the machine running that compose stack.

### 2. Turn it on in this app

Add to `.env` (see `.env.example`):

```
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=http://localhost:3000
```

Restart the server. That's it — `POST /chat` and `POST /chat/stream` now emit
traces. To turn it **off** again: remove the two keys (or set
`LANGFUSE_ENABLED=false`).

### What a trace contains

| Level | Data |
|---|---|
| trace | raw user query, resolved session id, optional `user_id`, final answer, cited sources, which mode ran, total latency |
| span `classify-intent` | the intent + how it was decided (`regex` / `llm` / `session`) |
| span `expand-query` | the query-rewrite variants that were searched |
| `retriever` observation `retrieve-context` | every reranked candidate — the ones kept **and** the ones dropped — each with a text preview + cosine similarity + rerank score + a `kept` flag (so "the right page was retrieved but reranked to 0.42" is visible at a glance) |
| `guardrail` observation `check-relevance-gate` | grounded vs refused, top rerank score, the threshold |
| `generation` `llm:{task}` (×N) | each LLM call: model, messages, completion, **token usage + cost**, latency, and a `task` label (`intent` / `rewrite` / `persona` / `qa` / `note` / `suggest` / `roleplay`) so you can filter cost/latency per task |

### Privacy / retention

Traces contain user queries and book excerpts. Self-hosted, that data stays on
your infra. API keys/secrets are never written into traces. To keep the trace
**structure** (spans, token usage, cost, latency, rerank scores, gate decision)
while redacting every query/excerpt/prompt/answer **text** field, set:

```
LANGFUSE_CAPTURE_IO=false
```

Retention is your call — configure it in the Langfuse project settings / its
data-retention job.

### Connecting evals to traces (scoring hook)

`app/core/tracing.py` exposes `score_trace(trace_id, name, value, comment=...)`,
a thin wrapper over Langfuse's score API. Use it to land quality signals on
logged real queries:

- a future frontend 👍/👎 button (POST the trace id back, call `score_trace`),
- a batch job that scores logged queries,
- the offline eval scripts (`eval_responses` / `eval_ragas`) emitting their
  per-question scores against traces of logged queries — attach the trace id to
  the run, then call `score_trace` per question.

No UI is built for this; only the function is provided.

## Project Structure

- `app/` — application code
  - `main.py` — FastAPI app factory
  - `core/config.py` — centralized settings (reads `.env`)
  - `api/router.py` — HTTP routes (`/health`, `/chat`, `/chat/stream`, `/conversations`)
  - `db/` — SQLAlchemy models and session (Postgres)
  - `services/` — orchestration layer (qa, note, roleplay, suggestion, intent classifier,
    session store)
  - `schemas/` — Pydantic request/response models
  - `rag/` — chunking, embedding, reranking, GPU-service client, query rewriting, Chroma
    client, retriever, generator, and the ingest script
- `scripts/` — one-time migration script and the `tune_threshold.py` refusal-threshold tuner
- `tests/` — pytest suite (`uv run pytest`). `tests/test_bengali_embedding.py` loads the real
  model and is skipped unless `RUN_MODEL_TESTS=1`.
- `chroma_db/` — generated vector store (gitignored; see "Reindexing" in `CLAUDE.md`)
- `pyproject.toml` / `uv.lock` — Python dependencies (cross-platform: Linux, macOS, Windows),
  managed with [uv](https://docs.astral.sh/uv/)

See [`CLAUDE.md`](./CLAUDE.md) for the full architecture, prompt contract, and
configuration reference.

## License

Specify your license here.
