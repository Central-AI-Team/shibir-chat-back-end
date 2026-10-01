# Shibir Chat Back-End

FastAPI back-end for **Shibir Chat**, a Bengali-language assistant that answers questions from a
curated library of books. Users can write in Bengali script, Banglish (romanized Bengali) or
English. The service retrieves the most relevant book excerpts and has an LLM answer **only from
those excerpts**, with numbered citations. When the library does not cover a question, it says so
instead of guessing.

This is a retrieval-augmented generation (RAG) service, not a general-purpose chatbot.

## Contents

- [How it works](#how-it-works)
- [Requirements](#requirements)
- [Quick start](#quick-start)
- [Configuration](#configuration)
- [API](#api)
- [Operations](#operations)
- [Development](#development)
- [Project layout](#project-layout)
- [Further documentation](#further-documentation)
- [License](#license)

## How it works

```
                    POST /chat  {"message": "..."}
                              │
                     intent classifier  (regex first, LLM fallback)
          ┌──────────────┬────┴─────────┬──────────────────┐
          ▼              ▼              ▼                  ▼
         QA         SUGGESTION        NOTE             ROLEPLAY
          │              │              │                  │
   query rewrite → embed (bge-m3)       │           persona + history
   → Chroma search → rerank             │           (no retrieval)
   (bge-reranker-v2-m3)                 │
          │              │     every page of the book,
   relevance gate (MIN_RERANK_SCORE)    map-reduced into a note
          │              │              │                  │
          └──────────────┴──────┬───────┴──────────────────┘
                                ▼
                  OpenAI gpt-5-mini → Bengali answer + sources
```

- **Content** lives in PostgreSQL (`categories` → `books` → `chapters` → `pages`, plus `articles`).
- **Embeddings** of chunked pages live in a local Chroma vector store (`chroma_db/`).
- **Embedding and reranking** run either on this machine's CPU or on the separate
  `shibir-chat-gpu-service` deployed on [Modal](https://modal.com). See [GPU service](#gpu-service).
- **The LLM** is OpenAI `gpt-5-mini`, called through a single module (`app/core/llm.py`).

## Requirements

| Requirement | Notes |
|---|---|
| Python 3.11+ | uv downloads a matching interpreter automatically if needed. |
| [uv](https://docs.astral.sh/uv/getting-started/installation/) | Manages the virtual environment and dependencies. |
| PostgreSQL | Must already contain the book content. This service reads it and does not seed it. |
| OpenAI API key | https://platform.openai.com/api-keys |
| **Either** ~6 GB free RAM and ~3 GB disk | For the local embedding and reranker models (CPU mode). |
| **or** a `shibir-chat-gpu-service` URL and API key | GPU mode. No local models are loaded. |

## Quick start

```bash
# 1. Clone and install dependencies (exact versions from uv.lock)
git clone <repository-url>
cd shibir-chat-back-end
uv sync --locked

# 2. Configure
cp .env.example .env          # Windows: copy .env.example .env
# edit .env: set OPENAI_API_KEY and DATABASE_URL (and GPU_SERVICE_URL / GPU_API_KEY for GPU mode)

# 3. CPU mode only: download the models once (~2.8 GB)
HF_HUB_OFFLINE=0 uv run python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; SentenceTransformer('BAAI/bge-m3'); CrossEncoder('BAAI/bge-reranker-v2-m3')"

# 4. Build the vector store (embeds new or changed pages only)
HF_HUB_OFFLINE=1 uv run python -m app.rag.ingest

# 5. Run the server on :9200
./run.sh
# or: HF_HUB_OFFLINE=1 uv run uvicorn app.main:app --host 0.0.0.0 --port 9200
```

Check it:

```bash
curl -s http://127.0.0.1:9200/health
CHAT_IDENTITY=$(curl -s -X POST http://127.0.0.1:9200/identity | python -c "import json,sys; print(json.load(sys.stdin)['token'])")
curl -s http://127.0.0.1:9200/chat -H "Content-Type: application/json" \
  -H "X-Chat-Identity: $CHAT_IDENTITY" \
  -d '{"message": "যাকাতের অর্থ কোন কোন খাতে ব্যয় করা যায়?"}'
```

Interactive API docs are served at `http://127.0.0.1:9200/docs`.

### Windows

`run.sh` is a bash script. Use WSL or Git Bash, or run uvicorn directly in PowerShell:

```powershell
$env:HF_HUB_OFFLINE = "1"
uv run uvicorn app.main:app --host 0.0.0.0 --port 9200
```

`psycopg2-binary` bundles `libpq`, and the CUDA/`uvloop` packages carry platform markers, so
`uv sync` works on Linux, macOS and Windows without extra system packages.

### Docker

```bash
docker compose up --build
```

`docker-compose.yml` starts the app together with an **empty** Postgres 16 container. Load the
content into it (or point `DATABASE_URL` at an existing database) and run the ingest step before
expecting answers. Chroma data and the Hugging Face model cache are kept in named volumes.

## Configuration

All settings are environment variables, read from `.env` by `app/core/config.py`. Only two are
required:

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY` | OpenAI key used for every LLM call. |
| `DATABASE_URL` | SQLAlchemy DSN, e.g. `postgresql+psycopg2://user:password@localhost:5432/shibir_chat`. |

Commonly changed optional settings:

| Variable | Default | Purpose |
|---|---|---|
| `GPU_SERVICE_URL` / `GPU_API_KEY` | *(empty)* | Enable GPU mode (see below). |
| `MIN_RERANK_SCORE` | `0.5` | Relevance gate. Below it, the answer is not grounded and `sources` is empty. |
| `TOP_K` / `FETCH_K` | `5` / `25` | Excerpts sent to the LLM / candidates fetched before reranking. |
| `LLM_REQUEST_TIMEOUT_SECONDS` | `90` | Timeout per LLM call. |
| `CORS_ALLOW_ORIGINS` | local Vite + production front-end | JSON list of allowed browser origins. |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` | *(empty)* | Enable tracing (see [docs/tracing.md](docs/tracing.md)). |

`.env.example` lists every variable with comments. The full reference, including why each default
was chosen, is in the Configuration section of [CLAUDE.md](CLAUDE.md#11-configuration-appcoreconfigpy).

### GPU service

Set `GPU_SERVICE_URL` and `GPU_API_KEY` to run embedding and reranking on `shibir-chat-gpu-service`
(Modal) instead of locally. In this mode the app never imports torch.

- The first call, or any call after `GPU_WARM_WINDOW_SECONDS` (240 s) of inactivity, gets a longer
  timeout (`GPU_COLD_TIMEOUT_SECONDS`, 120 s) to cover the container's cold start. Other calls
  use `GPU_TIMEOUT_SECONDS` (30 s).
- Connection failures, 5xx and 429 responses are retried with exponential backoff
  (`GPU_MAX_RETRIES`, default 2). Other errors fail immediately.
- Every response's `model` (and `dim` for embeddings) is checked against the configured models,
  so a misconfigured service can never write incompatible vectors into the index.
- If the service is unavailable, `/chat` returns `503` and `/chat/stream` emits an `error` event.

Leave `GPU_SERVICE_URL` empty to use the local models.

## API

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness check. Returns `{"status": "ok"}`. |
| `POST` | `/chat` | Main entry point. Classifies the message and returns an answer. |
| `POST` | `/chat/stream` | Same as `/chat`, as Server-Sent Events. |
| `GET` | `/conversations` | Only your durable conversations, most recent first. |
| `GET` | `/conversations/{id}/messages` | Messages in one session (`404` if unknown). |
| `DELETE` | `/conversations/{id}` | Delete a session (`204`, or `404` if unknown). |

All chat, conversation and memory endpoints require `X-Chat-Identity`. Obtain a
private credential from `POST /identity` once and retain it securely. Identity is
verified by its hashed opaque token; the request's `user_id` field never grants
ownership. The frontend creates and retains this credential automatically.

### `POST /chat`

Request:

```json
{ "message": "your question", "session_id": null, "user_id": null }
```

- `session_id`: omit to start a new session. Send back the returned value to continue it.
- `user_id`: deprecated compatibility field, ignored. Traces use the verified identity.

Response:

```json
{
  "mode": "qa",
  "answer": "… (always Bengali)",
  "sources": [
    {
      "book": "…", "chapter": "…", "source_db": "tarun",
      "content": "…", "similarity": 0.65, "rerank_score": 0.98
    }
  ],
  "session_id": "…",
  "response_time_ms": 842.17
}
```

| `mode` | Behavior | `sources` |
|---|---|---|
| `qa` | Answers from retrieved excerpts, with `[১]`-style citations. | Excerpts used, or `[]` if nothing passed the relevance gate. |
| `suggestion` | Book-grounded recommendation; clearly labelled as general advice when the books do not cover it. | As for `qa`. |
| `note` | Structured summary of a whole book named in the message. | `[]` |
| `roleplay` | Multi-turn conversation in a persona. Continues until an exit phrase such as "stop roleplay". | `[]` |

Errors: `400` for an empty message; `503` with a Bengali "temporarily unavailable" message when
the LLM or the GPU service fails.

### `POST /chat/stream`

Same request body. The response is `text/event-stream` with these events, in order:

1. `session`: `{"session_id", "request_id"}`, before generation starts.
2. `sources`: `{"sources": [...]}`.
3. Optional `web_results`: `{"results": [...]}` and `verification`: `{"verification": {...}}`,
   when provided by the server.
4. `token` (one or more): `{"text": "..."}`. QA streams tokens; other modes send
   the whole answer in one event.
5. `done`: the complete chat response, including `resources`, compatible top-level
   resource fields, `mode`, `session_id`, `answer`, and `response_time_ms`.

Failures emit an `error` event (`{"detail": "..."}`); partial resource snapshots
remain durable. Conversations are shared across workers and survive restarts.

## Conversation memory

Conversations, full message history, summaries and user preferences are stored in
four PostgreSQL tables. Startup creates missing chat tables and upgrades existing
chat message tables under an advisory lock; corpus tables are untouched. For a deployment role without DDL
permission, run `uv run python -m scripts.create_chat_tables` with a migration role
before starting the app.

All modes use bounded context (recent messages, a summary and relevant older
messages). Follow-up references are resolved before classification and retrieval.
English `it`/`that`, Bengali references, and short Banglish/Bengali acceptances such
as `dao`, `daw`, `দাও`, and `নোট দাও` use the latest topic or assistant offer.
Ambiguous targets ask for clarification. Bare acceptance in a fresh conversation
requires a local antecedent; it does not silently select an archived conversation.
Latin matching uses word boundaries so `habit`/`credit` do not trigger `it`.
Standalone requests naming their target continue normally.
Memory is context, never authoritative book evidence. `/chat` and `/chat/stream`
share the same preparation. A separate `memory` mode recalls your prior discussions
with no book citations. Roleplay personas remain local to their conversation.

Summaries and explicit preferences/goals are refreshed after responses. If the
model is unavailable, full transcripts remain searchable; the summary cursor is
not advanced and a later turn retries. `POST /conversations` accepts a client-chosen
UUID for idempotent creation; `/chat` never recreates an unknown/deleted ID. Use a
UUID `request_id` to replay a completed request or retry an interrupted one without
duplicating messages. Concurrent turns in one chat return 409. A worker claim
expires after 15 minutes if its process dies.

`GET /memories` lists saved facts. `DELETE /memories/{id}` forgets facts from that
source chat and excludes the source chat from cross-chat recall. A transcript
remains visible to its owner. `PATCH /conversations/{id}` supports `title` and
`memory_enabled`; disabling memory also removes facts sourced from that chat.
Deleting a conversation removes its messages and sourced memories. The frontend's
Memory panel exposes forgetting and excluding the current chat, and its active
conversation survives a page refresh.

Identity currently belongs to this browser/login-token scope, not a verified
account across devices. This backend does not implement the UI's account-login
endpoints. Clearing browser storage loses the guest credential. Existing ephemeral
sessions cannot be recovered after restart or assigned to an owner safely.

`CHAT_CONTEXT_TOKEN_BUDGET` defaults to 6000; UTF-8 byte accounting provides a
conservative upper bound on context tokens, reserving framing space.
`CHAT_MEMORY_RESULTS` defaults to 4. Archive retrieval uses keyword/suffix overlap
and recency over transcripts and summaries; it is not semantic vector retrieval.
New model-routing tasks are `context`, `memory`, and `memory_answer`, configurable
through `MODEL_BY_TASK` like existing tasks.

Run `uv run python -m scripts.context_memory_demo` for an isolated, mocked
before/after prompt inspection. Chat tests use temporary SQLite; set
`CHAT_TEST_DATABASE_URL` to an isolated PostgreSQL database to run them in temporary
schemas that are dropped after each test.

### Durable message resources

Each message has a versioned `resources` snapshot: `version: 1`, ordered `sources`,
`web_results`, and `verification`. Citations retain JSON metadata such as page,
section, URL and provider provenance. Normal responses, streaming completion,
request replay and conversation history return the same snapshot; top-level
`sources`, `web_results` and `verification` remain compatibility aliases. The
frontend uses the same normalization for live responses and restored history.

Resources are committed before their SSE events are sent. Stopping a response,
a lost connection, or a worker crash keeps its checkpoint; the conversation lease
prevents an older worker overwriting a newer response. Request options are saved
for regeneration, and reusing a request ID with different text/options returns 409.

Startup adds `chat_messages.resources` and `chat_messages.options` to an existing
chat database and backfills resources from stored citations. Repeated upgrades
preserve newer snapshots. Previously unsaved browser-only cards cannot be recovered.
The synthetic offline response was removed: connectivity failures now surface as
retryable errors instead of displaying unsaved citations or verification results.

This stores server-supplied web/verification resources; this backend does not yet
implement live web-search or claim-verification providers. File-upload persistence
is outside this resource contract. Run `uv run python -m scripts.message_resources_demo`
for a local reload/replay/interruption demonstration with explicit fixture cards.

## Operations

### Ingesting and reindexing

`uv run python -m app.rag.ingest` embeds every published page or article that is new or has
changed since it was last embedded (tracked by `embedded_at`), and first removes the chunks of any
row that has been unpublished or excluded since. Run it after content changes.

A **full reindex** (for example after changing the embedding model) requires both deleting
`chroma_db/` **and** resetting `embedded_at` to `NULL` in Postgres. Deleting only the directory
produces an empty index while the ingest reports success. The exact steps are in
[CLAUDE.md → Ingestion and reindexing](CLAUDE.md#9-ingestion-and-reindexing).

On the current corpus (about 4,100 pages, 12,000 chunks) a CPU reindex can take hours on a
memory-constrained machine. GPU mode is much faster.

### Adding a data source

New content goes **loader → Postgres → ingest**. Loaders only write Postgres rows; only
`app.rag.ingest` writes to the vector store.

```bash
# 1. Load into Postgres (new rows are drafts unless --publish; --dry-run rolls back)
uv run python -m scripts.load_tafheem --dry-run                     # data/tafheemul_quran.db
uv run python -m scripts.load_articles --source cs-posts --dry-run   # or --source pp-articles
uv run python -m scripts.load_articles --source cs-posts --publish

# 2. Embed whatever is new or changed
HF_HUB_OFFLINE=1 uv run python -m app.rag.ingest
```

Each loader prints what it parsed, skipped (malformed / too short / repetitive / duplicate) and
wrote, plus sample rows; read it before publishing. Re-running a loader is safe: rows are upserted
on their source id and only real changes are re-embedded. `--publish` also sets the older import of
the same source to draft. To write a new loader, follow
[CLAUDE.md → Adding a data source](CLAUDE.md#adding-a-data-source).

### Deployment

Production runs as the systemd unit `shibirgpt.service`, which executes `run.sh`. The
`deploy.yml` GitHub Actions workflow is still a placeholder. For scaling beyond one instance, see
[docs/deployment.md](docs/deployment.md).

### Observability

Optional Langfuse tracing records each request's pipeline stages, rerank scores, LLM calls, token
usage and cost. It is off by default and cannot affect requests when on. See
[docs/tracing.md](docs/tracing.md).

## Development

```bash
uv run pytest                                               # full suite, no network or models needed
RUN_MODEL_TESTS=1 uv run pytest tests/test_bengali_embedding.py   # loads the real bge-m3 model
TEST_DATABASE_URL=postgresql+psycopg2://user:pass@localhost:5432/shibir_chat_test uv run pytest
```

Tests that write to Postgres (the loader tests) use `TEST_DATABASE_URL`, falling back to
`DATABASE_URL`, and are skipped unless the database name contains `test`, so they can never touch
the real corpus. Each runs in a rolled-back transaction.

CI (`.github/workflows/ci.yml`) runs the test suite against a Postgres service on every pull
request to `main`.

Quality evaluation scripts (retrieval recall, threshold tuning, intent routing, answer quality)
live in `scripts/` and are described in [docs/evaluation.md](docs/evaluation.md).

See [CONTRIBUTING.md](CONTRIBUTING.md) for the branch, commit and review workflow.

## Project layout

```
app/
  main.py            FastAPI app, CORS, startup/shutdown hooks
  api/router.py      HTTP routes
  core/              settings (config.py), LLM client (llm.py), tracing (tracing.py)
  services/          qa, suggestion, note, roleplay, intent classifier, session store
  rag/               chunker, embedder, reranker, GPU client, query rewriter,
                     Chroma client, retriever, generator, ingest script
  loaders/           raw corpus readers and text cleaning for scripts/load_*.py
  db/                SQLAlchemy models and session
  schemas/           Pydantic request/response models
scripts/             data loaders, migrations, corpus cleanup, evaluation and debugging tools
tests/               pytest suite
docs/                deployment, evaluation and tracing guides
chroma_db/           generated vector store (not committed)
```

## Further documentation

| Document | Audience |
|---|---|
| [CLAUDE.md](CLAUDE.md) | Maintainers and AI coding agents: architecture, contracts that must not regress, full configuration reference, known gaps. |
| [CONTRIBUTING.md](CONTRIBUTING.md) | How to set up, branch, commit, test and open pull requests. |
| [docs/deployment.md](docs/deployment.md) | Scaling beyond a single instance. |
| [docs/evaluation.md](docs/evaluation.md) | Measuring retrieval and answer quality. |
| [docs/tracing.md](docs/tracing.md) | Langfuse tracing setup and trace contents. |
| [SECURITY.md](SECURITY.md) | Reporting vulnerabilities and handling secrets. |
| [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) | Expectations for working together. |

## License

No license has been chosen yet. Until one is added, all rights are reserved by the project owners
and the code may not be used or redistributed outside the team.
