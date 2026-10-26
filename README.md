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

### Query rewrite

The books are in Bengali, so the search query must be too. Before retrieval,
`expand_query()` (`app/rag/query_rewriter.py`) turns the question into one Bengali search string:

| Input | What happens |
|---|---|
| Entirely Bengali script | Used as is, with no LLM call. |
| Banglish, English, Arabic, or Bengali mixed with Latin/Arabic letters | An LLM call rewrites it into standard Bengali, using the books' own terms (prayer → নামাজ, riba → রিবা/সুদ) rather than a literal translation. |

For these inputs the rewrite normally comes from the same LLM call that classifies intent. The
reply must be at least 80 % Bengali; otherwise it is retried once, and if that also fails the raw
query is used and a warning is logged. The answer is still written in the language of the
question. Details and measurements: `CLAUDE.md` §5.

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
# or: HF_HUB_OFFLINE=1 uv run uvicorn app.main:app --host 127.0.0.1 --port 9200
```

Check it:

```bash
curl -s http://127.0.0.1:9200/health
curl -s http://127.0.0.1:9200/chat -H "Content-Type: application/json" \
  -d '{"message": "যাকাতের অর্থ কোন কোন খাতে ব্যয় করা যায়?"}'
```

Interactive API docs are off by default; set `ENABLE_DOCS=true` in `.env` to serve them at
`http://127.0.0.1:9200/docs` (local development only).

### Windows

`run.sh` is a bash script. Use WSL or Git Bash, or run uvicorn directly in PowerShell:

```powershell
$env:HF_HUB_OFFLINE = "1"
uv run uvicorn app.main:app --host 127.0.0.1 --port 9200
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
| `GET` | `/ready` | Readiness check: Postgres and Chroma answer. `503` naming the failed check otherwise. |
| `POST` | `/chat` | Main entry point. Classifies the message and returns an answer. |
| `POST` | `/chat/stream` | Same as `/chat`, as Server-Sent Events. |
| `GET` | `/conversations` | The caller's own sessions, most recent first. |
| `GET` | `/conversations/{id}/messages` | Messages in one of the caller's sessions (`404` if unknown or not theirs). |
| `DELETE` | `/conversations/{id}` | Delete one of the caller's sessions (`204`, or `404` if unknown or not theirs). |
| `POST` | `/auth/register` | Create an account and log in (open unless `AUTH_REGISTRATION_ENABLED=false`). |
| `POST` | `/auth/login` | Log in; sets the session cookie. |
| `GET` | `/auth/me` | The logged-in user, or a guest's remaining free messages. |
| `POST` | `/auth/logout`, `/auth/logout-all` | End this session, or every session of the user. |
| `POST` | `/auth/change-password` | Change the password; ends every other session. |

**Accounts and guests.** Full contract: [docs/auth.md](docs/auth.md). Without logging in, a
browser is a guest: the first request receives an HttpOnly `shibir_owner` cookie, its sessions
belong to that cookie, and it gets `GUEST_MESSAGE_LIMIT` (5) answers in total. After that `/chat`
returns 401 with `"code": "guest_limit_reached"` until the user registers or logs in. Guests cannot
request notes. A logged-in user's sessions follow the account across browsers, and a guest's chats
move to the account on login. Nobody else can list, read, continue or delete a session. The
browser must send cookies: call the API with `fetch(url, { credentials: "include" })`. Chat
history is stored in Postgres and survives restarts; a guest's conversations are purged after
`GUEST_HISTORY_RETENTION_DAYS` (30) without activity. For local development over plain `http://`, set
`COOKIE_SECURE=false`, or the browser drops the cookies. Create the first admin with
`uv run python -m scripts.create_user --email you@example.com --role admin` (the app creates its
account tables on startup).

### `POST /chat`

Request:

```json
{ "message": "your question", "session_id": null, "user_id": null }
```

- `session_id`: omit to start a new session. Send back the returned value to continue it. It must
  be an id this server issued to this browser. Any other value (malformed, unknown, deleted,
  or someone else's) starts a new session instead, so always use the
  `session_id` returned by the latest response. The server never creates a session from a
  client-chosen id.
- `user_id`: accepted for compatibility but ignored (it is client-supplied and spoofable).

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
| `roleplay` | Mentor mode: a role from the allow-list (বন্ধু, দায়িত্বশীল ভাই, শিক্ষক, অধ্যয়ন-সার্কেল পরিচালক) answers from the books in its tone, adapted to the reader's level, with next steps; `suggestions` carries 2–3 follow-up questions. Requests to speak as a real person are refused with a normal answer. Continues until an exit phrase such as "stop roleplay". | As for `qa`. |

Errors: `400` for an empty message; `503` with a Bengali "temporarily unavailable" message when
the LLM or the GPU service fails.

### `POST /chat/stream`

Same request body. The response is `text/event-stream` with these events, in order:

1. `sources`: `{"sources": [...]}`
2. `token` (one or more): `{"text": "..."}`. `qa` streams token by token; other modes send
   the whole answer in one event.
3. `done`: `{"mode", "session_id", "response_time_ms", "suggestions"}`

On an upstream failure a single `error` event (`{"detail": "..."}`) is sent instead.

Conversations and messages live in the Postgres tables of the same name (schema managed by
Alembic: `uv run alembic upgrade head` after pulling a change under `migrations/`). The
history returned to the model and by `GET /conversations/{id}/messages` is the last 20 turns;
every turn is stored. Sessions no longer pin the app to one uvicorn worker; the in-process
rate limits (`app/services/rate_limiter.py`) still count per worker.

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

Other loaders follow the same pattern: `scripts.load_books` (whole-book blobs),
`scripts.load_shaheed` (martyr list) and `scripts.load_leadership` (the central president /
secretary-general list from `data/shibir-central-president.csv`, loaded as one article). Each loader
has default input paths; pass `--sql` / `--db` / `--csv` when your files live elsewhere (for example
`data/archive-data/`).

Each loader prints what it parsed, skipped (malformed / too short / repetitive / duplicate) and
wrote, plus sample rows; read it before publishing. Re-running a loader is safe: rows are upserted
on their source id and only real changes are re-embedded. `--publish` also sets the older import of
the same source to draft. To write a new loader, follow
[CLAUDE.md → Adding a data source](CLAUDE.md#adding-a-data-source).

### Moving the corpus to another server

`uv run python -m scripts.export_domains` writes one re-runnable SQL file per domain to
`data/export/` (`books`, `tafheem`, `articles`, `shaheed`, `leadership`, `syllabus`,
`site-directory`). Restore them in any order, then ingest:

```bash
for d in books tafheem articles shaheed leadership syllabus site-directory; do
  docker compose exec -T db psql -U postgres -d shibir_chat < data/export/$d.sql
done
HF_HUB_OFFLINE=1 uv run python -m app.rag.ingest
```

`data/` is git-ignored: the files hold full book text, so copy them to the server directly.
Exports set `embedded_at` to `NULL`, so ingest embeds everything it restored.

### Deployment

Production runs with Docker Compose on a VPS and deploys automatically: every push to `main`
rebuilds it, runs `alembic upgrade head`, checks `/health` and rolls back to the previous commit
if the check fails. One-time
setup: [docs/auto-deploy.md](docs/auto-deploy.md). For scaling beyond one instance, see
[docs/deployment.md](docs/deployment.md).

**The app must never be exposed directly.** It relies on the proxy for TLS, so `docker-compose.yml`
publishes it on `127.0.0.1:9200` only, and a TLS reverse proxy is the single public entry point
(`https://askshibir-back.potropollob.com`, front-end origin `https://shibirgpt.potropollob.com`).
Ready-made configs, with HTTP→HTTPS redirect, SSE-safe proxying for `/chat/stream`, a 64 KB body
limit and per-IP rate limiting on `/chat`:

- nginx: [deploy/nginx/shibirgpt-api.conf](deploy/nginx/shibirgpt-api.conf)
- Caddy: [deploy/caddy/Caddyfile](deploy/caddy/Caddyfile) (rate limiting needs the `caddy-ratelimit` plugin)

`/docs`, `/redoc` and `/openapi.json` are disabled unless `ENABLE_DOCS=true`; leave it unset in
production. uvicorn trusts `X-Forwarded-*` only from the proxy (`FORWARDED_ALLOW_IPS`, set to the
pinned Compose network gateway `172.28.0.1` in `docker-compose.yml`), so `request.client.host` is
the real client IP.

#### Firewall

Allow only SSH, HTTP and HTTPS; port 9200 stays closed (it is also bound to loopback, so this is a
second layer). Note that Docker-published ports bypass `ufw` rules, which is why the loopback bind
in `docker-compose.yml` is the primary control.

```bash
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow 22/tcp
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable
sudo ufw status verbose   # 9200 must not be listed
```

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
scripts/             data loaders, corpus export, migrations, corpus cleanup, evaluation and debugging tools
tests/               pytest suite
docs/                auto-deploy, deployment, evaluation and tracing guides
chroma_db/           generated vector store (not committed)
```

## Further documentation

| Document | Audience |
|---|---|
| [CLAUDE.md](CLAUDE.md) | Maintainers and AI coding agents: architecture, contracts that must not regress, full configuration reference, known gaps. |
| [CONTRIBUTING.md](CONTRIBUTING.md) | How to set up, branch, commit, test and open pull requests. |
| [docs/auto-deploy.md](docs/auto-deploy.md) | One-time setup for automatic deploys to the VPS. |
| [docs/deployment.md](docs/deployment.md) | Scaling beyond a single instance. |
| [docs/evaluation.md](docs/evaluation.md) | Measuring retrieval and answer quality. |
| [docs/tracing.md](docs/tracing.md) | Langfuse tracing setup and trace contents. |
| [SECURITY.md](SECURITY.md) | Reporting vulnerabilities and handling secrets. |
| [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) | Expectations for working together. |

## License

No license has been chosen yet. Until one is added, all rights are reserved by the project owners
and the code may not be used or redistributed outside the team.
