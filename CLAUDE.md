# Shibir Chat Back-End — Claude Code Project Guide

Claude Code auto-loads `CLAUDE.md` from the repo root at the start of every session — that's
why this file exists. It's written so Claude Code can pick up this project cold and be
immediately useful. Read this before making changes.

## What this project is

A FastAPI backend for a Bengali-language Q&A chatbot. A user asks a question, the service
retrieves relevant excerpts from a religious/educational book library (stored as embeddings
in a Chroma vector store) and asks an LLM (OpenAI `gpt-5-mini`) to answer **strictly from
those excerpts** — it is instructed to refuse rather than hallucinate when the excerpts don't
cover the question. This is retrieval-augmented generation (RAG), not a general-purpose
chatbot.

## Architecture

```
app/
  main.py               FastAPI app factory. Just creates the app and includes the router.
  core/config.py         Single source of config — a pydantic-settings Settings object that
                          reads .env. Nothing else in the app should call os.getenv() or
                          load_dotenv() directly; import `settings` from here instead.
  core/llm.py             Shared OpenAI chat-completions client + per-task model routing —
                          see "LLM provider" below. The only sanctioned way to call the LLM;
                          no call site should construct an OpenAI client directly.
  core/tracing.py         Langfuse tracing, opt-in — see "Tracing" below.
  api/router.py           HTTP layer. Routes: GET /health, POST /chat, POST /chat/stream,
                          GET /conversations, GET /conversations/{id}/messages,
                          DELETE /conversations/{id}. Thin — just binds request schemas to
                          the service layer and returns the result. See "API contract" below.
  db/
    models.py                SQLAlchemy models: Category, Book, Chapter, Page, Article. All
                              content now lives in Postgres (migrated off the old per-source
                              SQLite files) — see "Data model" below.
    session.py                SessionLocal — plain sessionmaker bound to settings.database_url.
  services/
    qa_service.py               Orchestration: answer_question(query) calls
                                 retrieve_relevant_docs() then decides, from the top
                                 rerank_score alone, whether to ground generate_answer() in
                                 those citations or call it with an empty citation list. See
                                 "Prompt contract" below — this is the fix for the old
                                 "answer says no but sources are attached anyway" bug.
    note_service.py              generate_chapter_note(chapter_id) -> dict. NOT retrieval —
                                  pulls every published page of a chapter in reading order
                                  from Postgres and map-reduces it into one note. A chapter
                                  summary is an aggregation task; similarity search over a
                                  handful of chunks can't do that. generate_book_notes_from_text(text)
                                  is what POST /chat's NOTE intent actually calls: it resolves
                                  a book from free-form text (plain substring match against
                                  book names, not an LLM call), then runs generate_chapter_note()
                                  over every chapter of that book.
    roleplay_service.py          handle_roleplay(message, session) -> str. Extracts a persona
                                  from the user's first roleplay message (one LLM call, task
                                  "persona"), then holds a multi-turn conversation in that
                                  persona using the session's history. No RAG — see "Roleplay
                                  and suggestion services" below.
    suggestion_service.py        give_suggestion(query) -> (str, list[Citation]). Same
                                  retrieval + relevance gate as qa_service, but the prompt is
                                  explicitly allowed to phrase a recommendation/opinion instead
                                  of only restating cited facts — grounded when the gate
                                  passes, an explicit "not book-grounded" disclaimer when it
                                  doesn't. See "Roleplay and suggestion services" below.
    session_store.py              In-process dict of chat sessions (mode, persona, history).
                                   Backs POST /chat's session_id continuity AND the
                                   /conversations endpoints. Single-worker only — see
                                   "API contract" below.
    intent_classifier.py          classify_intent(message, has_active_roleplay_session) -> one
                                   of NOTE/ROLEPLAY/SUGGESTION/QA. Regex-first (Bengali +
                                   Banglish patterns), falls back to one LLM call (task
                                   "intent") only when nothing matches. An active roleplay
                                   session short-circuits to ROLEPLAY unless the message
                                   matches an explicit exit phrase.
  schemas/query.py           Pydantic models. ChatRequest (message/session_id/user_id) and
                              ChatResponse (mode/answer/sources/session_id/response_time_ms)
                              back POST /chat and /chat/stream — see "API contract" below.
                              Citation (book/chapter/source_db/content/similarity/rerank_score)
                              is the shared source shape. ConversationSummary and
                              ConversationMessage back the /conversations endpoints.
                              QueryRequest/QueryResponse/NoteRequest/NoteByTextRequest/
                              NoteByTextResponse/ChapterNote are leftover from the removed
                              /ask, /note, /note-by-text routes. QueryResponse is still used
                              internally as qa_service.answer_question()'s return type; the
                              other five are unused by any route or service today.
  rag/
    chunker.py                  normalize(text) (NFC + whitespace cleanup — apply at ingest
                                 AND query time) and chunk_text(text) (paragraph → danda →
                                 period → space splitting, ~900 chars/chunk, 150 overlap).
                                 Previously ingest.py embedded one whole DB page as one
                                 vector; pages now split into multiple chunks first.
    embedder.py                  embed_text(s) via sentence-transformers, BAAI/bge-m3
                                  (1024-dim, multilingual). Loaded once at import time (lru_cache).
    reranker.py                   rerank(query, docs, top_n) via a CrossEncoder,
                                   BAAI/bge-reranker-v2-m3. Scores are Sigmoid-activated,
                                   i.e. in [0, 1] — NOT raw logits (see config.py note on
                                   min_rerank_score before assuming a >2 "clearly relevant"
                                   cutoff; that heuristic does not apply here).
    query_rewriter.py             expand_query(query) -> tuple[str, ...]. Converts Banglish /
                                   English / Bengali input into Bengali-script search variants
                                   via one LLM call routed through complete("rewrite", ...) —
                                   see "LLM provider" below — cached (lru_cache). Falls back to
                                   the raw query on any failure — including on a truncated/empty
                                   LLM response, so watch for silent no-op fallback if you ever
                                   lower the token budget again (see the comment in that file
                                   for the incident this guards against).
    chroma_client.py               get_collection() — the ONLY place that constructs the
                                    Chroma PersistentClient. Collection is created with
                                    hnsw:space="cosine" (required for the similarity
                                    thresholding in retriever.py to make sense — Chroma's
                                    default metric is L2).
    retriever.py                   retrieve_relevant_docs(query) -> list[Citation]. Pipeline:
                                    expand_query() → embed all variants → Chroma fetch_k=25
                                    nearest neighbors per variant, merged and deduped → drop
                                    anything below min_similarity → cross-encoder rerank down
                                    to top_k=5. Returns citations carrying both similarity
                                    (cosine, cheap pre-filter) and rerank_score (cross-encoder,
                                    the real relevance signal).
    generator.py                   generate_answer(query, citations) -> str, routed through
                                    complete("qa", ...). Bengali system prompt, numbered
                                    excerpts for citation. No temperature override — gpt-5-mini
                                    is a reasoning model and rejects any value other than its
                                    fixed default (1); see "LLM provider" below. Also exposes
                                    stream_answer() for POST /chat/stream, same prompt/context,
                                    stream=True. See "Prompt contract" below.
    ingest.py                      Standalone script (uv run python -m app.rag.ingest). Reads
                                    published pages/articles from Postgres (NOT from SQLite —
                                    that migration is one-time and already done), chunks each,
                                    embeds in batches of 64, and upserts into Chroma. Deletes
                                    a row's old chunks before re-upserting (chunk count changes
                                    when content is edited, so upsert-only would leave stale
                                    orphaned chunks). Only processes rows where
                                    embedded_at IS NULL OR updated_at > embedded_at.

scripts/
  migrate_sqlite_to_postgres.py   One-time migration, already run. Do not re-run against a
                                   live corpus.
  tune_threshold.py                uv run python -m scripts.tune_threshold [questions.json]. Runs a
                                    labeled question set through the real retrieval pipeline
                                    and suggests a MIN_RERANK_SCORE from the score gap between
                                    answerable and unanswerable questions. See
                                    eval_questions.example.json for the input shape.

chroma_db/                Generated vector store (gitignored). See "Reindexing" below —
                           deleting this is only half of a reindex.
```

There are deliberately only three layers (api → services → rag/db). There is exactly one
vector store and one LLM provider, so no repository/interface abstraction was introduced —
it would add indirection with no real swappability benefit at this size.

## Data model

Content lives in Postgres now (`categories`, `books`, `chapters`, `pages`, `articles` —
`app/db/models.py`). The old per-source SQLite files (`Tarun_Associate.db`,
`Nobin_Associate.db`) were migrated in with `scripts/migrate_sqlite_to_postgres.py`; each
`Page` keeps `source_db` / `source_page_id` purely as provenance so the migration script is
idempotent, not for anything the running app reads.

Only `status == 'published'` rows are ever candidates for embedding (`ingest.py` filters on
this). `embedded_at` tracks the last successful embed per row — `NULL` means "never
embedded", and `updated_at > embedded_at` means "stale, needs re-embedding".

`Page` also has `excluded_from_rag` (bool) / `exclusion_reason` (text), added by
`scripts/clean_corpus.py`'s migration. `ingest.py._due_pages()` skips
`excluded_from_rag == true` rows in addition to the `published`/`embedded_at` filters above —
this is a reversible soft-exclude for pages a corpus analysis flagged as junk/duplicate (no
Postgres row is ever deleted; flipping the flag back + re-ingesting restores a page). See
"Known gaps" below for which flagged pages are actually excluded today.

Chroma chunk ids are `f"{prefix}_{row.id}_c{chunk_index}"` (e.g. `page_842_c0`, `page_842_c1`,
`article_12_c0`) with metadata `{"book", "chapter", "page_id", "chunk_index", "row_key",
"source_db", "book_id", "category"}`. `row_key` (`f"{prefix}_{row.id}"`) is what `ingest.py`
deletes-by before re-upserting a row's chunks. `book_id`/`category` let `retriever.py`
disambiguate two different books that happen to share the same display name (e.g. book_id 173
and 210 are both named "কর্মপদ্ধতি") by appending the category to the citation.

## Prompt contract (important — do not regress this)

`generator.py`'s system prompt instructs the model to:
1. Answer **only** from the numbered `উদ্ধৃত অংশ` (excerpt) blocks — no outside knowledge.
2. **Partial answers are allowed.** If the excerpts partially cover the question, answer with
   what's there and name the gap in one sentence, rather than refusing outright. Refuse only
   if the topic isn't mentioned at all.
3. Cite every claim with `[১]` / `[২]` style excerpt numbers.
4. Answer entirely in standard Bengali — no English sentences, no Banglish.
5. No temperature override — `gpt-5-mini` is a reasoning model and only accepts its fixed
   default (1); see "LLM provider" below.

**The relevance gate is decided upstream, in `qa_service.py`, by score — not by the LLM's
judgment — but the LLM is always called.** If the top citation's `rerank_score` is below
`settings.min_rerank_score`, `qa_service.answer_question()` still calls `generate_answer()`,
just with an empty citation list; the system prompt's rule 3 above then has the model say
plainly, in its own words, that the books don't cover the question, instead of the code
returning a canned refusal string. `sources` is `[]` either way — nothing grounded the answer,
so nothing is cited. This is deliberate: the old version always attached sources regardless of
whether the LLM actually used them, so the UI could show "not in the database" next to a
populated sources list. Retrieval quality and the answer text can no longer contradict each
other. If you change this gate, re-verify with an on-topic, an off-topic (Bengali), and a
Banglish on-topic question — see "Smoke test" below.

**Conversational shortcut.** `qa_service._is_conversational()` (mirrored by
`_qa_stream()`'s check in `router.py` for the streaming path) regex-matches short greetings /
well-wishes / thanks / farewells (Bengali and common Banglish spellings, e.g. "hello", "kemon
achen", "dhonnobad") *before* retrieval runs. A match short-circuits to a random canned
friendly reply from a small fixed set — no retriever call, no LLM call, `sources: []`. Anything
longer than six words falls through to normal retrieval, so a real question that happens to
open with "হ্যালো" still gets answered normally.

## LLM provider (`app/core/llm.py`)

OpenAI is the production LLM provider (`OPENAI_API_KEY` / `OPENAI_MODEL`, default
`gpt-5-mini`). All six call sites (intent classification, query rewriting, persona
extraction, QA generation, note map/reduce, suggestion generation) funnel through this one
module — no call site constructs an `openai.OpenAI` client directly:

- **`get_client()`** — the shared, cached OpenAI client for `settings.openai_api_key`.
- **`get_model(task)`** — looks up `task` in `settings.model_by_task`, falling back to
  `settings.openai_model` for any task not (yet) explicitly routed. `model_by_task` is empty
  by default, so every task uses `gpt-5-mini` until deliberately routed elsewhere.
- **`complete(task, messages, *, token_budget=None, **overrides)`** — the actual call. Resolves
  `task`'s model via `get_model()`, then applies that model's own parameter quirks via
  `MODEL_ADAPTERS` before calling `chat.completions.create()`. This is the only sanctioned way
  for a call site to hit the LLM.
- **`MODEL_ADAPTERS`** — a per-model table of `(provider, token_param)`. The reason it exists:
  `gpt-5-mini` (and other OpenAI reasoning models) need `max_completion_tokens`, not
  `max_tokens`, for a token cap, and only accept the fixed default `temperature=1` — passing
  any other value is a 400. A call site never hardcodes this; it passes `token_budget=N` to
  `complete()` and gets whichever kwarg name the resolved model actually needs.
- **Groq routing (`get_client_for`)** — a second, OpenAI-compatible provider, used today only
  by `scripts/eval_generation_ab.py` (A/B testing Bengali answer quality against
  `settings.generation_candidates`, judged by `settings.generation_judge_model`). Not reachable
  from the production request path.

Task names for `model_by_task`: `intent`, `rewrite`, `persona`, `qa`, `note`, `suggest`,
`roleplay`.

## Configuration (`app/core/config.py`)

All settings are read from `.env` (see `.env.example` for the full documented list).
Only `OPENAI_API_KEY` and `DATABASE_URL` are required; everything else has a working default:

| Env var | Default | Notes |
|---|---|---|
| `OPENAI_API_KEY` | *(required)* | See "LLM provider" above — the only sanctioned entry point for LLM calls is `app/core/llm.py`. |
| `OPENAI_MODEL` | `gpt-5-mini` | A reasoning model — it spends completion-token budget on internal reasoning before emitting visible output. Give it a generous token budget (2000+ for short JSON-style outputs); too low silently truncates and fails downstream parsing. |
| `DATABASE_URL` | `postgresql+psycopg2://...` | SQLAlchemy DSN format — **not** parseable by `psql` directly. Use a Python script with `app.db.session.SessionLocal` for one-off queries/admin, not raw `psql -c`. |
| `GROQ_API_KEY` / `GROQ_BASE_URL` | *(empty)* / `https://api.groq.com/openai/v1` | Eval-only — see "LLM provider" above. Empty by default; not read anywhere in the production request path. |
| `EMBEDDING_MODEL_NAME` | `BAAI/bge-m3` | Multilingual, 1024-dim, handles Bengali (unlike the old `all-MiniLM-L6-v2`, which tokenized Bengali entirely to `[UNK]`). Changing this requires a full reindex — see below. |
| `RERANKER_MODEL_NAME` | `BAAI/bge-reranker-v2-m3` | Cross-encoder used only at query time (not stored in Chroma), so changing it does NOT require a reindex. |
| `CHROMA_PERSIST_DIR` | `chroma_db` | Relative to the working directory the process is started from. |
| `CHROMA_COLLECTION_NAME` | `documents_bge_m3` | Name encodes the embedding model on purpose — changing `EMBEDDING_MODEL_NAME` without also bumping this would silently mix incompatible-dimension vectors into one collection (Chroma would just reject them with a dimension-mismatch error, which is the point). |
| `TOP_K` | `5` | Chunks sent to the LLM after reranking. |
| `FETCH_K` | `25` | Candidates pulled from Chroma before reranking (per query variant, then merged). |
| `MIN_SIMILARITY` | `0.25` | Cheap cosine pre-filter before paying for the cross-encoder. Loose on purpose. |
| `MIN_RERANK_SCORE` | `0.5` | The real "do we have an answer?" gate — see "Prompt contract" above. **Scores are Sigmoid-activated, in [0, 1]**, not raw bge-reranker logits. Smoke-tested on this corpus: on-topic questions scored 0.94–0.99, a fully off-topic question topped out at 0.011. Re-tune with `uv run python -m scripts.tune_threshold` once you have ~30 real questions (20 answerable, 10 not). |
| `TARUN_DB_PATH` / `NOBIN_DB_PATH` | `data/Tarun_Associate.db` / `data/Nobin_Associate.db` | Legacy — only used by the one-time `scripts/migrate_sqlite_to_postgres.py`, not by `ingest.py` anymore. |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` | *(empty)* / *(empty)* | Opt-in tracing — see "Tracing" below. With either empty, `app/core/tracing.py` is a complete no-op: the langfuse SDK is never imported, `get_client()` returns the plain OpenAI client, `/chat` behaves byte-identically to a build without tracing. |
| `LANGFUSE_HOST` | `http://localhost:3000` | Self-hosted or Langfuse Cloud endpoint. **Not** `LANGFUSE_BASE_URL` — that's the `langfuse-cli` tool's own env var convention, different from this app's `Settings.langfuse_host` field; setting the wrong one leaves tracing silently pointed at `localhost:3000` even with real Cloud keys configured (fail-safe design swallows the resulting connection errors — see "Tracing" below). |
| `LANGFUSE_ENABLED` | `true` | Master switch — tracing is active only when this is true **and** both keys are set. |
| `LANGFUSE_CAPTURE_IO` | `true` | Set `false` to redact every prompt/completion/query/excerpt text field in a trace while keeping structure, scores, token usage, cost and latency. |
| `LANGFUSE_RELEASE` | *(empty)* | Optional build marker (e.g. a git short sha) attached to every trace. |
| `LANGFUSE_ENVIRONMENT` | `development` | Separates real production traffic from dev/staging runs in the Langfuse UI (its default views filter to `production`). Set to `production` via the deployed box's `.env` (e.g. `shibirgpt.service`). |
| `HOST` / `PORT` | `0.0.0.0` / `9200` | Not currently read by `run.sh`'s uvicorn invocation (hardcoded there) — see Known gaps. |

## API contract

`GET /health` → `{"status": "ok"}`

`POST /chat` — the single entry point for every user-facing interaction. The old `/ask`,
`/note`, `/note-by-text` endpoints are **removed**; their underlying service functions are
unchanged and still called from here, just not exposed as separate routes.
`app.services.intent_classifier.classify_intent()` routes the free-text `message` into one of
four modes and dispatches internally — see "Roleplay and suggestion services" below for
ROLEPLAY/SUGGESTION, "Prompt contract" above for QA, and the Architecture tree above for NOTE.

Request:
```json
{"message": "your question in any language — Bengali script, Banglish, or English", "session_id": null, "user_id": null}
```
`session_id` is optional — omit it (or pass `null`) to start a new session; the response
echoes back the `session_id` to reuse on the next turn. `user_id` is optional and only used to
group Langfuse traces per end user when tracing is enabled (see "Tracing" below) — no request
logic reads it.

Response:
```json
{
  "mode": "qa",
  "answer": "... (always Bengali)",
  "sources": [
    {
      "book": "...", "chapter": "...", "source_db": "tarun",
      "content": "full chunk text (includes the বই:/অধ্যায়: header)",
      "similarity": 0.6471, "rerank_score": 0.9861
    }
  ],
  "session_id": "...",
  "response_time_ms": 842.17
}
```
`mode` is one of `"qa" | "note" | "roleplay" | "suggestion"` (whichever the classifier picked
for this turn). `sources` is `[]` for NOTE and ROLEPLAY, and for QA/SUGGESTION whenever
retrieval didn't clear `settings.min_rerank_score` — never populated alongside a "not found"
answer. See "Prompt contract" above. A failed upstream LLM call (quota, bad key, outage) comes
back as a `503` with a Bengali "temporarily unavailable" detail, not a bare `500`.

`POST /chat/stream` — same request shape and dispatch as `POST /chat`, but Server-Sent Events:
a `sources` event, then one or more `token` events (QA streams the answer token-by-token via
`generator.stream_answer()`; NOTE/ROLEPLAY/SUGGESTION emit their whole answer as a single
`token` event since they don't generate incrementally), then a `done` event carrying
`mode`/`session_id`/`response_time_ms`. An upstream LLM failure emits an `error` event instead.

`GET /conversations` → sidebar list, newest-activity-first:
```json
[{"id": "...", "title": "first user message, trimmed", "message_count": 4, "created_at": "...", "updated_at": "..."}]
```

`GET /conversations/{session_id}/messages` → `[{"role": "user"|"assistant", "content": "..."}]`
turns for that session, or `404` if the id is unknown.

`DELETE /conversations/{session_id}` → `204` on success, `404` if the id is unknown.

**`/conversations` reads `app.services.session_store`, an in-process module-level `dict` —
single-worker only.** Sessions are lost on process restart and invisible across workers (e.g.
`uvicorn --workers N` or multiple pods each get their own copy). Fine for the current
single-worker deployment; a shared store (Redis or a DB table) is required before scaling to
multiple workers.

Unknown extra fields in the request body are silently ignored (default Pydantic behavior).

## Roleplay and suggestion services

`app/services/roleplay_service.py` (`handle_roleplay`) — no RAG retrieval by design: it's a
free-form persona conversation, not a book lookup. On the first message of a roleplay session
it extracts a persona description with one LLM call (task `"persona"`), then holds a
multi-turn conversation in that persona using the session's history (`session_store.py`,
capped at `MAX_HISTORY` turns). `router.py`'s `classify_intent(...)` keeps an ongoing roleplay
session routed to ROLEPLAY on every follow-up turn unless the message matches an explicit exit
phrase (e.g. "stop roleplay" / "রোলপ্লে বন্ধ"), at which point the session's `persona` is
cleared so a future roleplay starts fresh.

`app/services/suggestion_service.py` (`give_suggestion`) — reuses `retrieve_relevant_docs()`
and the same relevance gate as `qa_service.answer_question()`, but with its own prompt: when
the gate passes, the model is explicitly allowed to phrase a recommendation/opinion grounded
in the retrieved excerpts (not just restate cited facts, the way QA's prompt requires); when
it doesn't, a different ungrounded prompt has the model say plainly that the books don't cover
this, then optionally offer a general (explicitly non-book) suggestion.

## Common tasks

- **Run the server**: `./run.sh` (syncs deps via uv, starts uvicorn on :9200), or manually:
  `HF_HUB_OFFLINE=1 uv run uvicorn app.main:app --host 0.0.0.0 --port 9200`.
- **Smoke test**: run three `/chat` calls with `{"message": "..."}` — an on-topic question in
  Bengali script, the same question in Banglish (must return the same sources, not an
  empty/weak result), and a deliberately off-topic question (must return `sources: []`, not
  populated, with `mode: "qa"`). Then one `/chat` call phrased as a note request (e.g. "এই
  বইয়ের নোট বানাও ...") against a real book title — check `mode: "note"` in the response. See
  `scripts/tune_threshold.py` for a scripted version of the retrieval half of this.
- **Tune the refusal threshold**: fill in `scripts/eval_questions.example.json` (copy it,
  don't edit in place) with ~30 real questions — 20 answerable from the corpus, 10 not — then
  `uv run python -m scripts.tune_threshold your_questions.json`.
- **Deploy**: `shibirgpt.service` is a systemd unit that runs `run.sh` with
  `WorkingDirectory=/home/lab/apps/shibirgpt`, `Restart=always`. Note: this is a manual/legacy
  path — see "Known gaps" below for the state of the GitHub Actions deploy workflow.

## Reindexing

**Both of the following are required. Deleting only one leaves the migration/reindex
silently broken:**

```bash
rm -rf chroma_db
```
```python
from sqlalchemy import text
from app.db.session import SessionLocal
s = SessionLocal()
s.execute(text("UPDATE pages SET embedded_at = NULL"))
s.execute(text("UPDATE articles SET embedded_at = NULL"))
s.commit()
```
```bash
HF_HUB_OFFLINE=1 uv run python -m app.rag.ingest
```

Why both: `ingest.py` only embeds rows where `embedded_at IS NULL OR updated_at >
embedded_at`. If you delete `chroma_db/` but leave `embedded_at` set on every row, the script
runs, logs "0 pages need embedding", exits successfully, and you're left with an **empty**
vector store that looks like a clean run. This is exactly how the old MiniLM index could
survive a supposed "reindex" undetected.

On this corpus (4,143 published pages, 0 articles), a full reindex produces roughly 12,000
chunks (~3 chunks/page at the current `chunk_size=900`/`overlap=150`) and is CPU-bound — plan
for a long run on a memory-constrained machine (multiple hours observed under swap pressure;
much faster with more free RAM or a GPU).

## Tracing (`app/core/tracing.py`)

Langfuse-based observability (SDK v4, OpenTelemetry-based — migrated 2026-09-19 off the old v2
"manual client" API; see the module's own docstring for why and what changed), wired into
`app/main.py`'s lifespan (`tracing.init()` on startup pays the langfuse SDK import cost up front
instead of on the first request; `tracing.shutdown()` flushes buffered events on graceful
shutdown) and into `app/core/llm.py` (every `complete()` call links to the active request's
root span).

**Opt-in and fail-safe.** With `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY` empty (the
default), every function in `tracing.py` is a no-op, the langfuse SDK is never imported, and
`get_client()` in `llm.py` returns the plain `openai.OpenAI` — `/chat` behaves byte-identically
to a build without this module. Every public helper also swallows its own exceptions (logs a
warning, never raises), so a broken or unreachable Langfuse instance can never break a request.

One root span is opened per `/chat` (or `/chat/stream`) request in `router.py`. Its context
(root span object, trace id, root observation id) is held in a `ContextVar` AND returned
explicitly, because ambient OpenTelemetry context only reliably survives `run_in_threadpool`
hops within a single request — it does **not** survive Starlette's SSE streaming path, where
`gen()` is pulled one `next()` at a time, each call getting a fresh context copy (see
`tracing.py`'s module docstring for the full reasoning and how `app/rag/generator.py`'s
`stream_answer()` works around it with an explicit `trace_id`/`parent_observation_id`). Every
nested pipeline stage and LLM call groups under the root span without threading it through
service signatures: `classify-intent` (span; how the message was classified), `expand-query`
(span; the rewritten query variants), `retrieve-context` (a `retriever`-typed observation —
every reranked candidate, kept and dropped, with cosine similarity + rerank score),
`check-relevance-gate` (a `guardrail`-typed observation: grounded vs. refused, top rerank
score, threshold), and one `generation` per LLM call (model, messages, completion, token usage,
cost, latency — named `llm:{task}`, e.g. `llm:qa`, `llm:intent`). `LANGFUSE_CAPTURE_IO=false`
redacts every text field (query, excerpts, prompts, completions, answer) while keeping
structure/scores/usage/cost. `LANGFUSE_ENVIRONMENT` (default `development`) separates real
traffic from dev/staging runs in the Langfuse UI — set to `production` on the deployed box.
`mode` (qa/note/roleplay/suggestion) lands in the root span's `metadata` at finalize, not as a
tag — it's only known after intent classification runs, and v4's tags must be set at
observation-creation time via `propagate_attributes()`, so metadata is the documented escape
hatch. A failed request sets `level=ERROR` on the root span instead of an ad hoc tag.

`llm.py`'s `openai_wrapper_active()` guard exists because Langfuse's OpenAI wrapper injects
extra kwargs (`name`, `metadata`, `trace_id`, `parent_observation_id`, ...) that only the
*wrapped* client understands — a plain `openai.OpenAI` client 400s on them. The guard is true
only once `langfuse.openai`'s import has actually patched `openai`'s chat-completions methods,
**not** merely when tracing is enabled — so if Langfuse is enabled but that import fails,
`complete()` strips those kwargs and tracing silently stays off for that call instead of turning
every LLM call in the app into a 400.

## Known gaps / things a future change might need to address

- `app/core/llm.py`'s `get_client()` constructs the OpenAI client with no request timeout — a
  slow or hanging upstream call can block a `/chat` request indefinitely. Fix this before the
  next production push.
- `.github/workflows/deploy.yml` (triggered on push to `main`) is still a placeholder — its
  one step is `echo "Add your deploy steps here (Vercel, AWS, Docker push, etc.)"`.
  `.github/workflows/ci.yml` gates PRs into `main` with a real Postgres-backed pytest run, but
  nothing actually deploys on merge yet.
- `app/services/session_store.py` is an in-process `dict` — single-worker only. A real
  multi-worker/multi-pod deployment needs a shared store (Redis or a DB table) first; see
  "API contract" above.
- Vector store is still ChromaDB. Migration to something like pgvector/Qdrant for scale (the
  corpus is a growing target of several million+ chunks) is planned but not started.
- No lint/format tooling — `pyproject.toml`'s `[dependency-groups] dev` only has `pytest`, no
  `ruff` or equivalent.
- `min_rerank_score` is a single global threshold. If the corpus grows to cover very
  different domains, a per-book or per-category threshold might separate "answerable" from
  "not answerable" better than one global cutoff — not needed at the current corpus size.
- `query_rewriter.expand_query()` costs one LLM call per unique query (cached via `lru_cache`,
  so repeats are free within a process lifetime, but the cache doesn't persist across
  restarts). If that call ever fails, it silently falls back to the raw query — fine for
  Bengali-script input, degrades Banglish retrieval quality back to pre-fix levels.
- Corpus cleanup (`scripts/clean_corpus.py`, `Page.excluded_from_rag`) is data-confirmed but
  only partially applied: `book203_cross_contamination.csv`'s 106 flagged duplicate pages are
  excluded (confirmed live in Postgres: `excluded_from_rag = true` with reason
  `book203_cross_contamination.csv:loser`). `orphaned_source_pages.csv` was a no-op — its one
  row never reached Postgres. `arabic_placeholder_pages.csv` (343 data rows) was deliberately **not**
  applied: on inspection most of its rows aren't actually broken (many are pristine pages that
  merely contain Arabic script, and a chunk of the rest still have a usable Bengali
  translation/tafsir alongside a lost Arabic original) — applying it as-is would delete
  legitimately useful pages and hurt recall. It needs regenerating with a real broken-page
  heuristic before it's safe to apply.
