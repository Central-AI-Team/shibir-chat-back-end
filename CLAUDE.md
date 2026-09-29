# Shibir Chat Back-End: Maintainer and Agent Guide

This is the technical reference for the codebase. Claude Code loads it automatically at the start
of every session, and `GEMINI.md` imports it for Gemini CLI. Human maintainers should read it too.
For setup and API usage, start with [README.md](README.md).

Read the **Prompt contract**, **GPU service** and **Reindexing** sections before changing
retrieval, generation or ingestion code. They describe behaviour that has regressed before.

## 1. What this project is

A FastAPI back-end for a Bengali-language question-answering assistant. A user sends a message in
Bengali script, Banglish or English. The service retrieves relevant excerpts from a library of
religious and educational books (chunked, embedded and stored in Chroma) and asks an LLM (OpenAI
`gpt-5-mini`) to answer **strictly from those excerpts**, refusing rather than inventing an answer
when they do not cover the question.

It is a retrieval-augmented generation (RAG) service, not a general-purpose chatbot.

## 2. Architecture

The code has three layers: `api` → `services` → `rag`/`db`. There is exactly one vector store and
one production LLM provider, so there are deliberately no repository or interface abstractions;
they would add indirection with no real swap-ability at this size.

```
app/
  main.py                 FastAPI app: CORS middleware, router, lifespan hooks
                          (tracing.init() on startup, tracing.shutdown() on shutdown).
  core/
    config.py             The single source of configuration: a pydantic-settings Settings
                          object reading .env. No other module calls os.getenv() or
                          load_dotenv(); import `settings` from here.
    llm.py                Shared OpenAI client and per-task model routing. The only
                          sanctioned way to call an LLM (§5).
    tracing.py            Opt-in Langfuse tracing (§10).
  api/
    router.py             HTTP routes (§7). Thin: validates input, dispatches to services,
                          maps upstream failures to 503 / SSE `error`.
  services/
    intent_classifier.py  classify_intent(message, has_active_roleplay_session) -> NOTE |
                          ROLEPLAY | SUGGESTION | QA. Regex first (Bengali + Banglish
                          patterns); one LLM call (task "intent") only if nothing matches.
                          An active roleplay session stays ROLEPLAY unless the message
                          matches an exit phrase ("stop roleplay", "রোলপ্লে বন্ধ", ...).
    qa_service.py         answer_question(query) -> QueryResponse. Retrieval, relevance gate,
                          grounded generation (§4).
    suggestion_service.py give_suggestion(query) -> (str, list[Citation]). Same retrieval and
                          gate as QA, different prompt (§8).
    note_service.py       generate_chapter_note(chapter_id) and
                          generate_book_notes_from_text(text). Not retrieval: loads every
                          published page of a chapter in reading order and map-reduces it.
                          The book is resolved from free text by substring match on book
                          names (no LLM call).
    roleplay_service.py   handle_roleplay(message, session) -> str. Persona extraction on the
                          first turn (task "persona"), then multi-turn chat. No retrieval (§8).
    session_store.py      In-process dict of sessions (mode, persona, history capped at
                          MAX_HISTORY = 20). Backs session continuity and /conversations.
                          Single worker only.
  rag/
    chunker.py            normalize(text): NFC + whitespace cleanup; apply at ingest AND
                          query time. chunk_text(text): paragraph → danda (।) → period →
                          space splitting, ~900 chars per chunk, 150 overlap.
    query_rewriter.py     expand_query(query) -> tuple[str, ...]. One LLM call (task
                          "rewrite") turns Banglish/English/Bengali into Bengali-script
                          search variants. lru_cached per process. Falls back to the raw
                          query on any failure, including a truncated or empty response;
                          see the incident comment in the file before lowering its token budget.
    embedder.py           embed_text / embed_texts / embed_query with BAAI/bge-m3 (1024-dim,
                          multilingual). GPU mode calls the external service (§6); otherwise
                          a local sentence-transformers model, imported and loaded lazily
                          under a lock, so GPU mode never imports torch.
    reranker.py           rerank(query, docs, top_n) -> [(index, score)] with
                          BAAI/bge-reranker-v2-m3. Same two backends. Scores are
                          sigmoid-activated, in [0, 1], never raw logits.
    gpu_client.py         post(path, json) to the GPU service: shared httpx.Client,
                          X-API-Key, cold/warm timeouts, selective retries, GPUServiceError.
    chroma_client.py      get_collection(): the ONLY place a Chroma PersistentClient is
                          created. The collection uses hnsw:space="cosine" (Chroma's default
                          is L2, which would break similarity thresholds).
    retriever.py          retrieve_relevant_docs(query) -> list[Citation]:
                          expand_query → embed all variants → Chroma FETCH_K neighbours per
                          variant → merge + dedupe → drop below MIN_SIMILARITY → rerank →
                          TOP_K. Citations carry `similarity` (cosine pre-filter) and
                          `rerank_score` (the real relevance signal).
    generator.py          generate_answer(query, citations) -> str (task "qa") and
                          stream_answer() for /chat/stream. Bengali system prompt with
                          numbered excerpts (§4).
    ingest.py             `uv run python -m app.rag.ingest`. Embeds published pages/articles
                          from Postgres into Chroma (§9).
  loaders/                Raw corpus file -> clean text, standard library only; used by
                          scripts/load_*.py (§9 "Adding a data source"). Ported from
                          shibir-chat-gpu-service/ingestion/.
    text.py               html_to_text, drop_repeated_sentences, is_repetitive, _fingerprint.
    sources.py            iter_mysql_values (MySQL dump rows), read_cs_posts, read_pp_articles.
  db/
    models.py             SQLAlchemy models: Category, Book, Chapter, Page, Article (§3).
    session.py            SessionLocal and engine, bound to settings.database_url.
    base.py               Declarative Base.
  schemas/query.py        Pydantic models. ChatRequest / ChatResponse (/chat, /chat/stream),
                          Citation (shared source shape), ConversationSummary /
                          ConversationMessage (/conversations). QueryResponse is
                          answer_question()'s internal return type. QueryRequest,
                          NoteRequest, NoteByTextRequest, NoteByTextResponse and ChapterNote
                          are unused leftovers from the removed /ask, /note, /note-by-text
                          routes.

scripts/                  Migrations, corpus cleanup, evaluation and debugging tools.
                          Run as `uv run python -m scripts.<name>`. See docs/evaluation.md.
  migrate_sqlite_to_postgres.py   One-time SQLite → Postgres migration. Already run; do not
                                  re-run against the live corpus.
  add_source_type_to_articles.py  One-time migration adding source_type/source_ref/
                                  source_metadata to articles.
  clean_corpus.py                 Soft-excludes flagged pages (§3).
  dedupe_pages.py                 Flags byte-identical duplicate pages.
  load_tafheem.py                 data/tafheemul_quran.db -> pages, one per ayah (§9).
  load_articles.py                data/{pp-articles,cs-posts}-*.sql -> articles (§9).
tests/                    pytest suite; no network or real models by default (§11).
docs/                     deployment.md, evaluation.md, tracing.md.
chroma_db/                Generated vector store, git-ignored. Deleting it is only half of
                          a reindex (§9).
```

## 3. Data model

All content lives in Postgres (`categories`, `books`, `chapters`, `pages`, `articles`). The
original per-source SQLite files (`Tarun_Associate.db`, `Nobin_Associate.db`) were migrated once;
`Page.source_db` / `Page.source_page_id` are kept only as provenance so the migration is
idempotent. The running app does not read them.

- Only rows with `status == 'published'` are embedded.
- `embedded_at` records the last successful embed. `NULL` means never embedded;
  `updated_at > embedded_at` means stale.
- `Page.excluded_from_rag` / `Page.exclusion_reason` (added by `scripts/clean_corpus.py`) are a
  reversible soft-exclude: `ingest.py` skips excluded pages, and no row is ever deleted.
  Clearing the flag and re-ingesting restores a page. See §14 for what is currently excluded.

**Chroma layout.** Chunk ids are `f"{prefix}_{row.id}_c{chunk_index}"` (for example
`page_842_c0`, `article_12_c0`). Metadata: `book`, `chapter`, `page_id`, `chunk_index`,
`row_key`, `source_db`, `book_id`, `category`.

- `row_key` (`f"{prefix}_{row.id}"`) is what `ingest.py` deletes by before re-upserting a row,
  and the unit of relevance in the evaluation sets.
- `source_db` is `Page.source_db` for pages and `Article.source_type` (e.g. `cs-post`,
  `pp-article`) for articles; chunks embedded before that change say `articles`.
- `book_id` and `category` let the retriever tell apart different books with the same display
  name (for example book ids 173 and 210 are both "কর্মপদ্ধতি"); the category is appended to the
  citation.

## 4. Prompt contract (do not regress)

`generator.py`'s system prompt requires the model to:

1. Answer **only** from the numbered `উদ্ধৃত অংশ` (excerpt) blocks, with no outside knowledge.
2. Give **partial answers** when the excerpts partly cover the question, naming the gap in one
   sentence. Refuse only when the topic is not mentioned at all.
3. Cite every claim with `[১]`, `[২]`, ... excerpt numbers.
4. Write entirely in standard Bengali: no English sentences, no Banglish.

No `temperature` is passed: `gpt-5-mini` is a reasoning model and rejects any value except its
default of 1 (§5).

**The relevance gate is decided in code, by score, and the LLM is always called.** If the top
citation's `rerank_score` is below `settings.min_rerank_score`, `qa_service.answer_question()`
calls `generate_answer()` with an **empty** citation list, and the prompt has the model explain in
its own words that the books do not cover the question. `sources` is `[]` in that case. There is
no canned refusal string.

This fixes an old bug where sources were always attached, so the UI could show "not in the
library" next to a populated source list. The answer text and the sources can no longer
contradict each other. After changing the gate, re-verify with an on-topic Bengali question, the
same question in Banglish, and an off-topic question (§12).

**Conversational shortcut.** `qa_service._is_conversational()` (mirrored in `router._qa_stream()`
for streaming) matches short greetings, thanks and farewells in Bengali and common Banglish
spellings ("hello", "kemon achen", "dhonnobad") **before** retrieval. A match returns a random
reply from a small fixed set with no retrieval, no LLM call and `sources: []`. Messages longer
than six words always go through normal retrieval, so a real question that starts with "হ্যালো" is
still answered.

## 5. LLM provider (`app/core/llm.py`)

OpenAI is the production provider (`OPENAI_API_KEY`, `OPENAI_MODEL`, default `gpt-5-mini`). Every
LLM call (tasks `intent`, `rewrite`, `persona`, `qa`, `note`, `suggest`, `roleplay`) goes through
this module; no call site constructs an `openai.OpenAI` client itself.

- **`get_client()`**: the shared, cached OpenAI client, built with
  `settings.llm_request_timeout_seconds` (default 90) and `settings.llm_max_retries` (default 1).
- **`get_model(task)`**: looks up `task` in `settings.model_by_task`, falling back to
  `settings.openai_model`. `model_by_task` is empty by default, so every task uses `gpt-5-mini`.
- **`complete(task, messages, *, token_budget=None, **overrides)`**: the call itself. It
  resolves the model, applies that model's parameter rules from `MODEL_ADAPTERS`, and calls
  `chat.completions.create()`.
- **`MODEL_ADAPTERS`**: per-model `(provider, token_param)`. Reasoning models such as
  `gpt-5-mini` need `max_completion_tokens` instead of `max_tokens`, and reject any
  `temperature` other than 1 with a 400. Call sites pass `token_budget=N` and never hard-code
  either parameter.
- **Groq (`get_client_for`)**: a second, OpenAI-compatible provider used only by
  `scripts/eval_generation_ab.py`. It is not reachable from the request path.

`gpt-5-mini` spends completion tokens on hidden reasoning before producing output. Give it a
generous budget (2000+ even for short JSON outputs). Too small a budget truncates silently and
breaks downstream parsing.

**Routing a task to a cheaper model** requires `scripts/eval_task_routing.py` first, and the
result should be recorded next to `model_by_task` in `config.py`. For `rewrite`, repeat each query
several times: `openai/gpt-oss-20b` passed a single run but misspelled key Banglish terms in 7 of
18 repeated rewrites, collapsing retrieval, and was reverted on 2026-09-24.

## 6. GPU service (`app/rag/gpu_client.py`)

Embedding and reranking can run on the separate `shibir-chat-gpu-service` repository, deployed on
Modal. It is enabled by setting `GPU_SERVICE_URL` (the base URL, for example
`https://<workspace>--shibir-chat-gpu-service-gpuservice-web.modal.run`, with no path). When it is
empty, the local CPU models are used.

Service API (header `X-API-Key: $GPU_API_KEY`):

- `POST /embed` `{texts[1..256], normalize, batch_size}` → `{model, dim, vectors}`
- `POST /rerank` `{query, documents[1..200], top_k, max_length}` →
  `{model, results: [{index, score, prob}]}`

Contract this back-end relies on (do not regress):

- **Embedding:** `embed_texts` sends slices of at most 256 texts with `normalize=True,
  batch_size=64`, and raises `GPUServiceError` unless every response has `dim == 1024`,
  `model == settings.embedding_model_name` and one vector per text. A different model would
  silently corrupt the existing index.
- **Reranking:** `rerank` sends slices of at most 200 documents with `max_length=1024` (matching
  the local CrossEncoder) and `top_k=top_n`, checks `model == settings.reranker_model_name`, maps
  indices back to the original list, merges, and returns **`prob`** (sigmoid, 0–1), never
  `score` (raw logit). `MIN_RERANK_SCORE` is calibrated on sigmoid scores.
- **Retries:** only connection failures (`ConnectError`, `ConnectTimeout`, `RemoteProtocolError`,
  `ReadError`, `WriteError`; both endpoints are idempotent), 5xx and 429 are retried, with
  exponential backoff (0.5 s, 1 s, ...). Other 4xx responses and read timeouts raise immediately.
- **Timeouts:** chosen per call. A call uses `gpu_cold_timeout_seconds` if it is the first in
  the process or more than `gpu_warm_window_seconds` after the last success; otherwise
  `gpu_timeout_seconds`. It is not a once-per-process flag: Modal scales to zero after 300 s
  idle, and a short timeout on a cold start would surface as a 503. Ingest calls
  `embed_texts(..., bulk=True)`, which raises the floor to `gpu_bulk_timeout_seconds`: a
  256-chunk `/embed` can exceed the 30 s warm timeout even on a warm container.
- **Errors:** `router.py` maps `GPUServiceError` to the same Bengali 503 used for LLM failures,
  and to an SSE `error` event on `/chat/stream`.
- **Tracing:** each `post()` is a Langfuse span `gpu:embed` / `gpu:rerank` with
  `{n_items, latency_ms, cold, attempts, status}`, never the texts.
- **Privacy:** request bodies and the API key are never logged.
- **No torch in GPU mode:** torch and sentence-transformers are imported lazily inside `_model()`.
  They remain dependencies for the local fallback (§14).
- Chroma still runs in this process; only model inference moved.

## 7. API contract

| Method | Path | Result |
|---|---|---|
| `GET` | `/health` | `{"status": "ok"}` |
| `POST` | `/chat` | `ChatResponse` |
| `POST` | `/chat/stream` | Server-Sent Events |
| `GET` | `/conversations` | `[ConversationSummary]`, most recent activity first |
| `GET` | `/conversations/{session_id}/messages` | `[{"role": "user" \| "assistant", "content"}]`, or 404 |
| `DELETE` | `/conversations/{session_id}` | 204, or 404 |

The old `/ask`, `/note` and `/note-by-text` routes were removed. Their service functions are still
called from `/chat`.

**`POST /chat` request**

```json
{"message": "question in Bengali script, Banglish or English", "session_id": null, "user_id": null}
```

`session_id` is optional; the response returns the id to reuse on the next turn. `user_id` is
optional and only groups Langfuse traces; no request logic reads it. Unknown fields are ignored.
An empty message returns 400.

**`POST /chat` response**

```json
{
  "mode": "qa",
  "answer": "… (always Bengali)",
  "sources": [
    {
      "book": "…", "chapter": "…", "source_db": "tarun",
      "content": "full chunk text, including the বই:/অধ্যায়: header",
      "similarity": 0.6471, "rerank_score": 0.9861
    }
  ],
  "session_id": "…",
  "response_time_ms": 842.17
}
```

- `mode` is `qa`, `note`, `roleplay` or `suggestion`.
- `sources` is `[]` for `note` and `roleplay`, and for `qa`/`suggestion` whenever retrieval did
  not pass the gate. It is never populated alongside a "not found" answer.
- An upstream failure (OpenAI `APIError` or `GPUServiceError`) returns **503** with a Bengali
  "temporarily unavailable" detail, never a bare 500.

**`POST /chat/stream`**: the same request and dispatch, returned as events `sources`, then one or
more `token`, then `done` (`mode`, `session_id`, `response_time_ms`). QA streams token by token
through `generator.stream_answer()`; the other modes send their whole answer as one `token`. An
upstream failure emits a single `error` event instead.

The front-end (`shibir-chat-front-end`) depends on these shapes. Keep changes backward-compatible.

**Sessions are single-worker.** `/conversations` and session continuity read
`app.services.session_store`, a module-level dict. Sessions are lost on restart and invisible to
other workers or pods. A shared store (Redis or a table) is required before running more than one
worker.

## 8. Roleplay and suggestion

**Roleplay** (`roleplay_service.handle_roleplay`) uses no retrieval by design; it is a persona
conversation, not a book lookup. The first roleplay message is used to extract a persona (task
`persona`). Later turns continue in that persona using the session history. The classifier keeps
the session in ROLEPLAY until an exit phrase is sent; `router.py` then clears `persona`, so the
next roleplay starts fresh.

**Suggestion** (`suggestion_service.give_suggestion`) uses the same retrieval and gate as QA, but
its grounded prompt allows the model to phrase a recommendation or opinion based on the excerpts
rather than only restating them. When the gate fails, a separate prompt has the model say the
books do not cover the topic and then, optionally, offer clearly labelled general advice.

## 9. Ingestion and reindexing

`uv run python -m app.rag.ingest`:

0. **sweeps** first: every page/article that still has chunks (`embedded_at` set) but is no longer
   eligible (not published, or a page with `excluded_from_rag`) gets its chunks deleted by
   `row_key` and `embedded_at` cleared (`updated_at` untouched). Republishing such a row later
   makes it due again. Chroma errors abort the run rather than being swallowed; `embedded_at` is
   cleared only per successfully deleted batch, so the next run resumes;
1. selects published, non-excluded pages and published articles where
   `embedded_at IS NULL OR updated_at > embedded_at`;
2. chunks each row and prefixes each chunk with a Bengali `বই:` / `অধ্যায়:` header;
3. deletes the row's existing chunks by `row_key` (a row's chunk count changes when its content
   changes, so upsert alone would leave orphans);
4. embeds and upserts in batches (64 locally, 256 in GPU mode) and sets `embedded_at`.

**A full reindex needs both steps below.** Doing only the first leaves an empty index while the
ingest reports "0 pages need embedding" and exits successfully. That is how the old MiniLM index
once survived a supposed reindex.

```bash
rm -rf chroma_db
```

```python
# DATABASE_URL is a SQLAlchemy DSN and cannot be passed to psql directly.
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

The current corpus (4,143 published pages, 0 articles) produces about 12,000 chunks. On CPU this
takes hours on a memory-constrained machine; in GPU mode it is much faster. Never run ingest while
another process (an evaluation script, a second ingest) has the same `chroma_db/` open.

### Adding a data source

Content always flows **loader → Postgres → `uv run python -m app.rag.ingest`**. Postgres is the
single source of truth and `ingest.py` is the only code that writes to Chroma; a loader never
embeds or touches `chroma_db/`.

A loader (`scripts/load_<name>.py`) should:

1. read the raw file with a reader in `app/loaders/` (standard library only);
2. clean it (`html_to_text`, `drop_repeated_sentences`) and skip junk (`is_repetitive`, too short,
   exact duplicates of published rows via `_fingerprint`);
3. upsert on a stable provenance key — `(source_db, source_page_id)` for pages,
   `(source_type, source_ref)` for articles — assigning a column only when its value changed, so
   `updated_at` (and so re-embedding) moves only for real edits;
4. insert as draft unless `--publish`, and support `--dry-run` (roll back);
5. keep the Postgres work in a function that takes a session, so tests can run it on the test DB.

Existing loaders:

| Loader | Source | Rows |
|---|---|---|
| `scripts.load_tafheem [--db] [--publish] [--dry-run]` | `data/tafheemul_quran.db` (reads only `alquran`, `expl`, `vumika_sura`, `surah_name`) | Book "তাফহীমুল কুরআন" in category "কুরআন ও তাফসীর"; one chapter per sura; page `sura*1000` = introduction, page `sura*1000+ayah` = translation + that ayah's footnotes; `source_db="tafheem"` |
| `scripts.load_articles --source pp-articles\|cs-posts [--sql] [--publish] [--dry-run]` | `data/pp-articles-modified.sql`, `data/cs-posts-modified.sql` | `articles` with `source_type` `pp-article` / `cs-post`, `source_ref` = original id |

Tafheem footnote numbers restart in every sura, so footnotes are joined on `(sura_id, expl_id)`,
never `expl_id` alone; a negative `expl_id` is a "(ক)" sub-note. Both loaders print what they
could not match or skipped — read that output before publishing.

**Superseding the legacy imports.** These three sources were first imported outside this repo as
`pages.source_db = 'tafheemul_quran'` (several ayahs per page) and articles with `source_type`
`chhatrasangbad_post` / `persxpect_article`. `--publish` sets those legacy rows to draft (only the
ids present in the file, for articles) instead of deleting them, and legacy rows are never used as
dedupe targets. The next ingest's sweep removes their chunks from Chroma.

Typical run:

```bash
uv run python -m scripts.load_articles --source cs-posts --dry-run   # read the report
uv run python -m scripts.load_articles --source cs-posts --publish
HF_HUB_OFFLINE=1 uv run python -m app.rag.ingest
```

## 10. Tracing (`app/core/tracing.py`)

Langfuse SDK v4 (OpenTelemetry-based; migrated from v2 on 2026-09-19, see the module docstring).
Setup and trace contents are documented in [docs/tracing.md](docs/tracing.md). What matters when
changing code:

- **Opt-in and fail-safe.** With either key empty, every helper is a no-op, the SDK is never
  imported, and `get_client()` returns the plain OpenAI client. Every helper catches its own
  exceptions, so Langfuse can never break a request.
- **One root span per `/chat` or `/chat/stream` request**, opened in `router.py`. Its context is
  stored in a `ContextVar` **and** returned explicitly, because ambient OpenTelemetry context
  survives `run_in_threadpool` but not Starlette's SSE streaming, which pulls `gen()` one `next()`
  at a time in a fresh context. `stream_answer()` and `finalize_request_trace()` therefore take
  the trace id, parent observation id and trace explicitly.
- Nested observations: `classify-intent`, `expand-query`, `retrieve-context` (retriever),
  `check-relevance-gate` (guardrail), `gpu:embed` / `gpu:rerank`, and one generation per LLM call
  named `llm:{task}`.
- `mode` is written to root-span **metadata** at finalization, not as a tag: it is only known
  after classification, and v4 tags must be set when an observation is created. Failed requests
  set `level=ERROR`.
- `llm.py`'s `openai_wrapper_active()` guard: Langfuse's OpenAI wrapper adds kwargs (`name`,
  `metadata`, `trace_id`, `parent_observation_id`, ...) that a plain client rejects with a 400.
  The guard is true only once `langfuse.openai` has actually patched the OpenAI client, not
  merely when tracing is enabled. If that import fails, `complete()` strips those kwargs and the
  call proceeds untraced.

## 11. Configuration (`app/core/config.py`)

All settings come from `.env`; `.env.example` documents each one. Only `OPENAI_API_KEY` and
`DATABASE_URL` are required.

**Core**

| Variable | Default | Notes |
|---|---|---|
| `OPENAI_API_KEY` | *(required)* | Used only through `app/core/llm.py`. |
| `OPENAI_MODEL` | `gpt-5-mini` | Reasoning model; see the token-budget note in §5. |
| `LLM_REQUEST_TIMEOUT_SECONDS` / `LLM_MAX_RETRIES` | `90` / `1` | Per LLM call. For streaming, the timeout bounds the gap between chunks. |
| `MODEL_BY_TASK` | `{}` | JSON map from task (`intent`, `rewrite`, `persona`, `qa`, `note`, `suggest`, `roleplay`) to model. Change only after an evaluation (§5). |
| `DATABASE_URL` | `postgresql+psycopg2://…` | SQLAlchemy DSN, **not** usable with `psql`. Use `app.db.session.SessionLocal` for one-off queries. |
| `CORS_ALLOW_ORIGINS` | local Vite (`:5173`) + `https://shibirgpt.potropollob.com` | JSON list. An explicit allow-list, not `*`. |
| `HOST` / `PORT` | `0.0.0.0` / `9200` | Not read by `run.sh` or the Dockerfile, which hard-code them (§14). |

**Retrieval**

| Variable | Default | Notes |
|---|---|---|
| `EMBEDDING_MODEL_NAME` | `BAAI/bge-m3` | Multilingual, 1024-dim. The previous `all-MiniLM-L6-v2` tokenized all Bengali to `[UNK]`. Changing it requires a full reindex **and** a new collection name. |
| `RERANKER_MODEL_NAME` | `BAAI/bge-reranker-v2-m3` | Used only at query time; changing it needs no reindex but does need `MIN_RERANK_SCORE` re-tuned. |
| `CHROMA_PERSIST_DIR` | `chroma_db` | Relative to the working directory. |
| `CHROMA_COLLECTION_NAME` | `documents_bge_m3` | Encodes the embedding model on purpose, so a model change without a new name fails loudly on dimension mismatch. |
| `TOP_K` | `5` | Excerpts sent to the LLM after reranking. |
| `FETCH_K` | `25` | Chroma candidates per query variant, before merging and reranking. |
| `MAX_VARIANTS` | `4` | Extra rewrite variants beyond the canonical Bengali form. Each costs one embed and one search; tune with `eval_query_expansion`. |
| `MIN_SIMILARITY` | `0.25` | Loose cosine pre-filter before the reranker. |
| `MIN_RERANK_SCORE` | `0.5` | The relevance gate (§4). Sigmoid scores in [0, 1]: on-topic questions scored 0.94–0.99, an off-topic one peaked at 0.011. The ">2 is relevant" logit heuristic quoted for this model does **not** apply. Re-tune with `tune_threshold` on at least 30 questions. |

**GPU service** (§6)

| Variable | Default | Notes |
|---|---|---|
| `GPU_SERVICE_URL` / `GPU_API_KEY` | *(empty)* | Setting the URL enables GPU mode. The key is sent as `X-API-Key`. |
| `GPU_TIMEOUT_SECONDS` / `GPU_COLD_TIMEOUT_SECONDS` | `30` / `120` | Warm and cold-start timeouts. |
| `GPU_BULK_TIMEOUT_SECONDS` | `300` | Minimum timeout for ingest's bulk `/embed` calls. |
| `GPU_WARM_WINDOW_SECONDS` | `240` | Idle time after which the next call is treated as cold. Keep it below the service's Modal `scaledown_window` (300). |
| `GPU_MAX_RETRIES` | `2` | Retries after the first attempt, for retryable failures only. |

**Tracing** (§10)

| Variable | Default | Notes |
|---|---|---|
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` | *(empty)* | Both are required to enable tracing. |
| `LANGFUSE_HOST` | `http://localhost:3000` | Self-hosted or Cloud URL. **Not** `LANGFUSE_BASE_URL` (that is `langfuse-cli`'s variable); using the wrong one silently points tracing at localhost. |
| `LANGFUSE_ENABLED` | `true` | Master switch. |
| `LANGFUSE_CAPTURE_IO` | `true` | `false` redacts all text while keeping structure, scores, usage, cost and latency. |
| `LANGFUSE_RELEASE` | *(empty)* | Optional build marker. |
| `LANGFUSE_ENVIRONMENT` | `development` | Set to `production` on the deployed server. |

**Evaluation and legacy** (not read on the request path)

| Variable | Default | Notes |
|---|---|---|
| `GROQ_API_KEY` / `GROQ_BASE_URL` | *(empty)* / `https://api.groq.com/openai/v1` | For `eval_generation_ab` only. |
| `GENERATION_CANDIDATES` / `GENERATION_JUDGE_MODEL` | `["gpt-5", "openai/gpt-oss-120b"]` / `qwen/qwen3.6-27b` | `eval_generation_ab`. The judge must not share a model family with any candidate. |
| `RAGAS_JUDGE_MODEL` / `RAGAS_EMBEDDING_MODEL` | `gpt-5-mini` / `text-embedding-3-large` | `eval_ragas`. |
| `TARUN_DB_PATH` / `NOBIN_DB_PATH` | `data/Tarun_Associate.db` / `data/Nobin_Associate.db` | Used only by the one-time SQLite migration. |

## 12. Common tasks

- **Run the server:** `./run.sh` (runs `uv sync --locked`, then uvicorn on `:9200` with
  `HF_HUB_OFFLINE=1`), or `HF_HUB_OFFLINE=1 uv run uvicorn app.main:app --host 0.0.0.0 --port 9200`.
- **Run tests:** `uv run pytest`. No network or real models are needed.
  `tests/test_bengali_embedding.py` loads the real bge-m3 model and runs only with
  `RUN_MODEL_TESTS=1`. GPU-mode tests use `httpx.MockTransport`. Tests that patch `sys.modules`
  or module globals must restore them on teardown. Tests using the `db_session` fixture
  (`tests/conftest.py`; loader tests) run on `TEST_DATABASE_URL`, else `DATABASE_URL`, and are
  **skipped unless the database name contains `test`** — locally `DATABASE_URL` is the real
  corpus. Each test runs in a transaction that is rolled back.
- **Smoke test** after any retrieval, gate or prompt change, using `POST /chat`:
  1. an on-topic question in Bengali script: expect `mode: "qa"` and populated `sources`;
  2. the same question in Banglish: expect the same sources, not an empty or weak result;
  3. an off-topic question: expect `mode: "qa"` and `sources: []`;
  4. a note request such as "এই বইয়ের নোট বানাও …" with a real book title: expect `mode: "note"`.
- **Tune the relevance gate:** copy `scripts/eval_questions.example.json` (don't edit it in place),
  add about 30 real questions (20 answerable, 10 not), and run
  `uv run python -m scripts.tune_threshold your_questions.json`.
- **Measure quality:** see [docs/evaluation.md](docs/evaluation.md).
- **Deploy:** the systemd unit `shibirgpt.service` runs `run.sh` from `/home/lab/apps/shibirgpt`
  with `Restart=always`. `docker-compose.yml` is an alternative single-host setup. Scaling
  guidance: [docs/deployment.md](docs/deployment.md).

## 13. Conventions for changes

- Keep configuration in `Settings`; add every new variable to `.env.example` and §11.
- Route every LLM call through `complete(task, ...)` with `token_budget=`.
- Apply `chunker.normalize()` to text at both ingest and query time.
- Comments in this codebase record *why*: incidents, measurements, rejected alternatives. Keep
  them accurate when changing the code they describe.
- Update this file when you change a contract, a default, or anything listed in §14.

## 14. Known gaps

- **Heavy dependencies in GPU mode:** torch, sentence-transformers and the `nvidia-*` packages are
  still required even when `GPU_SERVICE_URL` is set. Moving them to an optional extra is planned.
- **No deployment pipeline:** `.github/workflows/deploy.yml` is a placeholder (`echo`). CI
  (`ci.yml`) runs the Postgres-backed test suite on pull requests to `main`, but nothing deploys
  on merge.
- **Hard-coded host and port:** `run.sh` and the Dockerfile start uvicorn on `0.0.0.0:9200`
  regardless of `HOST`/`PORT`.
- **Single-worker sessions:** `session_store.py` is an in-process dict (§7).
- **No API authentication or rate limiting**, and `/conversations` exposes every session in the
  process to any caller. See [SECURITY.md](SECURITY.md).
- **Vector store:** still local Chroma. A move to pgvector or Qdrant is planned for a corpus of
  several million chunks, but not started.
- **No lint/format tooling:** the dev dependency group contains only `pytest`.
- **Single global gate:** `min_rerank_score` applies to every book. Per-category thresholds may be
  needed if the corpus spans very different domains.
- **Rewrite fallback:** `expand_query()` costs one LLM call per unique query (cached only for the
  process lifetime). If it fails, retrieval silently uses the raw query, which is fine for Bengali
  script but degrades Banglish recall.
- **Partial corpus cleanup:**
  - Applied: the 106 duplicate pages in `book203_cross_contamination.csv` are excluded (reason
    `book203_cross_contamination.csv:loser`).
  - No-op: `orphaned_source_pages.csv`, whose one row never reached Postgres.
  - Deliberately **not** applied: `arabic_placeholder_pages.csv` (343 rows). Most flagged pages
    are not actually broken (many just contain Arabic script, or keep a usable Bengali
    translation), so applying it would hurt recall. It needs regenerating with a real
    broken-page heuristic first.
