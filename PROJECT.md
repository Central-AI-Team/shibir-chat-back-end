# Shibir Chat Back-End — Project Guide for AI Tools

This file is written so any AI coding assistant (Claude, Cursor, Copilot, Codex, etc.) can
pick up this project cold and be immediately useful. Read this before making changes.

## What this project is

A FastAPI backend for a Bengali-language Q&A chatbot. A user asks a question, the service
retrieves relevant excerpts from a religious/educational book library (stored as embeddings
in a Chroma vector store) and asks an LLM (Gemini) to answer **strictly from those excerpts**
— it is instructed to refuse rather than hallucinate when the excerpts don't cover the
question. This is retrieval-augmented generation (RAG), not a general-purpose chatbot.

## Architecture

```
app/
  main.py               FastAPI app factory. Just creates the app and includes the router.
  core/config.py         Single source of config — a pydantic-settings Settings object that
                          reads .env. Nothing else in the app should call os.getenv() or
                          load_dotenv() directly; import `settings` from here instead.
  api/router.py           HTTP layer. One route: POST /ask. Thin — just binds the request
                          schema to the service layer and returns its result.
  services/qa_service.py    Orchestration layer: answer_question(query) calls
                             retrieve_relevant_docs() then generate_answer() and assembles
                             the QueryResponse. Add new orchestration flows here, not in the
                             router or in rag/*.
  schemas/query.py           Pydantic models: QueryRequest (request), Citation (one retrieved
                              chunk with book/chapter/source_db/content), QueryResponse.
  rag/
    embedder.py                embed_text(str) -> list[float], via sentence-transformers
                                (all-MiniLM-L6-v2, loaded once at import time).
    chroma_client.py             get_collection() — the ONLY place that constructs the Chroma
                                  PersistentClient. Both retriever.py and ingest.py call this;
                                  never construct a client anywhere else.
    retriever.py                  retrieve_relevant_docs(query, top_k=None) -> list[Citation].
                                   Embeds the query, queries Chroma for nearest neighbors, zips
                                   documents+metadatas into Citation objects.
    generator.py                   generate_answer(query, citations) -> str. Builds a
                                    grounding prompt (see "Prompt contract" below) and calls
                                    the LLM via the OpenAI SDK pointed at Gemini's
                                    OpenAI-compatible endpoint.
    ingest.py                       Standalone script (python -m app.rag.ingest). Reads both
                                     source SQLite DBs, cleans HTML, embeds each page, and
                                     upserts into the shared Chroma collection. Run this
                                     whenever the source .db files change.

data/
  Tarun_Associate.db      Source library #1 (3441 pages / ~3436 ingested chunks).
  Nobin_Associate.db      Source library #2 (708 pages / 708 ingested chunks).

chroma_db/                Generated vector store (gitignored). Delete + re-run
                           `python -m app.rag.ingest` to rebuild from scratch.
```

There are deliberately only three layers (api → services → rag). There is exactly one
vector store and one LLM provider, so no repository/interface abstraction was introduced —
it would add indirection with no real swappability benefit at this size.

## Data model / why IDs are namespaced

`Tarun_Associate.db` and `Nobin_Associate.db` share an identical schema
(`categories`, `books`, `chapters`, `pages`, each with integer primary keys) but they are
**independent libraries with overlapping IDs** — both start numbering at low integers, and
e.g. both have a `books.id = 1` that is a *different* book's content per source.

Because of this, `ingest.py` writes every chunk into Chroma with:
- **id**: `f"{source_key}_{page_id}"` (e.g. `tarun_1`, `nobin_1`) — never a bare page id.
- **metadata**: `{"book", "chapter", "page_id", "source_db"}` where `source_db` is
  `"tarun"` or `"nobin"`.

If you ever add a third source database, follow the same pattern: give it a short key,
prefix its Chroma ids with that key, and tag `source_db` accordingly. Do not reuse an
existing key.

## Prompt contract (important — do not regress this)

`generator.py`'s prompt explicitly instructs the model to:
1. Answer **only** from the `Context` block (the retrieved citations) — no outside
   knowledge, no guessing.
2. If the context doesn't contain the answer, say so clearly in Bengali instead of
   fabricating one.
3. Always answer in Bengali, regardless of the question's language.

This has been verified to work: on-topic questions get answers sourced from the DB, and
off-topic questions (e.g. general trivia) get a clear "not in the database" reply in
Bengali rather than a hallucinated answer. If you change this prompt, re-verify both cases.

## Configuration (`app/core/config.py`)

All settings are read from `.env` (see `.env.example` for the full documented list).
Only `GEMINI_API_KEY` is required; everything else has a working default:

| Env var | Default | Notes |
|---|---|---|
| `GEMINI_API_KEY` | *(required)* | Get one at https://aistudio.google.com/apikey |
| `GEMINI_MODEL` | `gemini-flash-latest` | `gemini-2.0-flash` hit a hard `limit: 0` free-tier quota on the key tested during setup — `gemini-flash-latest` worked. Re-check quota if you change this. |
| `GEMINI_BASE_URL` | `https://generativelanguage.googleapis.com/v1beta/openai/` | Gemini's OpenAI-compatible endpoint — lets us reuse the `openai` SDK instead of adding `google-generativeai` as a dependency. |
| `EMBEDDING_MODEL_NAME` | `all-MiniLM-L6-v2` | Must match whatever was used to build the current `chroma_db/` — changing it requires a full re-ingest. |
| `CHROMA_PERSIST_DIR` | `chroma_db` | Relative to the working directory the process is started from. |
| `CHROMA_COLLECTION_NAME` | `documents` | |
| `TOP_K` | `3` | Number of chunks retrieved per query. |
| `TARUN_DB_PATH` / `NOBIN_DB_PATH` | `data/Tarun_Associate.db` / `data/Nobin_Associate.db` | Used only by `ingest.py`. |
| `HOST` / `PORT` | `0.0.0.0` / `9200` | Not currently read by `run.sh`'s uvicorn invocation (hardcoded there) — see Known gaps. |

The real `.env` also still has unused leftover `GROQ_API_KEY` / `OPENAI_API_KEY` lines from
earlier providers that were tried before settling on Gemini. They're inert; safe to remove
or ignore.

## API contract

`POST /ask`

Request:
```json
{"query": "your question in any language"}
```

Response:
```json
{
  "query": "...",
  "answer": "... (always Bengali)",
  "sources": [
    {"book": "...", "chapter": "...", "source_db": "tarun", "content": "full chunk text"}
  ]
}
```

Unknown extra fields in the request body are silently ignored (default Pydantic behavior) —
this is intentional; a previous iteration had an unused `user_id` field that was removed.

## Common tasks

- **Run the server**: `./run.sh` (creates venv, installs deps, starts uvicorn on :9200), or
  manually: `source venv/bin/activate && HF_HUB_OFFLINE=1 uvicorn app.main:app --host 0.0.0.0 --port 9200`.
- **Rebuild the vector store** (after changing source `.db` files, the embedding model, or
  the chunking logic in `ingest.py`): `rm -rf chroma_db && HF_HUB_OFFLINE=1 python -m app.rag.ingest`.
- **Smoke test**: `curl -X POST http://127.0.0.1:9200/ask -H "Content-Type: application/json" -d '{"query": "..."}'`
- **Deploy**: `shibirgpt.service` is a systemd unit that runs `run.sh` with
  `WorkingDirectory=/home/lab/apps/shibirgpt`, `Restart=always`.

## Known gaps / things a future change might need to address

- No tests, no CI, no Dockerfile — none existed before this restructure and none were added
  (kept the change scoped to what was asked).
- `run.sh` hardcodes `--host 0.0.0.0 --port 9200` in its uvicorn invocation instead of
  reading `settings.host`/`settings.port` — those config fields exist but aren't wired
  through to the shell script.
- `app/models/` was deleted during the restructure — it was empty and unreferenced
  (likely SQLAlchemy scaffolding that was never built out). If you need an ORM layer over
  the source SQLite DBs, it doesn't exist yet; `ingest.py` uses raw `sqlite3`.
- Retrieval quality is plain nearest-neighbor over MiniLM embeddings with no reranking —
  fine for this corpus size (~4,144 chunks) but worth knowing if answers seem to miss an
  obviously-relevant chunk from the smaller `nobin` source in favor of `tarun` (which is
  ~5x larger and can dominate top-k results for broad queries).
