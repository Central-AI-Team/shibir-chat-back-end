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
  api/router.py           HTTP layer. Routes: GET /health, POST /ask, POST /note. Thin — just
                          binds request schemas to the service layer and returns the result.
  db/
    models.py                SQLAlchemy models: Category, Book, Chapter, Page, Article. All
                              content now lives in Postgres (migrated off the old per-source
                              SQLite files) — see "Data model" below.
    session.py                SessionLocal — plain sessionmaker bound to settings.database_url.
  services/
    qa_service.py               Orchestration: answer_question(query) calls
                                 retrieve_relevant_docs() then decides, from the top
                                 rerank_score alone, whether to call generate_answer() or
                                 return a hard refusal with empty sources. See "Prompt
                                 contract" below — this is the fix for the old "answer says
                                 no but sources are attached anyway" bug.
    note_service.py              generate_chapter_note(chapter_id) -> dict. NOT retrieval —
                                  pulls every published page of a chapter in reading order
                                  from Postgres and map-reduces it into one note. A chapter
                                  summary is an aggregation task; similarity search over a
                                  handful of chunks can't do that.
  schemas/query.py           Pydantic models: QueryRequest, Citation (book/chapter/source_db/
                              content/similarity/rerank_score), QueryResponse, NoteRequest.
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
                                   via one Gemini call, cached (lru_cache). Falls back to the
                                   raw query on any failure — including on a truncated/empty
                                   LLM response, so watch for silent no-op fallback if you ever
                                   lower max_tokens again (see the comment in that file for the
                                   incident this guards against).
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
    generator.py                   generate_answer(query, citations) -> str. Bengali system
                                    prompt, temperature=0.2, numbered excerpts for citation.
                                    See "Prompt contract" below.
    ingest.py                      Standalone script (python -m app.rag.ingest). Reads
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
  tune_threshold.py                python -m scripts.tune_threshold [questions.json]. Runs a
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

Chroma chunk ids are `f"{prefix}_{row.id}_c{chunk_index}"` (e.g. `page_842_c0`, `page_842_c1`,
`article_12_c0`) with metadata `{"book", "chapter", "page_id", "chunk_index", "row_key",
"source_db"}`. `row_key` (`f"{prefix}_{row.id}"`) is what `ingest.py` deletes-by before
re-upserting a row's chunks.

## Prompt contract (important — do not regress this)

`generator.py`'s system prompt instructs the model to:
1. Answer **only** from the numbered `উদ্ধৃত অংশ` (excerpt) blocks — no outside knowledge.
2. **Partial answers are allowed.** If the excerpts partially cover the question, answer with
   what's there and name the gap in one sentence, rather than refusing outright. Refuse only
   if the topic isn't mentioned at all.
3. Cite every claim with `[১]` / `[২]` style excerpt numbers.
4. Answer entirely in standard Bengali — no English sentences, no Banglish.
5. `temperature=0.2` — deterministic, grounded Q&A, not creative writing.

**Hard refusal is decided upstream, in `qa_service.py`, by score — not by the LLM's
judgment.** If the top citation's `rerank_score` is below `settings.min_rerank_score`,
`qa_service.answer_question()` returns a fixed Bengali "not found" message with `sources: []`
and never calls the LLM at all. This is deliberate: the old version always attached sources
regardless of whether the LLM actually used them, so the UI could show "not in the database"
next to a populated sources list. Retrieval quality and the answer text can no longer
contradict each other. If you change this gate, re-verify with an on-topic, an off-topic
(Bengali), and a Banglish on-topic question — see "Smoke test" below.

## Configuration (`app/core/config.py`)

All settings are read from `.env` (see `.env.example` for the full documented list).
Only `GEMINI_API_KEY` and `DATABASE_URL` are required; everything else has a working default:

| Env var | Default | Notes |
|---|---|---|
| `GEMINI_API_KEY` | *(required)* | Get one at https://aistudio.google.com/apikey. Free tier is capped at 20 requests/day **per Google Cloud project** — swapping to a different key under the same project does not reset this. |
| `GEMINI_MODEL` | `gemini-flash-latest` | A "thinking" model — it spends completion-token budget on internal reasoning before emitting visible output. Any Gemini call here needs a generous `max_tokens` (2000+ for short JSON-style outputs); too low silently truncates and fails downstream parsing. |
| `GEMINI_BASE_URL` | `https://generativelanguage.googleapis.com/v1beta/openai/` | Gemini's OpenAI-compatible endpoint — lets us reuse the `openai` SDK instead of adding `google-generativeai` as a dependency. |
| `DATABASE_URL` | `postgresql+psycopg2://...` | SQLAlchemy DSN format — **not** parseable by `psql` directly. Use a Python script with `app.db.session.SessionLocal` for one-off queries/admin, not raw `psql -c`. |
| `EMBEDDING_MODEL_NAME` | `BAAI/bge-m3` | Multilingual, 1024-dim, handles Bengali (unlike the old `all-MiniLM-L6-v2`, which tokenized Bengali entirely to `[UNK]`). Changing this requires a full reindex — see below. |
| `RERANKER_MODEL_NAME` | `BAAI/bge-reranker-v2-m3` | Cross-encoder used only at query time (not stored in Chroma), so changing it does NOT require a reindex. |
| `CHROMA_PERSIST_DIR` | `chroma_db` | Relative to the working directory the process is started from. |
| `CHROMA_COLLECTION_NAME` | `documents_bge_m3` | Name encodes the embedding model on purpose — changing `EMBEDDING_MODEL_NAME` without also bumping this would silently mix incompatible-dimension vectors into one collection (Chroma would just reject them with a dimension-mismatch error, which is the point). |
| `TOP_K` | `5` | Chunks sent to Gemini after reranking. |
| `FETCH_K` | `25` | Candidates pulled from Chroma before reranking. |
| `MIN_SIMILARITY` | `0.25` | Cheap cosine pre-filter before paying for the cross-encoder. Loose on purpose. |
| `MIN_RERANK_SCORE` | `0.5` | The real "do we have an answer?" gate — see "Prompt contract" above. **Scores are Sigmoid-activated, in [0, 1]**, not raw bge-reranker logits. Smoke-tested on this corpus: on-topic questions scored 0.94–0.99, a fully off-topic question topped out at 0.011. Re-tune with `python -m scripts.tune_threshold` once you have ~30 real questions (20 answerable, 10 not). |
| `TARUN_DB_PATH` / `NOBIN_DB_PATH` | `data/Tarun_Associate.db` / `data/Nobin_Associate.db` | Legacy — only used by the one-time `scripts/migrate_sqlite_to_postgres.py`, not by `ingest.py` anymore. |
| `HOST` / `PORT` | `0.0.0.0` / `9200` | Not currently read by `run.sh`'s uvicorn invocation (hardcoded there) — see Known gaps. |

The real `.env` may also have unused leftover `GROQ_API_KEY` / `OPENAI_API_KEY` lines from
earlier providers that were tried before settling on Gemini. They're inert; safe to remove
or ignore.

## API contract

`GET /health` → `{"status": "ok"}`

`POST /ask`

Request:
```json
{"query": "your question in any language — Bengali script, Banglish, or English"}
```

Response:
```json
{
  "query": "...",
  "answer": "... (always Bengali)",
  "sources": [
    {
      "book": "...", "chapter": "...", "source_db": "tarun",
      "content": "full chunk text (includes the বই:/অধ্যায়: header)",
      "similarity": 0.6471, "rerank_score": 0.9861
    }
  ]
}
```
`sources` is `[]` whenever the answer is a refusal — never populated alongside a "not found"
answer. See "Prompt contract" above.

`POST /note`

Request:
```json
{"chapter_id": 2509}
```

Response:
```json
{"book": "...", "chapter": "...", "pages_used": 139, "note": "... (structured Bengali summary)"}
```
404s with a Bengali error message if the chapter doesn't exist or has no published pages.
Note: large chapters (100+ pages) issue several sequential Gemini calls (map-reduce over
~12,000-char page groups) — mind the free-tier daily quota if testing repeatedly.

Unknown extra fields in the request body are silently ignored (default Pydantic behavior).

## Common tasks

- **Run the server**: `./run.sh` (creates venv, installs deps, starts uvicorn on :9200), or
  manually: `source venv/bin/activate && HF_HUB_OFFLINE=1 uvicorn app.main:app --host 0.0.0.0 --port 9200`.
- **Smoke test**: run three `/ask` calls — an on-topic question in Bengali script, the same
  question in Banglish (must return the same sources, not an empty/weak result), and a
  deliberately off-topic question (must return `sources: []`, not populated). Then one
  `/note` call against a real `chapter_id`. See `scripts/tune_threshold.py` for a scripted
  version of the retrieval half of this.
- **Tune the refusal threshold**: fill in `scripts/eval_questions.example.json` (copy it,
  don't edit in place) with ~30 real questions — 20 answerable from the corpus, 10 not — then
  `python -m scripts.tune_threshold your_questions.json`.
- **Deploy**: `shibirgpt.service` is a systemd unit that runs `run.sh` with
  `WorkingDirectory=/home/lab/apps/shibirgpt`, `Restart=always`.

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
HF_HUB_OFFLINE=1 python -m app.rag.ingest
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

## Known gaps / things a future change might need to address

- `run.sh` hardcodes `--host 0.0.0.0 --port 9200` in its uvicorn invocation instead of
  reading `settings.host`/`settings.port` — those config fields exist but aren't wired
  through to the shell script.
- No CI, no Dockerfile. `tests/test_bengali_embedding.py` exists (regression coverage for the
  MiniLM→bge-m3 migration: no `[UNK]` on Bengali input, unrelated sentences don't collapse to
  near-identical vectors, chunking splits long text, `normalize()` is idempotent) but nothing
  runs it automatically yet.
- `min_rerank_score` is a single global threshold. If the corpus grows to cover very
  different domains, a per-book or per-category threshold might separate "answerable" from
  "not answerable" better than one global cutoff — not needed at the current corpus size.
- `query_rewriter.expand_query()` costs one Gemini call per unique query (cached via
  `lru_cache`, so repeats are free within a process lifetime, but the cache doesn't persist
  across restarts). On a fully offline / air-gapped deployment this call will fail every time
  and silently fall back to the raw query — fine for Bengali-script input, degrades Banglish
  retrieval quality back to pre-fix levels.
