# Shibir Chat Back-End — Development & Architecture Mandates

This file is the single source of truth for repository-wide workflows, architectural guidelines, coding standards, and operational mandates. All development on the Shibir Chat Back-End MUST adhere strictly to these principles.

---

## 1. Core Architecture & Philosophy

Shibir Chat Back-End is a FastAPI-based Retrieval-Augmented Generation (RAG) Q&A service specifically tailored for Bengali-language library lookups. It is **not** a general-purpose conversational LLM chatbot. It utilizes structured documents stored in Postgres and vectorized chunks in ChromaDB to answer user questions strictly based on retrieved content.

### Architectural Layers
The codebase is structured into three clean, linear layers to prevent indirection:
1. **API Layer (`app/api/router.py`)**: Thin HTTP controllers mapping HTTP request schemas (Pydantic) to the service layer and returning responses. It contains endpoints like `/health`, `/chat`, `/chat/stream`, and `/conversations`.
2. **Service Layer (`app/services/`)**: Business logic orchestrators.
   - `qa_service.py`: Retrieval, relevance check, and LLM answer generation.
   - `note_service.py`: Book/chapter summarization via Postgres maps/reduces (non-retrieval).
   - `roleplay_service.py`: Conversational mode with persona extraction (non-retrieval).
   - `suggestion_service.py`: Opinionated or recommendation generation.
   - `intent_classifier.py`: Routes queries using RegEx pre-filtering and LLM fallback classification.
   - `session_store.py`: In-memory thread-safe chat sessions (supports single-worker deployment).
3. **RAG/Database Layer (`app/rag/`, `app/db/`)**: Data representation and ingestion logic.
   - `chunker.py`: Clean NFC normalization and chunk segmentation.
   - `embedder.py`: Model vector generation (BGE-M3 1024-dimension).
   - `reranker.py`: Cross-encoder scoring (BGE-Reranker-V2-M3 Sigmoid [0,1]).
   - `gpu_client.py`: External server client (Modal fallback).
   - `query_rewriter.py`: Converts Banglish/English/Bengali to high-recall Bengali queries.
   - `chroma_client.py`: Handles connections to the persistent Chroma client.
   - `models.py` & `session.py`: SQLAlchemy Postgres schemas and Session makers.

---

## 2. Engineering Standards & Conventions

### 2.1 Configuration Mandate
- **Single Source of Truth**: All configuration MUST be defined in `app/core/config.py` using Pydantic's `BaseSettings`.
- **No Direct Environment Calls**: Never import `os` to call `os.getenv()` or `load_dotenv()` directly in product or service files. Always import `settings` from `app.core.config`.
- **Dynamic Port & Host**: Production runs must read `settings.host` and `settings.port`.

### 2.2 LLM Call Constraints (`app/core/llm.py`)
- **Sanctioned Client Only**: No call site may construct `openai.OpenAI` or similar clients directly. All calls MUST route through `app/core/llm.complete(task, messages, **overrides)`.
- **Task-Based Routing**: Tasks (`intent`, `rewrite`, `persona`, `qa`, `note`, `suggest`, `roleplay`) look up their model in `settings.model_by_task`, defaulting to `settings.openai_model` (`gpt-5-mini`).
- **Token Cap & Reasoner Parameters**: OpenAI reasoning models (`gpt-5-mini`, `gpt-5`) use `max_completion_tokens` instead of `max_tokens` for token caps and ONLY accept `temperature=1`. To maintain cross-model compatibility, call sites MUST pass `token_budget=N` to `complete()`, which translates to the appropriate model-specific parameter automatically.

### 2.3 RAG Pipeline Constraints
- **Text Normalization**: All raw text MUST be passed through `chunker.normalize(text)` (which cleans up whitespaces and forces Unicode NFC normalization) at both ingestion time and query time.
- **Cosine Space Metric**: Chroma collection MUST be constructed with `hnsw:space="cosine"`. This ensures the similarity thresholding in `retriever.py` aligns correctly with mathematical distance expectations.
- **Reranker Scores (Sigmoid)**: The cross-encoder reranker utilizes Sigmoid activation, yielding scores strictly bounded in `[0, 1]`. Do **not** apply logits-based "greater than 2" heuristics to check relevance.

---

## 3. Grounded Generation & The Prompt Contract

Maintaining strict alignment between retrieved context and LLM outputs is critical to avoiding hallucinations.

1. **Strict Context Adherence**: The LLM system prompt in `generator.py` instructs the model to answer **only** from the provided excerpt (`উদ্ধৃত অংশ`) blocks.
2. **Standard Bengali Output**: Answers must be phrased entirely in standard, clear Bengali. No English sentences, no Banglish.
3. **Claim Citations**: Every factual claim must be backed by numerical citations matching the source block index, e.g., `[১]`, `[২]`.
4. **Partial Answers & Refusal**:
   - If the excerpts partially cover the question, answer with what's there and name the gap in one sentence, rather than refusing outright.
   - If the topic is entirely absent from the excerpts, refuse.
5. **Relevance Gate Coordination**:
   - The relevance decision is determined **upstream in code** by comparing the top rerank score against `settings.min_rerank_score`.
   - If the score falls below the gate, `qa_service` still calls `generate_answer` but passes an **empty citation list**. The prompt's refusal rules then prompt the LLM to emit a natural, user-friendly Bengali refusal, while the service returns `sources: []`.
   - **Crucial**: Never return empty/canned refusal strings directly from code if the gate fails, and never attach sources to a failed answer.

---

## 4. Observability & Tracing (`app/core/tracing.py`)

- **Opt-In & Fail-Safe**: Tracing via Langfuse v4 is optional. It is activated only when `LANGFUSE_ENABLED=true` and both `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are configured. If disabled, all tracing methods are safe no-ops, the SDK is not imported, and zero overhead is introduced.
- **Exception Isolation**: Tracing code must swallow its own exceptions and log warnings rather than causing requests or client creation to fail.
- **SSE Streaming Context Propagation**: Ambient OpenTelemetry trace contexts do not survive Starlette's asynchronous Server-Sent Events (SSE) generator lines. Therefore, the streaming generator (`stream_answer()`) MUST explicitly forward `trace_id` and `parent_observation_id` parameters to coordinate child observations under the HTTP root trace.
- **Privacy Controls**: When `LANGFUSE_CAPTURE_IO=false`, all prompt texts, retrieval texts, and completion outputs MUST be redacted from traces while preserving metadata, token counts, latency, and structure.

---

## 5. Development & Maintenance Runbooks

### 5.1 Verification & Tests
- **Testing Entry**: Run unit tests using `uv run pytest`.
- **No Side-Effects in Fixtures**: Any test module that mocks or overrides modules in `sys.modules` (e.g., `test_query_rewriter.py` mocking `app.core.llm`) MUST clean up and restore original modules upon teardown/yield to prevent breaking concurrent or subsequent test runs (e.g., `test_tracing.py`).

### 5.2 Server Execution
- **Local Startup**: Run `./run.sh` to sync environment dependencies via `uv` and start the FastAPI uvicorn server on port `9200`.
- **Systemd Service**: Deployed environments use `shibirgpt.service` located at `/etc/systemd/system/shibirgpt.service` directing to `/home/lab/apps/shibirgpt`.

### 5.3 Database & Indexing Maintenance
- **Only Published candidate Content**: Ingestion filters strictly on `status == 'published'`. Soft-exclusions like `excluded_from_rag == true` are bypassed.
- **The Reindexing Rule**: To completely re-index the database content into Chroma, deleting the `chroma_db` directory is **insufficient** on its own because `ingest.py` tracks processed rows via `embedded_at`. You MUST reset the database tracking column along with clearing Chroma:
  ```bash
  # 1. Clear Chroma Storage
  rm -rf chroma_db

  # 2. Reset Postgres tracking timestamps (via Python session or SQL client)
  # UPDATE pages SET embedded_at = NULL;
  # UPDATE articles SET embedded_at = NULL;

  # 3. Trigger Ingest
  HF_HUB_OFFLINE=1 uv run python -m app.rag.ingest
  ```

---

## 6. Known Architectural & Operational Gaps

When designing new features or extending the codebase, be aware of the following design traits:
1. **Single-Worker Session Store**: `app/services/session_store.py` keeps chat histories and session mappings in an in-memory dictionary. Scaling to multiple workers or pods requires replacing this with a shared state manager (e.g., Redis or a dedicated database table).
2. **Hard CUDA / Local Torch Dependencies**: Even when configuring `GPU_SERVICE_URL` to offload embedding/reranking to Modal, `torch`, `sentence-transformers`, and `nvidia-*` libraries remain hard requirements in `pyproject.toml`.
3. **No-op Deployment Flow**: The GHA workflow `.github/workflows/deploy.yml` is a placeholder. Production deploys are currently manual systemd service restarts.
4. **Query Expansion Fail-safe Fallback**: If query rewriting fails (LLM timeout, API quota), the pipeline silently falls back to using the raw user query. While robust, this degrades recall for Banglish and English inputs to pre-rewriting performance levels.
