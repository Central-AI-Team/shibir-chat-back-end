"""Settings.

CHANGED defaults:
  embedding_model_name  all-MiniLM-L6-v2 -> BAAI/bge-m3   (the critical one)
  top_k                 3 -> 5
NEW:
  reranker_model_name, fetch_k, min_similarity, min_rerank_score
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # generator.py / query_rewriter.py / note_service.py all call the shared
    # OpenAI client in app/core/llm.py with this key/model.
    openai_api_key: str
    openai_model: str = "gpt-5-mini"

    # Per-request timeout (seconds) for every LLM call in app/core/llm.py --
    # both the OpenAI client (get_client()) and the Groq-compatible client
    # (get_client_for()) are constructed with this. For a streamed answer it
    # bounds the gap between chunks, not the whole answer. The slowest call
    # observed live is a gpt-5-mini rewrite at ~40s on long input, so 90s
    # leaves headroom while still failing a stuck call in reasonable time.
    # (It was 450s, which made a stuck request look like a 7+ minute hang.)
    llm_request_timeout_seconds: int = 90

    # How many times the openai SDK retries a failed/timed-out LLM call. The
    # SDK default is 2, so worst case is (1 + retries) * timeout -- with the
    # old 450s timeout that was ~22 minutes. 1 retry covers a transient blip.
    llm_max_retries: int = 1

    # Latency knobs (see CLAUDE.md §5). gpt-5-mini spends completion tokens on
    # hidden reasoning before it writes anything; "minimal" cuts most of that
    # wait. Applied only to OpenAI gpt-5* models (Groq/other models reject it);
    # empty string = send nothing (the API default, "medium").
    openai_reasoning_effort: str = "minimal"
    # Per-task override of the above. Measured live (2026-10-05) on gpt-5-mini:
    # "minimal" misrouted plain questions as the intent classifier
    # ("রোযা কীভাবে আত্মসংযম শেখায়?" -> SUGGESTION 3/3, "যাকাতের গুরুত্ব নিয়ে
    # বিস্তারিত বলো" -> NOTE 3/3) and left Latin letters in a Banglish rewrite
    # ("শirk"), which corrupts retrieval. "low" matched the default on intent.
    # Tasks not listed use openai_reasoning_effort. JSON map in .env.
    openai_reasoning_effort_by_task: dict[str, str] = {"intent": "low", "rewrite": "low"}
    # Completion-token caps per mode, applied when a call site passes no
    # token_budget. Bengali tokenizes ~3-4x worse than English, so these are
    # generous: a cap that truncates an answer mid-sentence is worse than slow.
    # NOTE is a ceiling on note_service's own per-call budgets (4000 map / 6000 reduce).
    max_tokens_qa: int = 2500
    max_tokens_suggestion: int = 2000
    max_tokens_roleplay: int = 1500
    max_tokens_note: int = 6000
    # Excerpts / characters per excerpt put in the prompt. Defaults send
    # everything retrieval returns (top_k chunks are ~900 chars + header), so
    # lowering them is a speed-vs-grounding trade-off; if context_top_k < top_k
    # the sources list can show excerpts the model never saw.
    context_top_k: int = 5
    context_max_chars: int = 1200
    # Queries whose letters are at least this fraction Bengali script skip the
    # LLM rewrite (query_rewriter.expand_query).
    rewrite_skip_bengali_ratio: float = 0.6
    # When intent needs the LLM AND the query needs a rewrite, do both in one
    # call returning JSON; falls back to the two-call path on invalid JSON.
    combine_intent_rewrite: bool = True
    # LRU of query embeddings, keyed by the normalized rewritten query. 0 = off.
    embed_cache_size: int = 512
    # Embed + search + rerank a dummy query in the background at startup.
    warmup_on_startup: bool = True
    # Max concurrent LLM map calls when building a chapter note.
    note_map_concurrency: int = 4

    # Groq (OpenAI-compatible API, different base_url) -- used only by
    # scripts/eval_generation_ab.py for a different-provider challenger model
    # and an independent judge model. Not read anywhere in the production
    # request path. Empty default so importing settings never requires it.
    groq_api_key: str = ""
    groq_base_url: str = "https://api.groq.com/openai/v1"

    # scripts/eval_generation_ab.py: which models to A/B against openai_model
    # (always included as the baseline) for Bengali answer quality, and which
    # model judges them. The judge is deliberately NOT in this list -- it
    # must be a model/family none of the candidates are, so it isn't scoring
    # its own (or a sibling's) answers. See that script's module docstring.
    #
    # openai/gpt-oss-120b (not llama-3.3-70b-versatile) is the Groq-hosted
    # challenger -- confirmed live (Sept 2026) that this project's
    # GROQ_API_KEY does NOT have access to either Llama model on Groq
    # (404 "does not exist or you do not have access to it"), despite both
    # being listed on Groq's own docs as self-serve. Reconfirm live access
    # before adding any Groq model back to this list.
    generation_candidates: list[str] = ["gpt-5", "openai/gpt-oss-120b"]
    generation_judge_model: str = "qwen/qwen3.6-27b"

    # Per-task model routing (app/core/llm.get_model(task) / complete(task,
    # ...)). Keys are the six call-site task names: intent, rewrite, persona
    # (mechanical/structured -- a cheap fast model is usually enough, and
    # these run on EVERY request before the user sees anything) and qa,
    # note, suggest, roleplay (user-facing Bengali generation -- quality
    # matters, latency is less critical).
    #
    # Empty by default ON PURPOSE: get_model(task) falls back to
    # openai_model for any task not listed here, so today's single-model
    # behavior is EXACTLY unchanged until a task is deliberately added below.
    # Do not hand-fill these with a "cheap" model without first running
    # scripts/eval_task_routing.py (or equivalent) for that specific task --
    # a cheaper model can quietly raise structured-output parse-failure
    # rates on the mechanical tasks, which is worse than the latency it
    # saves. When you do fill one in, record here: the model, the date, and
    # the eval (+ its question-set size) that justified it, e.g.:
    #   "intent": "llama-3.1-8b-instant",  # 2026-09-10, eval_task_routing.py
    #                                      # vs gpt-5-mini on chat_intent_
    #                                      # test_questions.json (25 Qs):
    #                                      # accuracy unchanged, -420ms/call
    model_by_task: dict[str, str] = {}

    # scripts/eval_ragas.py's judge LLM + embeddings -- isolated eval tooling
    # (venv-ragas/, NOT the production app; see that script's module
    # docstring). RAGAS's internal prompts (claim decomposition, verdicts)
    # are English-centric, so this must be a Bengali-CAPABLE model, not a
    # weak default. gpt-5-mini validated live (Sept 2026): correct
    # faithfulness/answer_relevancy judgments on a Bengali sample via
    # LangchainLLMWrapper(..., bypass_temperature=True) -- REQUIRED for any
    # gpt-5-family judge, since RAGAS's own prompts pass their own default
    # temperature that gpt-5/gpt-5-mini reject. gpt-5 works identically but
    # at ~5x the cost for no measured latency or quality difference in that
    # check, so gpt-5-mini is the default; not qwen/qwen3.6-27b (this
    # project's GROQ_API_KEY hits a 1000 output-tokens/min rate limit that
    # RAGAS's multi-call-per-metric pattern blows through immediately).
    # text-embedding-3-large (not bge-m3) avoids a second multi-GB local
    # model load in the isolated eval environment -- it's multilingual and
    # good enough for answer_relevancy's cosine-similarity use, though a
    # bge-m3-backed HuggingFaceEmbeddings wrapper is a drop-in swap if you
    # have headroom to run it (see that script's docstring).
    ragas_judge_model: str = "gpt-5-mini"
    ragas_embedding_model: str = "text-embedding-3-large"

    # all-MiniLM-L6-v2 cannot tokenize Bengali -- every word became [UNK].
    embedding_model_name: str = "BAAI/bge-m3"
    reranker_model_name: str = "BAAI/bge-reranker-v2-m3"

    # --- External GPU service (shibir-chat-gpu-service on Modal) ----------
    # When gpu_service_url is set, app/rag/embedder.py and reranker.py call
    # its /embed and /rerank endpoints (via app/rag/gpu_client.py) instead of
    # loading bge-m3 / bge-reranker-v2-m3 locally -- torch is then never
    # imported. Empty (default) = the local sentence-transformers path.
    gpu_service_url: str = ""
    gpu_api_key: str = ""           # sent as the X-API-Key header
    gpu_timeout_seconds: float = 30
    # A call may hit a scaled-to-zero Modal container that has to boot and
    # load both models, so it gets a longer timeout: the first call in a
    # process, and any call more than gpu_warm_window_seconds after the last
    # successful one.
    gpu_cold_timeout_seconds: float = 120
    # Must stay below the service's Modal scaledown_window (300s): past this
    # much idle time the container may be gone, so assume a cold start.
    gpu_warm_window_seconds: float = 240
    # Floor for bulk /embed calls (ingest: 256 full ~900-char chunks per
    # request). The 30s warm timeout is sized for request-path calls of a
    # handful of short queries; a full ingest slice read-timed-out under it on
    # 2026-09-27 even though the container was warm.
    gpu_bulk_timeout_seconds: float = 300
    # Retries after the first attempt, only on connection errors / 5xx / 429.
    gpu_max_retries: int = 2

    database_url: str = "postgresql+psycopg2://user:password@localhost:5432/shibir_chat"

    chroma_persist_dir: str = "chroma_db"
    chroma_collection_name: str = "documents_bge_m3"  # new name = new index

    top_k: int = 5        # chunks sent to the LLM after reranking
    fetch_k: int = 25     # candidates pulled from Chroma before reranking (PER query variant, then merged)

    # Extra search strings app/rag/query_rewriter.expand_query() generates
    # beyond the canonical Bengali-script form (which is always kept). Each
    # extra variant costs one more embed + Chroma query, so this is a
    # recall-vs-cost knob -- tune with scripts/eval_query_expansion.py before
    # changing it.
    max_variants: int = 4

    # Cheap pre-filter on cosine similarity (1 - distance). Loose on purpose;
    # the reranker does the real filtering.
    min_similarity: float = 0.25

    # The real "do we have an answer?" gate.
    #
    # NOTE: sentence-transformers' CrossEncoder applies a Sigmoid activation by
    # default (confirmed: reranker._model().activation_fn == Sigmoid()), so
    # rerank() returns scores in [0, 1], NOT raw bge-reranker-v2-m3 logits.
    # Do not use the ">2 clearly relevant" logit heuristic sometimes quoted for
    # this model -- it does not apply to sigmoid-activated scores.
    #
    # Smoke-tested on this corpus: a clearly answerable question scored
    # 0.94-0.99 across its top-5 sources; a fully off-topic question topped out
    # at 0.0111. 0.5 sits well inside the gap between those two clusters.
    # TUNE THIS with scripts/tune_threshold.py once you have ~30 real questions
    # -- too high refuses valid questions, too low hallucinates from noise.
    min_rerank_score: float = 0.5

    # Hybrid retrieval (app/rag/lexical.py + retriever.py): a BM25 leg over the
    # same Chroma chunks, fused with the dense leg by Reciprocal Rank Fusion.
    # Off by default -- the dense-only path is unchanged until this is enabled.
    hybrid_enabled: bool = False
    bm25_fetch_k: int = 25      # BM25 candidates per query
    rrf_k: int = 60             # RRF damping: score = sum(1 / (rrf_k + rank))
    # Character n-gram sizes for the BM25 tokenizer. Bengali is suffix-heavy
    # (নামাজ / নামাজে / নামাজের), so whole-word matching alone misses inflections.
    ngram_min: int = 3
    ngram_max: int = 5
    # After reranking, keep at most this many chunks per page/article so one
    # long page cannot fill every slot (applies in hybrid mode only).
    max_chunks_per_page: int = 2

    tarun_db_path: str = "data/Tarun_Associate.db"
    nobin_db_path: str = "data/Nobin_Associate.db"

    # --- Langfuse tracing / observability (app/core/tracing.py) -----------
    # OPT-IN. With both keys empty, app/core/tracing.py is a complete no-op:
    # the langfuse SDK is never imported, get_client() returns the plain
    # OpenAI client, and /chat behaves exactly as a build without tracing --
    # no added latency, no background thread. Set both keys (self-hosted or
    # cloud) to turn tracing on. See the README section "LLM tracing with
    # Langfuse (self-hosted)".
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "http://localhost:3000"
    # Master switch: even with keys set, LANGFUSE_ENABLED=false keeps tracing
    # off. Tracing is active only when this is true AND both keys are set.
    langfuse_enabled: bool = True
    # Local latency table + logs/timing.jsonl. Only does anything when the
    # git-ignored app/core/timing_local.py exists (see app/core/timing.py).
    enable_timing: bool = False
    # Privacy lever. A trace otherwise stores the raw user query, the
    # retrieved book excerpts and the final answer (all on YOUR infra when
    # self-hosted). Set false to keep the trace/span structure, token usage,
    # cost, latency, rerank scores and the gate decision while redacting
    # every prompt / completion / query / excerpt text field.
    langfuse_capture_io: bool = True
    # Optional build marker attached to every trace (e.g. a git short sha).
    langfuse_release: str = ""
    # Separates real production traffic from local/dev/staging runs in the
    # Langfuse UI (its default views filter to "production") -- set to
    # "production" via the deployed box's .env, e.g. shibirgpt.service.
    # Applies client-wide (every trace this process sends), not per-trace.
    langfuse_environment: str = "development"

    # Browser same-origin policy blocks the front-end (a different origin --
    # the Vite dev server on :5173, or the deployed site) from calling this
    # API unless the response carries an Access-Control-Allow-Origin header
    # for that exact origin. app/main.py feeds this list to CORSMiddleware.
    # An explicit allow-list, not "*", because "*" is incompatible with
    # credentialed requests and we lose nothing by naming origins. Add
    # production/staging origins here (or via CORS_ALLOW_ORIGINS in .env --
    # pydantic-settings parses it as JSON, e.g. '["https://shibir.example"]').
    cors_allow_origins: list[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        # Production front-end (askshibir-back.potropollob.com is this API's
        # own domain; this is the site calling it).
        "https://shibirgpt.potropollob.com",
    ]

    host: str = "0.0.0.0"
    port: int = 9200

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()