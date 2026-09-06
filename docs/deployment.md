# Scalable Deployment Guide

This document describes how to take Shibir Chat Back-End from the current single-node
`docker-compose` setup (see repo root `Dockerfile` / `docker-compose.yml`) to a topology that
scales horizontally. It's written from this codebase's actual architecture and actual
constraints — not a generic "how Kubernetes works" tutorial. For request/response shapes and
the retrieval pipeline itself, see [`PROJECT.md`](../PROJECT.md); this file is about *running
more than one instance of it reliably*.

## Development-stage requirements (before any of the scaling below applies)

Everything past this section is about running *more than one instance* in production. Before
that's even a question, a single dev/staging box needs enough headroom to hold the embedder +
reranker + Postgres + Chroma in one process without swapping — this is a lower bar than the
production tiers below, but it's the one most likely to be silently under-provisioned, because
it's easy to assume "it's just one box" means "it doesn't need much."

| Resource | Recommended minimum | Why |
|---|---|---|
| RAM | **16 GB** | `bge-m3` (~2.2 GB) + `bge-reranker-v2-m3` (~2 GB) resident together already cost ~4.5 GB before Postgres, Chroma, an IDE, or a browser get anything — on an 8 GB box this reliably swap-thrashes or gets OOM-killed under any real retrieval load, not just under multiple concurrent requests. |
| CPU | 4+ cores | The embedder/reranker are CPU-bound but not core-hungry at dev-traffic levels; RAM headroom, not core count, is what determines whether a single retrieval call takes seconds or minutes. |
| Disk | **40–50 GB free**, dedicated to this project | `venv/` alone is 7.9 GB measured; + ~2.8 GB HF model cache; + `chroma_db/` (386 MB at 12,304 chunks currently); + a second isolated `venv-ragas/` (~1.3 GB) if running the RAGAS harness (`scripts/eval_ragas.py`) — see that script's module docstring for why it needs its own venv. Running low here doesn't fail gracefully — it fails mid-`pip install` or mid-eval-run. |
| GPU | Not required | CPU inference is fine at the current corpus size, *given* the RAM above isn't the constraint. Revisit only if the corpus grows an order of magnitude or eval-script iteration speed becomes the bottleneck. |

None of the production tiers (Redis, managed Postgres, a separate Chroma server, a split
inference service) are needed at this stage — one box running Postgres + Chroma + the app in a
single process is the right setup until you're actually adding replicas.

## Before anything else: read the statefulness audit

The single most important fact for scaling this service is that **the FastAPI process is not
fully stateless today**. Three things live inside the process that a naive "just run more
replicas behind a load balancer" approach will silently break:

| Component | Current implementation | Scales horizontally? |
|---|---|---|
| Chat/roleplay session state | `app/services/session_store.py` — a plain module-level `dict` | **No.** Its own docstring already says so: a session started on one worker/replica is invisible to another. Multi-worker or multi-replica today means a user's roleplay persona or conversation history randomly "disappears" mid-conversation depending on which process handles the next request. |
| Embedding + reranker models | `app/rag/embedder.py` / `app/rag/reranker.py` — `sentence-transformers` models loaded in-process via `@lru_cache`, CPU-bound | Scales, but **expensively** — every replica loads its own full copy of `BAAI/bge-m3` (~2.2 GB) and `BAAI/bge-reranker-v2-m3` (~2 GB+), so N replicas cost N× that RAM before serving a single request. |
| Vector store (Chroma) | `app/rag/chroma_client.py` — `chromadb.PersistentClient` writing to a local `chroma_db/` directory | **Not safely**, as-is. Chroma's persistent-client mode is not designed for multiple processes concurrently opening the same on-disk index (this has caused real corruption/contention in this project's own ingest workflow). A shared network volume mounted read-write from N replicas is not a fix — it's the same problem with extra latency. |
| Postgres (content, source of truth) | Standard SQLAlchemy engine, `app/db/session.py` | **Yes**, this is a normal RDBMS — connection pooling and read replicas work the usual way. |
| LLM calls (OpenAI, optionally Groq) | `app/core/llm.py`, external API calls, no local state | **Yes**, but each provider has its own rate limits that don't scale just because you added replicas — see [LLM provider layer](#tier-5-llm-provider-layer) below. |
| Corpus ingest (`python -m app.rag.ingest`) | Offline batch script, not in the request path | Not a live-traffic concern, but it competes for the same CPU/RAM as the embedder/reranker if run on the same host — schedule it separately (see [Ingest as its own job](#ingest-as-its-own-job)). |

Everything below is organized around fixing these three (session store, model duplication,
vector store) without a rewrite — each is an incremental, independently-shippable change.

## Recommended topology

```
                         ┌─────────────┐
                         │ Load balancer│
                         └──────┬──────┘
                                │
                 ┌──────────────┼──────────────┐
                 ▼              ▼              ▼
           ┌──────────┐   ┌──────────┐   ┌──────────┐
           │ API pod 1 │   │ API pod 2 │   │ API pod N │   <- stateless FastAPI,
           │(uvicorn)  │   │(uvicorn)  │   │(uvicorn)  │      horizontally autoscaled
           └─────┬─────┘   └─────┬─────┘   └─────┬─────┘
                 │               │               │
        ┌────────┼───────────────┼───────────────┼────────┐
        ▼        ▼               ▼               ▼        ▼
   ┌─────────┐ ┌───────────────────────┐ ┌───────────────┐ ┌──────────────┐
   │  Redis   │ │   Vector store tier    │ │   Postgres     │ │ LLM providers │
   │(sessions)│ │ (Chroma server OR a    │ │ (managed,      │ │ (OpenAI, Groq)│
   │          │ │  managed vector DB)    │ │  pooled)       │ │  external     │
   └─────────┘ └───────────────────────┘ └───────────────┘ └──────────────┘

   ┌──────────────────────────────────────────────────────┐
   │  Ingest job (python -m app.rag.ingest) — separate,     │
   │  scheduled/on-demand, writes to the vector store tier  │
   │  above. Never runs on an API pod.                      │
   └──────────────────────────────────────────────────────┘
```

The API layer is the only thing you autoscale reactively (on CPU/latency/queue depth).
Everything else in the diagram is a shared, independently-sized tier.

## Server specs per tier

Concrete starting points, not a ceiling — treat these as the floor to provision above, with
real headroom, not the target to hit exactly. Where a number below is measured rather than
estimated, it's called out as such.

| Tier | vCPU | RAM | Disk | Example instance class | Notes |
|---|---|---|---|---|---|
| **API replica** (models in-process, Tier 1's default) | 4 | 8 GB | 15 GB | AWS `m6i.xlarge` / GCP `n2-standard-4` | RAM floor is ~6 GB just for `bge-m3` + `bge-reranker-v2-m3` + PyTorch/Chroma-client overhead (see Tier 1) — 8 GB leaves working headroom instead of running at the edge. Disk: this project's own `venv/` with these deps is **7.9 GB measured** on Linux, plus ~2.8 GB for the two cached HF models (per `README.md`) — 15 GB comfortably covers venv + models + container image + logs. CPU-bound on rerank, not I/O-bound — don't undersize vCPU to save cost here. |
| **API replica** (models externalized to Tier 5's inference service) | 1–2 | 1–2 GB | 2 GB | AWS `t3.small`/`t3.medium` | Once the embedder/reranker live in a separate service, an API pod is a thin FastAPI/httpx layer — this is the payoff of doing that split. |
| **Embedding + reranker inference service** (if split out per Tier 5) | 4–8 (CPU) or 1× small GPU | 8–12 GB | 8 GB | CPU: AWS `c6i.2xlarge`; GPU: AWS `g4dn.xlarge` (T4) | A GPU meaningfully speeds up both bi-encoder embedding and cross-encoder reranking under sustained load; CPU-only is fine at the traffic levels a single-digit number of API replicas would generate. Only worth deploying once you've actually made the Tier 5 split — don't provision this ahead of needing it. |
| **Vector store — Chroma server** (Tier 3, Option A) | 2 | 2–4 GB | 10 GB, scale with corpus | AWS `t3.medium`/`m6i.large` | **Measured on this project's current corpus**: 12,304 chunks (~900 chars each, `bge-m3`'s 1024-dim vectors) occupy **386 MB** on disk — roughly 31 KB/chunk including the HNSW index and stored text. At this scale the vector-store tier is genuinely cheap; re-budget disk as `chunk_count × ~35 KB` (with headroom) if the corpus grows by an order of magnitude, and re-check after any chunk-size change in `app/rag/chunker.py` (chunk size directly changes chunk count for the same corpus). |
| **Vector store — pgvector** (Tier 3, Option B, same Postgres) | — | — | add ~35 KB/chunk to the Postgres disk estimate below | — | No separate tier — sizing folds into the Postgres row. |
| **Postgres** (managed) | 2 | 4 GB | 20 GB+, corpus-dependent | AWS RDS `db.t3.medium` / GCP Cloud SQL equivalent | Book/chapter/page content itself is text, not the bottleneck here — size mainly for connection count (pooled via PgBouncer, see Tier 4) and, if chosen, pgvector's addition above. |
| **Redis** (session store) | 1 | 512 MB–1 GB | 1 GB | AWS ElastiCache `cache.t3.micro` | Session payloads are small (a persona string + up to `MAX_HISTORY=20` chat turns per session) — this tier is sized for connection count and availability, not data volume. Set an eviction policy (`allkeys-lru` or similar) and rely on the TTL in the Tier 2 sketch rather than persistence. |
| **Ingest job** (batch, `python -m app.rag.ingest`) | 2–4 | 8 GB | 15 GB | Same class as an API replica, or a spot/preemptible instance | Needs the embedder resident (~2.2 GB) for the run's duration only — fine on a cheaper/spot instance since it's not latency-sensitive and can be retried. GPU optional (`requirements.txt` already carries CUDA extras behind platform markers) — only worth it for a large corpus's first full ingest, not incremental updates. |

The two numbers worth re-deriving for your own deployment rather than trusting verbatim are
the **API replica RAM floor** (re-measure if you change `embedding_model_name`/
`reranker_model_name` in `config.py` to different-sized models) and the **vector-store disk
estimate** (re-measure per-chunk size after any `chunker.py` change — chunk size/overlap
directly changes how many chunks the same corpus produces).

## Tier 1: the API layer (stateless FastAPI, horizontally scaled)

This is the easy part — `app/main.py` / `app/api/router.py` hold no state of their own once
the two items below are externalized. Run N replicas behind any load balancer (a managed one,
or `nginx`/`traefik` if self-hosting), with:

- **Autoscale on CPU**, not request count. The two CPU-heavy operations per request are the
  cross-encoder rerank (`app/rag/reranker.py`, up to `fetch_k=25` candidates scored per `/chat`
  call) and, for Banglish/English queries, the bi-encoder embed of every
  `query_rewriter.expand_query()` variant. Both are synchronous CPU work wrapped in
  `run_in_threadpool` (see `router.py`) — under load this saturates a pod's CPU well before
  memory does.
- **Size each replica for ~6 GB minimum RAM** while the embedder/reranker still run in-process
  (see [Tier 5](#tier-5-embeddingreranker-inference) for the alternative). This is not a
  theoretical number — `bge-m3` (~2.2 GB) + `bge-reranker-v2-m3` (~2 GB) + PyTorch/Chroma
  client overhead has been directly observed to cause OOM kills on boxes with less headroom
  than that, even for a single instance.
- **Readiness probe on `GET /health`**, but don't rely on it alone — `/health` returns
  `{"status": "ok"}` unconditionally and does not check that the embedder/reranker have
  finished loading (they load lazily on first use via `@lru_cache`). Either warm them at
  startup (a FastAPI `startup` event calling `embed_texts([""])`/`rerank("", [""], 1)` once)
  or accept a slow first request per replica after a rollout.
- **Set `--workers 1` per uvicorn process** if you keep the models in-process, and scale via
  more replicas/pods instead of more uvicorn workers per pod. Multiple uvicorn workers in one
  process group still each load their own copy of the models — you get the RAM cost of more
  replicas without the deployment-level flexibility (independent scheduling, rolling restarts)
  of actual pods/containers.

## Tier 2: shared session store (Redis)

This is the first thing to fix, and the lowest-risk change on this list — `session_store.py`
already exposes a narrow, three-function interface
(`get_or_create_session` / `update_session` / `append_history`), so swapping its backing store
doesn't touch any caller in `roleplay_service.py` or `router.py`.

Replace the module-level `_sessions: dict` with Redis, keeping the exact same function
signatures:

```python
# app/services/session_store.py (Redis-backed sketch)
import json
import uuid
import redis

MAX_HISTORY = 20
SESSION_TTL_SECONDS = 60 * 60 * 6  # expire idle sessions; a roleplay chat isn't forever

_r = redis.Redis.from_url(settings.redis_url, decode_responses=True)

def _key(session_id: str) -> str:
    return f"session:{session_id}"

def get_or_create_session(session_id: str | None) -> tuple[str, dict]:
    if session_id is None:
        session_id = str(uuid.uuid4())
    raw = _r.get(_key(session_id))
    session = json.loads(raw) if raw else {"mode": None, "persona": None, "history": []}
    return session_id, session

def update_session(session_id: str, **fields) -> dict:
    _, session = get_or_create_session(session_id)
    history = fields.pop("history", None)
    session.update(fields)
    if history is not None:
        session["history"] = history[-MAX_HISTORY:]
    _r.set(_key(session_id), json.dumps(session), ex=SESSION_TTL_SECONDS)
    return session
```

Add `redis_url: str = "redis://localhost:6379/0"` to `app/core/config.py`'s `Settings`
(same pattern as every other setting there), and a `redis` service to `docker-compose.yml`
(a managed Redis/ElastiCache/Memorystore instance in production — don't self-host Redis
persistence for session data that's expendable). This alone unblocks running more than one
API replica without roleplay sessions randomly resetting.

## Tier 3: vector store — pick one, don't run both

Chroma's local `PersistentClient` (what `chroma_client.py` uses today) is fine for a single
API instance but is the real horizontal-scaling blocker for the vector store. Two real paths
forward, not a checklist to do both:

**Option A — Chroma in client/server mode.** Run Chroma as its own service
(`chromadb.HttpClient` instead of `PersistentClient` in `chroma_client.py` — the rest of
`retriever.py`/`ingest.py` is unaffected, since they only ever call `collection.query()` /
`collection.upsert()` through the object `get_collection()` returns). One Chroma server pod,
one persistent volume behind it, every API replica connects over the network instead of
opening the file directly. Lowest-effort migration — same library, same data model, just a
network hop instead of local disk. Best if your corpus and query volume are moderate and you
don't need Chroma itself to scale past one node.

**Option B — migrate to a managed/clustered vector store** (pgvector on the *same* Postgres
you already run, or a dedicated service like Qdrant/Weaviate/Milvus). More upfront work
(`retriever.py`'s `collection.query(...)` call and `ingest.py`'s upsert path both need
rewriting against the new client), but this is the path that actually scales the vector store
itself horizontally, not just the API layer in front of it. `pgvector` is worth calling out
specifically here: this project already runs Postgres for content, so pgvector means one fewer
moving piece in the topology (no separate vector-store tier at all) — a real option if the
corpus size stays in the low-single-digit-millions-of-chunks range.

Whichever you pick, the corpus stays the single source of truth in Postgres — you're only
ever changing where the *derived* embeddings live, never the underlying book/chapter/page
content.

### Ingest as its own job

`python -m app.rag.ingest` is CPU-bound and reads Postgres / writes the vector store — it is
not part of the request-serving path and should never run inside an API pod. Run it as:

- a Kubernetes `Job`/`CronJob` (or the equivalent scheduled task in your platform) triggered
  after content changes, with its own resource request (it needs the embedder in memory too,
  same ~2.2 GB, but only for the duration of the run), or
- a manually-triggered CI step / one-off script run from an operator's machine against the
  same `DATABASE_URL` and vector-store endpoint the API tier uses.

Either way, size it separately from API replicas — it's a periodic batch cost, not a
steady-state one.

## Tier 4: Postgres

Nothing unusual here relative to any FastAPI+SQLAlchemy service: use a managed instance
(RDS / Cloud SQL / equivalent) rather than self-hosting, put a connection pooler (PgBouncer,
or the managed service's built-in pooling) in front of it once you have more than a handful of
API replicas each holding their own SQLAlchemy pool, and add a read replica if `/chat`'s NOTE
path (`note_service.py`'s per-chapter map-reduce, which issues its own Postgres reads) becomes
a meaningfully large fraction of traffic — it's a read-heavy, latency-tolerant path and a good
first candidate to route to a replica.

## Tier 5: LLM provider layer

`app/core/llm.py`'s per-task model routing (`get_model(task)` / `complete(task, ...)`,
configured via `settings.model_by_task`) is directly useful for scaling, not just cost: routing
the two tasks that run on *every* request before the user sees anything (`intent`, `rewrite`)
to a smaller/faster model reduces p50/p99 latency under load, while the user-facing generation
tasks (`qa`, `note`, `suggest`, `roleplay`) keep the stronger model. Validate any such routing
change with `scripts/eval_task_routing.py` before relying on it in production — a cheaper
model that raises the JSON-parse-failure rate on `rewrite` costs you more in retries/fallbacks
than the latency it saves (this is exactly what that script is for).

Two things that matter operationally, not just architecturally:

- **Per-provider rate limits are real and provider-specific**, and scaling replicas multiplies
  how fast you hit them, not how much headroom you have. (Concretely: a Groq-hosted model on
  this project's own API key hit a hard 1000-output-tokens/minute limit well before any
  meaningful concurrency — this is not a hypothetical.) If you route any task to a
  secondary/cheap provider for cost or latency, monitor that provider's rate-limit error rate
  separately from the primary one, and have a fallback (e.g., `model_by_task` defaulting back
  to `settings.openai_model` on repeated failures) rather than letting `/chat` 503 whenever the
  cheap tier is saturated.
- `router.py` already converts an `openai.APIError` into a clean `503` rather than a bare
  `500` — extend that same treatment to whatever exception a second provider's client raises,
  so a Groq outage/rate-limit doesn't look like an unhandled server bug to callers.

## Observability — what actually matters for this app

Generic "CPU/memory/request rate" dashboards miss the failure modes specific to this
architecture. At minimum, instrument:

- **Latency broken down by `/chat` mode** (`NOTE` / `ROLEPLAY` / `SUGGESTION` / `QA` — the
  `mode` field is already on every `ChatResponse`). `NOTE` is a multi-call map-reduce over an
  entire chapter and will have a wildly different latency profile than `QA`'s single
  retrieve+generate — averaging them together hides regressions in either one.
- **Rerank and embed latency separately from LLM latency.** The reranker (`app/rag/
  reranker.py`) scoring `fetch_k=25` candidates is pure local CPU work; if it slows down, that's
  a CPU/replica-sizing problem, not an upstream API problem — don't let one dashboard number
  ("total request latency") conflate the two when they need completely different fixes.
- **Per-(task, model) LLM cost and latency**, keyed off `settings.model_by_task` — the same
  breakdown `scripts/eval_generation_ab.py` and `scripts/eval_task_routing.py` already compute
  offline is exactly what you want tracked online, so a model swap's effect on production cost
  is visible immediately, not discovered at the next invoice.
- **Vector-store query latency and index size**, independent of API latency — whichever option
  you picked in Tier 3, its own growth (more chunks ingested over time) degrades query latency
  gradually, not as a discrete incident, and won't show up as an API-side alert until it's
  already been slow for a while.
- **Session-store hit rate and TTL evictions** (Redis) — a sudden drop in "existing session
  found" vs "new session created" after a deploy usually means a config/connectivity issue to
  Redis, not organic new-user growth.
- **RAM headroom on API replicas**, watched explicitly rather than assumed — the ~6 GB/replica
  figure in Tier 1 is a floor for this specific model combination, and a scheduler that packs
  pods tightly against that number with no margin will intermittently OOM-kill exactly the way
  a single under-provisioned box does.

## Migration path (incremental, not a rewrite)

Do these in order — each is independently shippable and leaves the system in a working state:

1. **Externalize the session store to Redis** (Tier 2). Zero API contract changes, unblocks
   running more than one replica at all.
2. **Put N API replicas behind a load balancer**, still pointed at a single Chroma
   `PersistentClient` instance/volume (one writer, i.e. don't run ingest concurrently with
   this, and treat this step as a stopgap, not the end state — it works because Chroma reads
   from an already-built index are safer than concurrent read+write, not because it's a
   supported multi-writer setup).
3. **Move the vector store to its own tier** (Tier 3, Option A or B) — removes the stopgap
   from step 2 and is the point at which ingest can safely run without any coordination with
   live API traffic.
4. **Move Postgres to a managed instance with pooling** (Tier 4) if not already there.
5. **(Optional, cost-driven) Extract embedding+rerank into a dedicated inference service** —
   only worth doing once API-replica count is high enough that N-times-model-RAM is a real
   cost line item; until then, keeping models in-process (Tier 1) is simpler and has one fewer
   network hop per request.

Steps 1–3 are what actually fix horizontal scaling; 4–5 are efficiency/cost refinements on top
of a topology that already scales.
