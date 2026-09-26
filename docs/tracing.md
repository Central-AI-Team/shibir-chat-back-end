# Tracing with Langfuse

The back-end can send one [Langfuse](https://langfuse.com) trace per `/chat` or `/chat/stream`
request. Each trace groups every pipeline stage and every LLM call made for that request. Use it
to:

- debug individual misses, for example "the right page was retrieved but reranked to 0.42, below
  the 0.5 gate";
- track cost, latency and token usage per task and per model over time;
- attach quality scores to real user queries.

Implementation: `app/core/tracing.py` (Langfuse Python SDK v4, OpenTelemetry-based).

## Safety guarantees

- **Off by default.** With either key unset, the Langfuse SDK is never imported, the plain OpenAI
  client is used, and requests behave exactly as they would without this feature.
- **Cannot break requests.** Every tracing helper catches and logs its own errors. Events are
  delivered from a background thread, so nothing is flushed on the request path.

## Setup

### 1. Get a Langfuse instance

Use [Langfuse Cloud](https://cloud.langfuse.com) or self-host it:

```bash
git clone https://github.com/langfuse/langfuse
cd langfuse
docker compose up -d
```

Then open `http://localhost:3000`, create a project, and create an API key pair under
**Project Settings → API Keys**.

> The self-hosted stack (Langfuse, Postgres, ClickHouse, Redis, MinIO) needs substantially more
> than 8 GB of RAM alongside this app. On a small development machine, use Langfuse Cloud or a
> separate host.

### 2. Configure this app

Add to `.env`:

```
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=http://localhost:3000     # or https://cloud.langfuse.com
LANGFUSE_ENVIRONMENT=development        # use "production" on the deployed server
```

Restart the server. To disable tracing, remove the keys or set `LANGFUSE_ENABLED=false`.

> The variable is `LANGFUSE_HOST`, not `LANGFUSE_BASE_URL`. The latter belongs to the
> `langfuse-cli` tool. Setting the wrong one leaves the app pointed at `localhost:3000`, and
> because tracing never raises, the failure is silent.

| Variable | Default | Purpose |
|---|---|---|
| `LANGFUSE_ENABLED` | `true` | Master switch. Tracing is active only when this is true **and** both keys are set. |
| `LANGFUSE_CAPTURE_IO` | `true` | `false` redacts all text (queries, excerpts, prompts, answers) and keeps structure, scores, usage, cost and latency. |
| `LANGFUSE_RELEASE` | *(empty)* | Optional build marker, for example a git short SHA. |
| `LANGFUSE_ENVIRONMENT` | `development` | Separates production traffic from development runs in the Langfuse UI. |

## What a trace contains

| Observation | Contents |
|---|---|
| Root span (`chat` / `chat_stream`) | User query, session id, optional `user_id`, final answer, sources, `mode` (in metadata), total latency. `level=ERROR` if the request failed. |
| `classify-intent` (span) | The detected intent and how it was decided (`regex`, `llm` or `session`). |
| `expand-query` (span) | The query variants produced by the rewriter. |
| `retrieve-context` (retriever) | Every reranked candidate, kept and dropped, with a text preview, cosine similarity, rerank score and a `kept` flag. |
| `check-relevance-gate` (guardrail) | Grounded or refused, top rerank score, threshold. |
| `gpu:embed` / `gpu:rerank` (span) | GPU mode only: item count, latency, cold start, attempts, status. Never the texts. |
| `llm:{task}` (generation) | One per LLM call: model, messages, completion, token usage, cost, latency. `task` is one of `intent`, `rewrite`, `persona`, `qa`, `note`, `suggest`, `roleplay`. |

## Privacy and retention

Traces contain user queries and book excerpts unless `LANGFUSE_CAPTURE_IO=false`. API keys and
other secrets are never written to traces. Configure retention in the Langfuse project settings.

## Attaching scores

`tracing.score_trace(trace_id, name, value, comment=...)` wraps Langfuse's score API. Possible
uses:

- a thumbs-up/down button in the front-end that sends the trace id back;
- a batch job that scores logged queries;
- the offline evaluation scripts (`eval_responses`, `eval_ragas`) writing per-question scores to
  the matching traces.

Only the function is provided; no UI or job uses it yet.

## Implementation notes

For the reasoning behind the design, see the module docstring of `app/core/tracing.py`. In
particular, it explains why the streaming path passes `trace_id` and `parent_observation_id`
explicitly instead of relying on ambient OpenTelemetry context.
