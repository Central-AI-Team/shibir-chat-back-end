# Contributing

This guide describes how the team works on `shibir-chat-back-end`: setting up, making changes,
testing them and getting them merged. The current contributors are Safaet, Naimur and Farabi;
roles are not fixed, and new members should start here.

Please also read the [Code of Conduct](CODE_OF_CONDUCT.md).

## Getting set up

Follow the [Quick start](README.md#quick-start) in the README. In short:

```bash
git clone <repository-url>
cd shibir-chat-back-end
uv sync --locked
cp .env.example .env        # set OPENAI_API_KEY and DATABASE_URL at minimum
uv run pytest               # should pass before you change anything
uvx pre-commit install      # runs ruff, gitleaks and file checks on every commit
```

Before starting work, confirm that:

- [ ] `GET /health` responds;
- [ ] the Chroma collection is populated (run `uv run python -m app.rag.ingest` if it is empty);
- [ ] you have read the "Prompt contract" and "Known gaps" sections of [CLAUDE.md](CLAUDE.md).

On a machine with 8 GB of RAM or less, set `GPU_SERVICE_URL` in `.env` rather than loading the
models locally.

## Branches

| Branch | Purpose |
|---|---|
| `main` | Stable, deployable code. No direct pushes; changes arrive from `dev` by pull request. |
| `dev` | Integration branch. All feature work merges here first. |
| `feature/<short-description>` | One branch per change, for example `feature/reranker-threshold-fix`. Use `fix/...` for bug fixes if you prefer. |

Every merge to `main` deploys automatically to the production VPS, which health-checks the new
build and rolls back if it fails. Treat merging to `main` as releasing. Setup and details:
[docs/auto-deploy.md](docs/auto-deploy.md).

```bash
git checkout dev
git pull origin dev
git checkout -b feature/reranker-threshold-fix
```

## Commit messages

Use [Conventional Commits](https://www.conventionalcommits.org/) prefixes and the imperative mood:

```
feat: add note generation for whole books
fix: re-apply the GPU cold-start timeout after idle
docs: document the GPU service configuration
refactor: route all LLM calls through complete()
test: cover retry behaviour of gpu_client
chore: bump langfuse to 4.15.4
```

Allowed prefixes: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`, `perf`. Keep the subject
under about 72 characters, and use the body to explain *why* when it isn't obvious.

## Coding guidelines

- **Match the surrounding code.** Ruff enforces only bug-class rules (syntax errors, undefined
  names, unused imports; see `ruff.toml`); no formatter or style rules are enforced. If you use a
  formatter locally, don't reformat code you aren't otherwise changing.
- **Follow PEP 8.** Use `snake_case` for functions and variables and `PascalCase` for classes.
  Order imports as standard library, then third-party, then local.
- **Add type hints** to new functions.
- **Keep the three layers:** `api` → `services` → `rag`/`db`. Routes stay thin.
- **Configuration** goes through `app.core.config.settings`. Never call `os.getenv()` or
  `load_dotenv()` in app code. Add any new setting to `Settings`, to `.env.example` and to the
  configuration table in CLAUDE.md.
- **LLM calls** go through `app.core.llm.complete(task, ...)`. Never construct an OpenAI client
  directly, and never pass `temperature` or `max_tokens` to reasoning models; use `token_budget=`.
- **Normalize text** with `chunker.normalize()` at both ingest and query time.
- **Comments explain why**, especially for thresholds, limits and workarounds. Many existing
  comments record an incident or a measurement; keep them accurate when you change the code.
- Take extra care with modules that every request passes through: `app/core/llm.py`,
  `app/rag/retriever.py`, `app/rag/generator.py` and `app/api/router.py`.

## Testing

```bash
uv run pytest                                                     # required for every PR
RUN_MODEL_TESTS=1 uv run pytest tests/test_bengali_embedding.py   # after embedding/model changes
```

- Tests must not call real external services. Mock at the service boundary, or use
  `httpx.MockTransport` for the GPU client, as `tests/test_gpu_client.py` does.
- If a test replaces entries in `sys.modules` or module globals, restore them on teardown (use
  `monkeypatch`). A leaked mock once broke an unrelated tracing test.
- Add or update tests with every behaviour change and every bug fix.

For quality changes that tests cannot catch, use the evaluation scripts described in
[docs/evaluation.md](docs/evaluation.md).

## Pull requests

1. Open the pull request against **`dev`** (only release PRs target `main`).
2. Fill in the pull request template: what changed, why, and how you tested it.
3. CI must pass. It runs three jobs on every pull request into `dev` or `main`, and all three
   must succeed: **lint** (ruff), **secret-scan** (gitleaks over the branch history) and
   **test** (pytest against Postgres).
4. At least one approval is required; Safaet approves merges.
5. Delete the feature branch after merging.

### Review checklist

**Every pull request**

- [ ] `uv run pytest` passes.
- [ ] No secrets, `.env` files, database dumps or generated data (`chroma_db/`, `eval_reports/`)
      are committed.
- [ ] `pyproject.toml` and `uv.lock` are updated together (`uv add <package>`) if dependencies changed.
- [ ] README.md, CLAUDE.md, `.env.example` or `docs/` are updated if behaviour, configuration or
      the API changed.

**Retrieval, embedding, reranking, prompt or model changes**

- [ ] `eval_retrieval` and/or `eval_responses` show no regression (include the numbers in the PR).
- [ ] `eval_intent_routing` passes if the intent classifier or its prompt changed.
- [ ] Banglish queries tested, not only Bengali script.
- [ ] If chunking or the embedding model changed, the PR explains the reindex required.
- [ ] If the reranker or chunking changed, `MIN_RERANK_SCORE` was re-checked with `tune_threshold`.

**API changes**

- [ ] The `/chat` response shape (`mode`, `answer`, `sources`, `session_id`,
      `response_time_ms`) and the SSE event names stay backward-compatible; the front-end
      (`shibir-chat-front-end`) depends on them.
- [ ] The front-end developers were told in advance about any breaking change.

## Secrets and environment

- Never commit `.env` or real keys. Add a placeholder and a comment to `.env.example` for any new
  variable, and tell the team where to obtain the real value.
- `GROQ_API_KEY` is only needed for the evaluation scripts; it is not read by the running service.
- Report a leaked credential immediately as described in [SECURITY.md](SECURITY.md).

## Tracking work

- The team tracksheet (Google Sheets) is the primary task tracker.
- GitHub Issues may be used for code-level bugs and features. Don't track the same item in both
  places; link to one from the other if needed.
