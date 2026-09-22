# Contributing Guide — shibir-chat-back-end

This document is for our internal team (Safaet, Naimur, Farabi) — how we contribute and review code for this project. Tasks aren't divided by fixed roles yet; more members may join later.

## Setup / Getting Started

```bash
git clone <repo-url>
cd shibir-chat-back-end

# install dependencies (uv manages the virtual environment automatically)
uv sync

# environment variables
cp .env.example .env
# set OPENAI_API_KEY and other required keys in .env

# database
# set the Postgres connection string in .env; run the migration/ingest script on first setup

# run the server
uv run uvicorn app.main:app --reload
```

Before starting work, make sure:
- [ ] the `/health` endpoint responds correctly
- [ ] the ChromaDB collection is loaded (run the ingest script if it's empty)

## Code Style

- **Formatter:** run Black before committing (`uv run black .`)
- **Import order:** standard lib → third-party → local (isort can automate this)
- **Type hints:** add type hints to new functions wherever reasonable
- **Naming:** snake_case for functions/variables, PascalCase for classes
- Break up large functions/modules when needed — be extra careful with changes to single-choke-point files like `app/core/llm.py`, since all LLM calls route through it

## Branch Strategy

- `main` — always stable, working code. No direct pushes.
- `dev` — integration branch; everyone's work merges here first.
- Feature branch naming: `feature/<short-description>`
  - Example: `feature/reranker-threshold-fix`, `feature/note-generation-v2`

```bash
git checkout dev
git pull origin dev
git checkout -b feature/reranker-threshold-fix
```

## Commit Message Convention

```
fix: reranker threshold tuning for Banglish queries
feat: add note-generation map-reduce endpoint
docs: update README (Gemini→OpenAI migration)
refactor: consolidate LLM calls through get_client()
chore: remove unused GROQ_API_KEY from .env
```

Prefixes: `feat`, `fix`, `docs`, `refactor`, `chore`, `test`

## Pull Request Process

1. Open PRs against `dev` (not `main`)
2. In the PR description, include:
   - What you changed
   - Why (which bug/feature)
   - How you tested it
3. At least one approval (Safaet) required before merging
4. Delete the feature branch after merging

## PR Review Checklist

**For retrieval / embedding / reranker / LLM changes**
- [ ] `scripts/eval_responses.py` run to check for quality regressions
- [ ] `eval_intent_routing.py` passes
- [ ] Banglish queries tested (query_rewriter path)
- [ ] if chunking/embedding changed, a note is included about re-indexing the existing ChromaDB collection

**For API / endpoint changes**
- [ ] the `/chat` endpoint's response shape (mode, session_id, sources) stays backward-compatible — the frontend repo (shibir-chat-front-end) depends on this shape
- [ ] the frontend side is notified ahead of any breaking change

**For every PR**
- [ ] `.env` not committed, no secrets leaked
- [ ] `pyproject.toml` / `uv.lock` updated if a new dependency was added
- [ ] README/PROJECT.md updated if relevant (currently stale — still describes the old Gemini setup even though it's migrated to OpenAI)

## Environment & Secrets

- `LLM_PROVIDER=openai` stays in `.env` as the fallback
- Unused keys (e.g. `GROQ_API_KEY`) should not be kept
- If a new secret is needed, let the team know and add a placeholder to `.env.example`

## Task Tracking

- Primary tracking is in the Google Sheets tracksheet
- Code-level bugs/features can also live in GitHub Issues, but avoid tracking the same thing in two places

## Before You Push

```bash
# run the eval harness to check nothing broke
uv run python scripts/eval_responses.py
uv run python scripts/eval_intent_routing.py
```
