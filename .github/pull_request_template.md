## What

<!-- What does this change do? One or two sentences. -->

## Why

<!-- The bug, feature or task this addresses. Link the tracksheet row or issue. -->

## How it was tested

<!-- Commands run and their results. For retrieval/prompt/model changes, paste the eval numbers
     before and after (see docs/evaluation.md). -->

## Checklist

- [ ] `uv run pytest` passes
- [ ] No secrets, `.env`, database dumps or generated data committed
- [ ] Docs updated if behaviour, configuration or the API changed (README.md, CLAUDE.md, `.env.example`, `docs/`)
- [ ] `/chat` response shape and SSE events unchanged, or the front-end was told in advance
- [ ] Retrieval/prompt/model change: evaluation run and results included above
- [ ] Chunking/embedding change: reindex steps described
