# Security

## Reporting a vulnerability

Do **not** open a public GitHub issue for a security problem. Report it privately to the
maintainer (Safaet) through the team's direct channel, including:

- what the issue is and where it is (file, endpoint or configuration);
- how to reproduce it;
- the potential impact.

You will get an acknowledgement as soon as possible. Fixes for confirmed issues are prioritized
over feature work.

## Leaked credentials

If an API key, database password or other secret is committed, pasted or shared by mistake:

1. Tell the maintainer immediately.
2. **Rotate the credential** at its provider (OpenAI, Groq, Langfuse, the GPU service, Postgres).
   Removing it from the latest commit is not enough; it stays in git history and in any clones.
3. Rewriting history is only worth doing after the credential has been rotated.

## Handling secrets

- Secrets live only in `.env`, which is git-ignored and excluded from Docker images by
  `.dockerignore`. `.env.example` contains placeholders only.
- Never log API keys, request bodies containing user queries, or book text. `app/rag/gpu_client.py`
  and `app/core/tracing.py` are written to avoid this; keep it that way.
- Langfuse traces contain user queries and excerpts unless `LANGFUSE_CAPTURE_IO=false`. Treat a
  Langfuse project as containing user data.
- Do not commit database dumps or other data exports. They can contain personal data, and git
  history is permanent.

## Current security posture

Be aware of these limitations when deploying:

- The API has **no authentication or rate limiting**. It relies on network placement and on CORS
  (`CORS_ALLOW_ORIGINS`) for browser callers. CORS does not stop non-browser clients.
- Chat sessions are held in process memory and are listed by `GET /conversations` without any
  per-user separation. Anyone who can reach the API can read and delete every session in that
  process.
- The GPU service is protected by a single shared API key (`X-API-Key`).
