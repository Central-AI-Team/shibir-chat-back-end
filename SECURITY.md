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

- Chat, conversation and memory endpoints require a private opaque `X-Chat-Identity`
  credential, whose SHA-256 hash is stored in PostgreSQL. Every operation checks ownership;
  caller-supplied `user_id` values do not grant access. Treat the browser-held credential
  as a bearer secret and use HTTPS outside local development.
- Account authentication, credential recovery and rate limiting are not implemented.
  `POST /identity` issues guest credentials. Browser storage loss means guest access is lost;
  the UI's existing account/demo-login token is not verified by this backend.
- Full transcripts and derived memories now persist in PostgreSQL. Protect and back up
  that database as user data; forgetting a memory excludes its source chat from cross-chat
  recall, while deleting a conversation removes its messages and sourced memories.
- CORS (`CORS_ALLOW_ORIGINS`) restricts browser origins but does not authenticate clients.
- The GPU service is protected by a single shared API key (`X-API-Key`).
