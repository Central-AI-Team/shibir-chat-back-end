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

- There are **no user accounts or logins**. Chat sessions are protected by **anonymous per-browser
  ownership**: the server issues a random token in the `shibir_owner` cookie (HttpOnly,
  SameSite=Lax, Secure unless `COOKIE_SECURE=false`, 30 days) and stores only its SHA-256 hash. A
  session is visible only to the owner that created it; an unknown session id and someone else's
  id get the same 404 on `/conversations`, and on `/chat` both start a new session under a fresh
  id, so neither reveals whether the id exists. Session ids are server-generated 256-bit random values. This is access
  control between anonymous browsers, not authentication: anyone can get a fresh identity, and
  whoever holds a browser's cookie is that owner.
- **Clearing cookies (or a different browser/device) loses access to earlier chats.** There is no
  recovery.
- Cross-site requests: CORS allows credentials only for the explicit `CORS_ALLOW_ORIGINS` list
  (`*` is rejected at startup), and `POST`/`DELETE` with an `Origin` outside that list get 403
  (CSRF), except same-origin requests (the API's own address, e.g. `/docs`). Requests without an
  `Origin` (curl, scripts) are allowed but only ever see their own anonymous sessions.
- **No rate limiting in Python yet**, and sessions have **no TTL**: only caps
  (`MAX_SESSIONS_PER_OWNER`=50, `MAX_SESSIONS_TOTAL`=10000, least-recently-updated evicted). A
  client that discards its cookie can create sessions until the global cap evicts other users'
  oldest sessions. Rate limiting is currently only at the proxy.
- **The app must not be exposed directly.** `docker-compose.yml` publishes it on `127.0.0.1:9200`
  only; a TLS reverse proxy (`deploy/nginx/` or `deploy/caddy/`) is the sole public entry point and
  adds HTTP→HTTPS redirect, a 64 KB body limit and per-IP rate limiting on `/chat` and
  `/chat/stream` (nginx; Caddy needs a plugin). The firewall allows only ports 22, 80 and 443 (see
  the README "Firewall" section).
- `/docs`, `/redoc` and `/openapi.json` are **off by default** (`ENABLE_DOCS=false`); the proxy
  also returns 404 for them. Enable them only for local development.
- Chat sessions are held in process memory: they are **lost on restart** and not shared between
  workers.
- Langfuse receives only a 12-character prefix of the owner hash as `user_id` and a hash of the
  session id; the client-supplied `user_id` field is ignored. The raw cookie, the full owner hash
  and the session id are never logged, traced or sent to an LLM.
- The GPU service is protected by a single shared API key (`X-API-Key`).
