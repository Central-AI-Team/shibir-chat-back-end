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

- **Accounts** (`app/auth/`, [docs/auth.md](docs/auth.md)): email + password, hashed with argon2id
  (library defaults). Login sessions are random 256-bit tokens in an HttpOnly, SameSite=Lax,
  `__Host-` cookie (Secure); only their SHA-256 is stored. Every login issues a fresh token,
  sessions expire after 7 days idle / 30 days absolute, logout and password change revoke them,
  and a user has at most 10. Login is throttled per email (5 / 15 min) and per IP (20 / h), and an
  account locks for 30 s to 15 min (never permanently) after 10 consecutive failures. Every
  failure kind gets the same 401 and, as far as possible, the same timing. Registration is open
  (3 per IP per hour, `AUTH_REGISTRATIONS_PER_DAY` overall) and has **no email verification,
  password reset or MFA**, so anyone can create accounts, which also weakens the guest limit.
  Security events go to the `app.auth.audit` log (no passwords, tokens, hashes, full emails or
  full IPs) and never to Langfuse.
- **Guests** (no login) keep the anonymous per-browser ownership: the server issues a random token
  in the `shibir_owner` cookie (HttpOnly, SameSite=Lax, Secure unless `COOKIE_SECURE=false`, 30
  days) and stores only its SHA-256 hash. Guests get `GUEST_MESSAGE_LIMIT` (5) responses in total
  per cookie and `GUEST_IP_LIMIT_PER_DAY` (15) per IP per 24 h, counted in Postgres
  (`guest_usage`, which holds IP addresses and is purged after 30 days idle), then must log in or
  register. Rotating IPs or using a VPN gets around the IP counter. Guests cannot request notes.
- A chat session is visible only to the owner that created it (a user, from any of their browsers,
  or one guest cookie); an unknown session id and someone else's id get the same 404 on
  `/conversations`, and on `/chat` both start a new session under a fresh id, so neither reveals
  whether the id exists. Session ids are server-generated 256-bit random values. A guest who logs
  in or registers takes their in-memory chats along.
- **A guest who clears cookies (or uses a different browser/device) loses access to earlier
  chats.** There is no recovery. A user's chats are kept in process memory and are lost on restart.
- Cross-site requests: CORS allows credentials only for the explicit `CORS_ALLOW_ORIGINS` list
  (`*` is rejected at startup), and `POST`/`DELETE` with an `Origin` outside that list get 403
  (CSRF), except same-origin requests (the API's own address, e.g. `/docs`). Requests without an
  `Origin` (curl, scripts) are allowed but only ever see their own anonymous sessions. On
  `/auth/*`, a `POST` that carries a login-session cookie but no `Origin` is also rejected.
- **In-app rate limits** (`app/services/rate_limiter.py`), each an HTTP 429 with `Retry-After`,
  enforced whichever proxy is in front:
  - `/chat` and `/chat/stream`: `RATE_LIMIT_PER_IP_PER_MINUTE` (30) and
    `RATE_LIMIT_PER_OWNER_PER_MINUTE` (20). Dropping the cookie escapes the owner limit, not the IP
    one.
  - `GET /conversations`, `GET /conversations/{id}/messages` and `DELETE`:
    `RATE_LIMIT_READ_PER_IP_PER_MINUTE` (120).
  - Note requests: `NOTE_LIMIT_PER_OWNER_PER_HOUR` (3); an SSE `error` event on `/chat/stream`.
  - Logged-in users: `USER_DAILY_MESSAGE_LIMIT` (100) chat requests per UTC day.
  - `/auth/login` and `/auth/register`: see Accounts above. The proxy does not rate-limit `/auth/*`.
  - `DAILY_LLM_REQUEST_BUDGET` (5000): a global count of chat requests per UTC day; past it every
    chat request gets 429 until midnight UTC. `0` = unlimited.

  The state is in process memory: it is per worker and **resets on restart** (including the daily
  budget).
- Sessions have **no TTL**: only caps (`MAX_SESSIONS_PER_OWNER`=50, `MAX_SESSIONS_TOTAL`=10000,
  least-recently-updated evicted). A client that discards its cookie can create sessions (now at
  most the per-IP chat rate) until the global cap evicts other users' oldest sessions.
- **Message length limits:** a chat message over `MAX_MESSAGE_CHARS` (4000) characters, or a note
  request over `MAX_NOTE_CHARS` (300), gets HTTP 413 (an SSE `error` event for a note on
  `/chat/stream`), on top of the proxy's 64 KB body limit.
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
