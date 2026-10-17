# Accounts, guests and access control

This is the contract for the front-end team and the operator notes for accounts. Code:
`app/auth/`, `app/api/auth_routes.py`, `app/api/deps.py`. Security summary: [SECURITY.md](../SECURITY.md).

## Overview

- **Guests** (not logged in) can chat without an account, up to `GUEST_MESSAGE_LIMIT` (5)
  answers in total. Then every chat request returns **401 `guest_limit_reached`** until the
  person registers or logs in. Guests cannot request notes.
- **Users** log in with email and password. They have no guest limit, only a daily cap
  (`USER_DAILY_MESSAGE_LIMIT`, 100). Their conversations follow the account to any browser.
- When a guest registers or logs in, the chats from *that browser* move to the account, so the
  conversation continues.
- There are no JWTs. Login state is an opaque, HttpOnly cookie, so JavaScript never sees a token.

## Setup

1. The tables (`users`, `auth_sessions`, `guest_usage`) are created automatically when the app
   starts, if they are missing. Existing tables are never altered. If the database is down at
   startup, the app still starts and logs a warning; guest chat and login then return 503 until
   the tables exist. To create them by hand instead (the file is idempotent):

   ```bash
   docker compose exec -T db psql -U postgres -d shibir_chat < docs/sql/auth_tables.sql
   ```

2. Create the first admin. The password comes from a prompt or the `CREATE_USER_PASSWORD`
   environment variable, never from the command line:

   ```bash
   uv run python -m scripts.create_user --email you@example.com --role admin
   ```

   The script applies the password policy and refuses to overwrite an existing account. The
   `admin` role exists for future admin endpoints; none exist yet.

Postgres is now on the request path for guests: if it is unreachable, guest chat requests fail
with 503 instead of skipping the limit (fail closed).

## Cookies, CORS and credentials

| Cookie | Set by | Purpose |
|---|---|---|
| `__Host-shibir_session` (`shibir_session` when `COOKIE_SECURE=false`) | login, register, change-password | Login session. HttpOnly, Secure, SameSite=Lax, Path=/, no Domain, Max-Age = `AUTH_SESSION_ABSOLUTE_DAYS`. |
| `shibir_owner` | any guest request to `/chat`, `/chat/stream`, `/conversations*` | Anonymous guest identity (unchanged). |

- Call the API with `fetch(url, { credentials: "include" })` on **every** request (including
  `GET /auth/me` and the chat routes), or the browser does not send or store the cookies. CORS
  allows credentials only for the origins in `CORS_ALLOW_ORIGINS`.
- **Send no custom request headers.** CORS allows only `Content-Type` (`allow_headers`), so an
  `Authorization`, `X-Requested-With` or `X-Session-Id` header fails the preflight and the request
  never reaches the API. There is no token to send anyway: the cookie is the login state.
- The only non-standard response header is `Retry-After` on a `429`; it is exposed to cross-origin
  JavaScript (`expose_headers`), so `response.headers.get("Retry-After")` works.
- Every `POST`/`DELETE` must come from an allowed `Origin` (browsers add it automatically). On
  `/auth/*`, a `POST` that carries the session cookie but **no** `Origin` header is rejected with
  403.
- Auth request bodies must be JSON (`Content-Type: application/json`, otherwise 415), with exactly
  the documented fields (unknown fields, or a missing one, are 422). An email longer than 254
  characters or a password longer than 128 is a policy error, not a 422: `400` on register and
  change-password, and on login the usual `401`.
- Every `/auth/*` response has `Cache-Control: no-store`.
- **There is no `session` SSE event** and no auth event on `/chat/stream`. The login state only
  changes through the `/auth/*` responses; call `GET /auth/me` after a login, register or logout to
  refresh the UI, or just use the body those endpoints return.
- **Local development: same host name or same origin.** Both cookies are `SameSite=Lax`, so the
  browser does not send them on a cross-site request, and a different host name is a different
  site: a front end on `http://localhost:5173` calling `http://127.0.0.1:9200` logs in and then
  looks logged out, because the cookie never comes back. Either proxy the API through the dev server
  (a Vite `server.proxy` entry for `/chat`, `/conversations`, `/auth` and `/health`, so the front
  end calls its own origin), or make sure both use the **same host name** (`localhost` on both
  sides; the port may differ). In production the API and the front end are on different
  subdomains of the same site, which Lax allows.
- Over plain `http://` (local development) set `COOKIE_SECURE=false`, or the browser drops the
  cookies; the session cookie is then called `shibir_session`.

## Endpoints

Errors use the FastAPI shape `{"detail": "..."}` unless stated otherwise. A `429` always has a
`Retry-After` header (seconds). A `503` means the database (or an upstream service) is down.

### `POST /auth/register`

```json
{ "email": "reader@example.com", "password": "at least ten characters" }
```

| Status | Meaning |
|---|---|
| `201` | `{"id": "<uuid>", "email": "reader@example.com", "role": "user"}`. The user is logged in (session cookie set). |
| `400` | Invalid email (shape, or longer than 254 characters), or the password breaks the policy (`detail` says which rule). |
| `404` | Registration is disabled (`AUTH_REGISTRATION_ENABLED=false`). |
| `409` | That email is already registered. |
| `429` | 3 registrations per IP per hour, or the global `AUTH_REGISTRATIONS_PER_DAY` is used up. Only well-formed attempts (past the 400 checks) count. |

Password policy: 10–128 characters, not only whitespace, not the same as the part of the email
before `@`. No composition rules (no "one digit, one symbol").

### `POST /auth/login`

```json
{ "email": "reader@example.com", "password": "..." }
```

| Status | Meaning |
|---|---|
| `200` | `{"id": "...", "email": "...", "role": "user" \| "admin"}`, and a fresh session cookie. |
| `401` | `{"detail": "invalid credentials"}`, identical for unknown email, wrong password (including one longer than 128 characters), disabled account and locked account. |
| `429` | 5 **failed** attempts per email per 15 minutes (successful logins do not count), or 20 attempts of any outcome per IP per hour. |

After 10 consecutive wrong passwords, the account is locked for 30 seconds, doubling with every
further failure up to 15 minutes. It is never locked permanently, and a successful login resets
the counter. A locked account gets the same 401 as a wrong password.

### `GET /auth/me`

Always `200`. Use it on page load to decide what to show.

```json
{ "authenticated": true, "id": "...", "email": "reader@example.com", "role": "user" }
```

```json
{ "authenticated": false, "guest_limit": 5, "guest_remaining": 3 }
```

It never creates a guest cookie. Without one, `guest_remaining` reflects the IP counter only.

### `POST /auth/logout` and `POST /auth/logout-all`

No body. `204` and the session cookie is cleared. `logout` ends this session; `logout-all` ends
every session of the account (all devices). `401` if not logged in.

### `POST /auth/change-password`

```json
{ "current_password": "...", "new_password": "..." }
```

| Status | Meaning |
|---|---|
| `200` | `{"id", "email", "role"}`. Every session of the account is ended, and this browser gets a new session cookie. |
| `400` | Wrong current password (`"invalid credentials"`) or a policy violation. |
| `401` | Not logged in. |
| `429` | Too many attempts (shares the login limits: wrong current passwords count against the per-account limit, every call against the per-IP one). |

## Chat for guests

`POST /chat` and `POST /chat/stream` are unchanged for logged-in users, apart from the daily cap.
For guests:

- Each answer counts against the quota. Empty or too-long messages (400 / 413), rate-limited
  requests (429) and requests that fail upstream (503, or an SSE `error` event) do not count.
- **Remaining quota** is in every answer:
  - `POST /chat` response: `"guest_remaining": 4, "guest_limit": 5` (both `null` for logged-in users);
  - `/chat/stream`: the same two fields in the `done` event.
- **Limit reached**: HTTP `401` with this body, on `/chat` and `/chat/stream` alike. On the
  streaming route it is a normal HTTP response returned before any event, not an SSE event.

  ```json
  {
    "detail": "অতিথি হিসেবে বার্তা পাঠানোর সীমা শেষ হয়েছে। চালিয়ে যেতে অনুগ্রহ করে লগইন বা নিবন্ধন করুন।",
    "code": "guest_limit_reached",
    "guest_limit": 5,
    "guest_remaining": 0
  }
  ```

  Show the login / register screen on `code == "guest_limit_reached"`. Do not rely on a 401
  alone: `/auth/logout` and friends also return 401 when not logged in.
- **Notes need an account.** A guest message classified as a note request gets HTTP `403` with
  a Bengali "please log in or register" `detail` on `/chat`, or an SSE `error` event on
  `/chat/stream`. It does not use up a guest message.
- Logged-in users get `429` with `Retry-After` (seconds to midnight UTC) after
  `USER_DAILY_MESSAGE_LIMIT` chat requests in a UTC day.

The guest quota is counted twice, and running out of either blocks the guest:

- per guest cookie (`shibir_owner`), `GUEST_MESSAGE_LIMIT` in total, forever;
- per client IP address (IPv6: per /64), `GUEST_IP_LIMIT_PER_DAY` per 24 hours. Clearing cookies
  does not reset the quota, and a shared network (a campus) gets more than one person's worth.

Both counters live in Postgres (`guest_usage`), so restarts and deploys do not reset them.
**`guest_usage` stores IP addresses**; rows not used for 30 days are deleted automatically.

## Access levels (deny by default)

Every route declares a level in `app/auth/policy.py`, and `tests/test_auth_policy.py` fails if a
route is missing from that map or lacks the matching dependency.

| Level | Who | Routes |
|---|---|---|
| `public` | anyone | `GET /health`, `POST /auth/register`, `POST /auth/login`, `GET /auth/me` |
| `guest_ok` | guests and users | `POST /chat`, `POST /chat/stream`, `GET /conversations`, `GET /conversations/{id}/messages`, `DELETE /conversations/{id}` |
| `user` | logged in (401 otherwise) | `POST /auth/logout`, `POST /auth/logout-all`, `POST /auth/change-password` |
| `admin` | role `admin` (401 guest, 403 user) | none yet |

A new endpoint must be added to the map with a level and the matching dependency
(`public_route`, `get_principal`, `require_user` or `require_role("admin")` from
`app/api/deps.py`).

## Settings

| Variable | Default | Notes |
|---|---|---|
| `AUTH_SESSION_IDLE_DAYS` | `7` | A session ends after this long without a request. Activity extends it (at most one DB write per 5 minutes). |
| `AUTH_SESSION_ABSOLUTE_DAYS` | `30` | A session ends this long after login, whatever the activity. Also the cookie's Max-Age. |
| `AUTH_REGISTRATION_ENABLED` | `true` | `false` makes `POST /auth/register` a 404. |
| `AUTH_REGISTRATIONS_PER_DAY` | `200` | Global sign-up circuit breaker per UTC day (in-process). `0` = unlimited. |
| `GUEST_MESSAGE_LIMIT` | `5` | Answers per guest cookie, in total. `0` = guests cannot chat at all. |
| `GUEST_IP_LIMIT_PER_DAY` | `15` | Guest answers per IP (IPv6 /64) per 24 hours. |
| `USER_DAILY_MESSAGE_LIMIT` | `100` | Chat requests per user per UTC day (in-process). `0` = unlimited. |
| `COOKIE_SECURE` | `true` | Existing setting; also decides the session cookie name and Secure flag. |

Fixed in code (`app/api/auth_routes.py`, `app/auth/sessions.py`): login 5 failures per email per
15 min and 20 attempts per IP per hour, registration 3 well-formed attempts per IP per hour, lockout
after 10 failures (30 s doubling to 15 min), at most 10 sessions per user. The reverse proxy adds
10 requests per minute per IP on `/auth/login` and `/auth/register` in front of all of these
(`deploy/nginx/shibirgpt-api.conf`; a commented-out equivalent in `deploy/caddy/Caddyfile`).

## Audit log

Each security event is one JSON line on the `app.auth.audit` logger (stderr): `login_success`,
`login_failure`, `register`, `logout`, `logout_all`, `password_changed`, `account_locked`,
`session_rejected`, `guest_limit_reached`, `forbidden_access`. Fields: `ts`, `event`, `user_id`
when known, `ip_prefix` (IPv4 /24, IPv6 /48) and, on failures, `email_ref` (the first 12 hex
characters of the normalised email's SHA-256). Passwords, tokens, hashes, full emails and full IPs
are never logged, and none of this goes to Langfuse. There is no request id yet.

## Limits (read before relying on this)

- **No email verification, password reset or MFA.** A forgotten password can only be reset by an
  operator (there is no tool for it yet). Because sign-up is open and unverified, anyone can
  create a fresh account to get past the guest limit; `AUTH_REGISTRATIONS_PER_DAY` and the
  per-IP limit only slow this down.
- **The guest limit is per cookie and per IP.** Rotating IPs (mobile networks, VPNs, IPv6
  addresses outside one /64) gets around the IP counter.
- **Single worker.** Login throttling, the user daily cap and the registration cap are in-process
  and reset on restart (the account lockout and the guest quota are in the database and survive).
  Run one worker until a shared store exists.
- **Chat sessions are still in memory.** They move to the account on login but are lost on restart,
  and logging out does not delete them (they come back on the next login, until a restart).
- A guest's chats are moved only from the browser that logs in, and only once; guest chats are
  never merged from other browsers.
- **`409` on register reveals whether an email is registered.** Anyone can check an address
  by trying to sign up with it. Login does not leak this (every failure is the same 401), but
  with open registration there is no way to hide it short of email verification ("we sent you
  a link" for new and existing addresses alike), which does not exist yet (point 1 below).
- Argon2id with library defaults uses about 64 MiB per hash. Many concurrent logins or sign-ups
  can use a lot of memory; the per-IP limits in the app and the proxy's 10 requests per minute
  per IP on the two routes are the only guards.

## Next steps (recommended)

1. Email verification on sign-up, and password reset by emailed one-time link.
2. MFA (TOTP) at least for admins.
3. Move sessions, limits and chat history to a shared store (Postgres or Redis) so more than one
   worker can run.
4. Admin endpoints (behind `require_role("admin")`) to disable accounts and unlock users.
