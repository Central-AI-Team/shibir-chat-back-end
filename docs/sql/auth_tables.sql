-- Account and guest-quota tables (app/db/models.py: User, AuthSession, GuestUsage).
-- The app creates these itself at startup (app.main.create_auth_tables). This file
-- is the same schema for creating them by hand. Idempotent: safe to run more than once.
--   docker compose exec -T db psql -U postgres -d shibir_chat < docs/sql/auth_tables.sql
-- Keep in sync with the models; tests/test_auth_schema.py compares the columns.

BEGIN;

CREATE TABLE IF NOT EXISTS users (
    id                  VARCHAR(36)  PRIMARY KEY,
    email               VARCHAR(254) NOT NULL UNIQUE,
    password_hash       VARCHAR(255) NOT NULL,
    role                VARCHAR(10)  NOT NULL DEFAULT 'user'
                        CONSTRAINT ck_users_role CHECK (role IN ('user', 'admin')),
    is_active           BOOLEAN      NOT NULL DEFAULT true,
    failed_login_count  INTEGER      NOT NULL DEFAULT 0,
    locked_until        TIMESTAMPTZ  NULL,
    created_at          TIMESTAMPTZ  NOT NULL,
    last_login_at       TIMESTAMPTZ  NULL,
    password_changed_at TIMESTAMPTZ  NOT NULL
);

CREATE TABLE IF NOT EXISTS auth_sessions (
    id                  SERIAL       PRIMARY KEY,
    user_id             VARCHAR(36)  NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token_hash          VARCHAR(64)  NOT NULL UNIQUE,
    created_at          TIMESTAMPTZ  NOT NULL,
    last_seen_at        TIMESTAMPTZ  NOT NULL,
    idle_expires_at     TIMESTAMPTZ  NOT NULL,
    absolute_expires_at TIMESTAMPTZ  NOT NULL,
    revoked_at          TIMESTAMPTZ  NULL,
    user_agent          VARCHAR(200) NULL,
    ip_prefix           VARCHAR(64)  NULL
);
CREATE INDEX IF NOT EXISTS ix_auth_sessions_user_id ON auth_sessions (user_id);

CREATE TABLE IF NOT EXISTS guest_usage (
    key          VARCHAR(255) PRIMARY KEY,
    count        INTEGER      NOT NULL DEFAULT 0,
    first_seen   TIMESTAMPTZ  NOT NULL,
    last_seen    TIMESTAMPTZ  NOT NULL,
    window_start TIMESTAMPTZ  NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_guest_usage_last_seen ON guest_usage (last_seen);

COMMIT;
