#!/usr/bin/env bash
# Deploy the latest origin/$BRANCH on the VPS, with a health check and rollback.
#
# Runs ON the VPS. GitHub Actions (.github/workflows/deploy.yml) triggers it
# over SSH with a key whose authorized_keys entry forces this script, so the
# command the client sends ($SSH_ORIGINAL_COMMAND) is ignored. It can also be
# run by hand: scripts/deploy.sh. One-time setup: docs/auto-deploy.md.
#
# Only tracked files change (git reset --hard, no git clean, no compose down),
# so .env, data/ and the Docker volumes (pgdata, chroma_db, hf_cache) survive.
# Rollback restores code only; it cannot undo database changes.
#
# Exit status: 0 = new commit deployed and healthy; 1 = anything else,
# including a successful rollback, so the Actions run shows the failure.

set -euo pipefail

BRANCH="${BRANCH:-main}"
HEALTH_URL="${HEALTH_URL:-http://localhost:9200/health}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-180}"  # seconds, after `compose up` returns
LOCK_FILE="${LOCK_FILE:-/tmp/shibir-chat-deploy.lock}"

log() { printf '[deploy %s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }

wait_healthy() {
    local deadline=$((SECONDS + HEALTH_TIMEOUT))
    while ((SECONDS < deadline)); do
        if curl -fsS --max-time 5 "$HEALTH_URL" >/dev/null 2>&1; then
            log "healthy: $HEALTH_URL"
            return 0
        fi
        sleep 3
    done
    log "not healthy after ${HEALTH_TIMEOUT}s; last app logs:"
    docker compose logs --tail=60 app || true
    return 1
}

# A failed build counts as unhealthy, so it rolls back too (under set -e it
# would otherwise exit before the rollback).
up_and_wait() {
    log "building and starting containers"
    if ! docker compose up -d --build --remove-orphans; then
        log "docker compose up failed"
        return 1
    fi
    wait_healthy
}

main() {
    # A forced SSH command starts in $HOME, not the repo.
    cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.."

    exec 9>"$LOCK_FILE"
    if ! flock -n 9; then
        log "another deploy is already running (lock: $LOCK_FILE)"
        return 1
    fi

    if [[ ! -f .env ]]; then
        log "refusing to deploy: .env is missing in $PWD"
        return 1
    fi

    local prev new
    prev="$(git rev-parse HEAD)"
    git fetch --prune origin "$BRANCH"
    git reset --hard "origin/$BRANCH"
    new="$(git rev-parse HEAD)"
    log "deploying ${new:0:7} (previous: ${prev:0:7})"

    if up_and_wait; then
        docker image prune -f || log "warning: docker image prune failed"
        log "deployed $(git rev-parse --short HEAD)"
        return 0
    fi

    if [[ "$new" == "$prev" ]]; then
        log "deploy of ${new:0:7} failed; it was already the running commit, nothing to roll back to"
        return 1
    fi

    log "deploy of ${new:0:7} failed; rolling back to ${prev:0:7}"
    git reset --hard "$prev"
    if up_and_wait; then
        log "rolled back to ${prev:0:7}; the deploy of ${new:0:7} FAILED"
    else
        log "ROLLBACK ALSO FAILED; manual intervention needed"
    fi
    return 1
}

# Everything above is only definitions. `git reset --hard` rewrites this file
# while bash is still reading it, so all logic runs from main(), which bash
# has fully parsed before the reset, and this last line is read in one go.
main "$@"; exit $?
