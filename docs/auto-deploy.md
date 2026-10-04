# Automatic deployment to the VPS

Every push to `main` (normally a merged release PR from `dev`) updates the VPS automatically:

1. `.github/workflows/deploy.yml` opens an SSH connection to the VPS with a dedicated key.
2. That key may only run [`scripts/deploy.sh`](../scripts/deploy.sh) (a forced command in
   `authorized_keys`). The script resets the checkout to `origin/main`, runs
   `docker compose up -d --build`, and polls `http://localhost:9200/health`.
3. If the build fails or `/health` doesn't respond within 180 s, the script resets to the
   previous commit, rebuilds it, and exits 1, so the Actions run fails and you are notified.

Deploys never overlap: Actions queues them (`concurrency: deploy-production`) and the script
holds a `flock`. You can also run a deploy by hand on the VPS: `scripts/deploy.sh`.

This page is the one-time setup. Placeholders such as `<vps-host>` and `/path/to/repo` stand for
your real values; never commit those values or any key.

## 1. Prepare the VPS

Do this as the user that will run deploys (the one in `VPS_USER` below).

1. **The clone already works with Docker Compose.** In `/path/to/repo`, `docker compose up -d`
   starts `db` and `app`, and `curl http://localhost:9200/health` returns `{"status":"ok"}`.
2. **The user can run Docker without sudo:** `sudo usermod -aG docker $USER`, then log in again.
3. **`.env` exists** in the repo root. The script refuses to deploy without it.
4. **`git fetch` works without a password.** Use a read-only credential: a GitHub *deploy key*
   (repo → Settings → Deploy keys, "Allow write access" unchecked) with an SSH `origin` URL, or
   a fine-grained read-only token. Test it: `git -C /path/to/repo fetch origin main`.
5. **No local edits to tracked files.** `git -C /path/to/repo status` must be clean except for
   untracked or ignored files. Deploys run `git reset --hard`, which discards edits to tracked
   files (for example a hand-edited `docker-compose.yml`). `.env`, `data/` and the Docker volumes
   (`pgdata`, `chroma_db`, `hf_cache`) are never touched.
6. **The script is executable:** `chmod +x /path/to/repo/scripts/deploy.sh`.

### Stop the old systemd service

Production used to run as `shibirgpt.service` (`run.sh` directly on the host). Left running, it
fights the `app` container for port 9200:

```bash
sudo systemctl disable --now shibirgpt.service
```

## 2. Create the deploy key

On your own computer, create a key used only for deploys, with no passphrase:

```bash
ssh-keygen -t ed25519 -f gh_deploy_key -N "" -C github-actions-deploy
```

On the VPS, add **one line** to `~/.ssh/authorized_keys` of the deploy user. It is the content of
`gh_deploy_key.pub`, prefixed with the forced command and restrictions:

```
command="/path/to/repo/scripts/deploy.sh",no-port-forwarding,no-agent-forwarding,no-pty,no-X11-forwarding ssh-ed25519 AAAA... github-actions-deploy
```

Whatever command the client sends, this key can only run `deploy.sh`. A leaked key can trigger a
redeploy of `main`, but it can't open a shell.

## 3. Configure GitHub

In the repository: **Settings → Environments → New environment**, name it **`production`**
(the workflow uses this exact name), and add these environment secrets:

| Secret | Value |
|---|---|
| `VPS_HOST` | The VPS hostname or IP address. |
| `VPS_USER` | The deploy user from step 1. |
| `VPS_PORT` | The SSH port. Optional; defaults to 22. |
| `VPS_SSH_KEY` | The full content of the **private** key `gh_deploy_key`, including the `BEGIN`/`END` lines. |
| `VPS_KNOWN_HOSTS` | The output of `ssh-keyscan -p 22 <vps-host>` (use your port if it isn't 22). |

Run `ssh-keyscan` **on your own computer**, and check that the fingerprints match the VPS
(`ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub` on the server). The workflow connects with
`StrictHostKeyChecking=yes`, so a wrong or missing entry fails the deploy instead of trusting an
unknown host.

Optional: under the environment's **Deployment protection rules**, add **Required reviewers**.
Each deploy then waits for a manual approval in the Actions tab.

Then **delete the private key from your computer** (`rm gh_deploy_key`). It only needs to exist
in the GitHub secret. Keep or delete `gh_deploy_key.pub`; it is not secret.

## 4. First deploy

Go to **Actions → Deploy → Run workflow** (branch `main`). The log shows the commit being
deployed, the health check and the final `deployed <commit>` line. After that, every push to
`main` deploys automatically.

## Notes

- **Rollback restores code only.** It can't undo database changes, so a release that changes the
  schema or data must be backward-compatible with the previous code. There are no migrations in
  this repo yet; adding them would need a step in `deploy.sh`.
- **After a rollback the VPS runs an older commit than `main`.** Fix forward with a new commit
  to `main`, or re-run the workflow once the cause (for example a missing `.env` variable) is fixed.
- **Rebuilds reuse the Docker layer cache.** Dependencies are reinstalled only when
  `pyproject.toml` or `uv.lock` change, so most deploys take seconds to a few minutes. The
  script runs `docker image prune -f` after a successful deploy to remove dangling images.
- **The health check only proves the app started.** `/health` loads no models (they load lazily
  on the first request), so a broken model, database or OpenAI setup can still pass it. Smoke-test
  `/chat` after deploys that touch retrieval or generation (see CLAUDE.md §12).
- **CI is not re-checked at deploy time.** Merges to `main` arrive through PRs that passed CI,
  but a manual "Run workflow" or a direct push deploys `main` as it is.
- **The first deploy after setup may be slow:** if the image has never been built on the VPS,
  dependencies (including torch) are installed from scratch. The job has a 30-minute timeout.
