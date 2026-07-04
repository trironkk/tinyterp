# shepherd-pr mechanics

Sandbox- and repo-specific facts the steps rely on. Kept out of SKILL.md so the steps stay legible.

## Watcher

`scripts/pr_watch.sh <owner/repo> <pr> [SINCE]` is the poller. Run it **detached** — a background
process exiting re-invokes the session with its final line, so it wakes the session by exiting.
Idle polling costs the session no context; only a real wake does. It reports only *that* activity
happened; the session pulls the details on wake (snapshot / API).

The gate is a **server timestamp**: it polls the PR's `updated_at` (GitHub's own clock, so no
local skew) and wakes on the first value strictly newer than `SINCE`. `SINCE` defaults to
`updated_at` at launch. `updated_at` bumps on comments, reviews, commits, edits, merge and close,
but **not** on CI check-runs — so CI churn never wakes the session.

Keep **exactly one watcher**. Each session response re-arms a fresh one (no `SINCE`, so it reads
`updated_at` after the response landed) — that is what stops the session from waking on its own
work: the response sits at the new gate, not after it.

## GitHub auth

The sandbox proxy injects credentials for github.com HTTPS traffic; `gh` is **not** logged in
(`gh auth status` reports logged-out — expected, not a failure).

- Read/write the API with curl: `curl -H "Accept: application/vnd.github+json" https://api.github.com/…`
  (POST with `-d @body.json` to open PRs, post replies, etc.).
- Push over HTTPS, naming the URL explicitly — SSH remotes fail in-sandbox:
  `git push https://github.com/<owner>/<repo>.git <branch>`.
- Force-push after an amend uses an explicit lease, since the push-URL tracking ref goes stale:
  `git push --force-with-lease=<branch>:<remote-sha> https://github.com/<owner>/<repo>.git <branch>`.

## Reply to a review thread

`POST /repos/<owner/repo>/pulls/<pr>/comments/<comment_id>/replies` with `{"body": "…"}`.
Get the `<comment_id>`s from `GET …/pulls/<pr>/comments`.

## Notebook run records

Changing a notebook's **code cells** stales its `runs/` record; the pre-commit hook and CI
(`scripts/check_runs.py`) block the commit until it is re-recorded. `make record NB=<nb>`
executes the notebook wherever it runs — the GPU-less sandbox yields a CPU-only record that
drops the GPU diagnostics the notebook is about. So re-record on the **GPU host** (direct mode
surfaces the new file back into the sandbox worktree), then commit. Notebook edits that touch
only markdown/prose cells do not change code cells and need no re-record.

## Follow-up vs amend

Before any review exists: amend for a clean single commit. Once the PR has been reviewed:
add a **follow-up** commit instead, so the review comments stay anchored to their lines and the
reviewer sees exactly what changed in response.
