# shepherd-pr mechanics

Sandbox- and repo-specific facts the steps rely on. Kept out of SKILL.md so the steps stay legible.

## Watcher

`scripts/pr_watch.sh <owner/repo> <pr> [SINCE]` is the poller. Run it **detached** — a background
process exiting re-invokes the session with its final line, so it wakes the session by exiting.
Idle polling costs the session no context; only a real wake does. It reports only *that* activity
happened; the session pulls the details on wake (snapshot / API).

The gate is a **server timestamp**: it polls the PR's `updated_at` (GitHub's own clock, so no
local skew) and wakes on the first value strictly newer than `SINCE`. `SINCE` defaults to a
*settled* `updated_at` at launch — read until two consecutive reads agree, because `updated_at`
can lag a just-made write by a few seconds; without the settle a re-armed watcher would wake on
the session's own reply once the lagging timestamp caught up. `updated_at` bumps on comments,
reviews, commits, edits, merge and close, but **not** on CI check-runs — so CI churn never wakes
the session.

Keep **exactly one watcher**. Each session response re-arms a fresh one (no `SINCE`, so it reads
`updated_at` after the response landed) — that is what stops the session from waking on its own
work: the response sits at the new gate, not after it.

## GitHub auth (sandbox and outside)

`pr_watch.sh` and `pr_snapshot.py` authenticate in whichever environment they run: they use
`GITHUB_TOKEN` / `GH_TOKEN` if set, else fall back to `gh auth token`, and send it as a bearer
token. If none is found the calls go out bare — which still works inside a sandbox (see below) and
for public repos anywhere.

- **Outside a sandbox:** `gh auth login` (or export `GITHUB_TOKEN`) and everything works with your
  normal credentials — the scripts pick the token up, and plain `git push` / `gh pr` handle writes.
- **Inside this sandbox:** the proxy injects credentials for github.com HTTPS traffic, so bare curl
  and `git push https://github.com/<owner>/<repo>.git <branch>` just work; `gh` is **not** logged
  in (`gh auth status` reports logged-out — expected) and SSH remotes fail, so name the HTTPS URL
  explicitly. Force-push after an amend needs an explicit lease (the push-URL tracking ref goes
  stale): `git push --force-with-lease=<branch>:<remote-sha> https://github.com/<owner>/<repo>.git <branch>`.

Read/write the REST API with curl (`POST -d @body.json` to open PRs and post replies, `PATCH` the
PR body); add `-H "Authorization: Bearer $TOKEN"` when a token is available.

## Reply to a review thread

`POST /repos/<owner/repo>/pulls/<pr>/comments/<comment_id>/replies` with `{"body": "…"}`.
Get the `<comment_id>`s from `GET …/pulls/<pr>/comments`.

## Follow-up vs amend

Before any review exists: amend for a clean single commit. Once the PR has been reviewed:
add a **follow-up** commit instead, so the review comments stay anchored to their lines and the
reviewer sees exactly what changed in response.

## Reset the workspace after a merge

Once the PR merges, its branch is dead weight — get the workspace back onto an up-to-date base so
the next task branches from the merged state, not a stale one.

- Switch off the merged branch and update the base: `git checkout <base>` (usually `main`), then
  `git fetch origin` and `git merge --ff-only origin/<base>`. Fast-forward only — a real merge here
  means the local base has drifted, which is worth surfacing rather than papering over.
- Optionally delete the merged local branch (`git branch -d <branch>` — the safe `-d` refuses if it
  isn't fully merged).
- A dirty working tree blocks the checkout; stop and flag it rather than stashing blindly.
