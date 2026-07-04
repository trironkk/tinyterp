# shepherd-pr mechanics

Sandbox- and repo-specific facts the steps rely on. Kept out of SKILL.md so the steps stay legible.

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
