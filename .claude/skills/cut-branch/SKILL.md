---
name: cut-branch
description: Cut a fresh branch off an up-to-date main. Use before work that lands via a PR, or when a skill needs a clean branch (e.g. notebook-arc).
---

Never commit on `main`; work lands via a PR. Sync `main`, then branch off it:

- `git fetch origin && git checkout main && git merge --ff-only origin/main`
- `git checkout -b <name>` — the caller names it for the work.

If the working tree is dirty or `main` has diverged (the `--ff-only` merge fails), stop and ask
rather than stashing or force-syncing.

Done when the named branch exists and is checked out.
