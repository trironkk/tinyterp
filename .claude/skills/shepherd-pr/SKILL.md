---
name: shepherd-pr
description: Shepherd a pull request from open to merge — arm a background watcher the moment a PR is created, and on each wake pull the details, address and reply to feedback, then re-arm. Use right after opening a PR, and whenever a shepherded PR wakes the session.
---

To shepherd a PR is to carry it from open to merge without the user chasing it. A detached
watcher polls the PR quietly and wakes the session only on new activity; the session then pulls
the details, addresses feedback, replies, and re-arms. Same process each run. Four steps, in
order — a later step runs only when its trigger fires.

Sandbox and repo mechanics each step depends on (GitHub auth, the watcher, replying to a thread,
notebook run records, follow-up vs amend) live in [reference.md](reference.md). Read it before
the first Address.

1. **Watch.** The instant a PR is created, arm the one-shot watcher.

   - Run `bash <this-skill>/scripts/pr_watch.sh <owner/repo> <pr>` **detached** (a background
     process), so idle polling costs the session no context.
   - It gates on the PR's current `updated_at` (GitHub's clock) and exits on the first activity
     strictly newer than that, which re-invokes the session. Everything up to launch — including
     whatever the session just did — sits at or before the gate and is ignored by construction.
   - Done when the detached watcher is running.

2. **Wake.** The watcher exits on new activity, re-invoking the session; it reports only *that*
   something changed, not what. Pull the details yourself.

   - Snapshot the PR (`scripts/pr_snapshot.py <owner/repo> <pr>`), PushNotification a one-line
     summary of what moved, and route:
   - New review feedback → Address.
   - `merged` / `closed` → Close.
   - Anything else (e.g. a comment needing no code change) → reply if warranted, then re-arm.
   - Done when the change is understood and routed.

3. **Address.** Resolve every open thread before re-arming.

   - Fetch: `GET /repos/<owner/repo>/pulls/<pr>/comments`.
   - Apply each requested change and verify it (run the affected code or repo checks, not just
     eyeball it).
   - Commit as a follow-up (never amend a reviewed PR), push over HTTPS, reply to each thread
     citing the commit sha. End every reply body with the `🤖 via Claude Code` marker (see
     [reference.md](reference.md)) so automated replies stay distinguishable from hand-typed ones.
   - Re-arm: launch a fresh `pr_watch.sh` with no SINCE — it reads `updated_at` after your push,
     so your own change sits at the gate and can never wake the session. Keep exactly one watcher;
     each response replaces it.
   - Done when every thread has a change and a reply, the follow-up is pushed, and a fresh watcher
     is armed.

4. **Close.** On merge or close, send a final PushNotification. Do not re-arm — shepherding is done.

   - On **merge**, reset the workspace to the freshly-merged base branch so the next task starts
     clean: switch off the merged branch (`git checkout <base>`), then `git fetch origin` and
     `git merge --ff-only origin/<base>`. If the work tree is dirty, stop and ask for guidance
     rather than stashing or discarding. On close-without-merge, leave the branch in place.
   - Done when the final PushNotification is sent and — if merged — the workspace sits on an
     up-to-date base branch.
