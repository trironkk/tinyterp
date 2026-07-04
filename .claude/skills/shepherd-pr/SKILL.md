---
name: shepherd-pr
description: Shepherd a pull request from open to merge — arm a watcher loop the moment a PR is created, then address and reply to review feedback each time it lands. Use right after opening a PR, and whenever feedback arrives on a shepherded PR.
---

To shepherd a PR is to carry it from open to merge without the user chasing it: a watcher
polls the PR, and every piece of review feedback is addressed, answered, and pushed. Same
process each run. Four steps, in order — a later step runs only when its trigger fires.

Sandbox and repo mechanics that each step depends on (GitHub auth, replying to a thread,
notebook run records, follow-up vs amend) live in [reference.md](reference.md). Read it before
the first Address.

1. **Watch.** The instant a PR is created, snapshot its baseline and arm a recurring watcher.

   - Baseline: `python3 <this-skill>/scripts/pr_snapshot.py <owner/repo> <pr> > $SCRATCHPAD/pr_<pr>_state.txt`.
   - Arm: CronCreate `*/5 * * * *`, `recurring: true`, prompt = the Poll step below with
     `<owner/repo>`, `<pr>`, and the absolute state-file and script paths inlined so the fire
     is self-contained. Keep the returned job id — Close needs it.
   - Done when the job is armed and the baseline file is written.

2. **Poll.** Each fire, re-snapshot and diff against the state file.

   - No diff → stop silently, no notification.
   - Diff → overwrite the state file, then PushNotification a one-line summary of what moved.
     Route it: new review feedback → Address; `merged=True` or `state=closed` → Close.
   - Done when the state file matches the live PR and any change is routed.

3. **Address.** On new review feedback, resolve every open thread before pushing.

   - Fetch: `GET /repos/<owner/repo>/pulls/<pr>/comments`.
   - Apply each requested change and verify it (run the affected code or repo checks, not just
     eyeball it). A changed notebook *code* cell stales its run record — see reference.md.
   - Commit as a follow-up (never amend a reviewed PR), push over HTTPS, then reply to each
     thread citing the commit sha.
   - Refresh the state file so your own commit and replies do not re-alert on the next Poll.
   - Done when every thread has a change and a reply, and the follow-up is pushed with checks green.

4. **Close.** On merge or close, send a final PushNotification, then CronDelete the watcher job.
   Done when the job is cancelled.
