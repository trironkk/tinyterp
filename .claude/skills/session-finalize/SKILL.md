---
name: session-finalize
description: Phase 5 of a build session (see session-arc) — local, reversible wrap-up (curriculum, make record, commit). Invoke only after the finished notebook is explicitly approved; not a standalone skill. Does not push.
---

**Finalize (local, reversible).** With the notebook approved, do the local wrap-up: update the
README curriculum (check off what the session covered, add resources consulted to Resources,
file newly surfaced concepts as new items), run `make record`, and commit. All of this stays on
the session branch and is reversible. Done when the curriculum reflects the session and the
work is committed locally. Then stop and hold — do not push.

Shipping is a separate skill. Invoke `session-ship` only after a separate, explicit go-ahead to
ship — approving the notebook, or approving this finalize work, is not approval to ship. The
user can say "not yet" here and the branch simply waits, its ship instructions unloaded.
