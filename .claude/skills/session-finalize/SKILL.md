---
name: session-finalize
description: Final phase of a notebook arc (see session-arc) — holistic review, then ship. Invoke only after the finished notebook is explicitly approved; not a standalone skill.
---

With the notebook approved, review it holistically before shipping. Dispatch subagents to read
the whole notebook end to end and report back: followups worth filing, and how to re-organize
cells for clarity. Work their suggestions through with the user — fold the clear wins in, file
the rest as curriculum items.

Then the local wrap-up: update the README curriculum (check off what the session covered, add
resources consulted to Resources, file newly surfaced concepts as new items), run `make record`,
and commit. All of this stays on the branch and is reversible.

Ship only after a separate, explicit go-ahead: push the branch, open the PR, and hand off to the
shepherd-pr skill. Approving the notebook, or the finalize work, is not approval to ship — the
user can say "not yet" and the branch simply waits.
