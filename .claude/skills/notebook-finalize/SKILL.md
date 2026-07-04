---
name: notebook-finalize
description: Review the finished notebook holistically, then ship it. Reached by notebook-build once the notebook is approved; not standalone.
---

With the notebook approved, review it holistically before shipping. Dispatch subagents to read
the whole notebook end to end and report back: followups worth filing, and how to re-organize
cells for clarity. Work their suggestions through with the user — fold the clear wins in, file
the rest as curriculum items.

Then the local wrap-up: update the README curriculum (check off what the notebook covered, add
resources consulted to Resources, file newly surfaced concepts as new items), run `make record`,
and commit. All of this stays on the branch and is reversible.

Ship only after a separate, explicit go-ahead: invoke `open-pr`. Approving the notebook, or the
finalize work, is not approval to ship — the user can say "not yet" and the branch simply waits.
