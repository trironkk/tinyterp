---
name: notebook-finalize
description: Review the finished notebook holistically, then ship it. Reached by notebook-build once the notebook is approved.
---

With the notebook approved, review it holistically before shipping. Dispatch subagents to read
the whole notebook end to end and report back: followups worth filing, and how to re-organize
cells for clarity (one concept per cell; split a cell doing two things), and where the prose or
the notebook's own TODO list can be thinned to a few high-value items. Work
their suggestions through with the user — fold the clear wins in, file the rest as curriculum
items.

Then the local wrap-up: update the README curriculum (check off what the notebook covered, add
resources consulted to Resources, file newly surfaced concepts as new items), run `make record`,
and commit. This commit is deliberately un-gated: it is local to the branch and reversible.

Finalize is done only when every subagent suggestion is folded in or filed, the README reflects
the notebook, `make record` has run, and the wrap-up is committed. Present that finished state,
then ship only on a go-ahead given against it — invoke `open-pr`. A "ship it" said back at the
build review does not carry through here; the user can still say "not yet" and the branch waits.
