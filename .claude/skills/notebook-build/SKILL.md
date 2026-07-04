---
name: notebook-build
description: Fill a notebook's skeleton cell by cell in lockstep, ending on a review gate. Reached by notebook-arc once the skeleton is approved; not standalone.
---

**Build.** In lockstep: fill one skeleton cell, update its logbook, narrate one sentence, ask
"Questions?", wait. Never move ahead of the user. A bare "continue" advances exactly one cell —
it authorizes the next cell only, never leaving Build. The logbook is the cell's markdown
description — what it does, why this approach, and the steering that shaped it, written as it
lands, never backfilled.

**Review gate.** When the final cell is filled, do not roll forward. Stop, present the
completed notebook for review, and hold. Build is done only when every cell is filled, the
logbook is current, and the user has explicitly approved the finished notebook. A bare
"continue" on the last cell is not that approval — it names a next cell that no longer exists;
the go-ahead to leave Build must name the notebook or the finish. Only then invoke
`notebook-finalize`.

## Logbook prose

- Formal lab-notebook register, structured as a research lab notebook: objective, method,
  measured results, conclusions. No conversational narration.
- Document why, not what: non-obvious considerations, measured results, rejected
  alternatives. Never narrate what the next line of code does, and never justify imports.
- No content that duplicates the README; link or omit.
- No em-dashes anywhere, including cell-title separators — use colons, commas, or
  parentheses (en-dashes in numeric ranges are fine). Grep for `—` before finishing.
- Collaboration stays implicit: decisions appear as recorded rationale ("min/max error
  bars were evaluated and rejected"), never as attributed dialogue ("Steering:", "the
  user asked").
- Diagnostic prints use the self-documenting f-string form `print(f"{expr=}")` so every
  logged value is prefixed by the code that generated it; format specs go after the `=`
  (`f"{diff=:.2e}"`). Prose conclusions and tables are exempt.
