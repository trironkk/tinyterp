---
name: notebook-build
description: Fill a notebook's approved skeleton cell by cell in lockstep. Reached by notebook-arc.
---

**Build.** In lockstep: fill one skeleton cell, update its logbook, narrate one sentence, ask
"Questions?", wait. A bare "continue" advances exactly one cell and never more. The logbook is
the cell's markdown description — what it does, why this approach, and the steering that shaped
it, written as it lands, never backfilled.

**Review gate.** The last cell has no next cell for "continue" to name, so do not roll into
shipping. Stop, present the completed notebook, and hold. Build is done only when every cell is
filled, the logbook is current, and the user has explicitly approved the finished notebook — an
approval that names the notebook or the finish, not the next cell. Only then invoke
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
