---
name: session-arc
description: Drive a build session from idea to finished notebook. Use when the user describes something they want to build, explore, or implement in this repo.
---

A session produces a single notebook covering a single concept. Five steps, in order — do not
start a step before the previous one's criterion is met.

1. **Scope.** Run the grilling skill: one pointed question at a time, anchored in the README
   curriculum — which item does this session advance? Done when the goal is a single concept —
   if scoping surfaces more than one, split it into multiple notebooks and take the first; if
   the concept is clearly too large for about an hour of building, flag it and propose a cut.
2. **Branch.** Cut a session branch off an up-to-date `main`, named for the scoped concept
   (mirror the notebook's `YYYY-MM-DD_topic` slug). Never build on `main` — the work lands via
   a PR. Done when the session branch exists and is checked out.
3. **Design.** Stub the skeleton: labeled cells (`# %% [A] Short description`) with empty
   bodies, so cells are referenceable ("split [C]", "merge [D] and [E]"). Done when the user
   has reviewed and approved the skeleton — no flesh before approval.
4. **Build.** In lockstep: fill one skeleton cell, update its logbook, narrate one sentence,
   ask "Questions?", wait. Never move ahead of the user. The logbook is the cell's markdown
   description — what it does, why this approach, and the steering that shaped it, written as
   it lands, never backfilled. Done when every skeleton cell is filled and the logbook is
   current.
5. **Conclude.** Update the README curriculum: check off what the session covered, add
   resources consulted to Resources, and file newly surfaced concepts as new items. Then push
   the session branch and open a PR, handing off to the shepherd-pr skill. Done when the
   curriculum reflects the session and the PR is open.

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
