---
name: session-arc
description: Drive a build session from idea to finished notebook. Use when the user describes something they want to build, explore, or implement in this repo.
---

A session produces a single notebook covering a single concept. Four steps, in order — do not
start a step before the previous one's criterion is met.

1. **Scope.** Run the grilling skill: one pointed question at a time, anchored in the README
   curriculum — which item does this session advance? Done when the goal is a single concept —
   if scoping surfaces more than one, split it into multiple notebooks and take the first; if
   the concept is clearly too large for about an hour of building, flag it and propose a cut.
2. **Design.** Stub the skeleton: labeled cells (`# %% [A] Short description`) with empty
   bodies, so cells are referenceable ("split [C]", "merge [D] and [E]"). Done when the user
   has reviewed and approved the skeleton — no flesh before approval.
3. **Build.** In lockstep: fill one skeleton cell, update its logbook, narrate one sentence,
   ask "Questions?", wait. Never move ahead of the user. The logbook is the cell's markdown
   description — what it does, why this approach, and the steering that shaped it, written as
   it lands, never backfilled. Done when every skeleton cell is filled and the logbook is
   current.
4. **Conclude.** Update the README curriculum: check off what the session covered, add
   resources consulted to Resources, and file newly surfaced concepts as new items. Done when
   the curriculum reflects the session.
