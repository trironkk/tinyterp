---
name: notebook-arc
description: Drive a notebook from idea to shipped. Use when the user wants to build or explore a notebook concept here; skill or script work is out of scope.
---

A notebook covers a single concept. Build it in phases, and do not load the next phase's skill
until the current phase is complete.

**Grill and branch.** Run the grilling skill on the notebook's objective and skeleton: one
pointed question at a time, anchored in the README curriculum (which item does this advance?).
Settle on a single concept — if more than one surfaces, take the first — and a skeleton of
labeled, empty cells (`# %% [A] Short description`) so cells stay referenceable. Then invoke
`cut-branch`, naming the branch for the notebook's `YYYY-MM-DD_topic` slug, and `notebook-build`.

`notebook-build` fills the skeleton cell by cell; `notebook-finalize` reviews the whole notebook
and ships it.
