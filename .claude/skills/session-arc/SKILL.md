---
name: session-arc
description: Drive a notebook from idea to shipped. Use when the user wants to build, explore, or implement a notebook concept in this repo (some sessions instead improve skills or scripts; this arc does not apply to those).
---

A notebook covers a single concept. Build it in phases, and do not load the next phase's skill
until the current phase is complete.

**Grill and branch.** Run the grilling skill on the notebook's objective and skeleton: one
pointed question at a time, anchored in the README curriculum (which item does this advance?).
Settle on a single concept — if more than one surfaces, take the first — and a skeleton of
labeled, empty cells (`# %% [A] Short description`) so cells stay referenceable. Cut a branch off
an up-to-date `main`, mirroring the notebook's `YYYY-MM-DD_topic` slug; the work lands via a PR,
never on `main`. Then invoke `session-build`.

`session-build` fills the skeleton cell by cell; `session-finalize` reviews the whole notebook
and ships it.
