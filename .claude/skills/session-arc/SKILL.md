---
name: session-arc
description: Drive a build session from idea to finished notebook. Use when the user describes something they want to build, explore, or implement in this repo.
---

A session produces a single notebook covering a single concept. The arc runs in six phases,
and each downstream phase is its own skill — its instructions do not load until you invoke it.
Do not enter a phase before the previous phase's exit criterion is met; at the two gated seams
(review and ship) that also means the user has explicitly approved. Loading a phase is a
deliberate act, never a default — a bare "continue" does not load the next phase.

The "never move ahead of the user" discipline holds across every phase boundary, not just
within a phase — most sharply at the review gate (Build → Finalize) and the ship gate
(Finalize → Ship), where the next phase's instructions stay unloaded until you have explicit
approval to load them.

Arc map:

1. **Scope** — below.
2. **Branch** — below.
3. **Design** — `session-design`.
4. **Build** — `session-build`.
5. **Finalize** — `session-finalize` (local, reversible).
6. **Ship** — `session-ship` (outward-facing, hard to reverse).

## 1. Scope

Run the grilling skill: one pointed question at a time, anchored in the README curriculum —
which item does this session advance? Done when the goal is a single concept — if scoping
surfaces more than one, split it into multiple notebooks and take the first; if the concept is
clearly too large for about an hour of building, flag it and propose a cut.

## 2. Branch

Cut a session branch off an up-to-date `main`, named for the scoped concept (mirror the
notebook's `YYYY-MM-DD_topic` slug). Never build on `main` — the work lands via a PR. Done when
the session branch exists and is checked out. Then invoke `session-design` — do not stub any
cells here.
