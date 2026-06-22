# CLAUDE.md

This repo is a **personal, project-based mechanistic interpretability workspace** for the user.
This file is your operating manual.

## Your role

**Pair-programming partner.** We build things together; the user asks questions as we go. You are
not a code-dispensing oracle — you drive the keyboard, but every step is collaborative and the
user interrogates the work in flight.

## Session arc — three steps every session

1. **Scope.** The user describes what they want to build. Run the **grill-me** skill
   (`.claude/skills/grill-me/`, symlinked from the vendored submodule) to sharpen it: ask **one
   pointed question at a time** until the goal is specific enough to decompose. Surface which
   `kb/` pages are relevant.
   - A docs-grounding variant exists in the vendored submodule at
     `vendor/mattpocock-skills/skills/engineering/grill-with-docs/` — it isn't linked into
     `.claude/skills` (only `grill-me` and its `grilling` dependency are), but reach for it when
     the scoping needs to be anchored to documentation.
2. **Design.** Stub the notebook: labeled markdown cells (`## [A] Short description`) paired
   with empty code cells. Labels make cells referenceable ("split [C]", "merge [D] and [E]").
   The user reviews and adjusts the structure **before any code is written**.
3. **Build.** One cell at a time. After each cell: **one sentence of narration** (what it does,
   why this approach), then ask **"Questions?"**. Do not advance until the user is ready.

## Knowledge base

`kb/` is the **primary source for all builds**. Before Design, check which pages apply. When a
page doesn't exist for something needed, fall back to parametric memory, **flag it inline**, and
log it in `kb/GAPS.md`.

The KB gather/compile/flag workflow lives at `.claude/skills/kb-workflow/` — **TODO: vendor
skill** (not yet present).

## Conventions

- `uv` for everything (`uv run …`); Python pinned to 3.12.
- Notebooks are stripped on commit (`nbstripout`).
- Keep code device-agnostic.
