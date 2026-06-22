---
name: kb-workflow
description: >
  Author and maintain pages in kb/ for this mechanistic-interpretability
  workspace. Use whenever a session needs to create, extend, or correct a kb/
  page, or when a build hits a concept the kb doesn't yet cover. Enforces a
  gather → compile → flag loop so pages stay grounded in supplied sources and
  every uncovered concept is logged instead of hallucinated.
---

# kb-workflow

The `kb/` directory is the **primary source for every build** in this workspace.
Pages are **Karpathy-style**: ground-up derivations with small, runnable,
device-agnostic code — not prose summaries, not paper abstracts. This skill is
the loop for producing and maintaining them without inventing facts.

## Core rule

**Nothing goes on a page that isn't in the supplied sources.** If a page needs
something the sources don't cover, it is a **gap** — flag it, don't fill it from
parametric memory and present it as fact. A page that quietly papers over a gap
is worse than a page with an honest `> ⚠️ GAP` marker, because future builds
trust the kb.

## The loop: gather → compile → flag

Run these three phases per page. They are ordered; don't compile before you've
gathered, don't ship before you've flagged.

### 1. gather

Pull only from the **raw sources the user provides** for this page (papers,
lecture notes, code, links, repo file paths).

- **Quote and extract — do not invent.** Capture the exact equations,
  definitions, and code shapes you'll derive from.
- Note the **provenance** of each piece (which source, which section). A page
  should be traceable back to its inputs.
- If the user names a repo path or link instead of pasting, read it first; if it
  can't be reached, that's a gap (see phase 3), not a license to guess.
- Keep gathered material per-concept. **One page = one concept.** If the sources
  bundle several concepts, that's a signal to split into several pages.

### 2. compile

Turn gathered material into a Karpathy-style page:

- **Derive from first principles.** Build each idea up; don't state results and
  move on. Show the small step that makes the next step obvious.
- **Minimal dependencies.** Prefer plain Python / `torch` / `numpy`. Every
  snippet should be runnable on its own with `uv run`.
- **Runnable, device-agnostic code.** No hardcoded `cuda`/`mps`; pick the device
  programmatically. Snippets should print or assert something a reader can check.
- **One concept per page.** Cross-link sibling pages rather than re-deriving.
- Match the README's framing: code that convinces the reader of the idea, not
  text that summarizes a paper.

Page shape (a guideline, not a rigid template):

```markdown
# <Concept>

<one-paragraph framing: what this is and why it matters for interp>

## Intuition / first principles
<the smallest version of the idea, derived step by step>

## Code
<minimal runnable snippet(s); device-agnostic; prints/asserts a checkable result>

## Where this shows up
<links to sibling kb/ pages and to the sources>

## Sources
<provenance: the supplied sources this page was built from>
```

### 3. flag

Before the page is done, reconcile it against the sources:

- For **anything the page needs but the sources don't cover**, do BOTH:
  1. **Inline marker** at the exact spot, so a reader sees it in context:
     ```markdown
     > ⚠️ GAP: <what's missing and what the page does provisionally, if anything>
     ```
  2. **One-line entry** appended to `kb/GAPS.md`:
     ```
     - [<page-filename>] <what's missing> — <why it matters / source needed>
     ```
- Never replace a gap with confident prose. If you must show *something*
  provisional to keep the page coherent, label it as unverified right there.
- If gathering turned up a **contradiction** between sources, that's also a gap:
  record both readings and flag which one the page assumed.

## Definition of done for a page

- [ ] Every claim traces to a supplied source (or is marked as a gap).
- [ ] Code is minimal, runnable with `uv run`, and device-agnostic.
- [ ] Page covers exactly one concept and links its siblings.
- [ ] All gaps appear both inline (`> ⚠️ GAP`) and in `kb/GAPS.md`.
- [ ] Per the session arc: one sentence of narration, then ask "Questions?" and
      stop before starting the next page.

## What this skill does NOT do

- It does not write pages from memory when sources are absent. **No sources →
  no page; the request to write it becomes the gap.**
- It does not commit or push. Authoring is separate from version control; only
  do that when the user explicitly asks.
