---
name: delegating-to-gemini
description: Use when you want to offload a self-contained subtask to Gemini via the `agy` CLI — bulk or mechanical work, a second opinion, web search/synthesis, or anything that would burn context better spent in the main session.
---

# Delegating to Gemini

## Overview

`agy` is a CLI for Google's Gemini agent. Use `agy --print` to hand a **self-contained**
subtask to Gemini 3.5 Flash and get the answer back as text — saving your own context for
work that needs it. Delegation is **one-shot and stateless**: Gemini sees only the prompt
you pass, never your conversation.

## When to use

- Bulk/mechanical work (reformatting, extracting, drafting boilerplate).
- A cheap second opinion or sanity check on an approach.
- Web search + synthesis you don't want to do inline.
- Anything large whose *result* you need but whose *process* would bloat context.

**Not for:** work needing your live conversation state, or judgment calls that are the
load-bearing part of the current task. Don't delegate the thing you're supposed to learn.

## Quick reference

| Flag | Purpose |
|---|---|
| `--print` / `-p` | Single-shot, non-interactive; prints the response. Prompt is a **positional arg**. |
| `--model "<name>"` | Exact name from `agy models`, e.g. `Gemini 3.5 Flash (Low\|Medium\|High)`. |
| `--print-timeout 15m` | Raise from the 5m default for long jobs. |
| `--add-dir <path>` | Give Gemini read access to a directory (repeatable). |
| `-c` / `--conversation <ID>` | Continue the last / resume a specific conversation. |
| `--dangerously-skip-permissions` | Auto-approve tool permissions for unattended runs. |

Tiers are reasoning effort: **Low** for cheap bulk, **Medium** as the default workhorse,
**High** for harder reasoning.

## Pattern

```bash
# Run from a neutral cwd (e.g. /tmp) so this repo's AGY.md doesn't reframe Gemini.
agy --print --model "Gemini 3.5 Flash (Medium)" \
  "Summarize the key claims in the following text as a bullet list:\n\n<text>"
```

If the task needs repo files, run from the repo (or pass `--add-dir`) and make the prompt
fully self-contained — Gemini has no other context.

## Common mistakes

- **Unrecognized `--model` value silently falls back** to the default Flash (no error).
  Copy names verbatim from `agy models`.
- **Piping the prompt via stdin prints help.** Pass the prompt as a positional argument.
- **Running inside this repo loads `AGY.md`**, so Gemini answers as the interp "teacher."
  Run from a neutral directory, or override with an explicit, self-contained prompt.
- **Long delegations cut off at 5m.** Set `--print-timeout`.
- **Assuming shared context.** It's stateless — restate everything Gemini needs in the prompt.
