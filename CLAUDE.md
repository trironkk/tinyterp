# CLAUDE.md

This repo is a personal project-based mechanistic interpretability workspace for the user.
This file is your operating manual.

## Your role

**Pair-programming partner.** We build things together; the user asks questions as we go. You
drive the keyboard, but every step is collaborative and the user interrogates the work in flight.

## Layout

- `notebooks/` — percent-format (`# %%`) Python files, date-prefixed (`YYYY-MM-DD_topic.py`),
  one topic/concept each. Plain `.py`, edited with ordinary tools, run cell-by-cell in VS
  Code's interactive window. No `.ipynb`, no output stripping, no notebook MCP tooling.
  **Notes live inline**: markdown cells (`# %% [markdown]`) carry the narrative — design
  decisions, measured results, rejected approaches, backlog — so each notebook is
  self-contained and stays current as the build evolves.
- `tinyterp/` — the package. Code graduates here from notebooks once it's settled (tokenizer,
  model, training entry points). Run modules with `uv run python -m tinyterp.<module>`.
- `artifacts/` (gitignored) — checkpoints, tokenizers, plots. Notebooks **load** trained models
  from here instead of retraining inline.
- `data/` (gitignored) — downloaded datasets and caches.

## Conventions

- `uv` for everything (`uv run …`); Python pinned to 3.12.
- Keep code device-agnostic.
