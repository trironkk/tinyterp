# tinyterp

A personal, project-based mechanistic interpretability project. The user builds interp concepts
from the ground up to convince themselves of their own understanding.

## Session arc

Every session follows three steps:

1. **Scope** — The user describes what they want to build; we sharpen it with pointed questions
   until it's specific enough to decompose.
2. **Design** — Stub the notebook as labeled markdown cells paired with empty code cells; the
   user reviews and adjusts the structure before any code is written.
3. **Build** — One cell at a time, with a one-sentence narration and a pause for questions after
   each.

## Notebooks

| Notebook | Description |
|----------|-------------|
| [microgpt-wiki](./notebooks/microgpt-wiki.ipynb) | Recreate Karpathy's microGPT, trained on Wikipedia (Simple English) with a from-scratch BPE tokenizer and a GPT-2-style Transformer built from raw PyTorch tensor ops. |

## Knowledge base

Builds are grounded in a Karpathy-style LLM wiki, managed by the
[karpathy-llm-wiki](./vendor/karpathy-llm-wiki/) skill: raw sources land in `raw/`, and the skill
compiles them into linked articles under `wiki/` (see `wiki/index.md`). Ingest a source, query
what you know, or lint for gaps — the LLM writes the wiki, you read and ask questions.

See [CLAUDE.md](./CLAUDE.md) for Claude's operating manual.
