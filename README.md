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
| [01_microgpt-wiki](./notebooks/01_microgpt-wiki.ipynb) | Recreate Karpathy's microGPT, trained on Wikipedia (Simple English) with a from-scratch BPE tokenizer and a GPT-2-style Transformer built from raw PyTorch tensor ops. |

## Future work

A rough roadmap beyond the current notebook(s), in arc order — train, post-train, then
interpret at increasing depth:

- **Post-training reinforcement learning** — take a pretrained model through an RL post-training
  stage (reward modeling / preference optimization) to study how behavior shifts from the base
  model.
- **Mechanistic interpretability probes** — the first interpretability pass on a trained model:
  - *Attention pattern visualization* — render the per-head `(n_head, T, T)` maps; see which
    heads do local vs long-range, positional vs content attention.
  - *Induction heads* — search for the `[A][B]…[A]→[B]` copy circuit with repeated-token inputs;
    the flagship finding on small Transformers.
  - *Logit lens* — project the residual stream at each layer through the (tied) unembed to watch
    the next-token prediction sharpen layer by layer.
  - *Embedding structure* — nearest neighbors in the token embedding, and the geometry of the
    learned position embeddings.
- **Circuit tracing** — move from observing patterns to identifying end-to-end circuits: which
  components compose to implement a specific behavior.
- **Sparse autoencoders (SAEs)** — decompose activations into interpretable, monosemantic
  features to get past the polysemanticity of raw neurons.

## Knowledge base

Builds are grounded in a Karpathy-style LLM wiki, managed by the
[karpathy-llm-wiki](./vendor/karpathy-llm-wiki/) skill: raw sources land in `raw/`, and the skill
compiles them into linked articles under `wiki/` (see `wiki/index.md`). Ingest a source, query
what you know, or lint for gaps — the LLM writes the wiki, you read and ask questions.

See [CLAUDE.md](./CLAUDE.md) for Claude's operating manual.
