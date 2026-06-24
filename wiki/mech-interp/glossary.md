# Mechanistic Interpretability Glossary (Neel Nanda)

> Sources: Neel Nanda, Unknown
> Raw: [Mech Interp Explainer & Glossary](../../raw/mech-interp/mech-interp-explainer-and-glossary.md)

## Overview

Neel Nanda's "Comprehensive Mechanistic Interpretability Explainer & Glossary" is a canonical
reference for the field's vocabulary, written to reduce "research debt" — the cost newcomers pay
re-deriving terminology and intuitions scattered across papers. It is a living document that defines
not just terms but the intuitions and connections behind them, spanning four areas: **core mech
interp concepts**, **ML fundamentals**, **transformer architecture**, and **attention heads**. This
page distills the definitions and points each cluster of terms at the wiki articles that develop
them in depth. It is the terminology onramp; the research-agenda onramp is
[200 Concrete Open Problems](open-problems.md).

## Core mechanistic interpretability vocabulary

The field is defined as **reverse engineering neural networks from the learned weights down to
human-interpretable algorithms**. The load-bearing concepts:

- **Feature** — a property of an input the model internally represents. The **features-as-directions
  hypothesis** holds that features are directions in **activation space**, accessible linearly; the
  stronger **features-as-neurons** hypothesis says each neuron *is* a feature.
- **Circuit** — the sub-part of a model that does some understandable computation, turning prior
  interpretable features into new ones; an **end-to-end circuit** spans inputs to outputs.
- **Basis vocabulary** — a **privileged basis** has coordinates with inherent meaning (typically
  forced by an elementwise nonlinearity like a ReLU), versus a **bottleneck dimension** with no
  privileged basis. An **interpretable basis** is one whose directions are each a feature.
- **Universality, motifs, equivariance** — the hope that the same circuits/features (**motifs**)
  recur across models (**universality**), and that analogous neurons form **families** so
  understanding one transfers.
- **Interventions** — **editing/intervening on an activation** (run, stop, edit, resume) and
  **pruning** (zeroing a neuron) are the basic causal tools; **causal scrubbing** formalizes testing
  whether a **computational subgraph** explains behavior.
- **Microscope AI** and **enumerative safety** — the aspirational payoffs: extract knowledge from a
  superhuman model rather than deploy it; understand *every* feature to catch undesired behavior.

These are the terms developed in [Mechanistic Interpretability: Features, Circuits, and
Superposition](circuits.md).

## Superposition vocabulary

A self-contained cluster for the phenomenon where a model represents **more than n features in n
dimensions** using non-orthogonal directions (an **overcomplete basis**):

- **Polysemanticity** (a neuron firing for multiple features) vs. a **monosemantic neuron**.
- **Bottleneck-dimension superposition** (storage, e.g. in the residual stream) vs. **neuron
  superposition** (computation across >n features).
- Geometry: **antipodal pairs**, **tegum products**, **correlated/anti-correlated features**, **local
  almost-orthogonal bases**, and the **asymmetric superposition motif** with cleanup neurons.
- The **feature importance curve** that drives how much superposition a model maintains.

These map onto [Superposition: Toy Models](superposition.md) and the dictionary-learning response in
[Sparse Autoencoders & Monosemanticity](sparse-autoencoders.md).

## ML fundamentals

A refresher layer so the mech-interp terms have ground to stand on: tensors, activations,
weights/biases, MLPs, activation functions (**ReLU**, **GELU**), softmax/logits, and losses
(cross-entropy, MSE). The **training-dynamics** sub-cluster is the most interpretability-relevant:
**phase transitions** (training-/data-/model-wise), **grokking** (memorize then suddenly generalize,
via a **memorisation circuit**, **circuit formation**, and **clean-up**), **scaling laws**,
**emergent capabilities**, **deep double descent**, and **path (in)dependence**. **Chinchilla** is
flagged as the compute-optimal-scaling result. Grokking and phase changes recur as worked examples in
the circuits literature.

## Transformer architecture

The glossary's transformer section doubles as a compact spec, much of it pinned to GPT-2 Small
(d_model 768, d_mlp 3072, d_head 64, n_heads 12, n_layers 12):

- **Residual stream** as the central object — the running sum of embedding plus every layer's
  output, framed as **shared bandwidth** / a memory bottleneck that all information flows through via
  **skip connections**.
- **Attention layers** move information *between* positions; **MLP layers** process *within* a
  position.
- Tokenization (**BPE**, BOS/EOS/PAD tokens), embeddings/unembeddings (and **tied embeddings**),
  positional schemes (**learned absolute**, **sinusoidal**, **shortformer**, **rotary/RoPE**), and
  **LayerNorm** (center, normalise, scale, translate).

This is the same architecture built up in [The Transformer
Architecture](../llm-from-scratch/transformer-architecture.md).

## Attention-head vocabulary

The QK/OV machinery, named the way circuits papers use it: **query/key/value** vectors and their
weights (W_Q, W_K, W_V) and biases; **attention score** (key·query) → **row-wise softmax** →
**attention pattern**; the **mixed value** → **result vector** via W_O added back to the residual
stream; **source/destination** positions and **information routing**. Two named circuits anchor the
vocabulary: the **previous-token head** and the **induction head** (detects and continues repeated
subsequences). These are exactly the objects formalized in [A Mathematical Framework for Transformer
Circuits](transformer-circuits-framework.md) and [Induction Heads and In-Context
Learning](induction-heads-and-in-context-learning.md).

## See Also

- [Mechanistic Interpretability: Features, Circuits, and Superposition](circuits.md) — the concepts
  the core vocabulary names.
- [Superposition: Toy Models](superposition.md) and
  [Sparse Autoencoders & Monosemanticity](sparse-autoencoders.md) — the superposition cluster.
- [A Mathematical Framework for Transformer Circuits](transformer-circuits-framework.md) and
  [Induction Heads and In-Context Learning](induction-heads-and-in-context-learning.md) — the
  attention-head and circuit terms in their original setting.
- [The Transformer Architecture](../llm-from-scratch/transformer-architecture.md) — the architecture
  the transformer/attention sections describe.
- [200 Concrete Open Problems in Mechanistic Interpretability](open-problems.md) and
  [Tooling & Curriculum: TransformerLens and ARENA](tooling-and-curriculum.md) — Nanda's companion
  research-agenda and practical onramps.
