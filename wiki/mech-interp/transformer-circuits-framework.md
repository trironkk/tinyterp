# A Mathematical Framework for Transformer Circuits

> Sources: Nelson Elhage, Neel Nanda, Catherine Olsson, Chris Olah, et al. (Anthropic), 2021-12-22
> Raw: [A Mathematical Framework for Transformer Circuits](../../raw/mech-interp/2021-12-22-a-mathematical-framework-for-transformer-circuits.md)

## Overview

The founding paper of Anthropic's **transformer Circuits** thread takes the conceptual program of
[Zoom In / Circuits](circuits.md) — features, circuits, superposition — and works out the
*mathematics* of applying it to transformers. It studies **attention-only** transformers (no MLPs)
with zero, one, and two layers, rewriting the standard computation into equivalent but interpretable
forms: the **residual stream** as a communication channel, attention heads as **independent additive
operations**, and each head as a product of two circuits — a **QK (query–key)** circuit that decides
*where* to attend and an **OV (output–value)** circuit that decides *what* to write. The payoff is
the discovery of **induction heads**, a composed two-head circuit that performs general in-context
copying — the result the [In-Context Learning and Induction Heads](induction-heads-and-in-context-learning.md)
follow-up elevates into a theory of in-context learning.

## The residual stream as a communication channel

Every component — token embedding, each attention head, the MLP (when present), the unembedding —
**reads** from the residual stream via a linear projection and **writes** back by *adding* a linear
projection in. Because reads are linear and writes are additive, the stream decomposes into separate
"communication channels" — **paths** through the model from input tokens to output logits. Components
communicate through **subspaces**; those in disjoint subspaces don't interact, and later components
can read what earlier ones wrote. This linear structure is what makes the whole analysis tractable:

> Transformers have an enormous amount of linear structure; one can learn a lot simply by breaking
> apart sums and multiplying together chains of matrices.

The main tools are **path expansion** (expand a product/sum of weight matrices into additive terms,
each a readable token→logit function) and **virtual weights / virtual attention heads** (effective
weights between non-adjacent components produced by composition).

## Attention heads as QK and OV circuits

An attention head is best understood as an **independent operation whose output is added into the
residual stream** (equivalent to, but clearer than, the usual concatenate-and-project formulation).
Each head factors into two largely independent circuits:

- **QK circuit** — governs the **attention pattern** (which positions attend to which), controlled by
  the combined low-rank matrix **W_Qᵀ W_K**.
- **OV circuit** — governs **what is written** to the destination when a position is attended to,
  controlled by the combined matrix **W_O W_V**.

Q/K/V vectors are just intermediate results; it is often cleaner to describe a head directly by these
two combined matrices.

## Zero, one, and two layers

- **Zero-layer** (embed → unembed): can only model **bigrams**. The table `W_U W_E` is readable
  straight off the weights.
- **One-layer attention-only**: an **ensemble of a bigram model and per-head "skip-trigram" models**
  of the form **[source]…[destination] → [prediction]** — a primitive in-context behavior (e.g.
  completing brackets/quotes). Because QK and OV are separable, one-layer heads have characteristic
  **bugs**: a head copying "…A…B → B" can't avoid also boosting "…B…A → A."
- **Two-layer**: unlocks **composition** between heads, in three forms named by which input of the
  second head receives the first head's output — **Q-composition** (queries), **K-composition**
  (keys), and **V-composition** (values). Composition creates **virtual attention heads** and
  qualitatively more powerful algorithms.

## Induction heads

Two-layer attention-only models reliably form **induction heads**, implementing the in-context rule
**given …[A][B]…[A], predict [B]** — find the previous occurrence of the current token and predict
whatever followed it last time. The mechanism is a two-head circuit:

1. A **previous-token head** (layer 0) writes, at each position, information about the *preceding*
   token (attends one step back, copies via its OV circuit).
2. An **induction head** (layer 1) uses **K-composition**: its keys read what the previous-token head
   wrote, so it matches the current token against positions whose *predecessor* equals it (**prefix
   matching**), then its OV circuit **copies** the matched token into the logits.

Unlike one-layer skip-trigrams, this generalizes to **novel repeated sequences** (even random
tokens), making it a genuinely general in-context-learning algorithm — and the launching point for
the next paper.

## See Also

- [Induction Heads and In-Context Learning](induction-heads-and-in-context-learning.md) — the
  follow-up arguing induction heads explain most in-context learning at scale.
- [Mechanistic Interpretability: Features, Circuits, and Superposition](circuits.md) — the vision-era
  "Zoom In" essay whose vocabulary this paper ports to transformers.
- [Sparse Autoencoders & Monosemanticity](sparse-autoencoders.md) and
  [Superposition](superposition.md) — the MLP/feature side of the same thread.
- [Mechanistic Interpretability Tooling & Curriculum](tooling-and-curriculum.md) — TransformerLens
  and ARENA, which let you find induction heads in code.
- [The Transformer Architecture](../llm-from-scratch/transformer-architecture.md) — the structure
  (residual stream, attention heads, W_Q/W_K/W_V/W_O) this paper reverse-engineers.
