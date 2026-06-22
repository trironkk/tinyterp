# A Mathematical Framework for Transformer Circuits

> Source: https://transformer-circuits.pub/2021/framework/index.html
> Collected: 2026-06-22
> Published: 2021-12-22

_Note: Captured from the transformer-circuits.pub article (now reachable after the host was added
to the network allowlist). Condensed faithfully, not verbatim. By Nelson Elhage, Neel Nanda,
Catherine Olsson, Tom Henighan, Nicholas Joseph, Ben Mann, Chris Olah, et al. (Anthropic). The
founding paper of the transformer Circuits thread._

## Goal and approach

The paper reverse-engineers small **attention-only** transformers (no MLP layers) with zero, one,
and two layers, as a first step toward understanding real models. The key move is to rewrite the
standard transformer computation into mathematically equivalent but more interpretable forms:
instead of "concatenate heads and multiply," decompose the model into explicit end-to-end paths
from input tokens to output logits. "Transformers have an enormous amount of linear structure; one
can learn a lot simply by breaking apart sums and multiplying together chains of matrices."

## The residual stream as a communication channel

Every component (token embedding, each attention head, MLP, unembedding) **reads** its input from
the residual stream by a linear projection and **writes** its result back by adding a linear
projection in. The residual stream is thus a shared bandwidth-limited communication channel; heads
and layers communicate by writing to and reading from **subspaces** of it. Because writes are
additive and reads are linear, the stream can be decomposed into separate "communication channels"
corresponding to paths through the model. Components in different subspaces don't interact;
later components can read what earlier ones wrote. (Linear structure also means some dimensions act
as "memory management," with heads clearing or moving information.)

## Attention heads are independent and additive

Each attention head can be understood as an **independent operation whose result is added into the
residual stream**. This contrasts with the usual "concatenate and project by W_O" formulation but
is mathematically equivalent and exposes the modular structure: heads act in parallel and sum.

## The QK and OV circuit decomposition

An attention head splits into two largely independent circuits:

- **QK (query–key) circuit**: governs the **attention pattern** — which positions attend to which.
  Controlled by the combined low-rank matrix **W_Q^T W_K** (written as a single bilinear form on
  pairs of token embeddings).
- **OV (output–value) circuit**: governs **what gets written** to the destination when a position
  is attended to — how an attended token affects the output. Controlled by the combined matrix
  **W_O W_V**.

Query, key, and value vectors are merely intermediate results; it is often cleaner to describe a
head directly via the two combined matrices W_Q^T W_K and W_O W_V, without reference to Q/K/V.
Tools used throughout: **path expansion** (expand the product/sum of matrices into additive terms,
each a readable input→logit function) and **virtual weights / virtual attention heads** (effective
weights between non-adjacent components produced by composition).

## Zero-layer transformers

With no attention, the model is just embedding → unembedding, so **W_U W_E** directly produces
**bigram statistics**: the most a zero-layer model can do is predict the next token from the current
token. The bigram table can be read straight off the weights.

## One-layer attention-only transformers

These are an **ensemble of a bigram model and "skip-trigram" models**. Path expansion gives a direct
("W_U W_E") bigram path plus, per head, an attention path. Each head implements **skip-trigrams** of
the form **[source]… [destination] → [prediction]**: it looks back to an earlier token and uses it
to boost predictions for the next token (a primitive form of in-context learning, e.g. completing
brackets/quotes, or "…keep… → keep"). Because the QK and OV circuits are **separable**, one-layer
heads have characteristic **bugs**: a head that copies "…A…B → B" can't help also boosting "…B…A →
A," producing systematic errors when the wrong skip-trigram fires.

## Two-layer transformers: composition

Two layers unlock **composition** between heads — three kinds, named by which input of the second
head receives the first head's output:

- **Q-composition**: first head's output feeds the second head's **queries**.
- **K-composition**: first head's output feeds the second head's **keys**.
- **V-composition**: first head's output feeds the second head's **values**.

Composition creates **virtual attention heads** and qualitatively more powerful inference-time
algorithms than one-layer models can express.

## Induction heads (the central discovery)

Two-layer attention-only models reliably form **induction heads**: a circuit implementing the
in-context rule **given …[A][B]…[A], predict [B]** — i.e., find the previous occurrence of the
current token and predict whatever followed it last time. Mechanism (a two-head circuit):

1. A **previous-token head** in layer 0 writes, into each position, information about the *preceding*
   token (it attends one step back and copies via its OV circuit).
2. An **induction head** in layer 1 uses **K-composition**: its keys read what the previous-token
   head wrote, so it matches the current token against positions whose *predecessor* equals the
   current token (**prefix matching**), then its OV circuit **copies** the token at the matched
   position into the output logits.

This is a genuinely different, more general in-context-learning algorithm than one-layer
skip-trigrams, and it generalizes to *novel* repeated sequences (including random tokens). The paper
argues induction heads are likely a major component of in-context learning in large models — the
thread picked up directly in the follow-up "In-Context Learning and Induction Heads."
