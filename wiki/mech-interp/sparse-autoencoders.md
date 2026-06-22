# Sparse Autoencoders & Monosemanticity (Dictionary Learning)

> Sources: Trenton Bricken, Adly Templeton, Joshua Batson, Chris Olah, et al. (Anthropic), 2023-10-04
> Raw: [Towards Monosemanticity: Decomposing Language Models With Dictionary Learning](../../raw/mech-interp/2023-10-04-towards-monosemanticity.md)

## Overview

If features live in [superposition](superposition.md), then individual **neurons are polysemantic**
and a poor unit of analysis. "Towards Monosemanticity" shows that a **sparse autoencoder (SAE)** — a
weak **dictionary-learning** method — can decompose a model's activations into a *larger* set of
**monosemantic, interpretable features** that the neurons themselves are not. It is the first
rigorous demonstration that dictionary learning can extract individually interpretable features from
a real (if tiny) language model, and it launched the SAE wave in mechanistic interpretability.

> Note: the source article exceeds the fetch tool's size limit, so the raw file is a faithful
> condensation from the abstract/summary plus a web search rather than a verbatim capture; treat
> specific counts as approximate.

## Setup

- **Model**: a **one-layer transformer** with a **512-neuron MLP**, trained on text.
- **Method**: train a **sparse autoencoder** on the **MLP activations**. The SAE has a wide hidden
  layer — an **overcomplete dictionary** of many more features than the 512 neurons (from ~512 up to
  tens of thousands across runs) — a ReLU, and an **L1 sparsity penalty** so only a few features fire
  at once. Each learned **feature** is a dictionary direction in MLP-activation space.

The learned features turn out to be far more interpretable than the underlying neurons.

## Why the features are real

1. **Specificity** — a feature fires on, and essentially only on, a human-describable concept (e.g.
   DNA sequences, Arabic script, base64, a token in a specific role).
2. **Causal effect** — pinning or ablating a feature changes the output as its interpretation
   predicts; the feature *causes* the corresponding behavior.
3. **Neurons fail the same tests** — the 512 neurons are polysemantic and not interpretable,
   confirming superposition.
4. **Automated interpretability** — feature descriptions can be generated and scored automatically
   with a larger LLM, validating the interpretations at scale.

## Key phenomena

- **Feature splitting** — as the dictionary grows (e.g. 512 → 4,096 → 16,384 → more), coarse features
  split into finer, more specific ones. Dictionary size is a knob on feature granularity, and the
  number of recoverable features vastly exceeds the neuron count.
- **Universality** — SAEs trained on **two different models/runs** recover largely the **same**
  features, evidence that features are real properties of the data/computation, not run artifacts.
- **Feature circuits** — features connect to one another (one firing raises/lowers others
  downstream), forming small circuits; some feature systems behave like **finite state automata** for
  generating structured text.

## Claims and limitations

- **Headline claim**: sparse-autoencoder dictionary learning can **resolve superposition** into
  monosemantic features, giving an interpretable decomposition that neuron-level analysis cannot.
- **Limitations**: shown only on a **one-layer** model; scaling SAEs to large multi-layer models,
  handling feature splitting/coverage, and very large dictionaries are left for later work. This is
  the proof of concept behind today's SAE-based interpretability.

## See Also

- [Superposition: Toy Models](superposition.md) — the problem (features packed into too few neurons)
  this method resolves.
- [Mechanistic Interpretability Tooling & Curriculum](tooling-and-curriculum.md) — ARENA trains
  sparse autoencoders to recover features from superposition; TransformerLens's SAE support moved to
  SAELens.
- [Mechanistic Interpretability: Features, Circuits, and Superposition](circuits.md) — the founding
  "features as directions" framing that monosemanticity vindicates.
- [A Mathematical Framework for Transformer Circuits](transformer-circuits-framework.md) — the
  attention-side companion in the same research thread.
