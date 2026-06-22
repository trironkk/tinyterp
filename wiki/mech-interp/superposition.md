# Superposition: Toy Models

> Sources: Nelson Elhage, Tristan Hume, Catherine Olsson, Chris Olah, et al. (Anthropic), 2022-09-14
> Raw: [Toy Models of Superposition](../../raw/mech-interp/2022-09-14-toy-models-of-superposition.md)

## Overview

**Superposition** is the phenomenon where a network represents **more features than it has neurons**
by storing them as **overlapping, non-orthogonal directions**. "Toy Models of Superposition" studies
tiny, fully-understood models to pin down *when* superposition occurs, *what geometry* it takes, and
*why it matters*. Its central payoff for [Zoom In / Circuits](circuits.md): superposition is the
mechanistic explanation for **polysemantic neurons**, and resolving it is what later motivates the
sparse-autoencoder program of [Towards Monosemanticity](sparse-autoencoders.md).

## Background concepts

- **Features as directions** (linear representation hypothesis): a layer's activation is a sparse
  weighted sum of feature direction vectors.
- **Privileged vs. non-privileged basis**: an elementwise nonlinearity (ReLU) gives a layer a
  **privileged basis** that nudges features to align with neurons; a purely linear space (the
  residual stream, word embeddings) has **no privileged basis**, so features need not align with
  coordinates. Superposition exploits this freedom of basis.
- **Monosemantic vs. polysemantic**: ideally one neuron = one feature; superposition forces neurons
  to be polysemantic.

## The toy model

A minimal **ReLU output model**: take `n` synthetic features, each with an **importance** weight and
an activation **sparsity**; linearly project them into a small **bottleneck** of `m < n` dimensions
(`h = Wx`); reconstruct with `ReLU(Wᵀh + b)`. Whether the model dedicates a clean dimension to a
feature or crams several into superposition is read off the Gram matrix `WᵀW` of feature directions.

## What drives superposition: sparsity and importance

- **Sparsity** — how rarely a feature is active. **Sparser features → more superposition**, because
  rarely-active features seldom collide, so the interference cost of sharing directions is low. With
  dense features the model instead keeps only the most important features (orthogonally) and drops
  the rest.
- **Importance** — how much a feature matters for the loss. Important features get dedicated,
  low-interference directions; less important ones get squeezed into superposition or ignored.

As sparsity rises, the model shifts from "represent the top-`m` features cleanly, ignore the rest" to
"represent (almost) all `n` features in superposition, tolerating interference."

## The geometry of superposition

Feature directions self-organize into strikingly regular **uniform polytopes** that minimize
interference — **antipodal pairs / digons**, **triangles**, **pentagons**, **tetrahedra**, and so on
— a **Thomson-problem**-like optimal packing of vectors on a sphere. The paper introduces **feature
dimensionality** (the fraction of a dimension a feature "uses"), which sticks to discrete values tied
to these polytopes. Transitions between configurations — and between representing vs. not representing
a feature — are sharp, physics-like **phase changes** as sparsity/importance cross thresholds.

## Computation in superposition

Superposition is not just storage: the **absolute-value model** shows a network *computing* `|x|` on
many features packed into fewer dimensions, extracting and transforming individual features through
nonlinearity despite the overlap. So real models likely compute on superposed representations, not
merely decode them at the end.

## Implications for interpretability

- **Polysemanticity is expected**, not an accident — single neurons represent multiple unrelated
  features.
- Interpretability cannot generally proceed neuron-by-neuron; it must reason about feature
  **directions** and **interference geometry**.
- It directly motivates methods to **unfold** superposition into monosemantic features — the
  dictionary-learning / sparse-autoencoder approach taken up in
  [Towards Monosemanticity](sparse-autoencoders.md).

## See Also

- [Sparse Autoencoders & Monosemanticity](sparse-autoencoders.md) — the dictionary-learning method
  that resolves superposition into interpretable features.
- [Mechanistic Interpretability: Features, Circuits, and Superposition](circuits.md) — where
  superposition first appears (vision models), now formalized here.
- [Mechanistic Interpretability Tooling & Curriculum](tooling-and-curriculum.md) — ARENA replicates
  these superposition results and trains sparse autoencoders in code.
- [A Mathematical Framework for Transformer Circuits](transformer-circuits-framework.md) — the
  attention-side companion in the same research thread.
