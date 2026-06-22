# Toy Models of Superposition

> Source: https://transformer-circuits.pub/2022/toy_model/index.html
> Collected: 2026-06-22
> Published: 2022-09-14

_Note: Captured from the transformer-circuits.pub article (now reachable after the host was added
to the network allowlist). Condensed faithfully, not verbatim. By Nelson Elhage, Tristan Hume,
Catherine Olsson, Nicholas Schiefer, Tom Henighan, Chris Olah, et al. (Anthropic)._

## Thesis

**Superposition** is the phenomenon where a neural network represents **more features than it has
dimensions (neurons)** by storing them as **non-orthogonal directions** that overlap. The paper
studies tiny, fully-understood toy models to characterize *when* superposition happens, *what
geometry* it takes, and *what it means* for interpretability. Superposition is the proposed
explanation for **polysemantic neurons** (neurons that fire for several unrelated things).

## Background concepts

- **Features as directions (linear representation hypothesis)**: features correspond to directions
  in activation space; a layer's activation is a sparse weighted sum of feature directions.
- **Privileged vs. non-privileged basis**: an architectural element like a ReLU/elementwise
  nonlinearity gives a layer a **privileged basis** (encourages features to align with neurons),
  whereas a pure linear space (e.g. the residual stream, word embeddings) has **no privileged
  basis**, so features need not align with coordinates. Superposition exploits the freedom of basis
  to pack extra features in.
- **Decomposability / monosemantic vs. polysemantic**: ideally each neuron is one feature
  (monosemantic); superposition forces neurons to be polysemantic.

## The toy model

A minimal **ReLU output model**: take `n` sparse synthetic features, each with an **importance**
weight and an activation **sparsity**, linearly project them down into a small **bottleneck** of
`m < n` dimensions via a weight matrix `W` (hidden = `W x`), then reconstruct with `ReLU(Wᵀ h + b)`.
The model is trained to recover the features. Whether the model dedicates a clean dimension to a
feature or crams several into superposition is read off from `WᵀW` (the Gram matrix of feature
directions).

## What drives superposition: sparsity and importance

Two knobs control the outcome:

- **Sparsity**: how rarely a feature is active. **Sparser features → more superposition** — because
  rarely-active features seldom collide, so the interference cost of sharing directions is low.
  With dense features the model instead represents only the most important ones orthogonally and
  drops the rest.
- **Importance**: how much a feature matters for the loss. The model preferentially gives important
  features dedicated, low-interference directions; less important features get squeezed into
  superposition or ignored.

As sparsity increases, the model transitions from "represent the top-`m` features cleanly, ignore
the rest" to "represent (almost) all `n` features in superposition, tolerating interference."

## Geometry of superposition

The feature directions arrange themselves into strikingly regular geometric configurations to
minimize interference. As sparsity rises, features organize into **uniform polytopes**:

- **antipodal pairs / digons** (two features sharing a line, pointing opposite ways),
- **triangles**, **pentagons**, **tetrahedra**, and other symmetric arrangements.

This is an instance of a **Thomson-problem**-like packing (spreading points/vectors on a sphere to
minimize overlap). The paper introduces **feature dimensionality** — the fraction of a dimension a
feature "uses" — and finds features stick to discrete, "sticky" fractional values (e.g. 1/2 for an
antipodal pair) corresponding to these polytopes. Transitions between configurations are **phase
changes**: sharp, physics-like jumps between representing vs. not representing a feature, or between
one geometric arrangement and another, as sparsity/importance cross thresholds.

## Computation in superposition

Superposition isn't only storage — networks can **compute** on features while they remain in
superposition. The **absolute-value model** shows a network computing `|x|` on many features packed
into fewer dimensions, extracting and transforming individual features through nonlinearity despite
the overlap. This implies real models likely do meaningful computation in superposed
representations, not just decode them at the end.

## Implications for interpretability

- **Polysemanticity is expected**, not a fluke: single neurons represent multiple unrelated features
  because of superposition.
- Interpretability cannot proceed neuron-by-neuron in general; one must consider feature
  **directions** and their **interference geometry**.
- It motivates methods to "unfold" superposition back into monosemantic features — directly setting
  up the dictionary-learning / sparse-autoencoder program of "Towards Monosemanticity."
- Open questions: whether large models are in a clean superposition regime, and whether we can
  enumerate or resolve their features.
