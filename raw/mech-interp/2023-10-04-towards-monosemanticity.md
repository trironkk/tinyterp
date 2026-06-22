# Towards Monosemanticity: Decomposing Language Models With Dictionary Learning

> Source: https://transformer-circuits.pub/2023/monosemantic-features/index.html
> Collected: 2026-06-22
> Published: 2023-10-04

_Note: The transformer-circuits.pub host is now reachable (added to the network allowlist), but this
particular article exceeds the fetch tool's content-size limit (its inline interactive
visualizations push the page past 10 MB), so it could not be captured verbatim. This file is a
faithful condensation compiled from the article's abstract/summary and corroborated against a web
search; treat specific numbers as approximate. By Trenton Bricken, Adly Templeton, Joshua Batson,
Brian Chen, Adam Jermyn, Tom Conerly, Chris Olah, et al. (Anthropic)._

## Thesis

If features are stored in **superposition** (per "Toy Models of Superposition"), then individual
**neurons are polysemantic** and a poor unit of analysis. This paper shows that a **sparse
autoencoder** — a weak **dictionary-learning** method — can decompose a model's activations into a
larger set of **monosemantic, interpretable features** that the neurons themselves are not. It is
the first rigorous demonstration that dictionary learning can extract individually interpretable
features from a real (if tiny) language model.

## Setup

- **Model**: a **one-layer transformer** with a **512-neuron MLP** layer, trained on a large text
  corpus.
- **Method**: train a **sparse autoencoder (SAE)** on the **MLP activations**. The SAE has a wide
  hidden layer (an **overcomplete dictionary** — many more features than the 512 neurons, e.g.
  ~512 up to ~131,072 learned features across runs), a ReLU, and an **L1 sparsity penalty** that
  forces only a few features to be active at once. Each learned **feature** is a direction
  (dictionary element) in MLP-activation space.
- Result: the learned features are far more interpretable than the underlying neurons.

## Evidence that features are real and interpretable

The paper argues, with multiple lines of evidence, that features (not neurons) are the right unit:

1. **Specificity** — a feature reliably fires on, and only on, a human-describable context/concept
   (e.g., DNA sequences, Arabic script, base64 strings, specific tokens in specific roles).
2. **Causal / downstream effect** — pinning or ablating a feature changes the model's output in the
   way the feature's interpretation predicts (the feature *causes* the corresponding behavior, e.g.
   raising the probability of relevant next tokens).
3. **Neurons are not interpretable** — the same analysis applied to the 512 neurons fails: neurons
   are polysemantic, confirming superposition.
4. **Automated interpretability** — feature interpretations can be generated and scored
   automatically (using a larger LLM), showing the activations and effects match the proposed
   descriptions at scale.

## Key phenomena

- **Feature splitting** — as the dictionary grows (e.g. 512 → 4,096 → 16,384 → more features), a
  coarse feature splits into multiple finer, more specific features (e.g. a generic "math" feature
  splits into features for specific notations). Dictionary size acts as a knob on feature
  granularity; the number of features that can be pulled out vastly exceeds the neuron count.
- **Universality** — training SAEs on **two different models/runs** yields largely **the same
  features**, evidence that features are a real property of the data/computation, not artifacts of a
  single run.
- **Feature connectivity / "finite-state-automata"-like circuits** — features connect to one another
  (a feature's firing raises/lowers others downstream), forming small circuits; the authors show
  feature-level systems that behave like finite state automata for generating structured text.

## Claims and limitations

- **Headline claim**: dictionary learning with sparse autoencoders can **resolve superposition**
  into monosemantic features, giving an interpretable decomposition of a model that examining
  neurons cannot.
- **Limitations**: demonstrated only on a **one-layer** model; scaling SAEs to large multi-layer
  models, handling feature splitting/coverage, and dealing with very large dictionaries are left as
  challenges (taken up in later Anthropic scaling work). This paper is the proof of concept that
  launched the SAE wave in mechanistic interpretability.
