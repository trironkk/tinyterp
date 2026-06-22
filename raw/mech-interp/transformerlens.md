# TransformerLens

> Source: https://github.com/TransformerLensOrg/TransformerLens
> Collected: 2026-06-22
> Published: Unknown

_Note: Captured from the repository README via GitHub; condensed faithfully, not verbatim.
Created by Neel Nanda (and Joseph Bloom); maintained by Bryce Meyer and Jonah Larson._

## What it is

TransformerLens is "a library for doing
[mechanistic interpretability](https://distill.pub/2020/circuits/zoom-in/) of GPT-2 Style language
models." The stated goal of mechanistic interpretability is "to take a trained model and reverse
engineer the algorithms the model learned during training from its weights."

The library lets you load **9,000+ open-source language models across 50+ architecture families**
and exposes the model's internal activations. You can **cache** any internal activation, and add
hook functions to **edit, remove, or replace** activations as the model runs — the core capability
needed for activation patching, ablations, and circuit analysis.

## Usage

```python
from transformer_lens.model_bridge import TransformerBridge

# Load a model (e.g. GPT-2 Small)
bridge = TransformerBridge.boot_transformers("gpt2", device="cpu")

# Run the model and get logits and activations
logits, activations = bridge.run_with_cache("Hello World")
```

`TransformerBridge` is the recommended 3.0 path (supports 9,000+ models). By default it preserves
raw HuggingFace weights, so logits/activations match HF — *not* the legacy `HookedTransformer`,
which folds LayerNorm and centers weights by default. Call `bridge.enable_compatibility_mode()`
for HookedTransformer-equivalent numerics. The legacy `HookedTransformer.from_pretrained` API
remains but is deprecated. Gated models (Llama, Mistral, Gemma) require `HF_TOKEN`.

## Research enabled (gallery, selected)

- Progress Measures for Grokking via Mechanistic Interpretability (Nanda et al., ICLR Spotlight 2023)
- Finding Neurons in a Haystack: Case Studies with Sparse Probing (Gurnee, Nanda et al.)
- Towards Automated Circuit Discovery for Mechanistic Interpretability (Conmy et al.)
- Othello-GPT's linear emergent world representation (Nanda)
- A circuit for Python docstrings in a 4-layer attention-only transformer (Heimersheim, Janiak)
- A Toy Model of Universality (Chughtai, Chan, Nanda, ICML 2023)
- An induction-heads phase-change replication of Anthropic's "In-Context Learning and Induction
  Heads" using TransformerLens

## Getting started in mechanistic interpretability

The README frames mech interp as "a very young and small field" with "a lot of open problems" and
a low bar for entry, and points to key resources (all on neelnanda.io / ARENA):

- A Guide to Getting Started in Mechanistic Interpretability
- **ARENA** Mechanistic Interpretability Tutorials (Callum McDougall) — written in TransformerLens,
  with exercises and solutions; notable units: Coding GPT-2 from scratch, Intro to Mech Interp &
  TransformerLens (via induction heads), and Indirect Object Identification (covering direct logit
  attribution, activation patching, and path patching)
- Mech Interp Paper Reading List
- **200 Concrete Open Problems in Mechanistic Interpretability**
- A Comprehensive Mechanistic Interpretability Explainer (the glossary of jargon)
- Neel Nanda's YouTube channel (paper walkthroughs and research walkthroughs)

## Notes

- **HookedSAETransformer** was removed in v2.0; sparse-autoencoder functionality moved to
  [SAELens](https://github.com/jbloomAus/SAELens).
- Experimental bridge adapters exist for **Mamba-1/Mamba-2 (SSMs)**, with bit-for-bit HF-equivalent
  forward passes and hook-based introspection.
- The library's interface was "heavily inspired by Anthropic's Garcon tool" (credit to Nelson
  Elhage and Chris Olah). Neel Nanda wrote it after leaving the Anthropic interpretability team,
  frustrated by the lack of open tooling to "dig into model internals and reverse engineer how they
  work." A noted upside of mech interp: "you don't need large models or tons of compute" — many
  open problems are tractable with a small model in a Colab notebook.

## Cite

```bibtex
@misc{nanda2022transformerlens,
  title = {TransformerLens},
  author = {Neel Nanda and Joseph Bloom},
  year = {2022},
  howpublished = {\url{https://github.com/TransformerLensOrg/TransformerLens}},
}
```
