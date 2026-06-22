# Mechanistic Interpretability Tooling & Curriculum: TransformerLens and ARENA

> Sources: Neel Nanda et al. (TransformerLens), Unknown; Callum McDougall et al. (ARENA), Unknown
> Raw: [TransformerLens](../../raw/mech-interp/transformerlens.md); [ARENA Curriculum](../../raw/mech-interp/arena-curriculum.md)

## Overview

If [Zoom In / Circuits](circuits.md) supplies the *concepts* of mechanistic interpretability, this
page covers the *practical stack* for doing it on transformer language models: **TransformerLens**
(the library that exposes and lets you intervene on a model's internals) and **ARENA** (the
exercise-driven curriculum that teaches the techniques on top of it). Together they are the standard
on-ramp — and both emphasize that mech interp is accessible: many open problems are tractable with a
small model in a Colab notebook, no large compute required.

## TransformerLens

TransformerLens (created by Neel Nanda; maintained by Bryce Meyer and Jonah Larson) is "a library
for doing mechanistic interpretability of GPT-2 style language models," whose goal it states as
"take a trained model and reverse engineer the algorithms the model learned during training from its
weights." It loads **9,000+ open-source models across 50+ architecture families** and exposes their
internal **activations**.

The core capability is **hooks**: you can **cache** any internal activation, and register functions
to **edit, remove, or replace** activations as the model runs — the machinery behind activation
patching, ablations, and direct logit attribution.

```python
from transformer_lens.model_bridge import TransformerBridge
bridge = TransformerBridge.boot_transformers("gpt2", device="cpu")
logits, activations = bridge.run_with_cache("Hello World")
```

The v3 `TransformerBridge` preserves raw HuggingFace weights by default (logits match HF); the
legacy `HookedTransformer` (now deprecated) instead folds LayerNorm and centers weights, and is
recoverable via `enable_compatibility_mode()`. Notes: sparse-autoencoder support
(`HookedSAETransformer`) moved out to **SAELens** in v2.0; experimental bridges exist for Mamba/SSM
models. The interface was inspired by Anthropic's internal **Garcon** tool; Nanda built it after
leaving Anthropic's interpretability team, frustrated by the lack of open internals tooling.

TransformerLens has powered a wide range of results — e.g. progress measures for **grokking**,
sparse probing ("Finding Neurons in a Haystack"), automated circuit discovery (ACDC), Othello-GPT's
linear world model, and replications of the **induction-heads phase change**.

## ARENA curriculum

ARENA (Alignment Research Engineer Accelerator), by Callum McDougall and collaborators, is a hands-on
curriculum of exercises and solutions, much of it written in TransformerLens.

- **Chapter 0 — Fundamentals:** convolutions from scratch, building/finetuning a ResNet, W&B
  hyperparameter sweeps, hand-written backprop, GANs/VAEs.
- **Chapter 1 — Transformer Interpretability:** build a transformer from scratch; use TransformerLens
  to **locate induction heads in a 2-layer model**; find the **Indirect Object Identification (IOI)**
  circuit in GPT-2 small (direct logit attribution, activation patching, path patching); interpret
  toy-task models (bracket matching, modular arithmetic); **replicate Anthropic's superposition
  results and train sparse autoencoders** to recover features from superposition; and steer GPT-2-XL
  with **steering vectors**. Only the first two sets are compulsory; the rest are
  prerequisite-free extensions.
- **Chapter 2 — Reinforcement Learning:** bandits, DQN, PPO, and RLHF on transformers.
- **Chapter 3 — LLM Evaluations** (with Apollo Research): designing MCQ evals, the UK AISI Inspect
  library, and building LLM agents (ReAct/reflexion).
- **Chapter 4 — Alignment Science:** forthcoming.

## How this connects to the concepts

ARENA's flagship exercises directly operationalize the [circuits](circuits.md) ideas on transformers:
**induction heads** (a two-head circuit that copies repeated sequences — the canonical first
transformer circuit), the **IOI circuit** (a worked real-model circuit), and **sparse autoencoders**
(the leading approach to resolving **superposition** into monosemantic features). The conceptual
Anthropic papers behind these (the transformer-circuits.pub thread) are not yet ingested due to the
current GitHub-only network policy; TransformerLens and ARENA are the route to engaging with their
results in code today.

## See Also

- [Mechanistic Interpretability: Features, Circuits, and Superposition](circuits.md) — the concepts
  these tools put into practice.
- [nanoGPT & nanochat](../llm-from-scratch/nanogpt-and-nanochat.md) and
  [The Transformer Architecture](../llm-from-scratch/transformer-architecture.md) — building and
  understanding the models that TransformerLens then instruments.
