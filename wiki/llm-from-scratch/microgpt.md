# microGPT: A Minimal GPT in ~200 Lines

> Sources: Andrej Karpathy, 2026-02-12
> Raw: [microGPT](../../raw/llm-from-scratch/2026-02-12-microgpt.md)

## Overview

microGPT is Andrej Karpathy's ~200-line, dependency-free Python implementation of GPT
training and inference. It strips a language model down to its algorithmic essence: a
scalar autograd engine, a GPT-2-style Transformer, a training loop, and autoregressive
sampling. The guiding thesis is that these 200 lines contain the complete algorithm and
"everything else is just efficiency" — production systems scale this same core with
larger data, subword tokenizers, GPU tensor ops, and post-training, but the underlying
math is unchanged.

## Data and Tokenization

The model trains on a dataset of 32,000 names. A character-level tokenizer maps text to
a vocabulary of 27 tokens: the 26 lowercase letters plus a special beginning-of-sequence
(BOS) marker. This keeps the input pipeline trivial while still exercising the full
training and generation machinery.

## Scalar Autograd Engine

A custom `Value` class provides automatic differentiation. Each `Value` records the
computation graph that produced it, and backpropagation walks that graph applying the
chain rule. The approach is functionally identical to PyTorch's autograd but operates on
individual scalars rather than tensors — making the gradient mechanics fully transparent.
This is the same idea as the micrograd engine from Karpathy's "Neural Networks: Zero to
Hero" course (see [See Also](#see-also)).

## Architecture

The network is a GPT-2-style Transformer composed of:

- **Token and position embeddings** — turning token IDs and positions into vectors.
- **Multi-head attention** — the mechanism by which tokens communicate.
- **MLP blocks** — per-position feed-forward computation.
- **Residual connections** — to ease gradient flow through depth.
- **RMSNorm** — normalization for training stability.

## Training Loop

Each step: (1) tokenize documents, (2) run a forward pass that builds the computation
graph, (3) compute cross-entropy loss, (4) backpropagate gradients, and (5) update
parameters with the Adam optimizer under a decaying learning rate.

## Inference

Once trained, the frozen model generates new names by sampling tokens autoregressively
from its learned next-token distribution. A temperature parameter controls how
deterministic versus diverse the samples are.

## Why It Matters

microGPT is a pedagogical artifact: by fitting an end-to-end GPT into a readable file, it
makes every component — autograd, attention, normalization, optimization, sampling —
inspectable in one place. It serves as a compact reference for mechanistic study, since
nothing is hidden behind framework abstractions.

## See Also

- [Neural Networks: Zero to Hero](nn-zero-to-hero.md) — Karpathy's course that builds up
  the same components (micrograd, makemore, GPT) incrementally.
