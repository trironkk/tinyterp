# The Transformer Architecture

> Sources: Jay Alammar, 2018-06-27; Harvard NLP (Rush et al.), 2018-04 / 2022-04
> Raw: [The Illustrated Transformer](../../raw/llm-from-scratch/2018-06-27-illustrated-transformer.md); [The Annotated Transformer](../../raw/llm-from-scratch/2022-04-the-annotated-transformer.md)

## Overview

The Transformer (Vaswani et al., "Attention Is All You Need," 2017) is the architecture underlying
modern LLMs. It replaces recurrence and convolution with **attention**, which lets every position
attend to every other position in parallel — the property that makes large-scale training practical.
This article synthesizes two canonical explainers: Jay Alammar's **Illustrated Transformer** (visual
intuition) and Harvard NLP's **Annotated Transformer** (a line-by-line PyTorch implementation with
the paper's text interleaved). Both describe the original encoder-decoder model; GPT-style LLMs use
the decoder half of this design.

> The primary source, Vaswani et al. 2017 (arXiv:1706.03762), is not yet ingested as its own raw
> file (the host is unreachable under the current GitHub-only network policy). The two explainers
> here faithfully reproduce its content.

## Encoder-decoder structure

The model is an **encoder stack** and a **decoder stack** (the paper uses N=6 layers each). Each
**encoder layer** has two sub-layers: multi-head **self-attention** followed by a **position-wise
feed-forward network**. Each **decoder layer** adds a third sub-layer — **encoder-decoder
attention** — between them, letting the decoder attend over the encoder's output. Every sub-layer is
wrapped in a **residual connection** followed by **layer normalization**: `LayerNorm(x + Sublayer(x))`.

Words become vectors via a learned **embedding** (e.g., dimension 512). A key structural property:
within self-attention, positions have dependencies, but the feed-forward layer processes each
position independently — so positions can be computed in parallel.

## Scaled dot-product attention

From each input vector, three vectors are produced via learned matrices W^Q, W^K, W^V: a **Query**,
a **Key**, and a **Value** (dimension 64 in the paper). For a given position:

1. **Score** it against every position by dotting its query with each key.
2. **Scale** the scores by 1/√(d_k) (= 1/8 for d_k = 64) for gradient stability, then apply
   **softmax** so weights are positive and sum to 1.
3. **Weight and sum** the value vectors by those softmax weights.

In matrix form: `Attention(Q, K, V) = softmax(QKᵀ / √d_k) · V`.

## Multi-head attention

The model runs **h = 8 attention heads** in parallel, each with its own Q/K/V projections of
dimension d_k = d_v = d_model/h = 64. Multiple heads (1) let the model attend to several positions
at once and (2) provide multiple representation subspaces. The heads' outputs are concatenated and
projected by W^O back to d_model. Attention appears in three roles: encoder self-attention,
**masked** decoder self-attention (each position attends only to earlier positions, preserving
autoregression), and encoder-decoder attention (queries from the decoder, keys/values from the
encoder).

## The other components

- **Position-wise feed-forward network:** two linear layers with a ReLU between them (inner
  dimension 2048), applied identically at each position.
- **Embeddings and softmax:** learned token embeddings (scaled by √d_model), with weights shared
  between the embedding layers and the pre-softmax output projection.
- **Positional encoding:** since there is no recurrence/convolution, fixed **sinusoidal** encodings
  of varying frequency are added to the embeddings to inject order and relative-distance information.
- **Output:** the decoder's final vector is projected by a linear layer into vocabulary-sized
  **logits**, and softmax turns these into a probability distribution over the next token.

## Training (from the Annotated Transformer)

The Annotated Transformer also walks through training: batching and masking, the **Adam** optimizer
with the paper's warmup-then-decay learning-rate schedule, **label smoothing** regularization (via
KL divergence), and a worked German→English translation example, plus notes on multi-GPU training,
checkpoint averaging, and attention visualization.

## See Also

- [microGPT](microgpt.md) and [nanoGPT & nanochat](nanogpt-and-nanochat.md) — from-scratch
  implementations of the (decoder-only) GPT variant of this architecture.
- [Neural Networks: Zero to Hero](nn-zero-to-hero.md) — Karpathy's course builds this architecture
  by hand in the "Building GPT from Scratch" lecture.
