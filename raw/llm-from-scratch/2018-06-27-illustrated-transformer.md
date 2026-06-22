# The Illustrated Transformer

> Source: https://jalammar.github.io/illustrated-transformer/
> Collected: 2026-06-22
> Published: 2018-06-27

_Note: Captured from the post's source markdown via GitHub (jalammar/jalammar.github.io);
condensed faithfully, not verbatim. Author: Jay Alammar._

## Premise

A visual walkthrough of **The Transformer** (Vaswani et al., "Attention is All You Need," 2017),
a model that uses attention to speed up training and lends itself to parallelization. The post
introduces concepts one at a time for readers without in-depth background. (A 2025 update notes
the post became Chapter 3 of an LLM book covering later developments like Multi-Query Attention
and RoPE.)

## High-level structure

Viewed as a black box, the Transformer maps an input sentence to an output (e.g., a translation).
Inside is an **encoding component** (a stack of encoders — the paper uses six) and a **decoding
component** (a stack of the same number of decoders).

Each **encoder** has two sub-layers: a **self-attention** layer followed by a **position-wise
feed-forward network**. Each **decoder** has those two plus an **encoder-decoder attention**
layer in between, which helps it focus on relevant parts of the input.

## From words to vectors

Each input word is turned into a vector via an embedding (e.g., size 512). Embeddings flow up
through the encoder stack. A key property: in the self-attention layer the paths between
positions have dependencies, but the feed-forward layer has none, so the various positions can
be processed in parallel through the feed-forward layer.

## Self-attention, step by step

Self-attention lets the model, when encoding one word, look at other words in the sequence for
clues to a better representation (e.g., resolving what "it" refers to).

1. From each input embedding, create three vectors via learned matrices (W^Q, W^K, W^V): a
   **Query**, a **Key**, and a **Value** (in the paper, dimension 64).
2. **Score** each word against the current word by taking the dot product of the query with each
   key.
3. **Scale** the scores by dividing by √(d_k) = 8 (the square root of the key dimension) for more
   stable gradients, then apply **softmax** so the scores are positive and sum to 1.
4. Multiply each value vector by its softmax score and **sum** them to produce the self-attention
   output for that position.

In practice this is done with matrices: pack embeddings into X, compute Q = X·W^Q, K = X·W^K,
V = X·W^V, then output = softmax(Q·Kᵀ / √d_k)·V.

## Multi-head attention

The paper uses **eight attention heads**, each with its own Q/K/V projection matrices. This
(1) expands the model's ability to focus on different positions and (2) gives the attention layer
multiple "representation subspaces." The eight resulting matrices are concatenated and multiplied
by an additional weight matrix W^O to collapse back to a single matrix for the feed-forward layer.

## Positional encoding

Because the model has no recurrence or convolution, **positional encodings** are added to the
input embeddings to give the model a sense of word order and distance. The paper uses a pattern
of sine and cosine functions of different frequencies.

## Residuals and layer normalization

Each sub-layer (self-attention, feed-forward) in each encoder and decoder has a **residual
connection** around it, followed by **layer normalization**. This applies in the decoder as well.

## The decoder side

The encoder stack outputs a set of attention key/value vectors K and V, which each decoder uses
in its **encoder-decoder attention** layer (queries come from the layer below it). The decoder
runs step by step, feeding each output back in as input for the next step. Decoder self-attention
is **masked** so each position can only attend to earlier positions.

## Final linear and softmax layer

The decoder stack outputs a vector that a **linear layer** projects into a large "logits" vector
over the vocabulary; a **softmax** turns logits into probabilities, and the highest-probability
cell selects the output word for that step.
