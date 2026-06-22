# Attention Is All You Need

> Source: https://arxiv.org/abs/1706.03762
> Collected: 2026-06-22
> Published: 2017-06-12

_Note: Captured from the arXiv HTML version (now reachable after arxiv.org was added to the network
allowlist). Condensed faithfully, not verbatim. By Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob
Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin (Google Brain / Google
Research / U. Toronto). The paper that introduced the Transformer._

## Abstract / contribution

The dominant sequence-transduction models used recurrence or convolution with attention. This paper
proposes the **Transformer**, an architecture based **solely on attention**, dispensing with
recurrence and convolution entirely. It is more parallelizable and trains faster, and sets new
state of the art on machine translation: **28.4 BLEU on WMT 2014 English→German** and **41.8 BLEU on
English→French** (the latter a single model trained in 3.5 days on eight P100 GPUs), beating prior
results including ensembles. It also generalizes to English constituency parsing.

## Architecture overview

Encoder–decoder, each a stack of **N = 6 identical layers**.

- **Encoder layer**: two sub-layers — multi-head **self-attention**, then a **position-wise
  feed-forward network**. Each sub-layer is wrapped as `LayerNorm(x + Sublayer(x))` (residual +
  layer norm). All sub-layers and embeddings output dimension **d_model = 512**.
- **Decoder layer**: three sub-layers — **masked** multi-head self-attention (each position attends
  only to earlier positions, preserving autoregression), **encoder–decoder attention** (queries from
  the decoder, keys/values from the encoder output), then the feed-forward network — each wrapped in
  residual + layer norm.

## Scaled dot-product attention

`Attention(Q, K, V) = softmax(QKᵀ / √d_k) · V`. Dot the query against all keys, **scale by 1/√d_k**
(prevents large dot products from pushing softmax into low-gradient regions when d_k is large),
softmax, then weight the values. (Additive attention is comparable but dot-product is faster.)

## Multi-head attention

Run **h = 8** attention heads in parallel, each projecting to **d_k = d_v = d_model / h = 64**
dimensions with its own learned W^Q, W^K, W^V. Concatenate the heads and project by W^O back to
d_model. Multiple heads let the model jointly attend to information from different representation
subspaces at different positions. Attention is used in three places: encoder self-attention, masked
decoder self-attention, and encoder–decoder attention.

## Other components

- **Position-wise feed-forward network**: `FFN(x) = max(0, xW₁ + b₁)W₂ + b₂` — two linear layers
  with a ReLU between, inner dimension **d_ff = 2048**, applied identically and independently at each
  position.
- **Embeddings & softmax**: learned token embeddings (scaled by √d_model); the embedding weights are
  **shared** between the two embedding layers and the pre-softmax output projection.
- **Positional encoding**: no recurrence/convolution, so order is injected by adding fixed
  **sinusoidal** encodings: `PE(pos, 2i) = sin(pos / 10000^{2i/d_model})`,
  `PE(pos, 2i+1) = cos(pos / 10000^{2i/d_model})` — different frequencies per dimension, allowing the
  model to attend by relative positions.

## Why self-attention (Table 1)

Compared per layer:

- **Self-attention**: complexity `O(n²·d)`, `O(1)` sequential operations, maximum path length `O(1)`.
- **Recurrent**: `O(n·d²)`, `O(n)` sequential ops, path length `O(n)`.
- **Convolutional**: `O(k·n·d²)`, `O(1)` sequential ops, path length `O(log_k n)`.

Self-attention connects all positions with a constant number of sequential operations (great
parallelism) and the shortest path length for learning long-range dependencies; it is also cheaper
than recurrence when sequence length n < representation dimension d. Attention also offers some
interpretability (heads appear to take on distinct syntactic/semantic roles).

## Training

- **Optimizer**: Adam, β₁ = 0.9, β₂ = 0.98, ε = 10⁻⁹.
- **Learning-rate schedule**: increase **linearly for warmup_steps = 4000** steps, then decay
  proportional to the **inverse square root** of the step number.
- **Regularization**: residual **dropout** P_drop = 0.1 (applied to each sub-layer output before the
  residual add, and to embedding+positional sums) and **label smoothing** ε_ls = 0.1 (hurts
  perplexity but improves accuracy and BLEU).
- **Compute**: base model 100k steps (~12 hours) on 8 P100 GPUs; big model 300k steps (~3.5 days).

## Results

- WMT 2014 EN→DE: **28.4 BLEU** (big), 27.3 (base). WMT 2014 EN→FR: **41.8 BLEU** (big), 38.1 (base).
- English constituency parsing: 91.3 F1 (WSJ-only), 92.7 F1 (semi-supervised) — generalizes beyond
  translation.
