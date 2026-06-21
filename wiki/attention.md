# Attention (GPT-2: fused QKV, multi-head, causal)

> **Motivated by** [`notebooks/02_transformer.ipynb`](../notebooks/02_transformer.ipynb) (roadmap
> topic 2). The overview ("heads read/write the residual stream, additively") is in
> [gpt2-forward-pass](./gpt2-forward-pass.md); this page is the **LA-layer mechanics** to
> reimplement and pin against HF: the fused `c_attn` split, the multi-head reshape, scaled
> dot-product, and the causal mask. QK/OV circuit decomposition is deferred to the circuits topic.
>
> Sources: [Vaswani et al. 2017](../RESOURCES.md#transformer-foundation-notebooks-0102),
> [Elhage et al. 2021 (Framework)](../RESOURCES.md#transformer-foundation-notebooks-0102),
> [HF GPT-2 source](../RESOURCES.md#transformer-foundation-notebooks-0102) (installed `.venv`).
> Verified 2026-06-19.

## The core: scaled dot-product attention

> "We compute the dot products of the query with all keys, divide each by √dk, and apply a softmax
> function to obtain the weights on the values. Attention(Q,K,V) = softmax(QKᵀ/√dk) V" — [Vaswani
> §3.2.1, Eq. 1](https://arxiv.org/abs/1706.03762)

The `√dk` scaling has a concrete numerical reason:

> "for large values of dk, the dot products grow large in magnitude, pushing the softmax function
> into regions where it has extremely small gradients. To counteract this effect, we scale the dot
> products by 1/√dk." — [Vaswani §3.2.1](https://arxiv.org/abs/1706.03762)

In GPT-2 small, `dk = head_dim = 768/12 = 64`, so the scale is `1/8`.

## The GPT-2 layout (what to match)

GPT-2 fuses Q, K, V into one projection and splits afterward. From the installed HF source:

```python
self.embed_dim = config.hidden_size          # 768
self.num_heads = config.num_attention_heads   # 12
self.head_dim  = self.embed_dim // self.num_heads   # 64
self.split_size = self.embed_dim              # 768
self.scaling   = self.head_dim ** -0.5        # 1/8
self.c_attn    = Conv1D(3 * self.embed_dim, self.embed_dim)   # 768 -> 2304

# forward:
query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)  # 2304 -> 3x768
shape = (*query.shape[:-1], -1, self.head_dim)   # (batch, seq, 12, 64)
query = query.view(shape).transpose(1, 2)         # (batch, 12, seq, 64)
# key, value likewise
```

Two layout facts:
1. **Fused `c_attn`** is one `Conv1D(768, 2304)` (see [conv1d-vs-linear](./conv1d-vs-linear.md));
   `.split(768, dim=2)` slices the 2304 output into Q, K, V (in that order).
2. **Multi-head reshape**: `(batch, seq, 768) → view (batch, seq, 12, 64) → transpose (batch, 12,
   seq, 64)`, so the matmuls run in parallel across the 12 heads. After attention, the inverse:
   `transpose` back and `reshape (batch, seq, 768)`, then the output projection `c_proj`.

This is Vaswani's multi-head recipe — *"linearly project the queries, keys and values h times …
perform the attention function in parallel … concatenated and once again projected"* ([§3.2.2](https://arxiv.org/abs/1706.03762)).

## Scaled dot-product, the HF way

```python
attn_weights = torch.matmul(query, key.transpose(-1, -2)) * scaling   # QKᵀ / √dk
if attention_mask is not None:
    attn_weights = attn_weights + attention_mask                       # additive causal mask
attn_weights = softmax(attn_weights, dim=-1)
attn_output  = torch.matmul(attn_weights, value)                       # weighted sum of V
```

Note HF scales by **multiplying** `QKᵀ` by `head_dim**-0.5` (== dividing by `√dk`) and applies the
mask as an **additive** term *before* softmax.

## The causal mask

GPT-2 is a decoder — each position may only attend to itself and earlier positions:

> "each token can only attend to previous tokens" — [HF GPT-2 docs](https://huggingface.co/docs/transformers/en/model_doc/gpt2)

Mechanically this is Vaswani's masking:

> "We … implement this inside of scaled dot-product attention by masking out (setting to −∞) all
> values in the input of the softmax which correspond to illegal connections." — [Vaswani §3.2.3](https://arxiv.org/abs/1706.03762)

Setting future-position scores to `−∞` makes them `0` after softmax, *preserving the
auto-regressive property*. **Implementation note:** the classic form (and the original OpenAI
TF code) registers a lower-triangular buffer (`torch.tril`) and fills the upper triangle with a
large negative value; current HF builds an **additive** mask (`0` for allowed, large-negative for
disallowed) and adds it before softmax (the `attn_weights + attention_mask` line above). Either
way the reimplementation target is the same: zero out attention to future tokens. For a
from-scratch build, a lower-triangular boolean mask + `masked_fill(mask, float("-inf"))` matches.

## Why the additive structure matters

Per the Framework, heads are independent and additive — the reason circuits are analyzable:

> "Attention heads can be understood as independent operations, each outputting a result which is
> added into the residual stream." — [Elhage et al. 2021](https://transformer-circuits.pub/2021/framework/index.html)
