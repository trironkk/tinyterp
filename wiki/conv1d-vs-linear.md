# `Conv1D` vs `nn.Linear` (GPT-2's weight layout)

> **Motivated by** [`notebooks/02_transformer.ipynb`](../notebooks/02_transformer.ipynb) (roadmap
> topic 2). Every projection in GPT-2 — `c_attn`, `c_proj`, `c_fc` — is a HuggingFace `Conv1D`,
> **not** an `nn.Linear`. Same math, *transposed weight*; getting the layout wrong is the easiest
> way to load the pretrained weights into a from-scratch reimplementation incorrectly.
>
> Sources: [HF `Conv1D` source](../RESOURCES.md#transformer-foundation-notebooks-0102)
> (`transformers/pytorch_utils.py`, read from the installed `.venv`),
> [OpenAI GPT-2 reference](../RESOURCES.md#transformer-foundation-notebooks-0102). Verified against
> the installed source on 2026-06-19.

## The one fact

HF's own docstring says it outright:

> "1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).
> Basically works like a linear layer but the weights are transposed." — `Conv1D` docstring,
> `transformers/pytorch_utils.py`

It is **not** a convolution in any meaningful sense — it is a pointwise affine map: a `kernel_size=1`
convolution is a linear layer applied at every position, and the OpenAI/HF code computes it as a
plain matmul. The `Conv1D` name is historical (it generalizes that kernel-size-1 view).

## The layout difference

| | weight shape | forward |
|---|---|---|
| `nn.Linear(in, out)` | `(out, in)` | `x @ W.T + b` |
| `Conv1D(nf=out, nx=in)` | `(in, out)` i.e. `(nx, nf)` | `x @ W + b` |

From the installed source:

```python
# __init__
self.weight = nn.Parameter(torch.empty(nx, nf))   # (in, out) — already transposed vs Linear
self.bias   = nn.Parameter(torch.zeros(nf))
# forward
x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)   # bias + (x @ weight)
```

So `addmm(bias, x, weight)` computes `x @ weight + bias` with **no transpose** — because the
weight is *stored* transposed relative to `nn.Linear`.

## Why it matters for the reimplementation

The `assert reimpl_logits ≈ hf_logits` check will catch a *wrong matmul* (shapes won't line up, or
the numbers diverge). What it can't catch is a wrong **mental model** that happens to still produce
a runnable shape. The model to hold:

- A GPT-2 checkpoint stores `c_attn.weight` as `(768, 2304)` = `(d_model, 3·d_model)`.
- If you build your attention with `nn.Linear`, its `.weight` is `(2304, 768)` — **transposed**.
  Either store your reimpl weights `Conv1D`-style and do `x @ W`, or load the checkpoint weight
  with a `.T`. Pick one and be consistent across `c_attn`, `c_proj`, `c_fc`.

The fused `c_attn` Q/K/V split and the multi-head reshape that consume this projection live in
[attention](./attention.md); the 4× MLP that uses `c_fc`/`c_proj` is in
[gpt2-forward-pass](./gpt2-forward-pass.md).
