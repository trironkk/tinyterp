# GELU (the `"gelu_new"` activation)

> **Motivated by** [`notebooks/02_transformer.ipynb`](../notebooks/02_transformer.ipynb) (roadmap
> topic 2). GPT-2's MLP is `c_fc → activation → c_proj`; the activation is GELU, and HF ships the
> **tanh approximation** specifically. This closes the notebook-01 gap where the activation was
> only read off the HF module name. The MLP that wraps it is in
> [gpt2-forward-pass](./gpt2-forward-pass.md).
>
> Sources: [Hendrycks & Gimpel 2016](../RESOURCES.md#transformer-foundation-notebooks-0102),
> [GPT-1, Radford et al. 2018](../RESOURCES.md#transformer-foundation-notebooks-0102),
> [HF GPT-2 source](../RESOURCES.md#transformer-foundation-notebooks-0102). Verified 2026-06-19.

## What GELU is

> "The GELU activation function is xΦ(x), where Φ(x) [is] the standard Gaussian cumulative
> distribution function." — [Hendrycks & Gimpel 2016, Abstract](https://arxiv.org/abs/1606.08415)

So `GELU(x) = x · Φ(x) = x · ½[1 + erf(x/√2)]` (§2). Intuitively it weights an input by the
probability that a standard-normal draw falls below it — a smooth gate, unlike ReLU's hard cutoff.

## The tanh approximation GPT-2 actually uses

The exact `erf` form is expensive, so the paper offers a tanh approximation:

> "Instead of using a xσ(x) to approximate Φ(x), we used 0.5x(1 + tanh[√(2/π)(x + 0.044715x³)])
> … and we used the former in every experiment in this paper." — [Hendrycks & Gimpel 2016, §2 /
> Discussion](https://arxiv.org/abs/1606.08415)

This is **exactly** what HF GPT-2 runs. The config sets `activation_function="gelu_new"`, which
maps to `NewGELUActivation` in the installed source:

```python
# transformers/activations.py — NewGELUActivation.forward
return 0.5 * input * (1.0 + torch.tanh(
    math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
```

Identical to the paper's formula, `0.044715` constant and all. So the reimplementation should use
the **tanh form**, not `torch.erf` / `F.gelu(approximate="none")` — the two differ slightly (the tanh form is itself an approximation of the erf form), enough
to matter for a tight `assert reimpl_logits ≈ hf_logits` tolerance.

## Where it sits in GPT-2

> "For the activation function, we used the Gaussian Error Linear Unit (GELU)." — [GPT-1, Radford
> et al. 2018, §4.1](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)

GPT-2 inherits this from GPT-1 (the GPT-2 paper §2.3 doesn't name the activation). It's applied
between the two MLP projections: `c_fc` (768→3072), GELU, `c_proj` (3072→768) — the 4× expansion in
[gpt2-forward-pass](./gpt2-forward-pass.md).
