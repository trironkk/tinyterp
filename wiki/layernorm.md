# LayerNorm

> **Motivated by** [`notebooks/02_transformer.ipynb`](../notebooks/02_transformer.ipynb) (roadmap
> topic 2). GPT-2 has three LayerNorms per block-stack worth knowing: `ln_1` (before attention),
> `ln_2` (before the MLP), and `ln_f` (final, before the unembedding). This page is the LA-layer
> definition to reimplement; the *placement* (pre-LN) lives in
> [gpt2-forward-pass](./gpt2-forward-pass.md).
>
> Sources: [Ba, Kiros & Hinton 2016](../RESOURCES.md#transformer-foundation-notebooks-0102),
> [HF GPT-2 source](../RESOURCES.md#transformer-foundation-notebooks-0102). Quotes verified against
> the primary source on 2026-06-19.

## The computation

LayerNorm normalizes **each token's activation vector over its own features**, then applies a
learned per-feature affine. For a vector with `H` features (`H = d_model = 768` in GPT-2 small):

> "We, thus, compute the layer normalization statistics over all the hidden units in the same
> layer as follows: μˡ = (1/H) Σ aˡᵢ , σˡ = sqrt[ (1/H) Σ (aˡᵢ − μˡ)² ]" — [Ba et al. 2016, §3,
> Eq. 3](https://arxiv.org/abs/1607.06450)

Then re-center/re-scale and apply gain `g` and bias `b`:

> "They also learn an adaptive bias b and gain g for each neuron after the normalization:
> hᵢ = f( (gᵢ/σᵢ)(aᵢ − μᵢ) + bᵢ )" — [Ba et al. 2016, §5.1](https://arxiv.org/abs/1607.06450)

So, per token: `y = g ⊙ (x − μ) / sqrt(σ² + ε) + b`, where `μ`, `σ²` are the mean and variance
**across the 768 features**, `g` (`weight`) and `b` (`bias`) are length-768 learned vectors, and
`ε` is a small constant for numerical stability (HF `layer_norm_epsilon`, default `1e-5`).

## The mental model: per-token, not per-batch

This is the load-bearing distinction from BatchNorm — the statistics come from a *single example*,
so LayerNorm is batch-size-independent:

> "In this paper, we transpose batch normalization into layer normalization by computing the mean
> and variance used for normalization from all of the summed inputs to the neurons in a layer on a
> single training case." — [Ba et al. 2016, Abstract](https://arxiv.org/abs/1607.06450)

> "Unlike batch normalization, layer normalization does not impose any constraint on the size of a
> mini-batch and it can be used in the pure online regime with batch size 1." — [Ba et al. 2016,
> §3](https://arxiv.org/abs/1607.06450)

For inference this matters concretely: a single forward pass with one sequence normalizes each of
its token positions independently of every other — there is no running mean/var, no train/eval
difference in the *statistics* (unlike BatchNorm). (Notation note: the paper uses `g`/`b` for
gain/bias in the main text — `γ`/`β` is the BatchNorm convention, used only in the supplement as
`α`/`β`.)

## HF layout to match

GPT-2 uses stock `nn.LayerNorm`:

```python
self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)   # before attn
self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)   # before mlp
self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon) # final
```

So `weight` = `g` (init 1), `bias` = `b` (init 0), normalizing over the last dim (768). To match HF
logits the reimplementation must use the **same ε inside the sqrt** and divide by the
**biased** variance (the `1/H` form above, matching `torch.var(..., unbiased=False)`), not the
`1/(H−1)` sample variance.
