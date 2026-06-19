# The GPT-2 forward pass

> **Motivated by** [`notebooks/01_gpt2_inference.ipynb`](../notebooks/01_gpt2_inference.ipynb) —
> the `GPT2LMHeadModel` module printout and the single forward call inside `generate`. This page
> is the *overview altitude* of the architecture (what each block does and why); the
> per-primitive reimplementations (attention, MLP, LayerNorm) get their own pages when the
> Transformer notebook (roadmap topic 2) builds them. Tokenization is upstream
> ([tokenization-bpe](./tokenization-bpe.md)); the generation loop that calls this is
> [hf-inference-path](./hf-inference-path.md).
>
> Sources: [GPT-2 paper (Radford et al. 2019)](../RESOURCES.md#transformer-foundation-notebooks-0102),
> [Vaswani et al. 2017](../RESOURCES.md#transformer-foundation-notebooks-0102),
> [Elhage et al. 2021, A Mathematical Framework](../RESOURCES.md#transformer-foundation-notebooks-0102),
> [HF GPT-2 docs](../RESOURCES.md#transformer-foundation-notebooks-0102). Quotes verified against
> the primary sources on 2026-06-15.

## The shape of it

GPT-2 is a **decoder-only Transformer**: *"We use a Transformer (Vaswani et al., 2017) based
architecture for our LMs"* ([GPT-2 paper §2.3](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)).
At the highest level the data flow is exactly three stages
([Framework, "High-Level Architecture"](https://transformer-circuits.pub/2021/framework/index.html)):

> "A transformer starts with a token embedding, followed by a series of 'residual blocks', and
> finally a token unembedding."

For GPT-2 small (the `gpt2` checkpoint, *117M* params as reported in the paper — the checkpoint
is ~124M by the standard parameter count, the paper's 117M being a known undercount): **12
layers, d_model 768**
([GPT-2 paper, Table 2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf);
HF config `n_layer`/`n_embd` *"defaults to `12`"*/`768`), vocab **50,257**, context **1024**.
The notebook's module printout matches: `wte=Embedding(50257,768)`, `wpe=Embedding(1024,768)`,
`12 x GPT2Block`, `ln_f`, `lm_head=Linear(768, 50257, bias=False)`.

## The residual stream is the backbone

The single most important mental model: there is one running vector per token position — the
**residual stream** — and every layer reads from it and writes back by *addition*
([Framework](https://transformer-circuits.pub/2021/framework/index.html)):

> "All components of a transformer (the token embedding, attention heads, MLP layers, and
> unembedding) communicate with each other by reading and writing to different subspaces of the
> residual stream." — *Virtual Weights and the Residual Stream as a Communication Channel*

> "Both the attention and MLP layers each 'read' their input from the residual stream (by
> performing a linear projection), and then 'write' their result to the residual stream by
> adding a linear projection back in." — *High-Level Architecture*

This additive structure is what makes layers composable and is the foundation the whole
curriculum (circuits, SAEs) is built on. It traces directly to Vaswani's sub-layer wrapper:

> "We employ a residual connection [11] around each of the two sub-layers, followed by layer
> normalization. That is, the output of each sub-layer is LayerNorm(x + Sublayer(x))." — [Vaswani §3.1](https://arxiv.org/abs/1706.03762)

**GPT-2's one twist** — it moves the LayerNorm to the *input* of each sub-block (pre-LN) and
adds a final one:

> "Layer normalization (Ba et al., 2016) was moved to the input of each sub-block, similar to a
> pre-activation residual network (He et al., 2016) and an additional layer normalization was
> added after the final self-attention block." — [GPT-2 paper §2.3](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

So a GPT-2 block is `x + Attn(LN(x))` then `x + MLP(LN(x))` (pre-LN), and `ln_f` is that extra
final norm before the unembedding — visible as `ln_1`, `ln_2`, `ln_f` in the module printout.

## Stage 1 — Embedding (into the stream)

Two learned lookup tables, summed:
- **`wte`** (`Embedding(50257, 768)`): token → vector.
- **`wpe`** (`Embedding(1024, 768)`): absolute position → vector. GPT-2 uses **learned**
  positional embeddings (the printout shows `wpe` as a plain `Embedding`), unlike Vaswani's
  fixed sinusoids — but the *reason* is Vaswani's:

  > "Since our model contains no recurrence and no convolution, in order for the model to make
  > use of the order of the sequence, we must inject some information about the ... position of
  > the tokens." — [Vaswani §3.5](https://arxiv.org/abs/1706.03762)

The sum `wte[token] + wpe[pos]` is the initial residual-stream value for each position.

## Stage 2 — The 12 residual blocks

Each `GPT2Block` is attention then MLP, each pre-normed and added back.

### Attention (`attn`)

The core is **scaled dot-product attention** ([Vaswani §3.2.1](https://arxiv.org/abs/1706.03762)):

> "We compute the dot products of the query with all keys, divide each by √dk, and apply a
> softmax function to obtain the weights on the values."
>
> Attention(Q, K, V) = softmax(QKᵀ / √dk) V   *(Eq. 1)*

The `√dk` scaling exists for a concrete numerical reason:

> "for large values of dk, the dot products grow large in magnitude, pushing the softmax
> function into regions where it has extremely small gradients. To counteract this effect, we
> scale the dot products by 1/√dk." — [Vaswani §3.2.1](https://arxiv.org/abs/1706.03762)

**Multi-head**: run several attentions in parallel on projected subspaces, concatenate, project
out — *"linearly project the queries, keys and values h times ... perform the attention function
in parallel ... These are concatenated and once again projected"* ([Vaswani §3.2.2](https://arxiv.org/abs/1706.03762)).
GPT-2 small uses **12 heads** of width `768/12 = 64` (HF config `n_head` *"defaults to `12`"*).
In the printout, `c_attn=Conv1D(nf=2304)` is the fused Q,K,V projection (`3 × 768`) and
`c_proj=Conv1D(nf=768)` is the output projection W^O. As a decoder, GPT-2 is **causal** — *"each
token can only attend to previous tokens"* ([HF docs](https://huggingface.co/docs/transformers/en/model_doc/gpt2)),
implemented exactly as Vaswani's masking: *"masking out (setting to −∞) all values in the input
of the softmax which correspond to illegal connections"* ([Vaswani §3.2.3](https://arxiv.org/abs/1706.03762)),
which *"preserve[s] the auto-regressive property."*

Per the Framework, heads compose cleanly because they are independent and additive:

> "Attention heads can be understood as independent operations, each outputting a result which
> is added into the residual stream." — [Framework](https://transformer-circuits.pub/2021/framework/index.html)

### MLP (`mlp`)

A **position-wise feed-forward network** — two linear maps with a nonlinearity, applied
identically at every position ([Vaswani §3.3](https://arxiv.org/abs/1706.03762)):

> "FFN(x) = max(0, xW1 + b1)W2 + b2 ... The dimensionality of input and output is dmodel = 512,
> and the inner-layer has dimensionality dff = 2048."

That's the **4× expansion** pattern. GPT-2 keeps the 4× (`c_fc=Conv1D(nf=3072)` = `4 × 768`,
`c_proj` back to 768) but swaps the activation: the printout shows `NewGELUActivation`, i.e.
GELU rather than Vaswani's ReLU. *(The paper text pulled here doesn't state the activation; this
is read off the reference model — see Open threads.)*

## Stage 3 — Unembedding (out of the stream)

After `ln_f`, the final residual stream is projected to vocabulary logits by `lm_head`
(`Linear(768, 50257, bias=False)`):

> "Prediction scores of the language modeling head (scores for each vocabulary token before
> SoftMax)" — shape `(batch, seq, vocab_size)` ([HF docs](https://huggingface.co/docs/transformers/en/model_doc/gpt2)).

The unembedding **reuses the embedding matrix** (weight tying) — *"a language modeling head on
top (linear layer with weights tied to the input embeddings)"* ([HF docs](https://huggingface.co/docs/transformers/en/model_doc/gpt2);
config `tie_word_embeddings` *"defaults to `True`"*), the same choice Vaswani made: *"we share
the same weight matrix between the two embedding layers and the pre-softmax linear
transformation"* ([Vaswani §3.4](https://arxiv.org/abs/1706.03762)). So `wte` literally bookends
the stream — it writes tokens in at Stage 1 and reads logits out at Stage 3. Those logits feed
the decoding loop in [hf-inference-path](./hf-inference-path.md).

## One more GPT-2 detail: residual scaling at init

> "A modified initialization which accounts for the accumulation on the residual path with model
> depth is used. We scale the weights of residual layers at initialization by a factor of 1/√N
> where N is the number of residual layers." — [GPT-2 paper §2.3](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

A *training* detail (it shapes the learned weights), not an inference-time op — but it's why the
residual stream stays well-scaled across 12 added contributions. Noted here for completeness;
not exercised by notebook 01.

## Open threads (→ Transformer notebook, roadmap topic 2)

- **Reimplement each primitive to the LA layer** and pin `assert reimpl_logits ≈ hf_logits`:
  LayerNorm (mean/var normalize + affine), the fused `c_attn` split into Q/K/V, the causal mask,
  multi-head reshape, GELU, weight-tied unembedding.
- **GELU vs ReLU**: confirm GPT-2's exact activation (`NewGELUActivation` = the tanh-approx
  GELU) from a primary source, not just the HF module name — currently read off the reference
  model. Seed during topic 2's Gather.
- **`Conv1D` vs `Linear`**: GPT-2's `Conv1D` is a transposed-weight linear layer; reconcile its
  weight layout when reimplementing so the matmuls line up.
- **Privileged basis / superposition**: the Framework's claim that the residual stream has no
  privileged basis is the bridge to the SAE topic; defer until then.
