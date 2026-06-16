# The HuggingFace inference path (GPT-2)

> **Motivated by** [`notebooks/01_gpt2_inference.ipynb`](../notebooks/01_gpt2_inference.ipynb) —
> the load → tokenize → `generate` → decode pipeline. This page is the *reference* path the
> later reimplementation will assert against (`assert reimpl_logits ≈ hf_logits`); the
> tensor mechanics it wraps live in [gpt2-forward-pass](./gpt2-forward-pass.md) and the
> tokenizer in [tokenization-bpe](./tokenization-bpe.md).
>
> Sources: [HuggingFace GPT-2 docs](../RESOURCES.md#transformer-foundation-notebooks-0102) and
> the HuggingFace generation-strategies guide. Quotes verified against the live docs on 2026-06-15.

## The five-line pipeline

```python
tokenizer = AutoTokenizer.from_pretrained("gpt2")           # byte-level BPE
model = AutoModelForCausalLM.from_pretrained("gpt2")        # GPT2LMHeadModel
model.eval()                                                # disable dropout
inputs = tokenizer(prompt, return_tensors="pt")             # text -> input_ids
output_ids = model.generate(**inputs, max_new_tokens=50)    # autoregressive loop
tokenizer.decode(output_ids[0], skip_special_tokens=True)   # ids -> text
```

## What each class is

- **`AutoModelForCausalLM`** resolves to `GPT2LMHeadModel` for the `gpt2` checkpoint — *"a
  causal transformer language model"* whose *"architecture uses a unidirectional (causal)
  attention mechanism where each token can only attend to previous tokens"*
  ([HF GPT-2 docs](https://huggingface.co/docs/transformers/en/model_doc/gpt2)). It is *"The
  GPT2 Model transformer with a language modeling head on top (linear layer with weights tied
  to the input embeddings)."* — i.e. the unembedding reuses the embedding matrix (see
  [gpt2-forward-pass](./gpt2-forward-pass.md#unembedding); HF config `tie_word_embeddings`
  *"defaults to `True`"*).
- **`AutoTokenizer`** resolves to the GPT-2 byte-level BPE tokenizer
  ([tokenization-bpe](./tokenization-bpe.md)).

## `eval()` — why it matters at inference

GPT-2 contains dropout layers (`attn_pdrop`, `resid_pdrop`, the MLP dropout — all
*"defaults to `0.1`"* per the HF config; visible in the notebook's module printout as
`Dropout(p=0.1)`). Dropout is a *training-only* regularizer; `model.eval()` switches these to
identity so a forward pass is **deterministic**. Without it, repeated runs on the same input
would differ — fatal for `assert reimpl_logits ≈ hf_logits`. (The notebook also wraps
generation in `torch.no_grad()`, which is orthogonal: it skips building the autograd graph to
save memory/time, not changing the numbers.)

## What `generate()` does: the autoregressive loop

`generate()` is not a single forward pass — it calls the model repeatedly, appending one token
at a time. The default decoding strategy is **greedy search**:

> "Greedy search is the default decoding strategy. It selects the next most likely token at
> each step." — [HF generation strategies](https://huggingface.co/docs/transformers/en/generation_strategies)

The HF docs spell out the *"Simple greedy decoding loop"* verbatim — this is the entire idea:

```python
while cur_length < max_length:
    logits = model(input_ids).logits          # (batch, seq, vocab)
    next_token_logits = logits[:, -1, :]       # logits for the *last* position only
    next_tokens = torch.argmax(next_token_logits, dim=-1)
    input_ids = torch.cat((input_ids, next_tokens[:, None]), dim=-1)
    cur_length += 1
```
— [HF generation strategies](https://huggingface.co/docs/transformers/en/generation_strategies)

Key points this loop makes concrete:

- The model emits **logits at every position** (shape `(batch, seq, vocab_size)`; HF: the
  *"Prediction scores of the language modeling head (scores for each vocabulary token before
  SoftMax)"*), but greedy generation only consumes the **last** position's row to pick the next
  token. Earlier positions' logits are unused during plain generation — they become central
  once we start *interpreting* the forward pass.
- `argmax` over the vocab axis = greedy. Sampling/beam search are opt-in (`do_sample=True`,
  `num_beams>1`) and would make output non-deterministic.
- **`max_new_tokens`** bounds the loop — 50 new tokens in notebook 01. (Left unset, greedy
  search *"generates a maximum of 20 new tokens."*)
- This is why GPT-2 repeats itself on long greedy runs (visible in the notebook's "quick brown
  foxes..." output): greedy *"breaks down when generating longer sequences because it begins to
  repeat itself."*

## The pad-token nudge

Notebook 01 prints: `Setting pad_token_id to eos_token_id:50256 for open-end generation.`
GPT-2's config defines `bos_token_id` and `eos_token_id` as **50256** (`<|endoftext|>`) but no
`pad_token_id` ([HF GPT-2 docs](https://huggingface.co/docs/transformers/en/model_doc/gpt2),
config). For single-prompt open-ended generation there is nothing to pad, so HF defaults the
pad token to EOS to silence the ambiguity — harmless here, relevant once batching prompts of
unequal length.

## Why this page exists

This is the **trusted reference**. The Transformer notebook (roadmap step 1) reimplements the
forward pass at the linear-algebra layer and pins it with `assert reimpl_logits ≈ hf_logits`
against exactly this path — so the contract is: *same `input_ids` in, same logits out*, with
dropout disabled and no sampling randomness.

## Open threads

- KV-caching (`past_key_values` / `use_cache`) makes the loop O(n) instead of O(n²) but doesn't
  change the logits. Deferred — notebook 01 doesn't depend on it.
- The exact `argmax` tie-breaking and float determinism across devices: only matters when
  tightening the `≈` tolerance in the reimplementation. Revisit in step 1.
