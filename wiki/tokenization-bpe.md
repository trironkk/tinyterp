# Byte-level BPE tokenization (GPT-2)

> **Motivated by** [`notebooks/01_gpt2_inference.ipynb`](../notebooks/01_gpt2_inference.ipynb) —
> the `tokenizer = AutoTokenizer.from_pretrained("gpt2")` / `tokenizer(prompt, return_tensors="pt")`
> step. This page covers *what the tokenizer is and why GPT-2 chose it*; the downstream
> `input_ids → embeddings` step lives in [gpt2-forward-pass](./gpt2-forward-pass.md).
>
> Sources: [GPT-2 paper (Radford et al. 2019)](../RESOURCES.md#transformer-foundation-notebooks-0102),
> [HuggingFace GPT-2 docs](../RESOURCES.md#transformer-foundation-notebooks-0102). All quotes
> verified against the primary sources on 2026-06-15.

## What it is

GPT-2 tokenizes text with **byte-level Byte Pair Encoding (BPE)**. HuggingFace's tokenizer
class states this directly: *"Construct a GPT-2 tokenizer. Based on byte-level
Byte-Pair-Encoding."* ([HF GPT-2 docs](https://huggingface.co/docs/transformers/en/model_doc/gpt2),
`GPT2Tokenizer`). The output is a sequence of integer token IDs (`input_ids`) drawn from a
vocabulary of **50,257** entries (HF config: *"vocab_size ... defaults to `50257`"*).

## Why BPE — the middle ground

BPE sits between two extremes, neither of which GPT-2 wants:

> "Byte Pair Encoding (BPE) (Sennrich et al., 2015) is a practical middle ground between
> character and word level language modeling which effectively interpolates between word level
> inputs for frequent symbol sequences and character level inputs for infrequent symbol
> sequences." — [GPT-2 paper §2.1, Input Representation](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

- **Word-level** vocabularies can't represent unseen words (out-of-vocabulary problem).
- **Character/byte-level** sequences are fully general but long and hard to model.

BPE starts from a base alphabet and greedily *merges* the most frequent adjacent pairs into new
tokens, so frequent words collapse to one token while rare words fall back to smaller pieces.

## Why *byte*-level specifically

A naïve BPE over Unicode code points needs an enormous base vocabulary before any merges:

> "Despite its name, reference BPE implementations often operate on Unicode code points and not
> byte sequences. ... This would result in a base vocabulary of over 130,000 before any
> multi-symbol tokens are added. ... In contrast, a byte-level version of BPE only requires a
> base vocabulary of size 256." — [GPT-2 paper §2.1](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

Operating on **raw bytes** caps the base alphabet at 256 and makes the tokenizer *universal*:
any Unicode string is representable, so the model never hits a true out-of-vocabulary token.

> "Since our approach can assign a probability to any Unicode string, this allows us to evaluate
> our LMs on any dataset regardless of pre-processing, tokenization, or vocab size." — [GPT-2 paper §2.1](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

## The one wrinkle: don't merge across categories

Pure byte-level BPE produces redundant merges (`dog`, `dog.`, `dog!`, `dog?` as separate
tokens), wasting vocabulary slots:

> "To avoid this, we prevent BPE from merging across character categories for any byte sequence.
> We add an exception for spaces..." — [GPT-2 paper §2.1](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

This is why GPT-2 tokens famously carry a leading-space marker — a space attaches to the word
that follows it rather than being merged into punctuation.

## What this means for notebook 01

- `tokenizer(prompt, return_tensors="pt")` returns `input_ids` of shape `(batch, seq_len)` —
  integer IDs in `[0, 50256]`.
- The special token `<|endoftext|>` is ID **50256**, used as both beginning- and end-of-stream
  (HF config: `bos_token_id` and `eos_token_id` both *"defaults to `50256`"*). This is the ID
  the generation step treats as the stop / pad token (see [hf-inference-path](./hf-inference-path.md)).
- `tokenizer.decode(...)` inverts the mapping: byte-level BPE is losslessly reversible, so the
  generated IDs decode back to text without information loss.

## Open threads

- Reimplementing the BPE merge algorithm itself (encode/decode from the merges + vocab files)
  is **out of scope for notebook 01** (it runs HF's tokenizer as a reference). Promote to its
  own notebook only if a later topic needs token-level surgery.
