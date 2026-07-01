# Notes — `01_microgpt-wiki.ipynb`

Design decisions and lessons for the microGPT-on-Wikipedia notebook.

## Cell map

The notebook uses labeled cells (`## [X] …`) so notes can reference them. Backlog items below
are tagged with the cell(s) they touch.

| | | | |
|---|---|---|---|
| [A] Setup & dependencies | [F] Encode / decode | [K] Multi-head causal self-attn | [P] Training loop |
| [B] Config | [G] Tokenize the corpus | [L] MLP block | [Q] Loss curves |
| [C] Load Wikipedia | [H] Batching | [M] Transformer block | [R] Sampling |
| [D] Inspect the corpus | [I] Embeddings | [N] GPT model | [S] Generate |
| [E] Train a minimal BPE | [J] RMSNorm | [O] Optimizer & LR schedule | [T] Save / load |

## Data loading — final design [C]

Download the **whole** `20231101.simple` config locally on first run, then sample
uniformly across all of it. Self-contained in the notebook (download once, load from disk
after):

- `snapshot_download(repo_id="wikimedia/wikipedia",
  allow_patterns="20231101.simple/*.parquet", local_dir=data_dir/"wikipedia_simple")` —
  a *targeted* download, guarded by a parquet-exists check so re-runs do zero network I/O
  (offline after first run). Files land in gitignored `data/wikipedia_simple/`.
- `load_dataset("parquet", data_files=..., split="train")` → memory-mapped, random-access.
- `idx = random.Random(data_seed).sample(range(len(ds)), n_articles)`; keep `articles`
  (the sampled texts) and `corpus = "\n\n".join(articles)`.

Config [B]: `n_articles`, `data_seed`. The download + parquet load stay cached, so
iterating on sampling (change `data_seed`/`n_articles` or the selection rule) is instant
and draws a genuinely global uniform sample.

## Lesson: the sample is biased, and BPE amplifies it (the `", American"` story)

BPE [E] had learned a token for `", American"`. Two findings, worth remembering:

1. **The raw dataset is ~alphabetical by title.** Taking the first N articles gives an
   A-heavy, topically skewed slice (April, August, Art, A, Air, Alan Turing…). Year/date
   pages (1991, 2006…) and "Events" lists also cluster near the front.
2. **BPE merges on *global frequency*, so a few list pages dominate.** `", American"` hit
   2,576 times but in only 90/2000 articles (~29 each), concentrated in birth/death list
   pages ("*June 3 – Name, American actor*"). Dense repetition in a handful of structured
   docs is enough to earn a merge — it's a **list-page artifact**, not broad usage.

Takeaway: a representative *uniform* sample (current design) is what keeps these artifacts
out of the long-token tail. Some biography phrasing may still appear (Wikipedia is
biography-heavy) but no longer dominates. Truly taming list pages would need document-level
dedup/normalization — not done.

## Research directions

The two large, measured threads. Everything smaller lives in **Backlog** below.

### Scale the corpus to ~1 epoch — needs a faster tokenizer ([B]/[E]/[G])

**Status: prototyped, deliberately NOT merged** — kept the simple overfitting build.

The step count and corpus are badly mismatched: 20000 steps × 32 × 128 = **82M tokens seen**,
but 2000 articles ≈ 0.9M tokens, so training makes **~90 passes** over the data — the overfit
engine (see **Overfitting** below; this is the root-cause, data-side lever). Measured
~450–490 tokens/article (2.54 bytes/token at vocab 1024), so **~1 epoch needs ~180k articles**
(~88M tokens; the `20231101.simple` dump has 241,787).

Blocker: [E]/[G]'s pure-Python *global* BPE is O(`num_merges` × `len`) — measured **127s to
train and 57s to tokenize just 2000 articles**, i.e. ~hours at 180k, and `list(corpus.encode())`
at 180k is ~1.7GB of Python ints.

Validated fix (scratch prototype tokenized 180k in ~25s):

- **Decouple BPE training** onto a small fixed slice (add `bpe_train_articles`, e.g. 2000) —
  the vocab stabilizes long before the full corpus, so [E] stays cheap and independent of
  `n_articles`. Only [G]'s tokenization then scales.
- Adopt **GPT-2 regex pre-tokenization** (option (b)): split into word / number / punctuation
  pre-tokens and merge only *within* a chunk. Makes the vocab word-bounded (also kills the
  cross-word `", American"` artifact) and — the actual point — makes encoding **cacheable per
  unique chunk**, so [G] pays the merge loop once per unique pre-token (~100k) instead of per
  token (~88M).
- Tokenize by streaming pre-tokens (`re.finditer`, lazy — no giant intermediate list over the
  ~210MB corpus) through the cache into a `uint16` `array` → `torch.frombuffer` tensor (~88M
  tokens ≈ 176MB).
- **GOTCHA to fix before trusting it:** the stdlib-`re` translation of GPT-2's pattern did
  *not* partition the corpus gap-free — `decode(tokenize(corpus)) == corpus` came back
  **False**, i.e. some characters are silently dropped (the per-article demo roundtrips fine,
  so it's an edge char class the alternation misses). Before adopting: find the gap and assert
  a full-corpus roundtrip, or pull in the `regex` module and use GPT-2's exact `\p{L}`/`\p{N}`
  pattern (known gap-free). Silent dropping = corrupted training data, so this is the blocker,
  not the speed.
- Leftover perf once correct: (c) incremental pair-count maintenance to speed **[E] training**
  (still ~250s on the 2k slice with regex, since it rescans every chunk per merge — only
  re-touch chunks containing the merged pair, minbpe-style); (d) confirm greedy `encode()`
  matches the training-time merge order on held-out text.

### Overfitting ([P]/[Q])

First run showed val cross-entropy *rising* while train kept falling — classic overfitting,
expected here: a tiny corpus (~2000 articles), a 4L/128D model with capacity to memorize,
multiple effective passes, and zero regularization.

**Root cause is corpus size.** The data-side fix — scale `n_articles` toward ~1 epoch instead
of the current ~90 re-passes — is the primary lever, and it's the **Scale the corpus** research
direction above. The items below are *complementary regularization* levers; measure each against
the [Q] train/val gap.

First, **confirm it's real vs `eval_iters=50` sampling noise** — check several eval points, maybe
raise `eval_iters`.

Regularization toolkit:

- *Early stopping / best-val checkpoint* — track the val minimum during [P] and have [T]
  save **those** weights, not the final (most-overfit) step. Cheapest, do first.
- *Weight decay* — decoupled AdamW-style decay on the matmul weights (skip norms/embeddings);
  the standard "shrink unused capacity" regularizer.
- *Dropout* — on attention probabilities, the MLP hidden, and the residual adds (GPT-2 used
  p≈0.1). Stochastic masking that stops co-adaptation.
- *Label smoothing* — soften the cross-entropy targets so the model can't drive train loss to
  ~0 by memorizing exact next tokens.
- *Stochastic depth / smaller model* — randomly skip whole blocks, or simply cut
  `n_layer`/`n_embd` so there's less capacity to memorize with.
- *Data augmentation* — vary `data_seed` / reshuffle article order between runs so the model
  never sees the exact same stream (and the val tail isn't a fixed set of articles).

The high-value trio is **more data + early stopping + weight decay**; dropout/label-smoothing
are the next layer if the gap persists.

## Backlog / chores

Small, well-scoped follow-ups. Each is tagged with the cell(s) it touches.

- **Config hygiene ([O]→[B]).** Promote `warmup_steps` / `min_lr` from [O] up into [B] so [B]
  stays the single source of truth for hyperparameters.
- **Gradient clipping ([P]).** Add global-norm gradient clipping to the [P] step (clip before
  `optimizer.step()`) — the standard stabilizer we left out; cheap insurance against loss spikes.
- **Ablations ([J]–[N]).** Toggle weight tying (tied vs separate LM head), swap learned-absolute
  positions for RoPE, and compare pre-norm vs post-norm — measure the loss impact of each.
- **Better sampling ([R]/[S]).** Add top-p / nucleus sampling and a repetition penalty to
  `generate`, and compare output quality against the current top-k.
- **KV cache ([K]/[S]).** Generation re-runs the full context every token; cache per-layer keys
  and values so each new token is O(1) work instead of O(T). Good systems exercise.
- **Scaling sweep ([B]).** Plot final loss vs `n_articles` and vs model size (`n_layer` /
  `n_embd`) — a mini "scaling laws" curve for this setup.

## Rejected approaches (don't retry)

- **`take(first N)` of the stream** — front-of-stream = alphabetical/list-page bias.
- **Buffered stream-shuffle** (`.shuffle(buffer_size=B).take(N)`) — only samples the first
  ~`B+N` rows, so it stays pinned to the alphabetical front; bias persisted at
  `buffer_size=10_000`.
- **Non-streaming `load_dataset(split="train")`** — *hangs* in this WSL2 env; stalls in
  `datasets` data-files resolution on this giant multi-config parquet repo (only `README.md`
  ever fetched). `snapshot_download` of just the simple-config parquet sidesteps it.

## Known code smells / housekeeping

- `import math` is imported in [A] but unused until [K]/[L]/[N]/[O]/[Q] (kept under the
  "all imports in [A]" convention).

Resolved:

- ~~`import json` unused~~ — removed from [A] (the old `.jsonl` caches are gone; nothing else
  referenced it).
- ~~`ids` reused across [E]/[F]~~ — [E]'s module-level corpus stream is now `stream` and [F]'s
  demo is now `demo_ids`, so the two no longer collide. The `ids` *parameters* inside
  `get_stats`/`merge`/`encode`/`decode` are unchanged (minbpe-style, intentional).
