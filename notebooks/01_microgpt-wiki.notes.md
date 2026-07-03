# Notes — `01_microgpt-wiki.ipynb`

Design decisions and lessons for the microGPT-on-Wikipedia notebook.

## Cell map

The notebook uses labeled cells (`## [X] …`) so notes can reference them. Backlog items below
are tagged with the cell(s) they touch.

| | | | |
|---|---|---|---|
| [A] Setup & dependencies | [G] Encode / decode | [M] MLP block | [S] Sampling |
| [B] Config | [H] Tokenize the corpus | [N] Transformer block | [T] Generate |
| [C] Load Wikipedia | [I] Batching | [O] GPT model | [U] Save / load |
| [D] Inspect the corpus | [J] Embeddings | [P] Optimizer & LR schedule | |
| [E] Pre-tokenization pattern | [K] RMSNorm | [Q] Training loop | |
| [F] Train a minimal BPE | [L] Multi-head causal self-attn | [R] Loss curves | |

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

## Tokenizer — final design [E]/[F]/[G]/[H]

GPT-2-style **regex pre-tokenization** (`regex` module, `\p{L}`/`\p{N}` classes) splits text
into word / number / punctuation chunks; BPE only ever merges a pair **within** a chunk. This
buys three things at once:

1. **Correctness.** The alternation is gap-free, so `pretokenize` partitions text with no
   dropped characters. [H] asserts a **full-corpus roundtrip** (`decode(tokenize(corpus)) ==
   corpus`) as the gate — the property the earlier byte-level `re`-translation silently lacked
   (see *Rejected approaches*).
2. **No cross-word artifacts.** Merges can't cross a space, so the `", American"` list-page
   token (below) is now structurally impossible.
3. **Speed, via caching.** Encoding is memoized per unique pre-token chunk (`_chunk_cache`), so
   tokenizing the whole corpus pays each unique chunk's merge cost once (~tens of thousands of
   uniques) instead of once per token (~tens of millions).

Structure:

- **[E] Pre-tokenization pattern** — the `GPT2_PAT` regex + `pretokenize()` + the shared
  low-level `get_stats`/`merge` primitives, with a gap-free assert on a demo string.
- **[F] Train a minimal BPE** — trains `vocab_size - 256` merges over the **whole corpus**.
  Dedup to unique chunks first (they grow *sublinearly* — ~0.9M uniques for 180k articles, only
  ~18× the 2k count), then **incremental** pair-count maintenance (minbpe-style): after each
  merge only the chunks that contained the merged pair are re-scanned. Full-corpus training is
  ~35s at 180k (measured: pretok ~15s + train ~20s over 861k unique chunks) — cheap enough that
  there's no need to train on a small slice, so the vocab's frequency tail stays representative.
- **[G] Encode / decode** — greedy merge-by-rank within each chunk (`encode_chunk`, cached),
  `decode` via the id→bytes table. Greedy-by-rank is the standard BPE encode (GPT-2/minbpe); it
  need not reproduce [F]'s exact training segmentation, but decode inverts it losslessly either
  way.
- **[H] Tokenize the corpus** — stream pre-tokens with `re.finditer` (lazy, no giant
  intermediate list) through the [G] cache into a compact `uint16` `array` → `torch.frombuffer`
  tensor (~2 bytes/token; [I]'s `get_batch` casts each batch to `long`). Full-corpus roundtrip
  assert, then 90/10 split.

Measured at full scale (`n_articles` 180k): BPE train **19.2s** over 861,118 unique chunks;
tokenize **86.3M tokens in 28.6s** at 2.31 bytes/token (vocab 1024); full-corpus roundtrip
assert passes.

## Lesson: the sample is biased, and byte-level BPE amplified it (the `", American"` story)

The old byte-level BPE (no pre-tokenization) had learned a token for `", American"`. Two
findings, worth remembering even though the regex split has since made the artifact impossible:

1. **The raw dataset is ~alphabetical by title.** Taking the first N articles gives an
   A-heavy, topically skewed slice (April, August, Art, A, Air, Alan Turing…). Year/date
   pages (1991, 2006…) and "Events" lists also cluster near the front. (Fixed by the uniform
   sample in [C].)
2. **Byte-level BPE merged on *global frequency*, so a few list pages dominated.** `", American"`
   hit 2,576 times but in only 90/2000 articles (~29 each), concentrated in birth/death list
   pages ("*June 3 – Name, American actor*"). Dense repetition in a handful of structured docs
   was enough to earn a merge — a **list-page artifact**, not broad usage.

Takeaway: the uniform sample keeps such artifacts out of the long-token tail, and regex
pre-tokenization ([E]) removes the *cross-word* class of them entirely (a merge can no longer
span the `", "` → `"American"` boundary). Truly taming list pages would still need
document-level dedup/normalization — not done.

## Research directions

### Scale the corpus to ~1 epoch — DONE (tokenizer reworked, `n_articles` = 180k) ([B]/[F]/[H])

**Status: implemented and merged.** The old build did 20000 steps × 32 × 128 = **82M tokens
seen** over only ~0.9M tokens of corpus (2000 articles) → **~90 passes**, the root-cause,
data-side driver of the overfitting below. Measured ~2.3 bytes/token, so **~1 epoch ≈ 180k
articles** (~88M tokens; the `20231101.simple` dump has 241,787). `n_articles` is now 180000.

The blocker was the old pure-Python *global* BPE: O(`num_merges` × `len`), ~127s to train and
~57s to tokenize just 2000 articles. Resolved by the tokenizer rework above:

- **Full-corpus BPE stays cheap** ([F]): dedup to unique chunks + incremental counts make
  training ~35s at 180k, so we train on the whole corpus and the vocab's frequency tail has no
  small-sample bias. The originally-planned decouple-onto-a-fixed-slice turned out unnecessary —
  **dedup, not slicing, is what kills the old O(`num_merges` × `len`) blocker** (unique chunks
  grow sublinearly, so the merge loop is bounded regardless of `n_articles`).
- **GPT-2 regex pre-tokenization** ([E]) makes encoding cacheable per unique chunk ([G]/[H]).
- **The correctness blocker is closed.** The earlier stdlib-`re` translation dropped characters
  (`decode(tokenize(corpus)) == corpus` → False); using the **`regex` module** with GPT-2's
  exact `\p{L}`/`\p{N}` pattern is gap-free, and [H] now asserts the full-corpus roundtrip.
- **Incremental pair counts** ([F], minbpe-style) close the leftover training-perf gap.

### Overfitting ([Q]/[R]) — RESOLVED by corpus scale

First run showed val cross-entropy *rising* while train kept falling — classic overfitting,
expected then: a tiny corpus (~2000 articles), a 4L/128D model with capacity to memorize,
multiple effective passes, and zero regularization.

**Root cause was corpus size, confirmed.** The scaled run (180k articles, 82M tokens seen vs
77.7M train tokens ≈ 1.06 passes; 20k steps in ~4 min on CUDA) shows **val decreasing
monotonically to the very end**: best val **2.609 at step 19999** (the final step), train there
2.673. The train/val gap over the last several evals bounces between −0.06 and +0.07 — pure
`eval_iters=50` sampling noise around zero. (2.609 nats/token ≈ perplexity 13.6 ≈ 1.63
bits/byte at 2.31 bytes/token.)

With the gap gone, the regularization toolkit below drops from *needed* to *optional hygiene* —
the levers that now bind are **model capacity and step count** (the loss curve was still
declining at 20k steps), i.e. the scaling-sweep backlog item.

Regularization toolkit (kept for reference; measure any of these against the [R] gap):

- *Early stopping / best-val checkpoint* — track the val minimum during [Q] and have [U]
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

More data alone closed the gap; early stopping / weight decay / dropout only become relevant
if a future config (bigger model, more steps, smaller corpus) reopens it.

## Backlog / chores

Small, well-scoped follow-ups. Each is tagged with the cell(s) it touches.

- **Config hygiene ([P]→[B]).** Promote `warmup_steps` / `min_lr` from [P] up into [B] so [B]
  stays the single source of truth for hyperparameters.
- **Gradient clipping ([Q]).** Add global-norm gradient clipping to the [Q] step (clip before
  `optimizer.step()`) — the standard stabilizer we left out; cheap insurance against loss spikes.
- **Ablations ([K]–[O]).** Toggle weight tying (tied vs separate LM head), swap learned-absolute
  positions for RoPE, and compare pre-norm vs post-norm — measure the loss impact of each.
- **Better sampling ([S]/[T]).** Add top-p / nucleus sampling and a repetition penalty to
  `generate`, and compare output quality against the current top-k.
- **KV cache ([L]/[T]).** Generation re-runs the full context every token; cache per-layer keys
  and values so each new token is O(1) work instead of O(T). Good systems exercise.
- **Scaling sweep ([B]).** Plot final loss vs `n_articles` and vs model size (`n_layer` /
  `n_embd`) — a mini "scaling laws" curve for this setup.
- **Wiki gap: BPE/tokenization page.** The KB has no dedicated article on byte-pair encoding or
  GPT-2 pre-tokenization (only microGPT/nanoGPT/transformer pages). Worth an `ingest`/`lint`
  pass — this build worked from parametric memory for the regex pattern.

## Rejected approaches (don't retry)

- **`take(first N)` of the stream** — front-of-stream = alphabetical/list-page bias.
- **Buffered stream-shuffle** (`.shuffle(buffer_size=B).take(N)`) — only samples the first
  ~`B+N` rows, so it stays pinned to the alphabetical front; bias persisted at
  `buffer_size=10_000`.
- **Non-streaming `load_dataset(split="train")`** — *hangs* in this WSL2 env; stalls in
  `datasets` data-files resolution on this giant multi-config parquet repo (only `README.md`
  ever fetched). `snapshot_download` of just the simple-config parquet sidesteps it.
- **Byte-level BPE with no pre-tokenization** — merged across word boundaries (the `", American"`
  artifact) and was O(`num_merges` × `len`), too slow to scale. Replaced by regex
  pre-tokenization + incremental training ([E]/[F]).
- **stdlib `re` translation of GPT-2's pattern** — not gap-free: silently dropped an edge char
  class, so `decode(tokenize(corpus)) == corpus` was False. Use the `regex` module with the
  exact `\p{L}`/`\p{N}` pattern instead.

## Known code smells / housekeeping

- `import math` is imported in [A] but unused until [L]/[M]/[O]/[P]/[R] (kept under the
  "all imports in [A]" convention).

Resolved:

- ~~pure-Python global BPE too slow / cross-word artifacts~~ — replaced by regex
  pre-tokenization + incremental within-chunk training ([E]/[F]); see *Tokenizer — final design*.
- ~~`import json` unused~~ — removed from [A] (the old `.jsonl` caches are gone; nothing else
  referenced it).
- ~~`ids` reused across [E]/[F]~~ — resolved in the tokenizer rework; the low-level
  `get_stats`/`merge`/`encode_chunk`/`decode` `ids` parameters are unchanged (minbpe-style,
  intentional).
