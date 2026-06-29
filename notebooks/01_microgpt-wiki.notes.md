# Notes — `01_microgpt-wiki.ipynb`

Design decisions and lessons for the microGPT-on-Wikipedia notebook.

## Data loading — final design (block [C])

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

BPE (block [E]) had learned a token for `", American"`. Two findings, worth remembering:

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

## Rejected approaches (don't retry)

- **`take(first N)` of the stream** — front-of-stream = alphabetical/list-page bias.
- **Buffered stream-shuffle** (`.shuffle(buffer_size=B).take(N)`) — only samples the first
  ~`B+N` rows, so it stays pinned to the alphabetical front; bias persisted at
  `buffer_size=10_000`.
- **Non-streaming `load_dataset(split="train")`** — *hangs* in this WSL2 env; stalls in
  `datasets` data-files resolution on this giant multi-config parquet repo (only `README.md`
  ever fetched). `snapshot_download` of just the simple-config parquet sidesteps it.

## Misc coherency

- `import math` is imported in [A] but unused until [J]/[K]/[O] (kept under the
  "all imports in [A]" convention). `import json` is currently unused (the old `.jsonl`
  caches are gone).
- `ids` is reused: [E] leaves it as the full-corpus token stream, [F] reassigns it to
  `encode(sample)`. Harmless since [G] re-encodes `corpus`.
