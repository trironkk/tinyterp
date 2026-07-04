# %% [markdown]
# # Training data acquisition: Simple English Wikipedia
#
# **Objective.** Pull a raw-text training corpus from HuggingFace, verify it
# lands in the shared cache (persisting nothing in the worktree), and
# characterize it as text. Tokenization is deferred to its own notebook; this
# stops at the word boundary.

# %% [markdown]
# **[A] Load.** `wikimedia/wikipedia` is the current canonical Wikipedia export
# on HuggingFace (the older `wikipedia` dataset is deprecated and requires
# `apache-beam` to build from raw dumps); config `20231101.simple` is the
# Simple English wiki as of that dump date, pre-cleaned to one article per row
# with wikitext markup already stripped. Full download, not streaming: the
# corpus is small enough (hundreds of MB) to characterize in full, and later
# cells need global passes (a corpus-wide word count) that streaming cannot
# random-access.
#
# The download lands in the shared HuggingFace cache
# (`~/.cache/huggingface/datasets`), not the worktree. `$HOME` is common across
# all worktrees, so the first pull populates the cache and every other worktree
# hits it for free: acquisition persists nothing local. `HF_DATASETS_CACHE` is
# printed to make that concrete.

# %% [A] Load: Simple English Wikipedia via HuggingFace datasets
import datasets
from datasets import load_dataset

ds = load_dataset("wikimedia/wikipedia", "20231101.simple")

print(f"{datasets.config.HF_DATASETS_CACHE=}")
print(f"{ds=}")
articles = ds["train"]
print(f"{len(articles)=}")

# %% [markdown]
# **[B] Schema & scale.** All four columns are plain strings: `id`, `url`,
# `title`, `text`. Only `text` (and to a lesser extent `title`) is training
# material; `id`/`url` are provenance the training pipeline will drop.
#
# Two sizes are worth separating. `download_size` is the compressed parquet
# actually fetched over the network (~150 MiB); `dataset_size` is the
# uncompressed Arrow the loader materializes and memory-maps (~278 MiB, one
# shard). The ~1.9x gap is parquet's columnar compression on natural-language
# text. Both live under the shared cache, so the 278 MiB is paid once per
# machine, not once per worktree.

# %% [B] Schema & scale: row structure, article count, on-disk cache size
n = len(articles)
print(f"{articles.features=}")
print(f"{n=}")
print(f"{articles.info.download_size / 2**20=:.1f}")  # MiB, compressed parquet
print(f"{articles.info.dataset_size / 2**20=:.1f}")  # MiB, uncompressed arrow
print(f"{articles.cache_files=}")

# %% [markdown]
# **[C] Peek.** Make the corpus tangible: this string content is exactly what a
# tokenizer will later see. Two articles bracket the range, a long one (`April`,
# ~16k chars) and a short one (`Dublin`, ~1k), each shown as a head snippet plus
# metadata.
#
# Caveat surfaced here: the "pre-cleaned" claim holds for prose but not for
# tables. Article bodies drawn from infoboxes or statistics tables retain raw
# wikitext (`rowspan`, `colspan`, `|-` row separators); e.g. the footballer
# `Hans-Jörg Butt` (idx 50000) is largely a career-stats table. Prose-only
# filtering is therefore a real downstream concern, not assumed away by the
# source.

# %% [C] Peek: render sample articles
def show(idx: int, head: int = 600) -> None:
    """Print one article's metadata and a head snippet of its text."""
    a = articles[idx]
    print(f"--- idx={idx} chars={len(a['text'])} ---")
    print(f"{list(a.keys())=}")
    print(f"{a['id']=}")
    print(f"{a['url']=}")
    print(f"{a['title']=}")
    print(f"{a['text'][:head] + ('...' if len(a['text']) > head else '')=}")
    print()


show(0)
show(100)

# %% [markdown]
# **[D] Characterize.** Per-article size in two units: raw character length and
# word count. "Word" is fixed here for the whole notebook as a lowercased run of
# Unicode letters (`\p{L}+`), so case folds together and punctuation, digits,
# and apostrophe/hyphen boundaries split (`don't` becomes `don`, `t`). This
# single definition is reused by [E]; alternatives (keeping digits, keeping
# intra-word apostrophes) were rejected to keep the vocabulary purely lexical.
#
# Method: one pass building both per-article lists, then percentile summaries
# and histograms. The word pass runs the regex over ~267M characters and takes
# a few seconds; the char pass is trivial.
#
# Results: both distributions are heavily right-skewed. Median article is 514
# chars / 76 words but the mean is 1106 / 164, pulled up by a long tail topping
# out near 237k chars (29.5k words). A stub tail sits at the other end (min 1
# char). Log-spaced bins with log-log axes render both tails; linear bins
# collapse nearly all mass into the first bucket. Totals across 241,787
# articles: ~267M characters, ~39.7M words, the corpus scale later token-budget
# planning starts from.

# %% [D] Characterize: per-article length distributions (chars, words)
import numpy as np
import regex as re
import matplotlib.pyplot as plt

WORD = re.compile(r"\p{L}+")


def words(text: str) -> list[str]:
    """Lowercased Unicode-letter runs: the notebook's single 'word' definition."""
    return WORD.findall(text.lower())


texts = articles["text"]
char_lens = [len(t) for t in texts]
word_counts = [len(words(t)) for t in texts]


def summary(name: str, xs: list[int]) -> None:
    """Print min, median, mean, p90, p99, max for a length series."""
    s = sorted(xs)
    n = len(s)
    q = lambda p: s[int(p * (n - 1))]
    print(f"{name}: min={s[0]} p50={q(.5)} mean={sum(xs) // n} p90={q(.9)} p99={q(.99)} max={s[-1]}")


summary("chars", char_lens)
summary("words", word_counts)
print(f"{sum(char_lens)=}")
print(f"{sum(word_counts)=}")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, (label, xs) in zip(axes, [("chars / article", char_lens), ("words / article", word_counts)]):
    bins = np.logspace(0, np.log10(max(xs)), 50)
    ax.hist(xs, bins=bins)
    ax.set(xlabel=label, ylabel="articles", xscale="log", yscale="log")
fig.suptitle("Simple English Wikipedia: per-article length distributions")
plt.show()

# %% [markdown]
# **[E] Word frequency by length.** Corpus vocabulary: total token count, unique
# type count, and the five commonest words of each character length. Reuses
# [D]'s `words()`, so the total here must equal [D]'s per-article word sum, a
# built-in consistency check across the two independent passes.
#
# Method: one global `Counter` over the letter-tokens, then bucket the frequency
# ranking by word length and keep the top five per bucket. Lengths 1 to 20 are
# printed; the observed max length is 85, a tail of concatenated artifacts not
# worth showing.
#
# Results: 39,651,554 tokens (matching [D]) across 592,840 unique types. Short
# buckets are the expected Zipf function words (`the`, `of`, `in`, `and`). The
# per-length view's real payoff is quantifying the table-markup contamination
# [C] flagged: `right` (195k), `align` (188k), `bgcolor` (186k), and the hex
# color `fefefe` (70k) rank among the commonest 5-7 letter "words", none of them
# prose. `references` (157k, section headers) and category-suffix plurals
# (`establishments`, `politicians`) likewise reflect Wikipedia boilerplate over
# Simple English usage. Conclusion: raw dumps plus a letter-only split yield a
# vocabulary whose head is partly structural, reinforcing that prose extraction
# and filtering are prerequisites for a clean training corpus, not optional
# polish.

# %% [E] Word frequency by length: total word count, commonest words per length
from collections import Counter, defaultdict

vocab = Counter()
for t in texts:
    vocab.update(words(t))

print(f"{sum(vocab.values())=}")
print(f"{len(vocab)=}")

by_len: dict[int, list[tuple[str, int]]] = defaultdict(list)
for w, k in vocab.most_common():
    if len(by_len[len(w)]) < 5:
        by_len[len(w)].append((w, k))

for length in range(1, 21):
    row = by_len.get(length, [])
    print(f"len {length:2d}: " + "  ".join(f"{w}({k})" for w, k in row))
