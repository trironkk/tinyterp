# %% [markdown]
# # Tokenization: byte-level BPE trained from scratch
#
# **Objective.** Implement byte-pair encoding.

# %% [markdown]
# **[A] Toy text.** Base alphabet is raw UTF-8 bytes (256 possible values).
#
# Training loop:
# 1) Compute word frequencies for the corpus.
# 2) Initialize the vocabulary with the byte alphabet.
# 3) Tokenize each unique word (greedy longest-token match).
# 4) Compute the frequency of adjacent token pairs.
# 5) Find the most common pair.
# 6) Add the pair to the vocabulary as a new token.
# 7) Repeat from step 3 until the vocabulary reaches the target size.
# %% [A] Toy text: GPT-2 regex pre-split + greedy byte-pair training loop
import regex as re
from collections import Counter
from itertools import pairwise

GPT2_PATTERN = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def build_tokenizer(vocab: dict[int, bytes]):
    """Sort the vocabulary longest token first once, and return a greedy tokenizer that
    reuses that ordering for every word."""
    ordered_tokens = [(token_bytes, token_id) for token_id, token_bytes in vocab.items()]
    ordered_tokens.sort(key=lambda token: len(token[0]), reverse=True)

    def tokenize(word: tuple[int, ...]) -> tuple[int, ...]:
        """Greedily cover the word's bytes: at each position take the longest vocabulary
        token that matches the remaining prefix, emit it, and advance past it. Single-byte
        tokens are always present, so a match always exists and the position always
        advances."""
        word_bytes = bytes(word)
        token_ids = []
        position = 0
        while position < len(word_bytes):
            remaining = word_bytes[position:]
            for token_bytes, token_id in ordered_tokens:
                if remaining.startswith(token_bytes):
                    token_ids.append(token_id)
                    position += len(token_bytes)
                    break
        return tuple(token_ids)

    return tokenize


def train_bpe(text: str, vocab_size: int) -> tuple[dict[int, bytes], dict[tuple[int, int], int]]:
    """Train BPE up to vocab_size; return the vocab and the merge table (the pairs
    learned, in order). Tokenization is greedy over the vocab, so `merges` is a
    training record, not something re-tokenizing depends on."""
    
    
    # Pre-split the text into words.
    words = GPT2_PATTERN.findall(text)

    # Count how often each word occurs
    word_freqs = Counter(tuple(word.encode("utf-8")) for word in words)

    # Initialize vocabular with each raw byte value.
    vocab = {token_id: bytes([token_id]) for token_id in range(256)}


    merges: dict[tuple[int, int], int] = {}
    while len(vocab) < vocab_size:
        # Tokenize each unique word once (build_tokenizer sorts the vocab once).
        tokenize = build_tokenizer(vocab)
        tokenized_words = {word: tokenize(word) for word in word_freqs}

        # Count adjacent token pairs across the corpus.
        pair_counts = Counter()
        for word, token_ids in tokenized_words.items():
            for pair in pairwise(token_ids):
                pair_counts[pair] += word_freqs[word]
        
        # Most common pair, breaking ties by the pair itself so the choice is deterministic
        # and does not depend on dict insertion order. Exact frequency ties are common (e.g.
        # repeated table markup in [D]), and a later cell reproduces this result incrementally.
        most_common_pair = max(pair_counts, key=lambda pair: (pair_counts[pair], pair))
        new_id = len(vocab)
        merges[most_common_pair] = new_id
        vocab[new_id] = vocab[most_common_pair[0]] + vocab[most_common_pair[1]]
    return vocab, merges


toy_text = (
    "the cat sat on the mat. the cat sat on the mat. "
    "the cat sat on the mat. the dog sat on the log."
)
toy_vocab_size = 270

toy_vocab, toy_merges = train_bpe(toy_text, toy_vocab_size)

for pair, new_id in toy_merges.items():
    print(f"{new_id=} {pair=} -> {toy_vocab[new_id]!r}")

# measure compression: tokenize the corpus with the trained vocabulary

# Count how often each word occurs.
toy_words = GPT2_PATTERN.findall(toy_text)
toy_word_freqs = Counter(tuple(word.encode("utf-8")) for word in toy_words)

toy_tokenize = build_tokenizer(toy_vocab)
toy_bytes = len(toy_text.encode("utf-8"))
toy_tokens = sum(len(toy_tokenize(word)) * count for word, count in toy_word_freqs.items())
print(f"{len(toy_vocab)=}")
print(f"{toy_bytes=}")
print(f"{toy_tokens=}")
print(f"compression ratio = {toy_bytes / toy_tokens:.2f} bytes/token")

# %% [markdown]
# **[B] Encode / decode.** Round-trip any text through the trained vocabulary.
#
# 1) Encode: pre-split into words, greedily tokenize each, concatenate the ids.
# 2) Decode: concatenate each token's bytes, decode UTF-8.
# 3) Verify decode(encode(text)) == text on training and held-out text.
# %% [B] Encode / decode + lossless round-trip


def encode(text: str, vocab: dict[int, bytes]) -> list[int]:
    """Encode text to token ids: pre-split into words and greedily tokenize each. Tokens
    never cross word boundaries, matching how the vocabulary was trained."""
    tokenize = build_tokenizer(vocab)
    words = GPT2_PATTERN.findall(text)
    token_ids = []
    for word in words:
        token_ids.extend(tokenize(tuple(word.encode("utf-8"))))
    return token_ids


def decode(token_ids: list[int], vocab: dict[int, bytes]) -> str:
    """Decode token ids back to text: concatenate each token's bytes and decode UTF-8."""
    token_bytes = b"".join(vocab[token_id] for token_id in token_ids)
    return token_bytes.decode("utf-8")


# Verify a lossless round-trip: decode(encode(text)) recovers the original exactly. The
# base vocabulary holds all 256 byte values, so any UTF-8 text round-trips, including
# characters never seen in training.
held_out_text = "the naïve cat 😺 sat — again."

for label, text in [("training", toy_text), ("held-out", held_out_text)]:
    token_ids = encode(text, toy_vocab)
    restored = decode(token_ids, toy_vocab)
    print(f"{label}: match={restored == text} tokens={len(token_ids)}")
    assert restored == text

# %% [markdown]
# **[C] Clean Wikipedia prose.** Train on a few hand-picked prose articles from Simple
# English Wikipedia (loaded via the pipeline in
# notebooks/2026-07-04_training_data_acquisition.py) at the same small vocab target as the
# toy text, so the only thing that changes from [A] is the corpus.
#
# Articles are hand-picked to be prose, not stats tables: the source keeps raw wikitext
# (rowspan, bgcolor, |-) in table-heavy articles like idx 50000 "Hans-Jörg Butt", which
# would otherwise surface as junk merges.
# %% [C] Clean Wikipedia subsample: train on hand-picked prose articles
from datasets import load_dataset

wiki_articles = load_dataset("wikimedia/wikipedia", "20231101.simple")["train"]

# Hand-picked clean prose (title in comment). Table-heavy articles are avoided on purpose.
prose_indices = [0, 1000, 2000, 3000]  # April, Chemical cell, Kofi Annan, Ice hockey
wiki_text = "\n\n".join(wiki_articles[idx]["text"] for idx in prose_indices)

# Same small target as the toy text, for a like-for-like compression comparison.
wiki_vocab_size = 270
wiki_vocab, wiki_merges = train_bpe(wiki_text, wiki_vocab_size)

for pair, new_id in wiki_merges.items():
    print(f"{new_id=} {pair=} -> {wiki_vocab[new_id]!r}")

# compression at this vocab target
wiki_bytes = len(wiki_text.encode("utf-8"))
wiki_tokens = len(encode(wiki_text, wiki_vocab))
print(f"{len(wiki_vocab)=}")
print(f"{wiki_bytes=}")
print(f"{wiki_tokens=}")
print(f"compression ratio = {wiki_bytes / wiki_tokens:.2f} bytes/token")

# %% [markdown]
# **[D] Raw sample, naive baseline.** Train the unoptimized train_bpe on a small random
# sample of raw articles: the baseline [E] reproduces and then scales past. Even a small sample
# takes ~10s at vocab 512, because the cost is the full re-tokenization per merge, not the
# corpus size. Raw-markup contamination is too rare in a small sample to surface here; it shows
# up at [E]'s larger scale.
#
# 1) Randomly sample a few raw articles (seeded).
# 2) Train the naive train_bpe; time it.
# 3) Report compression.
# %% [D] Raw sample: naive baseline (timed)
import random
import time

# A small random sample of raw articles (no prose filtering, unlike [C]). Seeded so the
# sample, and everything trained from it, is reproducible.
sample_indices = random.Random(0).sample(range(len(wiki_articles)), 25)
raw_text = "\n\n".join(wiki_articles[idx]["text"] for idx in sample_indices)

raw_vocab_size = 512
start = time.perf_counter()
raw_vocab, raw_merges = train_bpe(raw_text, raw_vocab_size)
elapsed = time.perf_counter() - start
print(f"trained vocab={len(raw_vocab)} on {len(raw_text)} chars in {elapsed:.1f}s")

# compression at this vocab target
raw_bytes = len(raw_text.encode("utf-8"))
raw_tokens = len(encode(raw_text, raw_vocab))
print(f"{raw_bytes=}")
print(f"{raw_tokens=}")
print(f"compression ratio = {raw_bytes / raw_tokens:.2f} bytes/token")

# %% [markdown]
# **[E] Incremental training.** train_bpe recomputes the pair counts from a full
# re-tokenization every merge, which is why even [D]'s small sample is slow. But a word's
# greedy tokenization can only change when the newly merged token appears in its bytes, so most
# words are untouched each merge. Keep every word's tokenization and the global pair counts,
# and each merge re-tokenize only the words the new token can appear in, adjusting the counts
# by the difference. Same greedy result as train_bpe (same deterministic tie-break), far less
# work.
#
# 1) Verify it reproduces [D]'s vocab and merges exactly, and report the speedup.
# 2) Scale to a corpus train_bpe could not finish in reasonable time (~10s here). At this
#    scale the raw table markup ([C] hand-picked around) surfaces in the vocabulary.
# %% [E] Incremental training: same result as train_bpe, far fewer recomputations
def train_bpe_incremental(text: str, vocab_size: int) -> tuple[dict[int, bytes], dict[tuple[int, int], int]]:
    """Faster train_bpe with the identical result. Keep each word's tokenization and the
    global pair counts; each merge, only re-tokenize the words whose bytes contain the new
    token and adjust the counts by the difference, instead of recomputing from scratch."""
    words = GPT2_PATTERN.findall(text)
    word_freqs = Counter(tuple(word.encode("utf-8")) for word in words)
    word_bytes = {word: bytes(word) for word in word_freqs}
    vocab = {token_id: bytes([token_id]) for token_id in range(256)}
    merges: dict[tuple[int, int], int] = {}

    # With only single-byte tokens, greedy tokenization of a word is just its raw bytes.
    tokenized_words = {word: tuple(word) for word in word_freqs}
    pair_counts = Counter()
    for word, token_ids in tokenized_words.items():
        for pair in pairwise(token_ids):
            pair_counts[pair] += word_freqs[word]

    while len(vocab) < vocab_size:
        if not pair_counts:
            break
        # Same deterministic tie-break as train_bpe, so the two produce identical merges.
        most_common_pair = max(pair_counts, key=lambda pair: (pair_counts[pair], pair))
        new_id = len(vocab)
        merges[most_common_pair] = new_id
        new_token = vocab[most_common_pair[0]] + vocab[most_common_pair[1]]
        vocab[new_id] = new_token

        # Only words whose bytes contain the new token can change their greedy tokenization.
        tokenize = build_tokenizer(vocab)
        for word, raw_bytes in word_bytes.items():
            if new_token not in raw_bytes:
                continue
            old_ids = tokenized_words[word]
            new_ids = tokenize(word)
            if new_ids == old_ids:
                continue
            # Move this word's contribution from its old pairs to its new pairs.
            freq = word_freqs[word]
            for pair in pairwise(old_ids):
                pair_counts[pair] -= freq
                if pair_counts[pair] <= 0:
                    del pair_counts[pair]
            for pair in pairwise(new_ids):
                pair_counts[pair] += freq
            tokenized_words[word] = new_ids
    return vocab, merges


# (1) Same corpus as [D]: identical result, without the full recompute each merge.
start = time.perf_counter()
fast_vocab, fast_merges = train_bpe_incremental(raw_text, raw_vocab_size)
fast_elapsed = time.perf_counter() - start
print(f"train_bpe             {elapsed:7.2f}s")
print(f"train_bpe_incremental {fast_elapsed:7.2f}s   speedup = {elapsed / fast_elapsed:.0f}x")
print(f"identical vocab:  {fast_vocab == raw_vocab}")
print(f"identical merges: {fast_merges == raw_merges}")
assert fast_vocab == raw_vocab and fast_merges == raw_merges

# (2) Scale to a corpus train_bpe could not finish in reasonable time.
big_indices = random.Random(0).sample(range(len(wiki_articles)), 4000)
big_text = "\n\n".join(wiki_articles[idx]["text"] for idx in big_indices)
start = time.perf_counter()
big_vocab, big_merges = train_bpe_incremental(big_text, raw_vocab_size)
big_elapsed = time.perf_counter() - start
print(f"incremental on {len(big_indices)} articles ({len(big_text)} chars): {big_elapsed:.1f}s")

# At this scale the raw table markup surfaces as learned tokens.
markers = [b"bgcolor", b"rowspan", b"colspan", b"align", b"fefefe", b"ffffff", b"style", b"px"]
matched = [t for t in big_vocab.values() if any(m in t for m in markers)]
print("markup-matching tokens:", sorted(t.decode("utf-8", "replace") for t in matched))

big_bytes = len(big_text.encode("utf-8"))
big_tokens = len(encode(big_text, big_vocab))
print(f"compression ratio = {big_bytes / big_tokens:.2f} bytes/token")

# %% [markdown]
# **[F] Full corpus: train and analyze.** Train on the entire corpus with the incremental
# trainer at vocab 2048, measure corpus-wide compression at several vocab sizes, and inspect
# the learned vocabulary. Merges are learned in order, so the smaller vocabularies are prefixes
# of the trained one and a single training run covers them all.
# %% [F] Full corpus: train at vocab 2048, compression per size, analyze
full_text = "\n\n".join(wiki_articles["text"])  # the entire corpus as one string
full_vocab_size = 2048
start = time.perf_counter()
full_vocab, full_merges = train_bpe_incremental(full_text, full_vocab_size)
print(f"trained vocab={len(full_vocab)} on {len(full_text)} chars in {time.perf_counter() - start:.0f}s")

# Compression at several vocab sizes. A smaller vocabulary is the first N ids of the trained
# one; truncate to that size and tokenize the corpus (deduplicated by word: each distinct word
# is tokenized once, so the token count equals encoding every occurrence).
full_word_freqs = Counter(tuple(word.encode("utf-8")) for word in GPT2_PATTERN.findall(full_text))
full_bytes = len(full_text.encode("utf-8"))
for size in [512, 1024, 1536, 2048]:
    vocab_n = {token_id: token for token_id, token in full_vocab.items() if token_id < size}
    tokenize = build_tokenizer(vocab_n)
    tokens = sum(len(tokenize(word)) * count for word, count in full_word_freqs.items())
    print(f"vocab {size}: {tokens} tokens, {full_bytes / tokens:.2f} bytes/token")

# --- analysis of the learned vocabulary ---
# Token length distribution (bytes per token).
length_counts = Counter(len(token) for token in full_vocab.values())
print("token lengths (bytes -> count):", dict(sorted(length_counts.items())))

# Whole-word tokens (leading space) vs word-internal pieces.
whole_words = [token for token in full_vocab.values() if token.startswith(b" ")]
print(f"tokens starting with a space (whole words): {len(whole_words)}")

# The longest tokens learned.
longest = sorted(full_vocab.values(), key=len, reverse=True)[:20]
print("longest tokens:", [token.decode("utf-8", "replace") for token in longest])

# Table-markup contamination that survives into the full-corpus vocabulary.
markers = [b"bgcolor", b"rowspan", b"colspan", b"align", b"fefefe", b"ffffff", b"style", b"px"]
matched = [token for token in full_vocab.values() if any(m in token for m in markers)]
print("markup-matching tokens:", sorted(token.decode("utf-8", "replace") for token in matched))

# All learned merges, in the order they were learned, 128 per line.
merge_tokens = [full_vocab[256 + i].decode("utf-8", "replace") for i in range(len(full_merges))]
print("all merges:")
for line_start in range(0, len(merge_tokens), 128):
    print(" ".join(merge_tokens[line_start:line_start + 128]))

# %% [markdown]
# **[F] Findings.**
# - Raw table markup is pervasive, not a tail: `||` (a table cell separator) lands among the
#   top merges of the whole corpus, and `bgcolor`, `align`, `fefefe`, and `References` all
#   become tokens. Raw Simple Wiki needs prose extraction, as [C] and the acquisition notebook
#   warned.
# - The vocabulary is mostly short subword pieces; the longest tokens are common whole words
#   (` television`, ` September`) alongside markup and stray proper nouns (` Socorro`, from the
#   asteroid tables).
# - Compression improves with vocabulary size but with diminishing returns (see the per-size
#   numbers above).
# - The full-corpus train finished in minutes via the incremental trainer; the naive train_bpe
#   could not have reached this scale.

# %% [markdown]
# **[G] Persist.** Save the trained tokenizer to artifacts/ (gitignored) and load it back, so
# later notebooks can restore it without retraining. Smaller vocabularies are prefixes of this
# one, so a future notebook can truncate rather than retrain.
# %% [G] Persist: save the tokenizer and load it back
import pickle
from pathlib import Path

artifacts_dir = Path("artifacts")
artifacts_dir.mkdir(exist_ok=True)
tokenizer_path = artifacts_dir / f"tokenizer_simplewiki_v{full_vocab_size}.pkl"
with tokenizer_path.open("wb") as f:
    pickle.dump({"vocab": full_vocab, "merges": full_merges}, f)
print(f"saved {tokenizer_path} ({tokenizer_path.stat().st_size} bytes)")

with tokenizer_path.open("rb") as f:
    loaded = pickle.load(f)
loaded_vocab, loaded_merges = loaded["vocab"], loaded["merges"]
print(f"loaded vocab={len(loaded_vocab)} merges={len(loaded_merges)}")

# The restored tokenizer matches, and still round-trips text losslessly.
sample = "The Socorro observatory discovered many minor planets in November."
restored = decode(encode(sample, loaded_vocab), loaded_vocab)
print(f"matches trained: {loaded_vocab == full_vocab and loaded_merges == full_merges}")
print(f"round-trip through loaded tokenizer: {restored == sample}")
assert loaded_vocab == full_vocab and loaded_merges == full_merges and restored == sample

# %% [markdown]
# ## TODO
#
# - Graduate the BPE tokenizer into the tinyterp/ package: train_bpe / train_bpe_incremental,
#   build_tokenizer, and encode / decode, so training and inference are importable rather than
#   living only in this notebook.
