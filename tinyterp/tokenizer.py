"""Byte-level BPE tokenizer: training, encode/decode, and persistence.

The base alphabet is the 256 raw UTF-8 byte values, so any text round-trips
losslessly regardless of what was seen in training. Text is pre-split with the
GPT-2 regex before training and encoding, so tokens never cross word boundaries.

Tokenization is greedy over the vocabulary (longest matching token first), so the
merge table is a training record, not something re-tokenizing depends on.

References:
    - notebooks/2026-07-05_tokenization.py
"""

import pickle
from collections import Counter
from itertools import pairwise
from pathlib import Path
from typing import Callable

import regex as re

# vocab maps token id -> the token's raw bytes; merges maps the learned pair ->
# the id it was assigned, in the order pairs were merged.
Vocab = dict[int, bytes]
Merges = dict[tuple[int, int], int]

GPT2_PATTERN = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def build_tokenizer(vocab: Vocab) -> Callable[[tuple[int, ...]], tuple[int, ...]]:
    """Sort the vocabulary longest token first once, and return a greedy tokenizer
    that reuses that ordering for every word."""
    ordered_tokens = [(token_bytes, token_id) for token_id, token_bytes in vocab.items()]
    ordered_tokens.sort(key=lambda token: len(token[0]), reverse=True)

    def tokenize(word: tuple[int, ...]) -> tuple[int, ...]:
        """Greedily cover the word's bytes: at each position take the longest
        vocabulary token that matches the remaining prefix, emit it, and advance
        past it. Single-byte tokens are always present, so a match always exists and
        the position always advances."""
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


def train_bpe(text: str, vocab_size: int) -> tuple[Vocab, Merges]:
    """Naive reference trainer. Train BPE up to vocab_size; return the vocab and the
    merge table (the pairs learned, in order). Recomputes the pair counts from a full
    re-tokenization every merge; `train_bpe_incremental` produces the identical result
    with far less work."""
    # Pre-split the text into words.
    words = GPT2_PATTERN.findall(text)

    # Count how often each word occurs.
    word_freqs = Counter(tuple(word.encode("utf-8")) for word in words)

    # Initialize the vocabulary with each raw byte value.
    vocab: Vocab = {token_id: bytes([token_id]) for token_id in range(256)}

    merges: Merges = {}
    while len(vocab) < vocab_size:
        # Tokenize each unique word once (build_tokenizer sorts the vocab once).
        tokenize = build_tokenizer(vocab)
        tokenized_words = {word: tokenize(word) for word in word_freqs}

        # Count adjacent token pairs across the corpus.
        pair_counts: Counter = Counter()
        for word, token_ids in tokenized_words.items():
            for pair in pairwise(token_ids):
                pair_counts[pair] += word_freqs[word]

        # Most common pair, breaking ties by the pair itself so the choice is
        # deterministic and does not depend on dict insertion order. Exact frequency
        # ties are common, and the incremental trainer reproduces this result exactly.
        most_common_pair = max(pair_counts, key=lambda pair: (pair_counts[pair], pair))
        new_id = len(vocab)
        merges[most_common_pair] = new_id
        vocab[new_id] = vocab[most_common_pair[0]] + vocab[most_common_pair[1]]
    return vocab, merges


def train_bpe_incremental(text: str, vocab_size: int) -> tuple[Vocab, Merges]:
    """Faster train_bpe with the identical result. Keep each word's tokenization and
    the global pair counts; each merge, only re-tokenize the words whose bytes contain
    the new token and adjust the counts by the difference, instead of recomputing from
    scratch. Same greedy result as train_bpe (same deterministic tie-break)."""
    words = GPT2_PATTERN.findall(text)
    word_freqs = Counter(tuple(word.encode("utf-8")) for word in words)
    word_bytes = {word: bytes(word) for word in word_freqs}
    vocab: Vocab = {token_id: bytes([token_id]) for token_id in range(256)}
    merges: Merges = {}

    # With only single-byte tokens, greedy tokenization of a word is just its raw bytes.
    tokenized_words = {word: tuple(word) for word in word_freqs}
    pair_counts: Counter = Counter()
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

        # Only words whose bytes contain the new token can change their greedy
        # tokenization.
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


def encode(text: str, vocab: Vocab) -> list[int]:
    """Encode text to token ids: pre-split into words and greedily tokenize each.
    Tokens never cross word boundaries, matching how the vocabulary was trained."""
    tokenize = build_tokenizer(vocab)
    words = GPT2_PATTERN.findall(text)
    token_ids: list[int] = []
    for word in words:
        token_ids.extend(tokenize(tuple(word.encode("utf-8"))))
    return token_ids


def decode(token_ids: list[int], vocab: Vocab) -> str:
    """Decode token ids back to text: concatenate each token's bytes and decode UTF-8.
    The base vocabulary holds all 256 byte values, so any text encoded with a matching
    vocab round-trips exactly."""
    token_bytes = b"".join(vocab[token_id] for token_id in token_ids)
    return token_bytes.decode("utf-8")


def show_tokenization(text: str, vocab: Vocab) -> str:
    """Print (and return) the token boundaries for `text`: each token decoded and
    joined by `|`, for eyeballing how a string splits. A token whose bytes are not
    valid UTF-8 on their own (a multi-byte character split across tokens) is shown with
    the Unicode replacement character rather than raising."""
    pieces = [vocab[token_id].decode("utf-8", "replace") for token_id in encode(text, vocab)]
    rendered = "|".join(pieces)
    print(rendered)
    return rendered


def save_tokenizer(path: str | Path, vocab: Vocab, merges: Merges) -> None:
    """Pickle the trained tokenizer (vocab and merge table) to `path`, creating the
    parent directory if needed. The counterpart to `load_tokenizer`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump({"vocab": vocab, "merges": merges}, f)


def load_tokenizer(path: str | Path) -> tuple[Vocab, Merges]:
    """Load a tokenizer saved by `save_tokenizer`, returning (vocab, merges).

    The contract is load, don't retrain: training the full-corpus tokenizer is
    minutes of work, so notebooks and entry points restore the artifact instead.
    Smaller vocabularies are prefixes of a trained one, so truncate to the first N ids
    rather than retraining for a smaller target."""
    with Path(path).open("rb") as f:
        loaded = pickle.load(f)
    return loaded["vocab"], loaded["merges"]
