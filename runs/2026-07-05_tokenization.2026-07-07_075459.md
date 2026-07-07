# Tokenization: byte-level BPE trained from scratch

**Objective.** Implement byte-pair encoding.

**[A] Toy text.** Base alphabet is raw UTF-8 bytes (256 possible values).

Training loop:

1. Compute word frequencies for the corpus.
2. Initialize the vocabulary with the byte alphabet.
3. Tokenize each unique word (greedy longest-token match).
4. Compute the frequency of adjacent token pairs.
5. Find the most common pair.
6. Add the pair to the vocabulary as a new token.
7. Repeat from step 3 until the vocabulary reaches the target size.

```python
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
```

```
new_id=256 pair=(97, 116) -> b'at'
new_id=257 pair=(116, 104) -> b'th'
new_id=258 pair=(257, 101) -> b'the'
new_id=259 pair=(32, 258) -> b' the'
new_id=260 pair=(115, 256) -> b'sat'
new_id=261 pair=(111, 110) -> b'on'
new_id=262 pair=(32, 261) -> b' on'
new_id=263 pair=(32, 260) -> b' sat'
new_id=264 pair=(109, 256) -> b'mat'
new_id=265 pair=(99, 256) -> b'cat'
new_id=266 pair=(32, 265) -> b' cat'
new_id=267 pair=(32, 264) -> b' mat'
new_id=268 pair=(111, 103) -> b'og'
new_id=269 pair=(108, 268) -> b'log'
len(toy_vocab)=270
toy_bytes=95
toy_tokens=31
compression ratio = 3.06 bytes/token
```

**[B] Encode / decode.** Round-trip any text through the trained vocabulary.

1. Encode: pre-split into words, greedily tokenize each, concatenate the ids.
2. Decode: concatenate each token's bytes, decode UTF-8.
3. Verify decode(encode(text)) == text on training and held-out text.

```python


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
```

```
training: match=True tokens=31
held-out: match=True tokens=26
```

**[C] Clean Wikipedia prose.** Train on a few hand-picked prose articles from Simple
English Wikipedia (loaded via the pipeline in
notebooks/2026-07-04_training_data_acquisition.py) at the same small vocab target as the
toy text, so the only thing that changes from [A] is the corpus.

Articles are hand-picked to be prose, not stats tables: the source keeps raw wikitext
(rowspan, bgcolor, |-) in table-heavy articles like idx 50000 "Hans-Jörg Butt", which
would otherwise surface as junk merges.

```python
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
```

```
/home/trironkk/agent-worktrees/github.com/trironkk/tinyterp/a01/.venv/lib/python3.12/site-packages/tqdm/auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
  from .autonotebook import tqdm as notebook_tqdm


Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.


new_id=256 pair=(32, 116) -> b' t'
new_id=257 pair=(104, 101) -> b'he'
new_id=258 pair=(97, 110) -> b'an'
new_id=259 pair=(114, 105) -> b'ri'
new_id=260 pair=(105, 110) -> b'in'
new_id=261 pair=(256, 257) -> b' the'
new_id=262 pair=(101, 114) -> b'er'
new_id=263 pair=(32, 65) -> b' A'
new_id=264 pair=(111, 110) -> b'on'
new_id=265 pair=(97, 114) -> b'ar'
new_id=266 pair=(101, 110) -> b'en'
new_id=267 pair=(112, 259) -> b'pri'
new_id=268 pair=(267, 108) -> b'pril'
new_id=269 pair=(263, 268) -> b' April'
len(wiki_vocab)=270
wiki_bytes=28381
wiki_tokens=24110
compression ratio = 1.18 bytes/token
```

**[D] Raw sample, naive baseline.** Train the unoptimized train_bpe on a small random
sample of raw articles: the baseline [E] reproduces and then scales past. Even a small sample
takes ~10s at vocab 512, because the cost is the full re-tokenization per merge, not the
corpus size. Raw-markup contamination is too rare in a small sample to surface here; it shows
up at [E]'s larger scale.

1. Randomly sample a few raw articles (seeded).
2. Train the naive train_bpe; time it.
3. Report compression.

```python
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
```

```
trained vocab=512 on 31112 chars in 11.1s
raw_bytes=31251
raw_tokens=15520
compression ratio = 2.01 bytes/token
```

**[E] Incremental training.** train_bpe recomputes the pair counts from a full
re-tokenization every merge, which is why even [D]'s small sample is slow. But a word's
greedy tokenization can only change when the newly merged token appears in its bytes, so most
words are untouched each merge. Keep every word's tokenization and the global pair counts,
and each merge re-tokenize only the words the new token can appear in, adjusting the counts
by the difference. Same greedy result as train_bpe (same deterministic tie-break), far less
work.

1. Verify it reproduces [D]'s vocab and merges exactly, and report the speedup.
2. Scale to a corpus train_bpe could not finish in reasonable time (~10s here). At this
   scale the raw table markup ([C] hand-picked around) surfaces in the vocabulary.

```python
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
```

```
train_bpe               11.08s
train_bpe_incremental    0.24s   speedup = 45x
identical vocab:  True
identical merges: True


incremental on 4000 articles (4688996 chars): 9.4s
markup-matching tokens: [' align', ' bgcolor']


compression ratio = 1.89 bytes/token
```

**[F] Full corpus: train and analyze.** Train on the entire corpus with the incremental
trainer at vocab 2048, measure corpus-wide compression at several vocab sizes, and inspect
the learned vocabulary. Merges are learned in order, so the smaller vocabularies are prefixes
of the trained one and a single training run covers them all.

```python
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
```

```
trained vocab=2048 on 267960633 chars in 704s


vocab 512: 142754446 tokens, 1.89 bytes/token


vocab 1024: 116178598 tokens, 2.32 bytes/token


vocab 1536: 105876091 tokens, 2.55 bytes/token


vocab 2048: 99708369 tokens, 2.70 bytes/token
token lengths (bytes -> count): {1: 256, 2: 343, 3: 437, 4: 372, 5: 263, 6: 166, 7: 97, 8: 61, 9: 24, 10: 14, 11: 10, 13: 3, 14: 1, 15: 1}
tokens starting with a space (whole words): 889
longest tokens: [' establishments', ' International', ' municipality', ' Championship', ' professional', ' television', ' politician', ' University', 'ternational', ' department', ' California', ' Spacewatch', ' government', ' population', ' footballer', 'References', ' September', ' establish', ' different', ' President']
markup-matching tokens: [' align', ' bgcolor', 'colspan', 'fefefe', 'rowspan']
all merges:
he  t  a in er an on or  the es || is en ar at ed  o  w  ||  s al it  b  of  in ic  c  f re as nd  p  1 ro  S  m  and le  A ing  d ou  T ion  C  2 ol ig  (  M il  to am  is  h om us ent id el  P �  I  19  B  H ct 00  L  was et ad st  � ch  l  n ir em iv ay  al ers ot ef ur  F  N  D  The  R ist  th  re th  G  g op im ov  J  on ce un ht  for ow  W  E  k ian ber ly 19 ation rom ag and ight  — ut ies |- ra  st  
  be  from
 O  K oc  as  e nt ia ign um  200  20 ces os ul mer  U  by art ith col  an right ter ak  It ep od ish av ev ican  |  are color gcolor  align =#  bgcolor  km  with eb  at  v ity ain  St  He  pl  com 's ap  wh if eren  that est ther ab ill The erences ew Ref References ip eop  it ary up  or ec ember merican eople ard  " ac  V res 20  he ld ng ust all  pro  Ch ment ect  201 ug ud  – ear ant rit ort ong  199  3 ame ate ub  - ich orn ri og  con ere  Un ount  ch  mov ran eat se ge ie ast  people EA  his ern  play so ore  Soc
ue ive ord orro ry ost  us  In  act our man ok IN land ated ial ey ine EAR  Socorro  LIN  LINEAR  18 her ited  American ), ates sit ver own  also ork sp  4  Y  has  sp  sh  un  Mar ave lish  were ell ors ebsit  Th ack irst uary  ex  websit  not  y age ome ik  5 ions ical  bir du  bec ure ide ). 200 ths  United eath  comp out iver  which  ab ff  Al  first ational –  can Other any amp  7 ory  198 efe fe fefe ond fefefe iz ice  websites oot les ex ren  have  births ied ang  States  6  8 ach ph ball aw  ar ound end ne ck per ept olit fter ician  part qu eg  Sp    ous ob
 197 to rs  ro ts  one ober  New ss 18 199 te  they cl der  had  cl  9 ision  ser  ra aus orld ree  kn ood eptember  j mun io ass American  death  other ics  their port ib ater ade sh  Ar  September  She ime row ean ootball ild  who  Le  name In ric  se lect pp ll  17  year ton  
 
  196  but  tw ward ally ral  2000 ounty ace é son tion mp nce ress  call uch ident old ah  her  time outh  Fran erman  Jan 201 oh ctober ark  known  about  movies are ased orm ake  this ould iving ind ook ings  Eng orth  polit  October low ments  used  movie  10  They  2001 ram  te lev  dis 21 ists  mus  called ugust  16 =" pri
 This  March  Sh ance  work  some ten ock ince born one ities oy  made erson  15  after  sing ted  two als  deaths  January  August ult ations   198  le ina oug ick ire clud und  194  195 ans ite  sc  Nov  football  de  most  May  song  its  Dec  many tle irect we uring  born wn red ile  Z 22  includ ery rent  ad apan ition ages  writ  14  German  December  there vent  November  Ind  been pril  tra ns arge ames  city  telev atch  193 lic gan  television roup ebr ike az overn ause ebruary  into mon span  Aust  r  found  cont  ag  all  died ool ague  13  more  im me  ac  count ury resident  April ru ays my iss raph iversit  11  when ms  12  up ife  February
une  politician ens  direct  pop  Jul nc ft  per ix 197 bum pl ablish |||| ritish  Com  she  est ty  cent ail ale  team ash  County  live  col umber  off  Japan  app  bet  over  became  World  reg ublic  Con inist  out  Pri ins Living iversity ase ction ason arri 10 reat  fam  do cer  years  1999  member form  elect tern  Is  commun 17  new ily  France vel 

  actors  album igh  series ier icip  rec  produ yp  group  July tic  Pro ve  pri  ph  War  popul  Sc  number  played  Col  sy  establish  dist  South resent inal  2002  York ner na  Joh  King  rele eral  so  dif arl ative  only aj  dep uth  than ween  As 23 ath ient  like ning  sec oth  start  him ose  North  League ugh
anad icial olog able  bu  player  And  English uk  large 16  make rict rench uc 15 ouse rough  stud  state  British sc  June ved ". hip ever rist air ee 196  Eur uss way ures  between angu  them  gen rib icipal  because ai  University  Cl de hool oman  place unicipal ars  co  go  bl  diffe  again 14 13 ks  City  po  during  Q  Cal eak  north  where  show ron ting  Ph til other  man ism  Bl  very       An 25  pre side  Pal  would  end  spec  players ital ica  may  won Ch  creat  2020 artment ys  Austral  such ope  mean  Ital "| ax med ually  town  qu 24  person  music aid  rel ough  released 28 ious con ned mb  Europe  establishments  sm ual  Rep 26  under ium 30
29  At Com ternational pen dom  met  River  three 27  24  different  fl ert  high  until  second  sup  long rowspan 000 rop  Part dition  gover  Award ution  will ians  Canad  John  National  main ese arly ines ograph  cap  south ries ven  world ox  later  med hen  then  actor 50  II  ter ters areer eor  book ung  child  Man ta  French fore  govern  1998 go ser  win ds  national  set  age  through arch iet  2003  life  area  President vince  West urg ampions ced  department  histo  25 ley  Af  Fl  Par  songs  Pl 194 au OS  Mon  same  singer ble omen Rel  Russ  small  band  loc its rid  30  Ad  sub  season  former  Bra  four ling cc itt ional ield lymp 11  game lp ls ise  kill  langu  use  2010
 Space  fol iforn  take ats  art ino sign time  On arac  2004 rian ifornia  His  being reen ject ience ove ring ", ural  Mus  chang People tor fess  well  
 
 
 
  Car com  mod  back  mon  Party  California  Olymp  There han ney Mov ess ants  pages  record cess  often 195  did  For lo  named  started  get  2005  district iation  rep  no lands  Cent  old  2019 ank ilm  region omar  based  give stem 34 stit ky  before ather ons  Ab  system  Roman á  family  municipal ded ined rt  2017  ret  Palomar  water  Germany  you  said  Minist  Geor  career  own  bro burg Related  Wh  since  Christ  England ane  against  29  form  

 oice 37 12  Kitt  cause  actress 80  commune ondon  Will vice  law  Peak eal owever gy pt  games  trans
 sur urch  near  air  23 round illion ement  Pol ae ful  2018 cent  Gu gram co very ink watch  2014  26  help  .  charac  typ  follow 36 Movies  any  Ed  21  London by aug EAT  Cup  Spacewatch ham  Afric  government  NEAT ask  ent mper  organ  directed  best ampionship  inter 39 ze  these  28 aul  anim ody 35 wer ara  century He ness  become 40  event  2006 ideo iew  population gs  All gin  rem  27 rd rama  Republic merica  appear uld  &  country  around  including  Some  orig 38  Sw  2016 ross  Kingdom Th  word ony anish  land  comed eed alk  Ser ata  home  both ma  sever  website  America  X  war tra  school  atta  US ö lf  gu  2015 90  could ains  cur ajor ird  Bar 60 ’ ves ny
rand Un  Mich ke lin On  22 ille act ights aint  190  Japanese  2021 tt  import ales sident 70  marri  Scot  La  build  list ized  Indian  profess  2011 ology  East  each  en  run la ria munes  car  2007 vol cy  Anderson Death ilit ones  usually  lived oin tin  Australia ü  children day  Off  examp tal oard rew  Her cture  sim  hum  Gen  power  Mes ible iam NE  defe writ istan  drama  Mc  last ald  devel  dec ating colspan 99 istrict ger ute  After  created ows zer  rock "|-  Minister ture ctors  Air  began  ob  county lege ipp  several  program  sign "|-||  2013  2008  club  India graph aking  though ower  video  House  You ublish  attack  oper  while  common  gra nal Hist lie ular  inst  writer ummer Communes  vers  '
ocra í Eng  Italian soc arth  2012  res attle  important 33  suc umb  single  perform  Bel  official  we ilt amb  if 45  million ired  Lou ensus ivision  film ti  Mesa  went  rul ington  open  Pet  El ek hy  Bro  bus  State 
  
 
  
 
 
 
   2009 oss ull  Ex  William tes  municipality iction Sta Cities ni  language que zil 95  These fer ence  popular yle  now ó ilitary omb  compose It 47 icians cial  example  Phil 98  Pre 96 Deaths nts 48 ues  capital 97  design History ama ." the times aught  head  countries  early  Hol ried  LO ford  stat  short  president  San  However  ()  2022  Tur  ep west  Sec  compet  Se 46 speople  Be  Ber  groups  west  paint NEOS English  LONEOS  include ockey  Med  day  Ange  office ivil
ris  join  came van  species  Championship  Govern ..  means  leade  One  Swed mina  written conom riend ridge  Miss  Sch  left ration ead  publish  way rote  Sup  contro wards  Per ication  Charl  general ico  every  1997 
 
  history FA 2000  When ki isod  refer ka ya ided  pass ertain  down  times  inf  manage orts ncy  much  Que  east  Har dy ä  married  women 88 cept  ann  Dav  province 75 oon  what  Canada  cancer  famous  major |||||||| è 65  current  var  comedy  original ps utch uff line  support  still  voice  class iness  dri  professional  activ itle  members ease  1990 duc  river alth view oviet  built  father Sport  1996 eh  School mar  prese  NY  desc ounce  rece  vill  Gree  International Actors  Tex emocra  sold ville  how ush 2010 ography yd anne
 match itar ived  Rec  producer ered pe ffect 85  does  company  took roduc  largest  public idd east 31 ida yl  Saint  house ius  good  represent  business icago  Mex  moved  wrote urn raft  100 32  another cade  stage  belie  31 66 ael  Port  Paul  footballer  Net itch ournal work  hap rime  third do val  engin  success berg  Italy ext  George  held  son  poin ij 87  title  NYS  arch  body  feat Polit  Pen  China  Louis rest ler aster  lead gress  James  big  result iod  Union  even af  1995  need  close  order  Russian  director  given  comple  proble  Canadian bor  want  Cath  albums  top vers  Kore  deb graphy 94  phys  young ets atural 05  great uthor  political  Texas  Tra  books  Brazil urt isco 86  six  just  Rober  stru oe istics  rad  Its
```

**[F] Findings.**

- Raw table markup is pervasive, not a tail: `||` (a table cell separator) lands among the
  top merges of the whole corpus, and `bgcolor`, `align`, `fefefe`, and `References` all
  become tokens. Raw Simple Wiki needs prose extraction, as [C] and the acquisition notebook
  warned.
- The vocabulary is mostly short subword pieces; the longest tokens are common whole words
  (` television`, ` September`) alongside markup and stray proper nouns (` Socorro`, from the
  asteroid tables).
- Compression improves with vocabulary size but with diminishing returns (see the per-size
  numbers above).
- The full-corpus train finished in minutes via the incremental trainer; the naive train_bpe
  could not have reached this scale.

**[G] Persist.** Save the trained tokenizer to artifacts/ (gitignored) and load it back, so
later notebooks can restore it without retraining. Smaller vocabularies are prefixes of this
one, so a future notebook can truncate rather than retrain.

```python
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
```

```
saved artifacts/tokenizer_simplewiki_v2048.pkl (38181 bytes)
loaded vocab=2048 merges=1792
matches trained: True
round-trip through loaded tokenizer: True
```

## TODO

- Graduate the BPE tokenizer into the tinyterp/ package: train_bpe / train_bpe_incremental,
  build_tokenizer, and encode / decode, so training and inference are importable rather than
  living only in this notebook.
