# %% [markdown]
# # RL'ing behaviors out of a base model with DPO: a uniform catalog
#
# **Objective.** Post-training, the *preference optimization* curriculum item. Fine-tune the base
# transformer directly against synthetic preference pairs with the DPO loss — no separate reward
# network. The implicit reward of a sequence is `beta * (logp_policy - logp_ref)`; the loss is a
# classifier pushing the trainable *policy* to score `chosen` above `rejected` relative to a frozen
# *reference* copy of itself:
#
# ```
# L = -logsigmoid( beta * ( (logp_pol(chosen) - logp_ref(chosen))
#                         - (logp_pol(rej)    - logp_ref(rej)) ) )
# ```
#
# where `logp` sums the log-probability of the *continuation* tokens under each model.
#
# **This notebook is a catalog.** One shared setup + machinery block, then the same three-cell
# template repeated per behavior we want to suppress: **(1) construction** of `(prompt, chosen,
# rejected)` pairs where `rejected` *is* the behavior; **(2) the RL loop** (DPO from a fresh policy);
# **(3) metrics & samples** — held-out preference accuracy (base→DPO, with 95% CIs), the reward-margin
# decomposition, a shared generation health panel, and base-vs-DPO greedy samples.
#
# Six behaviors, all built from the corpus + the model itself — no human labels, no LLM generation:
#
# | # | behavior (the `rejected` side) | construction |
# |---|---|---|
# | A | scrambled word order | shuffle the real continuation's words |
# | B | dropped words | delete ~30% of the real continuation's words |
# | C | off-distribution self-samples | the base model's *own* sampled continuation |
# | D | repetition loops (suggestion 1) | tile the continuation's first half |
# | E | metadata / boilerplate collapse (suggestion 2) | a mined "References / Categories" tail |
# | F | topic drift (suggestion 5) | a *different* article's continuation |
#
# The shared **generation panel** (`rep` = repeated-3-gram fraction = looping/degeneracy; `meta` =
# fraction of samples emitting Wikipedia metadata tokens like "References / Categories"; `overlap` =
# prompt↔continuation content-word Jaccard = on-topicality) lets us read each behavior's *own* target
# metric move — and any cross-effects — on the same axes. (`rep` and `meta` are deliberately separate:
# newline-loop degeneracy shows up in `rep`, not `meta`, so the two failure modes don't conflate.)

# %% [markdown]
# **[S1] Setup.** Two copies of the 27.5M base checkpoint: a frozen `ref` (grads off, the fixed anchor)
# and — per run, inside `train_dpo` — a fresh trainable `policy`. One shared pool of `(prompt, chosen)`
# examples: split each Simple Wikipedia article into sentences, take an adjacent pair (prompt = a
# sentence, chosen = the next). Every behavior reuses this pool and differs only in how `rejected` is
# built. Prompt and continuation are tokenized independently (continuation with a leading space) and
# concatenated, so the continuation-token span is exact.
# %% [S1] Setup: frozen reference, shared corpus, shared (prompt, chosen) pool
import math
import random
import re
from collections import defaultdict

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from datasets import load_dataset

from tinyterp import cache_path, encode, get_device, load_model, load_tokenizer

device = get_device()
SEED = 0
random.seed(SEED)
torch.manual_seed(SEED)

vocab, merges = load_tokenizer(cache_path("tokenizer_simplewiki_v2048.pkl"))
BASE_CKPT = cache_path("transformer_simplewiki_e40ffbb37f24.pt")

ref, config, _ = load_model(BASE_CKPT)
ref.to(device).eval()
for p in ref.parameters():
    p.requires_grad_(False)

articles = load_dataset("wikimedia/wikipedia", "20231101.simple")["train"]

MIN_WORDS = 6  # continuations shorter than this make the distortions trivial


def decode_ids(ids: list[int]) -> str:
    return b"".join(vocab[t] for t in ids).decode("utf-8", "replace")


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.replace("\n", " "))
    return [p.strip() for p in parts if p.strip()]


def build_pool(n_examples: int, seed: int = 0) -> list[dict]:
    """One (prompt, chosen) example per article, tokenized once, until we have n_examples."""
    rng = random.Random(seed)
    order = list(range(len(articles)))
    rng.shuffle(order)
    pool: list[dict] = []
    for ai in order:
        sents = split_sentences(articles[ai]["text"])
        valid = [i for i in range(len(sents) - 1) if len(sents[i + 1].split()) >= MIN_WORDS]
        if not valid:
            continue
        i = rng.choice(valid)
        prompt_ids = encode(sents[i], vocab)
        chosen_ids = encode(" " + sents[i + 1], vocab)
        if not prompt_ids or len(prompt_ids) + len(chosen_ids) > config.block_size:
            continue
        pool.append(
            {
                "prompt_text": sents[i],
                "prompt_ids": prompt_ids,
                "chosen_text": sents[i + 1],
                "chosen_ids": chosen_ids,
            }
        )
        if len(pool) >= n_examples:
            break
    return pool


N_POOL = 5000
N_TRAIN = 4000
pool = build_pool(N_POOL, seed=SEED)
train_slice, held_slice = pool[:N_TRAIN], pool[N_TRAIN:]
panel_prompts = held_slice[:128]  # fixed prompt set for the generation panel (behavior-independent)
n_params = sum(p.numel() for p in ref.parameters())
print(f"{device=}  {n_params/1e6=:.1f}  block={config.block_size}")
print(f"{len(pool)=}  {len(train_slice)=}  {len(held_slice)=}  {len(panel_prompts)=}")

# %% [markdown]
# **[S2] Shared machinery.** Everything the per-behavior cells call. `masked_logp` is the core
# primitive (summed log-prob over continuation tokens only; the mask zeroes prompt + right-padding,
# safe under causal attention). `dpo_batch` is the loss plus diagnostics (reward margin, and the
# policy's log-prob *drift* from the reference on each side — the honest read on whether the margin
# moves by lifting chosen or crushing rejected). `train_dpo` starts a fresh policy from the base
# checkpoint every call, so behaviors are independent and comparable. The **generation panel**
# (`generation_report`, defined here) batches greedy decoding (length-bucketed — this model has no
# attention-mask input, so padding would corrupt positions) and scores the three health metrics (see
# the title cell) on the same 128 prompts.
# %% [S2] Machinery: masked_logp, dpo_batch, train_dpo, generation, panel, eval
BETA, LR, STEPS, BATCH = 0.1, 1e-5, 300, 16
N_GEN = 40
TRIGGERS = ("References", "Other websites", "Related pages", "Living people", "births", "deaths")


def collate(batch: list[dict], key: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack `prompt_ids || batch[key]` into right-padded [B, L]; mask is 1 on continuation tokens."""
    seqs = [ex["prompt_ids"] + ex[key] for ex in batch]
    length = max(len(s) for s in seqs)
    seq = torch.zeros(len(seqs), length, dtype=torch.long)
    mask = torch.zeros(len(seqs), length)
    for i, ex in enumerate(batch):
        s = ex["prompt_ids"] + ex[key]
        seq[i, : len(s)] = torch.tensor(s)
        mask[i, len(ex["prompt_ids"]) : len(s)] = 1.0
    return seq.to(device), mask.to(device)


def masked_logp(model, seq: torch.Tensor, cont_mask: torch.Tensor) -> torch.Tensor:
    """Summed log-prob of the masked (continuation) tokens under `model`. Returns [B]."""
    logp = F.log_softmax(model(seq), dim=-1)
    target = seq[:, 1:]
    tok_logp = logp[:, :-1, :].gather(-1, target.unsqueeze(-1)).squeeze(-1)
    return (tok_logp * cont_mask[:, 1:]).sum(-1)


def dpo_batch(policy, ref, batch: list[dict], beta: float = BETA) -> tuple[torch.Tensor, dict]:
    ch_seq, ch_mask = collate(batch, "chosen_ids")
    rej_seq, rej_mask = collate(batch, "rejected_ids")
    pol_ch, pol_rej = masked_logp(policy, ch_seq, ch_mask), masked_logp(policy, rej_seq, rej_mask)
    with torch.no_grad():
        ref_ch, ref_rej = masked_logp(ref, ch_seq, ch_mask), masked_logp(ref, rej_seq, rej_mask)
    gap = beta * ((pol_ch - pol_rej) - (ref_ch - ref_rej))
    loss = -F.logsigmoid(gap).mean()
    stats = {
        "loss": loss.item(),
        "margin": gap.mean().item(),
        "acc": (gap > 0).float().mean().item(),
        "chosen_drift": (pol_ch - ref_ch).mean().item(),
        "rejected_drift": (pol_rej - ref_rej).mean().item(),
    }
    return loss, stats


def train_dpo(train_pairs, steps=STEPS, lr=LR, beta=BETA, batch=BATCH, seed=0):
    """Fresh policy from the base checkpoint, DPO-trained against the frozen `ref`."""
    torch.manual_seed(seed)
    rng = random.Random(seed)
    policy, _, _ = load_model(BASE_CKPT)
    policy.to(device).train()
    opt = torch.optim.AdamW(policy.parameters(), lr=lr)
    for step in range(steps):
        loss, stats = dpo_batch(policy, ref, rng.sample(train_pairs, batch), beta)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 50 == 0 or step == steps - 1:
            print(f"  step {step:3d}  loss {stats['loss']:.4f}  margin {stats['margin']:+.3f}  "
                  f"drift ch {stats['chosen_drift']:+.2f} rej {stats['rejected_drift']:+.2f}")
    policy.eval()
    return policy


@torch.no_grad()
def generate(model, prompt_text, max_new_tokens=N_GEN, temperature=0.8, greedy=False, seed=0):
    model.eval()
    torch.manual_seed(seed)
    block = config.block_size
    ids = torch.tensor([encode(prompt_text, vocab)], device=device)
    start = ids.shape[1]
    for _ in range(max_new_tokens):
        logits = model(ids[:, -block:])[:, -1, :]
        nxt = logits.argmax(-1, keepdim=True) if greedy else \
            torch.multinomial(F.softmax(logits / temperature, dim=-1), 1)
        ids = torch.cat([ids, nxt], dim=1)
    return decode_ids(ids[0, start:].tolist())


@torch.no_grad()
def batched_generate(model, prompt_ids_list, n_new=N_GEN, seed=0):
    """Greedy continuations for many prompts; length-bucketed so each batch runs unpadded."""
    torch.manual_seed(seed)
    model.eval()
    block = config.block_size
    cap = block - n_new
    conts: list = [None] * len(prompt_ids_list)
    buckets = defaultdict(list)
    for i, p in enumerate(prompt_ids_list):
        buckets[len(p[-cap:])].append(i)
    for _, idxs in buckets.items():
        ids = torch.tensor([prompt_ids_list[i][-cap:] for i in idxs], device=device)
        lp = ids.shape[1]
        for _ in range(n_new):
            logits = model(ids[:, -block:])[:, -1, :]
            ids = torch.cat([ids, logits.argmax(-1, keepdim=True)], dim=1)
        for j, i in enumerate(idxs):
            conts[i] = ids[j, lp:].tolist()
    return conts


def rep_fraction(ids, n=3) -> float:
    """Fraction of repeated n-grams: 1 - distinct-n. Higher = more looping."""
    if len(ids) <= n:
        return 0.0
    grams = [tuple(ids[k : k + n]) for k in range(len(ids) - n + 1)]
    return 1 - len(set(grams)) / len(grams)


def content_words(text: str) -> set:
    return set(re.findall(r"[a-z]{4,}", text.lower()))


def generation_report(model, prompts, seed=0) -> dict:
    """Health panel over a fixed prompt set, from greedy generations: `rep` = repeated-3-gram fraction
    (looping/degeneracy), `meta` = genuine metadata-token emission ("References / Categories"),
    `overlap` = prompt↔continuation content-word Jaccard (on-topicality). `meta` is trigger tokens
    only — newline-loop degeneracy lands in `rep`, so the two failure modes stay separated."""
    conts = batched_generate(model, [ex["prompt_ids"] for ex in prompts], seed=seed)
    rep = meta = overlap = 0.0
    for ex, c in zip(prompts, conts):
        rep += rep_fraction(c)
        txt = decode_ids(c)
        meta += 1.0 if any(t in txt for t in TRIGGERS) else 0.0
        pcw, ccw = content_words(ex["prompt_text"]), content_words(txt)
        overlap += len(pcw & ccw) / len(pcw | ccw) if (pcw | ccw) else 0.0
    n = len(prompts)
    return {"rep": rep / n, "meta": meta / n, "overlap": overlap / n}


@torch.no_grad()
def pref_accuracy(model, pairs, bs=64) -> float:
    """Fraction of pairs the model scores chosen > rejected (raw continuation log-prob)."""
    correct = 0
    for i in range(0, len(pairs), bs):
        b = pairs[i : i + bs]
        ch = masked_logp(model, *collate(b, "chosen_ids"))
        rej = masked_logp(model, *collate(b, "rejected_ids"))
        correct += (ch > rej).sum().item()
    return correct / len(pairs)


@torch.no_grad()
def margin_stats(policy, ref, pairs, beta=BETA, bs=64) -> dict:
    tot = len(pairs)
    agg = dict(margin=0.0, chosen_drift=0.0, rejected_drift=0.0)
    for i in range(0, tot, bs):
        b = pairs[i : i + bs]
        pc, pr = masked_logp(policy, *collate(b, "chosen_ids")), masked_logp(policy, *collate(b, "rejected_ids"))
        rc, rr = masked_logp(ref, *collate(b, "chosen_ids")), masked_logp(ref, *collate(b, "rejected_ids"))
        agg["margin"] += (beta * ((pc - pr) - (rc - rr))).sum().item()
        agg["chosen_drift"] += (pc - rc).sum().item()
        agg["rejected_drift"] += (pr - rr).sum().item()
    return {k: v / tot for k, v in agg.items()}


def ci95(p, n):
    return 1.96 * math.sqrt(p * (1 - p) / n)


RESULTS: dict[str, dict] = {}


def report_behavior(name, policy, held_pairs):
    """Uniform metrics + samples block: pref-acc base→DPO (+CI), margin/drift, generation panel
    base→DPO, and four base-vs-DPO greedy samples. Records into RESULTS for the closing summary."""
    n = len(held_pairs)
    base_acc, dpo_acc = pref_accuracy(ref, held_pairs), pref_accuracy(policy, held_pairs)
    ms = margin_stats(policy, ref, held_pairs)
    panel = generation_report(policy, panel_prompts)
    print(f"[{name}]  preference accuracy  base {base_acc:.3f} ± {ci95(base_acc, n):.3f}"
          f"  ->  DPO {dpo_acc:.3f} ± {ci95(dpo_acc, n):.3f}   (n={n})")
    print(f"   reward margin {ms['margin']:+.2f}   drift  chosen {ms['chosen_drift']:+.2f}"
          f"   rejected {ms['rejected_drift']:+.2f}")
    print(f"   gen panel   rep {BASE_PANEL['rep']:.3f}->{panel['rep']:.3f}"
          f"   meta {BASE_PANEL['meta']:.3f}->{panel['meta']:.3f}"
          f"   overlap {BASE_PANEL['overlap']:.3f}->{panel['overlap']:.3f}")
    RESULTS[name] = {"base_acc": base_acc, "dpo_acc": dpo_acc, **ms, "panel": panel}
    for ex in held_pairs[:4]:
        pt = ex["prompt_text"]
        print("   PROMPT:", pt[:96])
        print("     base:", repr(generate(ref, pt, greedy=True)))
        print("     DPO :", repr(generate(policy, pt, greedy=True)))


BASE_PANEL = generation_report(ref, panel_prompts)
print(f"base generation panel: rep {BASE_PANEL['rep']:.3f}  meta {BASE_PANEL['meta']:.3f}"
      f"  overlap {BASE_PANEL['overlap']:.3f}")

# %% [markdown]
# ## Behavior A — scrambled word order
# **[A1] Construction.** `rejected` = the real continuation with its words shuffled (retry so it
# differs). The reward-model notebook found this is ~100% linearly separable on frozen features — the
# base already ranks ordered text above scrambled, so expect little headroom (base pref-acc near 1).
# %% [A1] Construct scrambled-word-order pairs
def shuffle_words(text, rng):
    words = text.split()
    out = words[:]
    for _ in range(10):
        rng.shuffle(out)
        if out != words:
            break
    return " ".join(out)


def attach_distortion(examples, fn, seed=0):
    rng = random.Random(seed)
    out = []
    for ex in examples:
        rej = fn(ex["chosen_text"], rng)
        out.append({**ex, "rejected_text": rej, "rejected_ids": encode(" " + rej, vocab)})
    return out


shuffle_pairs = attach_distortion(pool, shuffle_words, seed=SEED)
shuffle_tr, shuffle_he = shuffle_pairs[:N_TRAIN], shuffle_pairs[N_TRAIN:]
for ex in shuffle_he[:2]:
    print("CHOSEN:", repr(ex["chosen_text"]))
    print("REJECT:", repr(ex["rejected_text"]), "\n")

# %% [A2] RL loop
policy_shuffle = train_dpo(shuffle_tr, seed=SEED)

# %% [A3] Metrics & samples
report_behavior("A: word-shuffle", policy_shuffle, shuffle_he)

# %% [markdown]
# ## Behavior B — dropped words
# **[B1] Construction.** `rejected` = the real continuation with ~30% of words deleted (always at
# least one). Corrupts completeness rather than order. Note this is the one construction that is *not*
# length-matched — deletion shortens `rejected`, and summed log-prob is length-biased (fewer negative
# terms), so the base actually *prefers* the truncated version (pref-acc below chance). Read the
# base→DPO move here partly as DPO overcoming that length bias, not purely a completeness preference.
# %% [B1] Construct word-deletion pairs
def delete_words(text, rng, frac=0.3):
    words = text.split()
    keep = [w for w in words if rng.random() > frac]
    if len(keep) == len(words):
        del keep[rng.randrange(len(keep))]
    return " ".join(keep) if keep else words[0]


delete_pairs = attach_distortion(pool, delete_words, seed=SEED)
delete_tr, delete_he = delete_pairs[:N_TRAIN], delete_pairs[N_TRAIN:]
for ex in delete_he[:2]:
    print("CHOSEN:", repr(ex["chosen_text"]))
    print("REJECT:", repr(ex["rejected_text"]), "\n")

# %% [B2] RL loop
policy_delete = train_dpo(delete_tr, seed=SEED)

# %% [B3] Metrics & samples
report_behavior("B: word-delete", policy_delete, delete_he)

# %% [markdown]
# ## Behavior C — off-distribution self-samples (on-policy)
# **[C1] Construction.** `rejected` = the base model's *own* sampled continuation from the same
# prompt; `chosen` = the real one, truncated to the same token count (≤48) so the preference can't be
# won on length. Unlike the distortions, the base does *not* already win here — it assigns its own
# samples high probability — so this is where real headroom (and a real accuracy boost) lives.
# Bucketed by prompt length so sampling runs unpadded.
# %% [C1] Sample on-policy rejects from the base model (length-matched)
L_OP = 48


@torch.no_grad()
def sample_onpolicy(examples, max_len=L_OP, temperature=1.0, seed=0):
    torch.manual_seed(seed)
    block = config.block_size
    buckets = defaultdict(list)
    for idx, ex in enumerate(examples):
        buckets[len(ex["prompt_ids"])].append(idx)
    out: list = [None] * len(examples)
    for lp, idxs in buckets.items():
        ks = {i: min(len(examples[i]["chosen_ids"]), max_len) for i in idxs}
        n_tokens = min(max(ks.values()), block - lp)
        ids = torch.tensor([examples[i]["prompt_ids"] for i in idxs], device=device)
        for _ in range(n_tokens):
            logits = ref(ids[:, -block:])[:, -1, :]
            nxt = torch.multinomial(F.softmax(logits / temperature, dim=-1), 1)
            ids = torch.cat([ids, nxt], dim=1)
        gen = ids[:, lp:]
        for j, i in enumerate(idxs):
            k = ks[i]
            rej, chosen = gen[j, :k].tolist(), examples[i]["chosen_ids"][:k]
            out[i] = {**examples[i], "chosen_ids": chosen, "rejected_ids": rej,
                      "chosen_text": decode_ids(chosen), "rejected_text": decode_ids(rej)}
    return out


print("sampling on-policy rejects ...", flush=True)
onpolicy_pairs = sample_onpolicy(pool, seed=SEED)
onpolicy_tr, onpolicy_he = onpolicy_pairs[:N_TRAIN], onpolicy_pairs[N_TRAIN:]
for ex in onpolicy_he[:2]:
    print("CHOSEN:", repr(ex["chosen_text"]))
    print("REJECT:", repr(ex["rejected_text"]), "\n")

# %% [C2] RL loop
policy_onpolicy = train_dpo(onpolicy_tr, seed=SEED)

# %% [C3] Metrics & samples
report_behavior("C: on-policy", policy_onpolicy, onpolicy_he)

# %% [markdown]
# ## Behavior D — repetition loops (suggestion 1)
# **[D1] Construction.** The base's #1 generation defect (see the base panel `rep`). `rejected` =
# the continuation's first half tiled to full length ("He was a career soldier. He was a career
# soldier."), length-matched to `chosen`. Since `rejected` shares its first half with `chosen`, the
# shared prefix cancels in the margin — the signal is purely "continue vs re-emit."
# %% [D1] Construct repetition pairs
def make_repetition(examples):
    out = []
    for ex in examples:
        c = ex["chosen_ids"]
        h = max(4, len(c) // 2)
        rej = (c[:h] * (len(c) // h + 1))[: len(c)]
        out.append({**ex, "rejected_ids": rej, "rejected_text": decode_ids(rej)})
    return out


rep_pairs = make_repetition(pool)
rep_tr, rep_he = rep_pairs[:N_TRAIN], rep_pairs[N_TRAIN:]
for ex in rep_he[:2]:
    print("CHOSEN:", repr(ex["chosen_text"]))
    print("REJECT:", repr(ex["rejected_text"]), "\n")

# %% [D2] RL loop
policy_rep = train_dpo(rep_tr, seed=SEED)

# %% [D3] Metrics & samples
report_behavior("D: repetition", policy_rep, rep_he)

# %% [markdown]
# ## Behavior E — metadata / boilerplate collapse (suggestion 2)
# **[E1] Construction.** The base bails from prose into Wikipedia scaffolding ("References / Other
# websites / 1987 births / Living people"). Mine those tails from the corpus (present in ~64% of
# articles), and set `rejected` = a mined metadata tail tiled/truncated to `chosen`'s length; `chosen`
# = the real prose continuation. Teaches "stay in prose, don't dump metadata mid-body."
# %% [E1] Mine boilerplate tails; construct metadata pairs
def mine_boilerplate(n_chunks, seed=0):
    rng = random.Random(seed)
    order = list(range(len(articles)))
    rng.shuffle(order)
    chunks = []
    for ai in order:
        t = articles[ai]["text"]
        m = re.search(r"\n\n(References|Other websites|Related pages)\b", t)
        if not m:
            continue
        ids = encode(" " + t[m.start():].strip(), vocab)[:96]
        if len(ids) >= 8:
            chunks.append(ids)
        if len(chunks) >= n_chunks:
            break
    return chunks


boiler_pool = mine_boilerplate(2500, seed=SEED)


def make_metadata(examples, chunks, seed=0):
    rng = random.Random(seed)
    out = []
    for ex in examples:
        n = len(ex["chosen_ids"])
        ch = rng.choice(chunks)
        rej = (ch * (n // len(ch) + 1))[:n]
        out.append({**ex, "rejected_ids": rej, "rejected_text": decode_ids(rej)})
    return out


meta_pairs = make_metadata(pool, boiler_pool, seed=SEED)
meta_tr, meta_he = meta_pairs[:N_TRAIN], meta_pairs[N_TRAIN:]
print(f"{len(boiler_pool)=}")
for ex in meta_he[:2]:
    print("CHOSEN:", repr(ex["chosen_text"]))
    print("REJECT:", repr(ex["rejected_text"]), "\n")

# %% [E2] RL loop
policy_meta = train_dpo(meta_tr, seed=SEED)

# %% [E3] Metrics & samples
report_behavior("E: metadata", policy_meta, meta_he)

# %% [markdown]
# ## Behavior F — topic drift (suggestion 5)
# **[F1] Construction.** Fluent but off-prompt continuations ("There are many ways to get to the
# Fairbanks of the United States"). `rejected` = a *different* article's continuation (length-matched
# by tile/truncate); `chosen` = the true one. The classic relevance contrast: prefer on-topic over
# fluent-but-unrelated.
# %% [F1] Construct topic-drift pairs
def make_topic(examples, seed=0):
    rng = random.Random(seed)
    out = []
    N = len(examples)
    for i, ex in enumerate(examples):
        j = rng.randrange(N)
        while j == i:
            j = rng.randrange(N)
        other, n = examples[j]["chosen_ids"], len(ex["chosen_ids"])
        rej = (other * (n // len(other) + 1))[:n]
        out.append({**ex, "rejected_ids": rej, "rejected_text": decode_ids(rej)})
    return out


topic_pairs = make_topic(pool, seed=SEED)
topic_tr, topic_he = topic_pairs[:N_TRAIN], topic_pairs[N_TRAIN:]
for ex in topic_he[:2]:
    print("PROMPT:", ex["prompt_text"][:80])
    print("CHOSEN:", repr(ex["chosen_text"]))
    print("REJECT:", repr(ex["rejected_text"]), "\n")

# %% [F2] RL loop
policy_topic = train_dpo(topic_tr, seed=SEED)

# %% [F3] Metrics & samples
report_behavior("F: topic-drift", policy_topic, topic_he)

# %% [markdown]
# **[G] Cross-behavior summary.** Preference-accuracy gains and the generation panel, side by side.
# The story is two-dimensional: *headroom* (did the base already win the preference?) and *transfer*
# (did suppressing the behavior move generation?). The two axes come apart — a behavior can be
# already-separable yet still reshape generation, or move the score with little visible effect.
# %% [G] Summary table + bar chart
names = list(RESULTS.keys())
print(f"{'behavior':16s} {'base_acc':>8s} {'dpo_acc':>8s} {'margin':>7s} {'ch_drift':>9s} "
      f"{'rep':>14s} {'meta':>14s} {'overlap':>14s}")
for k in names:
    r = RESULTS[k]
    print(f"{k:16s} {r['base_acc']:8.3f} {r['dpo_acc']:8.3f} {r['margin']:7.2f} "
          f"{r['chosen_drift']:9.2f} "
          f"{BASE_PANEL['rep']:.3f}->{r['panel']['rep']:.3f}  "
          f"{BASE_PANEL['meta']:.3f}->{r['panel']['meta']:.3f}  "
          f"{BASE_PANEL['overlap']:.3f}->{r['panel']['overlap']:.3f}")

fig, ax = plt.subplots(1, 2, figsize=(15, 4))
x = range(len(names))
ax[0].bar([i - 0.2 for i in x], [RESULTS[k]["base_acc"] for k in names], 0.4, label="base")
ax[0].bar([i + 0.2 for i in x], [RESULTS[k]["dpo_acc"] for k in names], 0.4, label="DPO")
ax[0].axhline(0.5, ls="--", c="gray")
ax[0].set(title="held-out preference accuracy", xticks=list(x))
ax[0].set_xticklabels(names, rotation=30, ha="right")
ax[0].legend()
for metric, c in (("rep", "C3"), ("meta", "C1"), ("overlap", "C2")):
    ax[1].plot(names, [RESULTS[k]["panel"][metric] for k in names], "o-", c=c, label=f"DPO {metric}")
    ax[1].axhline(BASE_PANEL[metric], ls="--", c=c, alpha=0.6, label=f"base {metric}")
ax[1].set(title="generation panel (dashed = base)")
ax[1].set_xticklabels(names, rotation=30, ha="right")
ax[1].legend(fontsize=8)
fig.tight_layout()

# %% [markdown]
# **[H] Findings & backlog.**
#
# Base generation panel (128 held-out prompts): **rep 0.244, meta 0.477, overlap 0.071**. The base
# repeats (a quarter of its 3-grams) and emits Wikipedia metadata ("References / Categories") on
# roughly half of prompts. Held-out preference accuracy (base → DPO, n=1000) and the panel shift
# (bold = the metric each behavior targets):
#
# | behavior | base→DPO acc | margin | ch_drift | rep | meta | overlap |
# |---|---|---|---|---|---|---|
# | A word-shuffle | 0.996 → 1.000 | +5.2 | −7.1 | 0.244→0.174 | 0.477→0.555 | 0.071→0.048 |
# | B word-delete  | 0.451 → 0.649 | +1.6 | +0.5 | 0.244→0.254 | 0.477→0.133 | 0.071→0.069 |
# | C on-policy    | 0.444 → 0.597 | +1.3 | +5.8 | 0.244→0.365 | 0.477→**0.000** | 0.071→**0.096** |
# | D repetition   | 0.643 → 0.930 | +4.3 | −24.7 | 0.244→**0.130** | 0.477→0.438 | 0.071→0.025 |
# | E metadata     | 0.451 → 0.919 | +8.8 | −4.2 | 0.244→0.295 | 0.477→**0.109** | 0.071→**0.109** |
# | F topic-drift  | 0.602 → 0.743 | +2.2 | −12.0 | 0.244→0.298 | 0.477→0.461 | 0.071→**0.056** |
#
# - **Headroom is uncorrelated with success.** *No headroom*: word-shuffle (base 0.996 — length-matched
#   and already solved, DPO can only sharpen). *Negative headroom*: word-delete and metadata sit at
#   ~0.45 because the base *prefers* the reject (delete for the length reason in [B1]; metadata because
#   the model likes "References / 1987 births" tokens). *Real headroom*: on-policy (0.444, the model
#   ranks its own samples highly). DPO moves all of them — but the accuracy number says nothing about
#   whether generation improved (below).
# - **Targeted metrics move — two of three suggestions land.** Anti-**repetition** (D) gives the least
#   repetitive model (rep 0.244→0.130): Fairbanks base loops "the Fairbanks of the United States" → DPO
#   "There are two types of temperature in the southern United States. Their winter temperatures vary
#   greatly." Anti-**metadata** (E) is the standout — biggest margin (+8.8), meta 0.477→0.109, *and* the
#   best on-topicality (overlap 0.109): Fairbanks → "The highest temperature in Fairbanks is −10 °F".
#   **Topic-drift (F) missed its own target** — overlap went *down* (0.071→0.056), see the drift bullet.
# - **Repetition and metadata are competing degeneracy modes (the corrected systemic finding).** With
#   the two failure modes separated (`rep` vs `meta` — an earlier combined "boilerplate" metric that
#   also counted newline spam had wrongly shown D/F flooding into metadata; they do not), the pattern is
#   a *trade*: suppressing metadata inflates repetition and vice-versa. B/C/E cut meta hard (0.13/0.00/
#   0.11) but push rep up (0.25/0.37/0.30); A/D leave meta flat-to-up (0.56/0.44) while cutting rep
#   (0.17/0.13). The model has cheap fallback modes and squeezing one inflates the other — a single
#   scalar objective cannot see the trade. Only **E (metadata) and C (on-policy)** improve on-topicality
#   at all, and both point the same way: prefer real corpus prose over the model's own output.
# - **`chosen_drift` is the tell for healthy vs degenerate — F is not a trade-off, it is reward
#   hacking.** The margin sign hides *how* it was won; the drift decomposition reveals it. Healthy
#   transfers win by holding or lifting the real text and crushing only the reject: C `chosen +5.8`, B
#   `+0.5`, E `−4.2` against `rejected −91.7` (≈22:1). The collapses win by dragging the real text down
#   hard: D `−24.7`, F `−12.0`, both only ≈2.7:1 vs their reject — most of their "win" is collateral
#   suppression of good text, which *is* the generative collapse. D still won its own metric; **F did
#   not**: its "real ≻ random-other" preference has no clean feature except prompt-lexical-overlap, so
#   DPO hacked it by collapsing onto prompt-token echoes ("Army\n\nArmy\n\nArmy", "Falls\n\nFalls") —
#   failing even its own overlap target. Large negative `chosen_drift` is the warning light.
#
# **Caveats (what would harden this).** Single seed per behavior — the `ci95` bars are binomial over
# held-out pairs, *not* training-run variance, so the cross-behavior ranking is one trajectory each.
# The panel is greedy-only (the regime that most exaggerates repetition) and its `rep`/`meta`/`overlap`
# are means over 128 generations with no CI, so small overlap moves (C 0.096 vs E 0.109) are within
# noise. Filed as curriculum items (multi-seed, a temperature panel, a null/placebo run).
#
# **Next notebook (the interpretability follow-up).** These are all *behavioral* readings — what the
# tuned model does. The natural next step is to open the hood: diff the DPO'd policies against the base
# *in weight space* and localize what tuning changed (which layers / heads / MLP directions moved, how
# `chosen_drift` maps onto parameter deltas, whether the healthy signals E/C touch different weights
# than the degenerate F/D). This notebook produces the policies and the behavioral labels that follow-up
# will explain mechanistically.
#
# **Deferred (real future work).** Every `rejected` here is still synthetic — string surgery or
# "gold ≻ the model's own guess". These are fluency/faithfulness priors, not a learned *intent*. The
# next step is a properly learned reward model that extracts preference structure from data, then DPO
# (or PPO) against *that*. Other backlog: (a) regenerate on-policy rejects from the *current* policy
# each epoch (true online DPO, not one-shot from base); (b) regularized DPO (DPOP/IPO) with an NLL term
# on `chosen` to prevent the `chosen_drift` collapse seen in D/F; (c) a multi-objective run against
# several behaviors at once, to test whether the repetition↔metadata trade balances out.
#
# **References.** DPO (Rafailov et al. 2023); the `chosen_drift` collapse is DPO likelihood
# displacement (Razin et al. 2024; Pal et al. 2024, DPO-positive) and motivates IPO (Azar et al. 2023);
# repetition/degeneracy and the distinct-n metric (Welleck et al. 2019; Holtzman et al. 2019); F's
# reward hacking is reward overoptimization (Gao et al. 2023). Links in the README Resources.
