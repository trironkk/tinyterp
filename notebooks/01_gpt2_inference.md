# 01 — GPT-2 inference (sidecar)

Memory aids for [`01_gpt2_inference.ipynb`](./01_gpt2_inference.ipynb). **None of this is
evidence** — it's scaffolding for future-TK to reconstruct the session. The notebook's job is
to prove the **environment is configured** for later notebooks; GPT-2's architecture is *not*
the subject here (that's topic 2).

## Resume here

- **Cleared this session:** the notebook reads cleanly as an **environment smoke test** across
  three independent failure surfaces — (1) device plumbing, (2) eval/determinism, (3) the
  HuggingFace download/cache pipeline.
- **In flight:** none — session wrapped at a clean point.
- **Next thread to push:** roadmap **topic 2 (Transformer)** — reimplement the forward pass to
  the LA layer and pin `assert reimpl_logits ≈ hf_logits` against the pretrained `gpt2` (the
  download verified here is that reference oracle). Optional warm-up: the deferred **tokenizer
  round-trip** follow-up below.

## Transcript (pull-only archive — don't re-read for review)

Fast Socratic, three environment dimensions. No thin-push cold cues (topic 1 has no priors).

1. **Device plumbing** — `device = "cuda" if torch.cuda.is_available() else "cpu"`, then
   `.to(device)` on model *and* inputs.
   - *Drop `.to(device)` on inputs only:* tensors on different devices; torch does **not**
     eagerly migrate — raises at the first mixing op (the `wte` embedding lookup) during the
     `model.generate(**inputs)` cell: `RuntimeError: Expected all tensors to be on the same
     device...`. Loud failure.
   - *Wrong device choice (CPU when a GPU exists):* **silent** — no exception, just a cold GPU
     and slow runs. The ternary *fails open* to CPU, so `print(torch.cuda.is_available())` is
     the only signal, and only because a human eyeballs the `True`. A stricter notebook would
     `assert torch.cuda.is_available()`. **Asymmetry surfaced:** device *mismatch* fails loud,
     device *choice* fails silent.
2. **`.eval()` / determinism** — printout is full of `Dropout(p=0.1)`.
   - Train mode keeps dropout **stochastic**: two identical forward passes differ *from each
     other*, run to run, at random. `.eval()` makes the forward pass **deterministic** (runs
     the full network, the thing dropout approximates in expectation).
   - Precise framing: eval is **not** "less overfit" — regularization already happened at
     *training* time; eval just stops thinning the net. Cleaner predictions, not less overfit.
   - Precise framing: forgetting `.eval()` is a **mode/state** bug, not a *wrong-math* bug —
     the forward pass is faithfully computing a randomly-thinned subnet. Mode bugs are silent
     and don't throw.
   - Amplification: `generate()` here is **greedy** (argmax, no tolerance band) — a single
     dropout-flipped argmax early **cascades** through the autoregressive loop into a wholly
     different continuation. Small noise → large-looking output divergence.
3. **Download / cache pipeline** — `from_pretrained("gpt2")` + the `HF_TOKEN` warning + the
   `Loading weights: 0/148` bar.
   - This is the **only** line exercising the HuggingFace data pipeline (network egress, hub
     client, anon-vs-`HF_TOKEN` auth, `transformers`/`safetensors` download→deserialize→
     `nn.Module`, disk-cache write) — a separate failure surface from "does torch see the GPU."
   - Cache lands in `~/.cache/huggingface/hub/` (`HF_HOME`). Second run = cache hit, **no bar**,
     but the **warning persists** because that's an *auth* warning on a hub **metadata/revision
     check** (still phones home unless `HF_HUB_OFFLINE=1`); only the weight blobs are local.
   - Precise framing: "not needed when training from scratch" doesn't apply to this roadmap —
     the roadmap is **not** training GPT-2; topic 2 checks the reimpl against these **pretrained
     weights**, so the download is the **reference oracle**, load-bearing for verification.

## Retrieval cues (questions — no answers; the thin-push pool)

- You keep `.to(device)` on the model but drop it on the inputs. Does it fail, and if so, at
  which cell and on which operation?
- Which environment failure mode of this notebook is **loud**, and which is **silent**? Why is
  the bare `print(torch.cuda.is_available())` there at all?
- You run the `generate` cell twice without calling `.eval()`. What do you see, and is it a
  wrong-math bug or a wrong-mode bug?
- Why can dropout in train mode turn a *tiny* logit perturbation into a *completely different*
  generated sentence under greedy decoding?
- Does `.eval()` make the model "less overfit"? If not, what does it actually change?
- Which single line in the notebook exercises a failure surface that none of the pure-torch
  cells touch? Name the things it depends on.
- On a second run the download progress bar is gone but the HF warning still prints. Why the
  split?
- Why is `from_pretrained("gpt2")` load-bearing for the topic-2 verification plan, given you're
  not training GPT-2 from scratch?

## Follow-ups (depth-on-demand backlog)

- **Tokenizer round-trip.** *Question:* trace `tokenizer(prompt, return_tensors="pt")` → ids →
  `model.generate` → `tokenizer.decode`, and explain the `Setting pad_token_id to
  eos_token_id:50256` warning — why does an open-end generation need a pad token at all?
  *Why deferred:* session wrapped before the 4th dimension; it's the one untouched environment
  surface in this notebook. *Pointer:* cells 4–5 of the notebook; the BPE mechanics live in
  [`wiki/tokenization-bpe.md`](../wiki/tokenization-bpe.md) (anti-hallucination reference, not
  study material). Pulling this spawns `01b_tokenizer_round_trip`.
