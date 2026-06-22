# 02 — The GPT-2 forward pass (sidecar)

Memory aids for [`02_transformer.ipynb`](./02_transformer.ipynb). **None of this is evidence** —
scaffolding for future-TK to reconstruct the session. The notebook's job: rebuild GPT-2-small's
forward pass from raw tensor ops and pin it to HF (`max abs diff ~1e-5`, `argmax` matches).

## Resume here

- **The one-sentence goal (re-read this if lost):** GPT-2 is a function *text in → a score
  (logit) for every possible next token out*. This notebook **rebuilds that exact function by
  hand** — matmuls, softmax, layernorm, gelu — using GPT-2's **real trained weights** (pulled
  from `state_dict`), and proves the rebuild is right by checking its logits match the HF black
  box. The win is "I can reproduce the forward pass from scratch and it agrees."
- **Cleared this session (cell `[1]`–`[4]` partial):**
  - *Two things we extract from the loaded `model`:* the **reference output** (`model(...).logits`
    → `hf_logits`, the oracle) and the **trained parameters** (`state_dict` → `sd`, the numbers
    we do arithmetic on). One is what it *computes*, the other is what it *is*.
  - *`eval()` vs `no_grad()`:* eval = **which math runs** (dropout off); no_grad = **whether torch
    records the math** for a backward pass. Independent axes.
  - *Why the pin isn't bitwise:* float32 non-associativity → ~1e-5 drift → `atol=1e-4`. `argmax
    matches: True` is the stronger signal that the drift is pure rounding.
  - *Fused QKV:* `c_attn` is one `(768, 2304)` matmul instead of three `(768, 768)`. **Packaging,
    not math** — `x @ [Wq|Wk|Wv] = [x@Wq | x@Wk | x@Wv]`, then `.split`. Perf (one big GEMM).
  - *What "logit" is:* raw pre-softmax score per vocab token; name from log-odds; softmax is
    shift-invariant so only *differences* matter (→ argmax is the natural read).
- **In flight — re-anchor needed:** a single line (`unbiased=False` in `layer_norm`) pulled the
  session into a long variance/Bessel detour, leaving the notebook's overall purpose worth
  restating on resume (see the goal sentence above). The detour resolves as: *LayerNorm
  normalizes the 768 features by their literal spread (Case A,
  whole population) → ÷H, not ÷(N−1); torch defaults to ÷(N−1) assuming sample-inference, so we
  override.* **No sampling happens in this notebook** (neither statistical-estimator sampling nor
  next-token sampling — it stops at logits + argmax).
- **Next thread to push (resume here):** re-anchor on the goal sentence above, then the *parked*
  question — **why does the 0.13% ÷768-vs-÷767 wobble break the `atol=1e-4` assert instead of
  washing out?** (Hint: LayerNorm sits *inside* the residual stream and feeds every downstream
  matmul; the error doesn't stay 0.13% — it propagates and the model's logit gaps are tight.)
  After that, the interrogation has only covered cells `[1]`–`[4]`; **attention (`[5]`), the MLP
  + full forward (`[6]`), and the pin (`[7]`) are still untouched.**

## Transcript (pull-only archive — don't re-read for review)

Mode: a **block-by-block walkthrough with questions** (worked-walkthrough + Socratic). Got
through cells `[1]`–`[4]` before a variance detour. The sharpened framings:

- **Thin-push cold cue (topic 1):** "forget `.to(device)` on inputs — fail where?" Loud failure,
  but not *upfront* — torch doesn't eagerly migrate; it raises at the **first mixing op** (the
  `wte` embedding lookup).
- **Cell `[1]` (load):** `from_pretrained` loads the **trained weights**, not just shape —
  `state_dict` *is* those numbers; hence the two-extractions framing (oracle output vs.
  parameters). `eval()` removes a source of non-determinism (dropout).
- **`no_grad` vs `eval`:** independent axes — `no_grad` = autograd bookkeeping; `eval` =
  mode-dependent layer behavior.
- **"bitwise correct?":** correct to float noise, with argmax as the robust check.
- **Fused QKV:** perf, packaging not math — HF ships Q/K/V pre-concatenated; there were never
  three matrices.
- **LayerNorm `unbiased=False` → variance detour:** what "bias" means here, rebuilt from
  scratch: estimator-bias vs the `+b` bias; variance = avg squared deviation (worked `[2,4,6]` →
  8/3); Bessel's correction (deviations-from-sample-mean run small; `N−1` patches it; proof
  `E[Σ(x−x̄)²]=(N−1)σ²`); the A/B/C table (population / sample-with-true-mean /
  sample-with-sample-mean). Crystallization: "estimating variance from the true mean vs a
  sample." A "map to LLMs — are we sampling next-token probs?" reconnect disambiguated the **two
  senses of 'sampling'** (statistical-estimator vs next-token), neither present here — at which
  point the detour had drifted off the notebook's purpose and the session wrapped.

## Retrieval cues (questions — no answers; the thin-push pool)

- In one sentence: what function is this notebook rebuilding, and how does it prove the rebuild
  is correct?
- We load one `model` object — name the **two distinct things** we pull out of it, and which cell
  uses each.
- `.eval()` and `torch.no_grad()` — which one turns off dropout, and which one stops autograd from
  recording? What does each have to do with the forward pass we run?
- Why is the pin `allclose(atol=1e-4)` and not `==`? What single printed line tells you the
  remaining difference is *just rounding* and not a logic bug?
- `c_attn` is one `(768, 2304)` weight. Why fused instead of three `(768, 768)`? Does fusing
  change the math?
- What is a logit? Why does adding a constant to every logit change nothing?
- LayerNorm divides the squared deviations by **H (768)**, not **H−1 (767)**. Why is the ÷H
  ("biased") form the *correct* one here, not a shortcut?
- Variance from the *true* mean vs from the *sample* mean — which one needs Bessel's `N−1`
  correction, and why exactly?
- The word "sampling" means two different things around LLMs. Name both. Which (if either) happens
  in this notebook?

## Follow-ups (depth-on-demand backlog)

- **The 0.13% propagation question (parked, pull first on resume).** *Question:* why does a 0.13%
  variance error (÷768 vs ÷767) break the `atol=1e-4` pin instead of washing out? *Why deferred:*
  session ended before answering. *Pointer:* `layer_norm` in cell `[4]`; think about LayerNorm
  feeding every downstream matmul in the residual stream + how tight the logit gaps are. No new
  notebook needed — it's an in-place experiment (perturb the divisor, watch the diff blow past
  1e-4).
- **Softmax + next-token sampling (out of scope here).** *Question:* what do softmax, temperature,
  top-k/top-p actually do to a logit vector, and how does that connect to the topic-1 greedy
  `generate`? *Why deferred:* this notebook stops at logits/argmax; decoding is downstream.
  *Pointer:* the final `x @ wte.T`; topic-1 sidecar `generate` thread. Pulling spawns
  `02b_logits_to_tokens`.
