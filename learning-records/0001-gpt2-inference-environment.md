# 0001 — GPT-2 inference as an environment smoke test

*Experience log for roadmap topic 1 ([`01_gpt2_inference.ipynb`](../notebooks/01_gpt2_inference.ipynb)).
Ungated memory aid — not evidence. 2026-06-19.*

## What this session was

The Interrogate + Document phases of topic 1, the retroactive dry-run of the full loop on the
already-existing notebook 01. Gather and Build were done in prior sessions.

## The reframe that set the direction

I opened the interrogation aimed at GPT-2's *architecture* (weight tying, residual stream) —
TK corrected me mid-session: **notebook 01's purpose is to confirm the environment is
configured for future notebooks, not to study GPT-2's shape.** Updated the README roadmap
step-1 goal to say so (working environment = the goal; inference = the smoke test; architecture
= out of scope, deferred to topic 2). Also logged: **the wiki is anti-hallucination scaffolding
for me, not study material for TK** — so I stopped citing it inside the interrogation.

Lesson for next time: read the *intent* of a retroactive notebook before picking interrogation
angles. A notebook can look like an architecture lesson but be a toolchain check.

## How the interrogation went

Fast Socratic, three environment failure surfaces. The pace stayed high on a strong systems
foundation, so the value was in **sharpening framings**, not teaching basics:

- **Device:** the mismatch-vs-choice distinction — mismatch fails *loud* (raises at the
  embedding op) while wrong-choice fails *silent* (cold GPU), which is why the bare
  `print(is_available())` exists.
- **Eval/dropout:** what `.eval()` actually changes — stochastic→**deterministic**;
  regularization already happened at training time (so eval is *not* "less overfit"); it's a
  **mode** bug, not a **math** bug; and greedy decoding makes a single flipped argmax
  **cascade** into a different sentence.
- **Download/cache:** the metadata-vs-payload split (warning persists because the auth check
  still phones home; bar gone because blobs are local), and why the pretrained weights are the
  **reference oracle** for topic 2's `assert reimpl ≈ hf` rather than "unnecessary when training
  from scratch" — nothing is trained here.

## Where the depth was

- *Low-friction:* the mechanics (caching, device placement, dropout's existence) — a solid
  systems foundation carries these.
- *Where precision mattered:* tightening intuitive first-pass phrasings to exact mechanisms
  (e.g. what `.eval()` changes vs. what training already did). Per the method, a fluent
  exchange is not evidence — topic 2's reimplementation is the real test.

## Honest status

A fast, fluent exchange. Per the method, **fluency is not evidence** — this session certifies
nothing on its own. The real test is topic 2: reimplement the forward pass and make
`reimpl_logits ≈ hf_logits` hold. The environment is now confirmed wired up to support exactly
that.

## Loose ends

- Tokenizer round-trip (the 4th environment surface) deferred → sidecar Follow-ups
  (`01b_tokenizer_round_trip` if pulled).
- Gemini-delegation in Gather still broken (README `## Backlog`) — unrelated to this session but
  blocks running topic-2 Gather as designed.
