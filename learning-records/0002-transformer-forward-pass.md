# 0002 — Reimplementing the GPT-2 forward pass (interrogation, partial)

*Experience log for roadmap topic 2 ([`02_transformer.ipynb`](../notebooks/02_transformer.ipynb)).
Ungated memory aid — not evidence. 2026-06-21.*

## What this session was

The Interrogate phase of topic 2 (Gather + Build were done in prior sessions; the notebook
already pins green). The session ran a **block-by-block walkthrough with questions**, covering
cells `[1]`–`[4]` before a long statistics detour pulled focus off the notebook's overall
purpose; it wrapped there to resume cleanly.

## How it went

Fast start, then a slow drift. The mechanics-level cells went smoothly and the value was
sharpening framings (same pattern as topic 1):

- *Two extractions* (oracle output vs. trained params), *eval vs no_grad*, *not-bitwise/argmax*,
  *fused QKV is packaging not math* — all covered, each tightened to the precise mechanism.
- The tightenings: pretrained **means** the trained weights (not just shape); `eval` is mode
  behavior (not activation caching); the pin is correct-to-float-noise (not bitwise).

## The variance detour — the lesson

One line — `unbiased=False` in `layer_norm` — opened the question of what "bias" means here,
which unspooled into a full from-scratch derivation of variance → Bessel's correction →
estimator bias → the two senses of "sampling." Good teaching in isolation, but the cost was
that the detour **pulled focus off the notebook's overall purpose** before the walkthrough
finished.

**Lesson:** a single in-the-weeds flag can swallow a session. When a clarifying question opens a
deep tangent (here: estimator theory behind one normalization flag), I should have **time-boxed
it and explicitly re-anchored to the goal** before going three levels deep — or parked it as a
follow-up. The detour was good teaching in isolation but broke the forest/trees balance the
walkthrough was supposed to keep. The notebook's purpose should have been re-stated *before*, not
*after*, the variance dive.

A useful reconnect mid-tangent — "map this back to LLMs: are we sampling next-token probs?" —
surfaced that the answer is "nothing is being sampled here at all," a sign the detour had
drifted off the notebook entirely.

## Where the depth was

- *Low-friction:* systems-level mechanics (loading, device, eval/dropout, the fused matmul as a
  perf trick) — a solid foundation carries these.
- *Deeper:* the variance/Bessel derivation behind LayerNorm's `unbiased=False` (worked from
  `[2,4,6]` up). A statistics-prerequisite thread, not a GPT-2 one — surfaced by the
  `unbiased=False` line.

## Honest status

Partial interrogation — per the method it certifies nothing on its own, and the walkthrough is
unfinished (attention, MLP, full-forward, and the pin still untouched). The variance excursion
covered a useful prerequisite but pulled focus off the topic. Resume by re-grounding on the
one-sentence goal, then the parked 0.13%-propagation question, then cells `[5]`–`[7]`.

## Loose ends

- Parked: why 0.13% variance error breaks `atol=1e-4` (sidecar Follow-ups, pull first).
- Deferred: softmax + next-token sampling → `02b_logits_to_tokens` if pulled.
- Cells `[5]` attention, `[6]` MLP/full-forward, `[7]` pin — **never reached** this session.
