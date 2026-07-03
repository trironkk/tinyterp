# Notes — `02_induction-heads.ipynb`

Design decisions and results for the induction-heads hunt on the 01 model.

## Headline claim

> I located the induction head(s) in the 8L/8H/320D model I trained from scratch, and proved
> they carry its in-context copying.

Proof bar: three gates, in order — **behavioral** (loss gap on repeated random sequences),
**correlational** (per-head prefix-matching scores + attention patterns), **causal**
(zero-ablation collapses the copying score with ordinary loss roughly intact; mean-ablation as
robustness check). The claim stands only if all three hold.

**Induction score** = mean per-position loss on the first copy − mean on the second copy
(more positive = stronger copying). This is the 128-token-context analogue of the paper's
in-context-learning score (loss@500 − loss@50); grounded in
`wiki/mech-interp/induction-heads-and-in-context-learning.md`.

**Prefix-matching score** (per head) = average attention weight from each second-copy position
back to the token *after* its previous occurrence — the "attend to what followed me last time"
signature from `wiki/mech-interp/transformer-circuits-framework.md`.

## Cell map

| | | | |
|---|---|---|---|
| [A] Setup & dependencies | [E] Instrumented forward pass | [I] Gate 2 — prefix-matching scores | [M] Gate 3 — mean-ablation check |
| [B] Config | [F] Sanity: instrumentation ≡ | [J] Gate 2 — attention patterns | [N] Stretch — previous-token heads |
| [C] Load checkpoint | [G] Repeated random sequences | [K] Natural-text baseline | [O] Verdict |
| [D] Tokenizer from merges | [H] Gate 1 — loss gap | [L] Gate 3 — zero-ablation | |

## Scoping decisions (fixed)

- **Standalone.** Loads `artifacts/microgpt_wiki.pt` (merges + params + config); no corpus
  download, no retraining, no imports from 01. 01 stays untouched.
- **Instrumented rewrite is the first lesson.** [E] copies 01's raw-tensor forward and rewrites
  it to cache per-head attention patterns and accept a per-(layer, head) ablation spec. No
  TransformerLens; device-agnostic.
- **Out of scope** (future notebooks): QK/OV composition analysis, logit lens, neuron/MLP
  browsing, SAEs. The stretch cell [N] (previous-token-head scores at offset 1) is the hard
  stop — it previews the two-head circuit *without* composition math.

## Design decisions made at stub time (veto-able)

- **Natural-text proxy [K].** Gate 3 needs "ordinary next-token loss stays intact" and
  mean-ablation needs per-head mean outputs over ordinary text — but the checkpoint carries no
  data and 02 downloads nothing. Plan: model-generated samples (temperature-sampled from the
  reloaded model) plus a small hardcoded English snippet serve as the natural-text eval set.
  Self-generated text is slightly easy for the model (it's on-policy), but for a *differential*
  measurement — loss before vs. after ablation — that bias cancels.
- **Ablation point.** "Zero-ablate a head" = zero that head's output *before* the output
  projection (equivalently its contribution to the residual stream), not its attention pattern.
  Mean-ablation substitutes the [K] mean vector at the same point.
- **Ablation control.** [L] also ablates an equal-sized set of random non-candidate heads, so
  "copying collapses" can be attributed to *which* heads were removed, not to removing heads
  per se.
- **One instrumented forward, two features.** Pattern-caching and ablation live in the same
  forward function (flags/arguments), so gates 2 and 3 exercise the same code path that [F]
  verified against the un-instrumented model.

## Pre-registered failure path

If **gate 1 fails** (no meaningful loss gap on repeated sequences): keep the notebook shape,
flip the headline to a documented negative result, and run exactly **one** bounded diagnostic —
sweep the repeat length, and check for sub-threshold prefix-matching heads — recorded here as
the seed for a follow-up notebook. No retraining inside 02.

## Results

*(to be filled during the build)*
