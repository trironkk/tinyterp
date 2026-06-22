# Induction Heads and In-Context Learning

> Sources: Catherine Olsson, Nelson Elhage, Neel Nanda, Chris Olah, et al. (Anthropic), 2022-03-08
> Raw: [In-Context Learning and Induction Heads](../../raw/mech-interp/2022-03-08-in-context-learning-and-induction-heads.md)

## Overview

This follow-up to [A Mathematical Framework for Transformer Circuits](transformer-circuits-framework.md)
advances a single bold hypothesis: **induction heads are the primary mechanism behind most
in-context learning** in transformer language models, at every scale. Since exact circuit analysis is
only feasible for tiny attention-only models (the **microscopic** domain), the paper builds an
indirect, six-pronged case linking the microscopic mechanism to **macroscopic** observables — loss
curves, scaling, and training dynamics — that are all we can measure in large models.

## Measuring in-context learning

**In-context learning** is defined operationally: the model predicts later tokens better given more
context — "tokens later in the context are easier to predict than tokens earlier." It is quantified
by an **in-context learning score**: the loss on the **500th** token minus the loss on the **50th**
token of the context, averaged over examples (more negative = stronger).

## Induction heads, recap

An induction head, on a repeated sequence, does **prefix matching** (attends back to the token that
previously *followed* the current one) plus **copying** (its OV circuit boosts the attended token's
logit), yielding **[A][B]…[A] → [B]**. It works even on **random/novel** sequences, so it is general
copy-and-continue, realized by the previous-token-head + matching-head circuit from the framework
paper.

## The phase change

Early in training, models with **more than one layer** undergo an abrupt **phase change** over a
narrow window of steps, visible as a small **bump in the training loss**. In that window, induction
heads form *and* the in-context-learning score sharply improves — the two coincide tightly. One-layer
models, which cannot form induction heads, show neither.

## Six lines of evidence

1. **Macroscopic co-occurrence** — across model sizes, induction heads form exactly when the
   in-context-learning score jumps (the phase change).
2. **Macroscopic co-perturbation** — an architectural tweak that lets induction heads form *earlier*
   shifts the in-context-learning improvement earlier too; they stay locked together.
3. **Direct ablation** — knocking out induction heads at test time in small models substantially
   reduces in-context learning.
4. **Generality** — though defined via literal copying, induction heads empirically also perform more
   abstract in-context tasks (pattern completion, few-shot behavior, translation-like copying), in
   small *and* large models.
5. **Mechanistic plausibility** — in small models the mechanism is fully understood and generalizes
   naturally: because the rule decouples A and B, the head can do **"fuzzy" / nearest-neighbor**
   matching **[A\*][B\*]…[A] → [B]** with A\* ≈ A and B\* ≈ B in representation space — plausibly
   extending to soft, semantic in-context learning.
6. **Continuity** — induction heads and in-context learning vary smoothly and analogously from small
   to large models, consistent with a shared mechanism.

## Microscopic vs. macroscopic

For tiny attention-only models, exact circuit decomposition is possible. For large models with MLPs
it is not, so the authors rely on perturbations, ablations, and training dynamics to build the case —
explicitly analogous to neuroscience studying development and lesions rather than full wiring
diagrams. The evidence is preliminary and circumstantial but mutually reinforcing.

## See Also

- [A Mathematical Framework for Transformer Circuits](transformer-circuits-framework.md) — where
  induction heads are mechanistically derived in two-layer models.
- [Mechanistic Interpretability Tooling & Curriculum](tooling-and-curriculum.md) — ARENA's flagship
  exercise locates induction heads in a 2-layer model with TransformerLens.
- [Mechanistic Interpretability: Features, Circuits, and Superposition](circuits.md) — the conceptual
  roots of the circuits program.
