# In-Context Learning and Induction Heads

> Source: https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html
> Collected: 2026-06-22
> Published: 2022-03-08

_Note: Captured from the transformer-circuits.pub article (now reachable after the host was added
to the network allowlist). Condensed faithfully, not verbatim. By Catherine Olsson, Nelson Elhage,
Neel Nanda, Nicholas Joseph, Nova DasSarma, Tom Henighan, et al. (Anthropic). Follow-up to "A
Mathematical Framework for Transformer Circuits."_

## Thesis

**Induction heads** — the circuit discovered in two-layer attention-only models — are hypothesized
to be **the primary mechanism behind most in-context learning in transformer language models of all
sizes**. The paper builds an indirect, multi-pronged case for this, bridging the **microscopic**
domain (full mechanistic understanding, feasible only for tiny attention-only models) and the
**macroscopic** domain (observable phenomena like loss curves and scaling, the only thing tractable
for large models with MLPs).

## Defining and measuring in-context learning

**In-context learning** = the model gets better at predicting later tokens given more context:
"tokens later in the context are easier to predict than tokens earlier in the context." It is
operationalized by an **in-context learning score**: the loss on the **500th token** of the context
minus the loss on the **50th token**, averaged over examples. A more negative score = stronger
in-context learning.

## Induction heads: definition

An induction head is an attention head that, on a repeated sequence, exhibits two properties:

1. **Prefix matching** — it attends back to the token that previously *followed* the current token
   (i.e., finds an earlier [A] whose successor we want), and
2. **Copying** — its OV circuit increases the logit of the attended-to token.

Together: **[A][B]…[A] → [B]**. Crucially, this works even on **random/novel** repeated sequences,
so it is a general copy-and-continue algorithm, not memorized bigrams. Mechanistically it is the
two-head (previous-token head + matching/copying head) circuit from the framework paper.

## The phase change

Early in training, models with **more than one layer** undergo an abrupt **phase change** over a
narrow window of training steps, visible as a small **bump in the training loss**. During this
window: (a) induction heads form, and (b) the model's in-context learning score improves sharply.
The two events coincide tightly. One-layer models, which cannot form induction heads, show neither
the bump nor the same in-context-learning jump.

## Six lines of evidence (arguments 1–6)

The paper presents six complementary arguments that induction heads drive in-context learning:

1. **Macroscopic co-occurrence** — across model sizes, induction heads form at the same time
   (the phase change) that the in-context learning score sharply improves.
2. **Macroscopic co-perturbation** — an architectural tweak (a "smeared key" / skip-trigram
   modification) that lets induction heads form *earlier* shifts the in-context-learning improvement
   to occur earlier too; the two stay locked together.
3. **Direct ablation** — knocking out induction heads at test time in small models substantially
   reduces in-context learning.
4. **Generality (specific examples)** — although defined via literal copying, induction heads
   empirically also do more abstract in-context tasks (e.g., pattern completion, few-shot–style
   behavior, translation-like copying), in both small and large models.
5. **Mechanistic plausibility (small models)** — for small models the mechanism is fully understood,
   and it naturally generalizes: because the rule decouples A and B, the head can do **"fuzzy" /
   nearest-neighbor** matching **[A*][B*]…[A] → [B]** where A* ≈ A and B* ≈ B in representation
   space, plausibly extending to soft, semantic in-context learning.
6. **Continuity from small to large** — the behavior and properties of induction heads and of
   in-context learning vary smoothly and analogously across scales, consistent with a shared
   mechanism.

## Macroscopic vs. microscopic

For tiny attention-only models, exact circuit-level decomposition is possible (microscopic). For
large models with MLPs, exact decomposition is infeasible, so the authors rely on perturbations,
training dynamics, and ablations to build an indirect case — explicitly analogous to neuroscience
studying development and lesions rather than full wiring diagrams. The evidence is preliminary and
circumstantial but mutually reinforcing.
