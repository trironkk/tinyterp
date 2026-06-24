# 200 Concrete Open Problems in Mechanistic Interpretability

> Sources: Neel Nanda, 2022-12-28
> Raw: [200 Concrete Open Problems in Mech Interp (Introduction)](../../raw/mech-interp/2022-12-28-200-concrete-open-problems-in-mech-interp.md)

## Overview

Neel Nanda's "200 Concrete Open Problems in Mechanistic Interpretability" is a research-agenda
sequence aimed at newcomers: it maps the field's big open-problem areas, argues why they matter,
gives advice on how to do the research, and lists concrete problems ranked by difficulty. Where the
[Glossary](glossary.md) is the terminology onramp and [TransformerLens & ARENA](tooling-and-curriculum.md)
are the practical stack, this is the **"what should I actually work on"** onramp. This page captures
the framing and category map from the introductory post.

## Why the field matters

Nanda gives two motivations:

- **Alignment and safety** — as systems get more capable, it matters whether a model is genuinely
  helpful or deceptive. Mech interp examines *how* a model solves a task, not just *that* it does.
- **Scientific understanding** — "there exist computer programs like GPT-3 that can essentially speak
  English at a human level, but we have no idea how to write these programs in normal code."
  Reverse-engineering them would be a major scientific result. The animating claim is that the
  inscrutable stack of matrices **can be decompiled to a human-interpretable algorithm**.

## Difficulty ratings

Problems are tagged A–D (often as ranges, since most can be done at varying depth):

- **A** — beginner, days to ~two weeks.
- **B** — substantial beginner project, several weeks.
- **C** — harder, for experienced researchers; potentially multi-month.
- **D** — ambitious/exploratory, needs significant scoping.

## How to approach the research

- Bring reasonable starting skills *and* realistic expectations — projects are learning
  opportunities, not failures if they don't break ground.
- Hands-on work grounds the reading: even for concepts best learned from papers, building intuitions
  in code gives context.
- Lean on community — Discord, mentorship, and collaboration meaningfully improve both quality and
  enjoyment.

## The problem categories

The sequence's areas, each a post with motivation, tips, resources, and specific problems:

1. **Toy language models** — reverse-engineer 1–4 layer models where full circuit analysis is
   tractable.
2. **Real model circuits** — scale the techniques to actual LMs like GPT-2 Small.
3. **Algorithmic problems** — clean, ground-truth tasks (e.g. modular addition) to develop and
   validate techniques.
4. **Polysemanticity and superposition** — how models pack many features into few dimensions.
5. **Training dynamics** — how circuits form and evolve during training.
6. **Techniques and tooling** — better methods for forming accurate beliefs about internals.
7. **Image models** — extending mech interp from language to vision.
8. **Reinforcement learning** — decision-making and goal-formation in RL agents.
9. **Learned features** — cataloguing and understanding MLP-layer representations.

## How this connects to the wiki

The categories line up with the rest of the mech-interp topic: toy-model and real-model circuits map
to [A Mathematical Framework for Transformer Circuits](transformer-circuits-framework.md) and
[Induction Heads and In-Context Learning](induction-heads-and-in-context-learning.md);
polysemanticity/superposition to [Superposition](superposition.md) and [Sparse
Autoencoders & Monosemanticity](sparse-autoencoders.md); and "techniques and tooling" to
[TransformerLens & ARENA](tooling-and-curriculum.md). Nanda explicitly points newcomers to his
[Glossary](glossary.md) for terminology and to **TransformerLens** for getting hands-on quickly —
many of these problems are tractable with a small model in a Colab notebook.

## See Also

- [Mechanistic Interpretability Glossary (Neel Nanda)](glossary.md) — the companion terminology
  onramp from the same author.
- [Tooling & Curriculum: TransformerLens and ARENA](tooling-and-curriculum.md) — the practical stack
  for actually attempting these problems; ARENA's exercises operationalize several categories.
- [Superposition](superposition.md) and [Sparse Autoencoders & Monosemanticity](sparse-autoencoders.md)
  — the polysemanticity/superposition category.
- [A Mathematical Framework for Transformer Circuits](transformer-circuits-framework.md) and
  [Induction Heads and In-Context Learning](induction-heads-and-in-context-learning.md) — the
  toy-model and real-model circuit categories.
