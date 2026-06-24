# 200 Concrete Open Problems in Mechanistic Interpretability: Introduction

> Source: https://www.lesswrong.com/posts/LbrPTJ4fmABEdEnLf/200-concrete-open-problems-in-mechanistic-interpretability
> Collected: 2026-06-24
> Published: 2022-12-28

_Note: By Neel Nanda. The introductory post to the "200 Concrete Open Problems in Mechanistic
Interpretability" sequence. Captured now that the host is reachable (added to the network
allowlist). Condensed faithfully; framing and structure preserved._

## What is Mechanistic Interpretability?

Mechanistic interpretability is "the study of reverse engineering neural networks." You take an
inscrutable stack of matrices that you know works, and reverse-engineer how it works. The core
claim motivating the field: this inscrutable stack of matrices can be decompiled to a
human-interpretable algorithm.

## Why This Sequence Was Created

Two core motivations:

- **Alignment and safety.** As systems grow more capable, it matters whether a model achieves a
  task through genuine helpfulness or through deception. Mech interp provides tools to examine
  *how* a model solves a problem, not just *that* it does.
- **Scientific understanding.** "It is a fact about today's world that there exist computer
  programs like GPT-3 that can essentially speak English at a human level, but we have no idea how
  to write these programs in normal code." Reverse-engineering these systems would be a profound
  scientific accomplishment.

The sequence targets researchers new to the field, offering concrete, approachable problems across
difficulty levels. It maps out the big areas of open problems, how to think about them, why they
matter, and how to approach research there, then lists problems ranked by difficulty.

## Problem Difficulty Ratings

A four-tier system (problems often have ranges reflecting varied depth):

- **A-Level** — Beginner problems achievable in days to ~two weeks.
- **B-Level** — Substantial beginner projects, requiring several weeks minimum.
- **C-Level** — Harder problems for experienced researchers; potentially multi-month.
- **D-Level** — Ambitious and exploratory, requiring significant scoping work.

## Research Approach Recommendations

Guidance addressing common misconceptions:

- Research needs both reasonable starting skills and realistic expectations about difficulty.
- Treat projects as learning opportunities rather than failures if they don't produce
  groundbreaking results. Even for concepts best learned from papers, hands-on intuitions help
  ground the reading and give context.
- Community support matters: Discord communities, mentorship, and collaboration significantly
  improve research quality and enjoyment.

## The Problem Categories

The sequence covers these areas, each with motivation, practical tips, resource recommendations,
and specific problem suggestions:

1. **Toy Language Models** — reverse-engineering one- to four-layer models for tractable circuit
   analysis.
2. **Real Model Circuits** — scaling insights to actual LMs like GPT-2 Small.
3. **Algorithmic Problems** — clean, ground-truth tasks (e.g., modular addition) for technique
   development.
4. **Polysemanticity and Superposition** — how models compress multiple features into limited
   dimensions.
5. **Training Dynamics** — how circuits emerge and evolve during training.
6. **Techniques and Tooling** — better methods for forming accurate beliefs about network
   internals.
7. **Image Models** — extending mech interp beyond language to vision.
8. **Reinforcement Learning** — decision-making and goal-formation in RL agents.
9. **Learned Features** — cataloging and understanding MLP-layer representations.

## Key Resources

Nanda points newcomers to his mechanistic interpretability explainer & glossary for unfamiliar
terminology, and to the TransformerLens library for practical research on GPT-style models —
emphasizing tools that enable rapid iteration and hands-on learning.
