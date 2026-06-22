# ARENA (Alignment Research Engineer Accelerator) Curriculum

> Source: https://github.com/callummcdougall/ARENA_3.0 (https://learn.arena.education/)
> Collected: 2026-06-22
> Published: Unknown

_Note: Captured from the repository README via GitHub; condensed faithfully, not verbatim.
By Callum McDougall and collaborators (evaluations chapter with Apollo Research)._

## What it is

ARENA is a hands-on curriculum of exercises and Streamlit/Colab notebooks for AI alignment and
mechanistic-interpretability engineering. The GitHub repo hosts the exercises (the "3.0" name is
kept for backwards compatibility but it is now the maintained, latest version). Each chapter
comes with exercises and solutions; much of the mech-interp material is written in
TransformerLens.

## Chapters

**Chapter 0: Fundamentals** — a grounding in deep-learning fundamentals (first ~5 days). Highlights:
building your own 1D/2D convolution functions; building and loading weights into a ResNet and
finetuning it; hyperparameter optimization with Weights & Biases; implementing your own
backpropagation; building GANs and VAEs.

**Chapter 1: Transformer Interpretability** — covers transformers (what they are, how they are
trained, how they generate output) and mechanistic interpretability (what it is, the most important
results so far, why it matters for alignment), plus function vectors and model steering. Highlights:

- Build your own transformer from scratch and sample autoregressively
- Use TransformerLens to **locate induction heads in a 2-layer model**
- Find a circuit for **Indirect Object Identification (IOI)** in GPT-2 small (a replication of
  "interpretability in the wild," covering direct logit attribution, activation patching, and path
  patching)
- Interpret models trained on toy tasks (bracket-string classification, modular arithmetic)
- **Replicate Anthropic's superposition results** and train **sparse autoencoders** to recover
  features from superposition
- Use **steering vectors** to induce behavioral changes in GPT-2-XL

Only the first two exercise sets (building/training transformers; basic mech interp with induction
heads & TransformerLens) are compulsory; the rest are optional, prerequisite-free extensions. The
authors weakly recommend IOI (experimentalists), superposition (theorists), or function vectors
(engineers).

**Chapter 2: Reinforcement Learning** — RL fundamentals with OpenAI Gym. Highlights: multi-armed
bandits (Sutton & Barto methods), implementing DQN and PPO on CartPole, and applying RLHF to
autoregressive transformers.

**Chapter 3: LLM Evaluations** — what evals are for and how to design/build them. Highlights:
designing an MCQ eval from scratch using LLMs (Anthropic's model-written-eval method); using the
UK AISI **Inspect** library; building an LLM agent (e.g., a Wikipedia-racing agent); implementing
ReAct and reflexion elicitation methods. Written with Apollo Research.

**Chapter 4: Alignment Science** — "Coming soon."

## Why it fits the wiki

ARENA is the standard practical on-ramp into mechanistic interpretability — it operationalizes the
concepts (induction heads, superposition, IOI circuits, sparse autoencoders) into runnable
exercises built on TransformerLens, bridging the conceptual Circuits-thread papers and the tooling.
