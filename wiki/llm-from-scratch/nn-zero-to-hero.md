# Neural Networks: Zero to Hero (Karpathy Course)

> Sources: Andrej Karpathy, Unknown
> Raw: [nn-zero-to-hero](../../raw/llm-from-scratch/nn-zero-to-hero.md)

## Overview

"Neural Networks: Zero to Hero" is Andrej Karpathy's video-plus-notebook course that builds
neural networks from first principles up to a working GPT. It assumes only basic Python and
a vague recollection of calculus, and proceeds incrementally: from a scalar autograd engine,
through character-level language models of increasing sophistication, to a Transformer and
its tokenizer. Each lecture pairs a YouTube video with a Jupyter notebook.

## Curriculum

The course progresses through eight lectures, each layering one new idea onto the last:

1. **Micrograd — backpropagation fundamentals.** Build a minimal automatic differentiation
   engine and train tiny neural nets, making the chain rule concrete.
2. **Makemore Part 1 — bigram language models.** Introduce PyTorch tensors and
   character-level modeling with bigrams; establish the train/sample/evaluate-loss loop.
3. **Makemore Part 2 — multilayer perceptrons.** Move to an MLP character model and cover
   hyperparameter tuning, train/test splits, and overfitting.
4. **Makemore Part 3 — activations, gradients & BatchNorm.** Inspect the internal statistics
   of deep nets, build diagnostic visualizations, and introduce Batch Normalization for
   training stability.
5. **Makemore Part 4 — manual backpropagation.** Backpropagate by hand without autograd to
   build intuition for gradient flow through the compute graph.
6. **Makemore Part 5 — WaveNet architecture.** Develop a hierarchical/convolutional model and
   practice a realistic deep-learning development workflow.
7. **Building GPT from Scratch.** Construct a Generatively Pretrained Transformer following
   "Attention Is All You Need" and the GPT-2/3 architectures.
8. **GPT Tokenizer.** Build the Byte Pair Encoding tokenization pipeline and explain the LLM
   quirks that trace back to tokenizer design.

## Throughline

A consistent set of primitives recurs and deepens across the course: autograd (micrograd),
the names/character-modeling task (makemore), and the Transformer (GPT). Later lectures
reuse and generalize earlier code, so the curriculum doubles as a layered reference for how
each component fits into a full language model.

## Why It Matters

The course is the canonical from-scratch path into LLM internals for mechanistic
interpretability work: every abstraction is built by hand before any framework convenience is
introduced, so the learner sees exactly what attention, normalization, and backprop are doing.

## See Also

- [microGPT](microgpt.md) — Karpathy's ~200-line implementation that packages the same
  micrograd → makemore → GPT progression into a single dependency-free file.
