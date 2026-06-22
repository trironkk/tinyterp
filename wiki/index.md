# Knowledge Base Index

## llm-from-scratch

Building neural networks and language models from first principles, dependency-free.

| Article | Summary | Updated |
|---------|---------|---------|
| [microGPT: A Minimal GPT in ~200 Lines](llm-from-scratch/microgpt.md) | Karpathy's ~200-line dependency-free GPT: scalar autograd, GPT-2-style Transformer, training, and sampling on a names dataset. | 2026-06-22 |
| [Neural Networks: Zero to Hero (Karpathy Course)](llm-from-scratch/nn-zero-to-hero.md) | Karpathy's 8-lecture video+notebook course building from micrograd through makemore to a GPT and its tokenizer; includes the standalone "Let's build GPT" lecture repo. | 2026-06-22 |
| [nanoGPT & nanochat: Minimal GPT Training in PyTorch](llm-from-scratch/nanogpt-and-nanochat.md) | Karpathy's hackable PyTorch codebases: nanoGPT reproduces GPT-2, nanochat covers the full tokenize→pretrain→SFT→RL→chat pipeline on one `--depth` dial. | 2026-06-22 |
| [The Transformer Architecture](llm-from-scratch/transformer-architecture.md) | The encoder-decoder Transformer (attention, multi-head, positional encoding, residuals) via the primary "Attention Is All You Need" paper plus the Illustrated and Annotated Transformer explainers. | 2026-06-22 |

## mech-interp

Reverse-engineering the algorithms learned inside trained neural networks — concepts, results, and tooling.

| Article | Summary | Updated |
|---------|---------|---------|
| [Mechanistic Interpretability: Features, Circuits, and Superposition](mech-interp/circuits.md) | Olah et al.'s founding "Zoom In" essay: features as directions, circuits as readable weight-algorithms, superposition/polysemanticity, universality, and interpretability as a natural science. | 2026-06-22 |
| [A Mathematical Framework for Transformer Circuits](mech-interp/transformer-circuits-framework.md) | Anthropic's founding transformer-circuits paper: residual stream as communication channel, attention heads as QK/OV circuits, zero/one/two-layer analysis, composition, and the discovery of induction heads. | 2026-06-22 |
| [Induction Heads and In-Context Learning](mech-interp/induction-heads-and-in-context-learning.md) | The phase change during training where induction heads form, and six lines of evidence that they are the mechanism behind most in-context learning at all scales. | 2026-06-22 |
| [Superposition: Toy Models](mech-interp/superposition.md) | How networks pack more features than neurons into non-orthogonal directions; sparsity/importance drivers, the polytope geometry, phase changes, and computation in superposition. | 2026-06-22 |
| [Sparse Autoencoders & Monosemanticity (Dictionary Learning)](mech-interp/sparse-autoencoders.md) | Using a sparse autoencoder to decompose a one-layer transformer's MLP into many monosemantic features; specificity, feature splitting, and universality. | 2026-06-22 |
| [Mechanistic Interpretability Tooling & Curriculum: TransformerLens and ARENA](mech-interp/tooling-and-curriculum.md) | The practical stack: TransformerLens for caching/patching internals, and ARENA's exercises on induction heads, IOI circuits, superposition, and sparse autoencoders. | 2026-06-22 |
