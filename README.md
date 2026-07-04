# tinyterp

A personal repository for project-based mechanistic interpretability exploration.

## Layout

| Path         | Purpose                                                                                      |
| ------------ | -------------------------------------------------------------------------------------------- |
| `notebooks/` | Experiments and exploration notebooks.                                                       |
| `tinyterp/`  | The package. Settled code graduates here from notebooks.                                     |
| `artifacts/` | Gitignored. Checkpoints, tokenizers, plots. Notebooks load from here rather than retraining. |
| `data/`      | Gitignored. Downloaded datasets and caches.                                                  |

## Curriculum

### Engineering

- [ ] Dependency management
- [ ] Hardware detection
- [ ] Training data acquisition

### Fundamentals & Theory

- [ ] Matrix multiplication
- [ ] Softmax
- [ ] Tokenization
- [ ] Cross-entropy
- [ ] Backpropagation
- [ ] Optimization (SGD → Adam)

### Transformer

- [ ] Attention
- [ ] Network architecture
- [ ] Training
- [ ] Evaluation
- [ ] Sampling

### Post-training

- [ ] Reward model
- [ ] Preference optimization

### Interpretability

- [ ] Probe attention patterns
- [ ] Probe induction heads
- [ ] Probe embeddings
- [ ] Probe logits
- [ ] Circuits
  - [ ] Activation patching
  - [ ] Circuit tracing
- [ ] Superposition toy model
- [ ] Sparse Autoencoders
  - [ ] Training
  - [ ] Feature interpretation

## Resources

- https://karpathy.ai/zero-to-hero.html
- https://www.3blue1brown.com/?topic=neural-networks
- https://colah.github.io/
