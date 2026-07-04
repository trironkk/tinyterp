# tinyterp

A personal repository for project-based mechanistic interpretability exploration.

## Layout

| Path         | Purpose                                                                                      |
| ------------ | -------------------------------------------------------------------------------------------- |
| `notebooks/` | Experiments and exploration notebooks.                                                       |
| `tinyterp/`  | The package. Settled code graduates here from notebooks.                                     |
| `artifacts/` | Gitignored. Checkpoints, tokenizers, plots. Notebooks load from here rather than retraining. |
| `data/`      | Gitignored. Downloaded datasets and caches.                                                  |

## Recipes

Run a package module:

```sh
uv run python -m tinyterp.<module>
```

Run a notebook end to end (or cell-by-cell in VS Code's interactive window):

```sh
uv run python notebooks/<file>.py
```

Convert a notebook to `.ipynb`, e.g. for Colab (the jupytext `.py` files are the source of truth; no `.ipynb` is committed):

```sh
uvx jupytext --to ipynb notebooks/<file>.py
```

Add a dependency:

```sh
uv add <package>
```

Format markdown (checked in CI):

```sh
git ls-files '*.md' | xargs uv run --only-group lint mdformat
```

## Curriculum

### Engineering

- [x] Dependency management
- [x] Hardware detection
  - [2026-07-03_hardware_detection.py](notebooks/2026-07-03_hardware_detection.py)
- [ ] Training data acquisition
- [ ] Colab bootstrap cell

### Fundamentals & Theory

- [x] Matrix multiplication
  - [2026-07-03_hardware_detection.py](notebooks/2026-07-03_hardware_detection.py)
- [x] Numerical precision
  - [2026-07-03_hardware_detection.py](notebooks/2026-07-03_hardware_detection.py)
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
