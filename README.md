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
uv run jupytext --to ipynb notebooks/<file>.py
```

Add a dependency:

```sh
uv add <package>
```

Format markdown (checked in CI):

```sh
git ls-files '*.md' | xargs uv run --only-group lint mdformat
```

Record a full notebook run in `runs/` for posterity (committed, unlike `artifacts/`), rendered as GitHub-friendly markdown with inline plots:

```sh
make record NB=notebooks/<file>.py
```

Install the pre-commit hook that blocks committing a notebook whose latest `runs/` record is missing or stale (CI checks the same thing):

```sh
make hooks
```

Run the repo checks locally (run records match notebook sources; markdown formatted):

```sh
make check
```

## Curriculum

### Engineering

- [x] Dependency management
- [x] Hardware detection
  - [2026-07-03_hardware_detection.py](notebooks/2026-07-03_hardware_detection.py)
- [ ] Training data acquisition
- [ ] Concurrency
- [ ] Colab bootstrap cell

### Fundamentals & Theory

- [x] Matrix multiplication
  - [2026-07-03_hardware_detection.py](notebooks/2026-07-03_hardware_detection.py)
- [x] Numerical precision
  - [2026-07-03_hardware_detection.py](notebooks/2026-07-03_hardware_detection.py)
- [ ] Softmax
- [ ] Tokenization
- [ ] Embeddings
- [ ] Cross-entropy
- [ ] Backpropagation
- [ ] Optimization (SGD → Adam)
- [ ] Layer normalization

### Transformer

- [ ] Attention
- [ ] Positional encodings
- [ ] Residual stream
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
- [ ] Logit lens
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
- [Goldberg, "What Every Computer Scientist Should Know About Floating-Point Arithmetic"](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html)
- [NVIDIA, "Floating Point and IEEE 754"](https://docs.nvidia.com/cuda/floating-point/)
