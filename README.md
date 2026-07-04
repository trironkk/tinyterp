# tinyterp

A personal repository for project-based mechanistic interpretability exploration.

## Layout

| Path         | Purpose                                                                                      |
| ------------ | -------------------------------------------------------------------------------------------- |
| `notebooks/` | Experiments and exploration notebooks.                                                       |
| `tinyterp/`  | The package. Settled code graduates here from notebooks.                                     |
| `artifacts/` | Gitignored. Checkpoints, tokenizers, plots. Notebooks load from here rather than retraining. |
| `data/`      | Gitignored. Downloaded datasets and caches.                                                  |

## Dependencies

Managed with `uv`: Python is pinned to 3.12 and dependencies are declared in
`pyproject.toml`. Everything runs through `uv run …` (package modules via
`uv run python -m tinyterp.<module>`); notebooks assume the project environment
rather than installing anything themselves.

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
- [ ] Numerical precision
  - fp32 accumulation noise vs TF32/bf16; surfaced by [2026-07-03_hardware_detection.py](notebooks/2026-07-03_hardware_detection.py)
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

## Colab

Notebooks will be Colab-compatible by convention, not machinery:

- The jupytext `.py` files are the source of truth; no `.ipynb` is committed.

- To run one in Colab, convert it ad hoc and upload the result:

  ```sh
  uvx jupytext --to ipynb notebooks/<file>.py
  ```

- Code stays device-agnostic (`get_device()` falls through cuda > cpu), so Colab GPU runtimes work unchanged.

## Resources

- https://karpathy.ai/zero-to-hero.html
- https://www.3blue1brown.com/?topic=neural-networks
- https://colah.github.io/
