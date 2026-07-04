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

- [x] Dependency management (covered by the workspace scaffold: uv, pyproject, pinned Python)
- [x] Hardware detection (notebooks/2026-07-03_hardware_detection.py)
- [ ] Training data acquisition

### Fundamentals & Theory

- [ ] Matrix multiplication (throughput benchmarked in the hardware notebook; the operation itself still open)
- [ ] Numerical precision (fp32 accumulation noise vs TF32/bf16; surfaced by the hardware notebook's tolerance question)
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

Notebooks are Colab-compatible by convention, not machinery:

- The percent-format `.py` files are the source of truth; no `.ipynb` is committed.
- To run one in Colab, convert it ad hoc: `uvx jupytext --to ipynb notebooks/<file>.py`, then upload.
- Code stays device-agnostic (`get_device()` falls through cuda > mps > cpu), so Colab GPU runtimes work unchanged.
- Planned: a guarded bootstrap cell (`"google.colab" in sys.modules` gates a pip install), deferred until a Colab account/free tier is set up.

## Resources

- https://karpathy.ai/zero-to-hero.html
- https://www.3blue1brown.com/?topic=neural-networks
- https://colah.github.io/
