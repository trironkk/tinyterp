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
- [x] Training data acquisition
  - [2026-07-04_training_data_acquisition.py](notebooks/2026-07-04_training_data_acquisition.py)
- [ ] Corpus cleaning / prose extraction
- [ ] Concurrency
- [ ] Colab bootstrap cell
- [ ] Pulling open-weight models (GPT-2, Gemma, DeepSeek)

### Fundamentals & Theory

- [x] Matrix multiplication
  - [2026-07-03_hardware_detection.py](notebooks/2026-07-03_hardware_detection.py)
- [x] Numerical precision
  - [2026-07-03_hardware_detection.py](notebooks/2026-07-03_hardware_detection.py)
- [x] Softmax
  - [2026-07-07_transformer_forward_pass.py](notebooks/2026-07-07_transformer_forward_pass.py)
- [ ] Tokenization
  - [ ] BPE tokenizer training
- [x] Embeddings
  - [2026-07-07_transformer_forward_pass.py](notebooks/2026-07-07_transformer_forward_pass.py)
- [x] Cross-entropy
  - [2026-07-07_transformer_forward_pass.py](notebooks/2026-07-07_transformer_forward_pass.py)
- [ ] Backpropagation
  - [ ] Autograd engine from scratch (micrograd)
- [ ] Optimization (SGD → Adam)
- [ ] Weight initialization
- [x] Layer normalization
  - [2026-07-07_transformer_forward_pass.py](notebooks/2026-07-07_transformer_forward_pass.py)
- [ ] Batch normalization

### Transformer

- [ ] RNN/LSTM/GRU baselines
- [x] Attention
  - [2026-07-07_transformer_forward_pass.py](notebooks/2026-07-07_transformer_forward_pass.py)
- [x] Positional encodings
  - [2026-07-07_transformer_forward_pass.py](notebooks/2026-07-07_transformer_forward_pass.py)
- [x] Residual stream
  - [2026-07-07_transformer_forward_pass.py](notebooks/2026-07-07_transformer_forward_pass.py)
- [x] Network architecture
  - [2026-07-07_transformer_forward_pass.py](notebooks/2026-07-07_transformer_forward_pass.py)
- [ ] Mixture of experts
- [x] Training
  - [2026-07-07_transformer_forward_pass.py](notebooks/2026-07-07_transformer_forward_pass.py)
- [ ] Evaluation
- [x] Sampling
  - [2026-07-07_transformer_forward_pass.py](notebooks/2026-07-07_transformer_forward_pass.py)
- [x] Scaling behavior (data vs parameters)
  - [2026-07-07_transformer_forward_pass.py](notebooks/2026-07-07_transformer_forward_pass.py)
- [ ] KV caching

### Post-training

- [ ] Reward model
- [x] Preference optimization
  - [2026-07-13_preference_optimization_dpo.py](notebooks/2026-07-13_preference_optimization_dpo.py)
- [ ] Regularized DPO (DPOP / IPO) against chosen-likelihood displacement
- [ ] Multi-objective preference optimization (train against several behaviors at once)
- [ ] Post-training regression control (held-out perplexity / KL-to-reference on neutral prompts)

### Interpretability

- [x] Probe attention patterns
  - [2026-07-10_attention_induction_heads.py](notebooks/2026-07-10_attention_induction_heads.py)
- [x] Probe induction heads
  - [2026-07-10_attention_induction_heads.py](notebooks/2026-07-10_attention_induction_heads.py)
- [x] Head ablation and significance (zero- vs mean-ablation)
  - [2026-07-10_attention_induction_heads.py](notebooks/2026-07-10_attention_induction_heads.py)
- [x] Head pruning and redundancy (are sixteen heads better than one?)
  - [2026-07-10_attention_induction_heads.py](notebooks/2026-07-10_attention_induction_heads.py)
- [ ] Copying score (OV circuit): does a head write the attended token to the logits?
- [ ] Probe embeddings
- [ ] Probe MLP feedforward memories
- [ ] Logit lens
- [ ] Probe logits
- [ ] Representation geometry (t-SNE / PCA / UMAP)
- [ ] Feature visualization (activation maximization)
- [ ] Attribution & saliency maps
- [ ] Circuits
  - [ ] Activation patching
  - [ ] Path patching / K-composition (previous-token head feeding the induction head)
  - [ ] Circuit tracing
- [ ] Superposition toy model
- [ ] Sparse Autoencoders
- [ ] Weight-space diff of fine-tuning: localize what DPO changed (which layers / heads / MLP
  directions moved; does the `chosen_drift` health signal map onto parameter deltas?)
  - [ ] Training
  - [ ] Feature interpretation

## Resources

- https://karpathy.ai/zero-to-hero.html
- [HuggingFace `wikimedia/wikipedia` dataset](https://huggingface.co/datasets/wikimedia/wikipedia)
- https://www.3blue1brown.com/?topic=neural-networks
- https://colah.github.io/
- [Goldberg, "What Every Computer Scientist Should Know About Floating-Point Arithmetic"](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html)
- [NVIDIA, "Floating Point and IEEE 754"](https://docs.nvidia.com/cuda/floating-point/)
- [Olsson et al., "In-context Learning and Induction Heads"](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)
- [Elhage et al., "A Mathematical Framework for Transformer Circuits"](https://transformer-circuits.pub/2021/framework/index.html)
- [Michel et al., "Are Sixteen Heads Really Better than One?"](https://arxiv.org/abs/1905.10650)
- [Rafailov et al., "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"](https://arxiv.org/abs/2305.18290)
- [Razin et al., "Unintentional Unalignment: Likelihood Displacement in DPO"](https://arxiv.org/abs/2410.08847)
- [Pal et al., "Smaug: Fixing Failure Modes of Preference Optimisation with DPO-Positive"](https://arxiv.org/abs/2402.13228)
- [Azar et al., "A General Theoretical Paradigm to Understand Learning from Human Preferences" (IPO)](https://arxiv.org/abs/2310.12036)
- [Welleck et al., "Neural Text Generation with Unlikelihood Training"](https://arxiv.org/abs/1908.04319)
- [Holtzman et al., "The Curious Case of Neural Text Degeneration"](https://arxiv.org/abs/1904.09751)
- [Gao et al., "Scaling Laws for Reward Model Overoptimization"](https://arxiv.org/abs/2210.10760)
- [Ouyang et al., "Training language models to follow instructions with human feedback" (InstructGPT)](https://arxiv.org/abs/2203.02155)
