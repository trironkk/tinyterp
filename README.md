# tinyterp

This repository will hold my personal explorations of the mechanistic interpretability domain. I'll
be re-implementing concepts on a smaller scale to convince myself of my own understanding,
minimizing the dependencies on the critical path of training and inference.

## Learning Workflow

This repo is run as a teaching workspace: I write notebooks with Claude (see
[CLAUDE.md](./CLAUDE.md) for its operating manual as teacher), then interrogate them. The
guiding principle is **fast default, depth on demand** — move quickly because I have a
foundation, but drop into stronger loops when I need more thorough understanding.

**Per-session loop:**

1. **Gather** — collect canonical sources, record them in `RESOURCES.md`, compile the
   relevant `wiki/` pages *with citations*. Nothing is taught from memory.
2. **Build** — Claude writes a short notebook (`notebooks/NN_topic.ipynb`) and sidecar
   (`notebooks/NN_topic.md`) derived from the wiki. Reimplemented to the linear-algebra
   layer; correctness pinned by inline asserts against a reference (e.g. matching HF
   logits).
3. **Interrogate** — fast Socratic Q&A by default; escalate on demand to
   *reconstruct-on-demand* (rebuild a blanked function) or *predict-before-reveal* (call
   the tensor shapes/values before running). These rungs, plus spacing and interleaving of
   old material, build storage strength rather than the fluency illusion.
4. **Distill** — sharp exchanges go into the sidecar; deferred depth goes into its
   **Follow-ups** backlog (pulling one spawns a companion notebook `NNb_…`); demonstrated
   understanding earns a `learning-records/` entry; recurring concepts graduate to `wiki/`.

**Artifacts:** `MISSION.md` (the *why*, grounds everything) · `wiki/` (source-grounded input
knowledge) · `RESOURCES.md` (curated sources) · `notebooks/NN_*.{ipynb,md}` (exploration +
memory aid) · `learning-records/` (demonstrated understanding, ADR-style) · a glossary
(compressed output, terms added only once understood). `tinyterp/` and `tests/` stay empty
until duplication forces an extraction.

**Curriculum spine:** Transformer → Circuits → Sparse Autoencoders → Evaluations. (Circuits
precede SAEs so that disentangling features is *motivated* by a circuit that uses them.)

**Bootstrapping plan:**

1. **Examine the mission** — interview past "convince myself" to a concrete real-world
   outcome; write `MISSION.md`. Gates everything downstream.
2. **Initialize the knowledge base** — stand up `wiki/` + `RESOURCES.md`, seeded by a
   source search.
3. **Retroactively run the workflow on notebook 01** — dry-run the full loop end-to-end on
   existing material before applying it to new topics.

## Development Notes

### Setup

```shell
git clone --recurse-submodules https://www.github.com/trironkk/tinyterp
cd tinyterp
# Install uv at https://docs.astral.sh/uv/getting-started/installation/
# torch is gated behind a per-machine extra — pick one (see Troubleshooting):
uv sync --extra cu130   # desktop with a CUDA GPU
# uv sync --extra cpu   # laptop / no usable GPU
uv run nbstripout --install --attributes .gitattributes
```

If you've already cloned without submodules, initialize them:

```shell
git submodule update --init --recursive
```

### Recipes

```shell
# Run tests
uv run pytest

# Check NVIDIA detected hardware
nvidia-smi

# Start VSCode
code .
```

## Project Logs

### Environment Setup

```shell
uv --version
# uv 0.11.14 (x86_64-unknown-linux-gnu)

uv python pin
# 3.12

$ uv run jupyter nbconvert --to markdown --execute --stdout notebooks/01_gpt2_inference.ipynb
# Runs the notebook
```

### Transformer

> TODO

### Circuits

> TODO

### Sparse Autoencoders

> TODO

### Evaluations

> TODO

## Side Quests

### Automatic Differentiation

Decided to defer this until more precise understanding of linear algebra implementation would be
more valuable.

> TODO

## Troubleshooting

### Picking a torch build per machine

torch ships as hardware-specific wheels, but a Linux laptop and a Linux desktop look identical to
dependency markers (both `linux` / `x86_64`) — there's no marker for "has a CUDA GPU." So the project
exposes torch through two mutually exclusive extras (see `[project.optional-dependencies]` and
`[tool.uv] conflicts` in `pyproject.toml`), and you select one per machine at sync time:

```shell
uv sync --extra cu130   # desktop — NVIDIA RTX 5060 Ti (Blackwell), CUDA wheel
uv sync --extra cpu     # laptop  — NVIDIA MX150 (Pascal), CPU-only wheel
```

One committed `uv.lock` encodes both resolutions, so this stays reproducible across both machines.
Verify what landed:

```shell
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# desktop: 2.12.0+cu130 True
# laptop:  2.12.0+cpu   False
```

Keep notebooks device-agnostic regardless, so the same code runs on either machine:

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
inputs = inputs.to(device)
```

### Torch not compiled with CUDA enabled / `torch.cuda.is_available()` is False

```
AssertionError: Torch not compiled with CUDA enabled
```

You synced (or are running) the CPU-only build (`torch ...+cpu`) on a machine with a usable GPU.
Re-sync with the CUDA extra: `uv sync --extra cu130`.

### cudaErrorNoKernelImageForDevice

```
Search for `cudaErrorNoKernelImageForDevice' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
```

The GPU is detected but the installed wheel bundles no kernel image for its compute capability.

- **Desktop (RTX 5060 Ti, Blackwell `sm_120`):** needs a cu128-or-newer wheel; older indexes
  (cu118, cu126) lack `sm_120` kernels. The `cu130` extra covers it.
- **Laptop (MX150, Pascal `sm_61`):** the *opposite* problem — recent wheels have dropped Pascal
  kernels. Use the `cpu` extra (small models, CPU inference finishes in seconds), or pin a legacy
  cu118 wheel that still ships Pascal kernels:

  ```toml
  [[tool.uv.index]]
  name = "pytorch-legacy-cuda"
  url = "https://download.pytorch.org/whl/cu118"
  explicit = true

  [tool.uv.sources]
  torch = { index = "pytorch-legacy-cuda" }  # also torchvision/torchaudio if used
  ```

  with `"torch==2.7.1"` pinned (cu118 wheels through 2.7.1 include Pascal kernels). Ad-hoc:

  ```shell
  uv pip install torch==2.7.1 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu118 --python .venv
  ```

### undefined symbol: ncclCommResume

```
ImportError: .../torch/lib/libtorch_cuda.so: undefined symbol: ncclCommResume
```

NCCL ABI mismatch between the installed torch wheel and the NCCL it expects at runtime. Typically
surfaces after torch gets upgraded past a pinned version. Re-pin to a known-good CUDA wheel or switch
to the `cpu` extra.

## References

> TODO
