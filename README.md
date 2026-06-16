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
4. **Document** — the step fans out into distinct artifacts, each with its own job: a
   **transcript** of the exchange (fidelity, lightly curated) and **retrieval cues**
   (questions without answers, for spaced re-quiz) go into the sidecar; deferred depth goes
   into the sidecar's **Follow-ups** backlog (pulling one spawns a companion notebook
   `NNb_…`); a short **briefing** lands in this README's `## Project Logs` as the rolling
   cross-topic progress view; demonstrated understanding earns a `learning-records/` entry;
   recurring concepts graduate to `wiki/`.

**Artifacts:** `MISSION.md` (the *why*, grounds everything) · `wiki/` (source-grounded input
knowledge) · `RESOURCES.md` (curated sources) · `notebooks/NN_*.{ipynb,md}` (exploration +
sidecar: transcript / retrieval cues / follow-ups) · `## Project Logs` below (rolling
progress briefing) · `learning-records/` (demonstrated understanding, ADR-style) · a glossary
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

## Roadmap

Each topic runs the full per-session loop. The four subtasks below map to the workflow
phases: **gather** (sources → `RESOURCES.md`, wiki pages with citations), **build**
(`NN_topic.ipynb` + sidecar, reimplemented to the LA layer, asserts against a reference),
**interrogate** (Socratic loop until satisfaction — escalate on demand), **document** (fan
out into artifacts: transcript + retrieval cues + follow-ups in the sidecar, a briefing in
`## Project Logs`, demonstrated understanding → `learning-records/`, recurring concepts → `wiki/`).

### 0. Inference on a pre-trained model *(retroactive — notebook 01 already exists)*

Dry-run the full loop end-to-end on existing material before applying it to new topics:
stand up the environment and run inference on a pre-trained model (GPT-2).

- [x] **Gather** — wiki pages for the GPT-2 forward pass / tokenization / HF inference path,
  cited in `RESOURCES.md` ([tokenization-bpe](./wiki/tokenization-bpe.md),
  [gpt2-forward-pass](./wiki/gpt2-forward-pass.md), [hf-inference-path](./wiki/hf-inference-path.md);
  two residual claims logged to `RESOURCES.md` `## Gaps`).
- [x] **Build** — `01_gpt2_inference.ipynb` (exists; backfill the sidecar
  `01_gpt2_inference.md` derived from the wiki).
- [ ] **Interrogate** — run the Socratic loop on the existing notebook until satisfaction.
- [ ] **Document** — sidecar (transcript / retrieval cues / follow-ups) + `## Project Logs`
  briefing; `learning-record` only on demonstrated understanding.

### 1. Transformer

Reimplement the forward pass to the linear-algebra layer; pin against HF logits.

- [ ] **Gather** — embeddings, attention, MLP, residual stream, LayerNorm.
- [ ] **Build** — reimplement the forward pass; `assert reimpl_logits ≈ hf_logits`.
- [ ] **Interrogate** — reconstruct-on-demand on attention; predict-before-reveal on shapes.
- [ ] **Document** — sidecar (transcript / retrieval cues / follow-ups) + `## Project Logs` briefing; promote recurring primitives to `wiki/`.

### 2. Circuits

Disentangle a concrete circuit first, so feature-finding (SAEs) is *motivated*.

- [ ] **Gather** — induction heads, QK/OV, path patching / activation patching.
- [ ] **Build** — isolate a circuit; verify by intervention.
- [ ] **Interrogate** — Socratic, spaced re-quiz of transformer internals (interleave).
- [ ] **Document** — sidecar (transcript / retrieval cues / follow-ups) + `## Project Logs` briefing; `learning-record` on demonstrated understanding.

### 3. Sparse Autoencoders

Decompose the residual stream into interpretable features.

- [ ] **Gather** — superposition, SAE architecture, sparsity penalties, dictionary learning.
- [ ] **Build** — train/load an SAE on captured activations; inspect features.
- [ ] **Interrogate** — Socratic, interleave circuits + transformer.
- [ ] **Document** — sidecar (transcript / retrieval cues / follow-ups) + `## Project Logs` briefing; promote recurring concepts to `wiki/`.

### 4. Evaluations

Measure whether the interpretability claims actually hold.

- [ ] **Gather** — faithfulness/completeness metrics, ablation studies.
- [ ] **Build** — evaluate a circuit or SAE against a metric.
- [ ] **Interrogate** — Socratic, interleave all prior topics.
- [ ] **Document** — sidecar (transcript / retrieval cues / follow-ups) + `## Project Logs` briefing; `learning-record` on demonstrated understanding.

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

The rolling **briefing** the Document phase writes to — a short per-topic progress view,
skimmable to re-orient when coming back cold. Fidelity lives in each notebook's sidecar; this
is the cross-topic summary.

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

## Backlog

Cross-cutting work that isn't a curriculum topic — process/tooling fixes surfaced by a
session, tracked here until pulled. (Topic-specific deferrals live in each notebook
sidecar's *Follow-ups*; source gaps live in [`RESOURCES.md ## Gaps`](./RESOURCES.md#gaps).)

### Fix Gemini delegation in the Gather phase

The per-session loop's Gather step is meant to stay *context-cheap* by fanning out one Gemini
worker per source ([`dispatching-parallel-agents`](./CLAUDE.md) + [`delegating-to-gemini`])
to read-and-extract under a strict citation contract, with a second pass reviewing the drafted
page for gaps. During notebook 01's Gather this **failed**: the `agy`/Gemini extraction workers
produced no output within the timeout, and the drafted-page review pass returned a meta
non-answer instead of engaging. Citation rigor still held — every quote was verified directly
against its primary source — but the labor-saving delegation did not, so the whole Gather ran
by hand. Until this is fixed, Gather defaults to direct self-verification (correct but not
context-cheap).

- [ ] Diagnose `agy --print` on large prompts and with `--add-dir` (does it engage tools / read
  the provided context, or fall back to a bare model reply?).
- [ ] Re-validate the `delegating-to-gemini` worker contract end-to-end on one source.
- [ ] Re-validate the drafted-page gap-review pass returns substantive findings.
- [ ] Confirm the orchestration (`dispatching-parallel-agents`) handles a worker timing out.

Tracked in [`RESOURCES.md ## Gaps`](./RESOURCES.md#gaps) ("Delegated-extraction reliability").
Resolve before step 1's Gather so the Transformer topic can run the loop as designed.

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
