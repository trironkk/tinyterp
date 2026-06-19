# tinyterp

This repository will hold my personal explorations of the mechanistic interpretability domain. I'll
be re-implementing concepts on a smaller scale to convince myself of my own understanding,
minimizing the dependencies on the critical path of training and inference.

## Learning Workflow

This repo is run as a teaching workspace: I write notebooks with Claude (see
[CLAUDE.md](./CLAUDE.md) for its operating manual as teacher), then interrogate them. The
system has **two layers**: a *subjective* learning loop (Gather / Build / Interrogate /
Document) whose every artifact is a **memory aid** that certifies nothing, and an *objective*
layer — **producing results through experimentation** — which is the only thing that convinces
me I understand. The loop's guiding principle is **fast default, depth on demand**: I move
quickly because I have a foundation and reach for deeper interaction modes when *I* decide I
need them. The loop ends on my own judgment; assessment lives only in experimental results.

**Per-session loop:**

1. **Gather** — collect canonical sources, record them in `RESOURCES.md`, compile the
   relevant `wiki/` pages *with citations*. Nothing is taught from memory.
2. **Build** — Claude writes a short notebook (`notebooks/NN_topic.ipynb`) and sidecar
   (`notebooks/NN_topic.md`) derived from the wiki. Reimplemented to the linear-algebra
   layer; inline asserts check *correctness* against a reference (e.g. matching HF logits).
   A green assert certifies the code, not my understanding — conviction comes later, from
   generative experimentation.
3. **Interrogate** — an *interaction* layer, not assessment. Fast Socratic by default; I pull
   a deeper **interaction mode** when I want to chew harder (open menu, starter set:
   *worked walkthrough*, *predict-before-reveal*, *reconstruct-on-demand*, *teach-back*,
   *free exploration*). The session ends when **I** say I've got it — no confirmation check.
   Claude opens with 1–2 cold cues from prior topics as *instrumentation* for my own
   calibration, never as a gate.
4. **Document** — the step fans out into memory aids, each with its own job (none is
   evidence): a **`## Resume here` bookmark** in the sidecar for mid-thread clock-exits; a
   **transcript** (a pull-only on-demand archive); **retrieval cues** (questions without
   answers, feeding the cold-cue push); the sidecar's **Follow-ups** backlog (pulling one
   spawns a companion notebook `NNb_…`); a short **briefing** in this README's `## Project
   Logs`; an ungated **`learning-records/`** experience-log entry; and recurring concepts
   graduate to `wiki/`.

**Artifacts:** `MISSION.md` (the *why*, grounds everything) · `wiki/` (source-grounded input
knowledge) · `RESOURCES.md` (curated sources) · `notebooks/NN_*.{ipynb,md}` (exploration +
sidecar: resume bookmark / transcript / retrieval cues / follow-ups; **experimental results
are the only evidence**) · `## Project Logs` below (rolling progress briefing) ·
`learning-records/` (ungated experience log) · a glossary (gut-added vocabulary aid). Every
loop artifact is a memory aid, not evidence. `tinyterp/` and `tests/` stay empty until
duplication forces an extraction.

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

Each topic runs the full per-session loop. **Topic numbers align with notebook numbers**
(topic *N* ↔ `notebooks/0N_…`), so there's no off-by-one between the roadmap and the files.
The four subtasks below map to the workflow
phases: **gather** (sources → `RESOURCES.md`, wiki pages with citations), **build**
(`NN_topic.ipynb` + sidecar, reimplemented to the LA layer, asserts against a reference),
**interrogate** (Socratic by default, deeper interaction modes on demand, ends on my own
judgment), **document** (fan out into memory aids: resume bookmark + transcript + retrieval
cues + follow-ups in the sidecar, a briefing in `## Project Logs`, an ungated
`learning-records/` entry, recurring concepts → `wiki/`).

### 1. Inference on a pre-trained model *(notebook `01` — retroactive, already exists)*

Dry-run the full loop end-to-end on existing material before applying it to new topics.
The **goal is a working environment** — confirm the toolchain (`uv`, torch on the right
device, `transformers`, model/tokenizer download) is configured end-to-end and future
notebooks can rely on it. Running GPT-2 inference is the **smoke test** that proves it, not
the subject of study; GPT-2's architecture is out of scope here (it's the topic-2
reimplementation).

- [x] **Gather** — wiki pages for the GPT-2 forward pass / tokenization / HF inference path,
  cited in `RESOURCES.md` ([tokenization-bpe](./wiki/tokenization-bpe.md),
  [gpt2-forward-pass](./wiki/gpt2-forward-pass.md), [hf-inference-path](./wiki/hf-inference-path.md);
  two residual claims logged to `RESOURCES.md` `## Gaps`).
- [x] **Build** — `01_gpt2_inference.ipynb` (exists; sidecar
  [`01_gpt2_inference.md`](./notebooks/01_gpt2_inference.md) backfilled as session memory aids).
- [x] **Interrogate** — fast Socratic on the notebook *as an environment smoke test* (device
  plumbing / eval-determinism / HF download-cache); ended on my own judgment. Tokenizer
  round-trip deferred to the sidecar Follow-ups.
- [x] **Document** — sidecar (resume bookmark / transcript / retrieval cues / follow-ups) +
  `## Project Logs` briefing below; learning-record
  [`0001-gpt2-inference-environment`](./learning-records/0001-gpt2-inference-environment.md).

### 2. Transformer *(notebook `02`)*

Reimplement the forward pass to the linear-algebra layer; pin against HF logits.

- [ ] **Gather** — embeddings, attention, MLP, residual stream, LayerNorm. **First actions,
  carried over from notebook 01:** (a) add **GPT-1 (Radford 2018)** to `RESOURCES.md` and re-cite
  the **GELU** and **learned positional embedding** claims to a primary source — they're currently
  read off the HF module printout ([`RESOURCES.md` Gaps](./RESOURCES.md#gaps)); (b) re-verify the
  existing [`gpt2-forward-pass`](./wiki/gpt2-forward-pass.md) page rather than inheriting it as
  "done". For the **per-primitive** pages (LayerNorm, fused `c_attn` Q/K/V split, causal mask,
  multi-head reshape, weight-tied unembed, `Conv1D`-vs-`Linear` weight layout) weight toward the
  **implementation sources** (HF / OpenAI code) for exact tensor layout, papers for the *why* —
  layout facts the `≈` assert will catch anyway can stay provisional; a wrong *mental model* it
  can't catch cannot.
- [ ] **Build** — reimplement the forward pass; `assert reimpl_logits ≈ hf_logits`.
- [ ] **Interrogate** — Socratic; pull *reconstruct-on-demand* on attention or *predict-before-reveal* on shapes for depth.
- [ ] **Document** — sidecar (resume bookmark / transcript / retrieval cues / follow-ups) + `## Project Logs` briefing; promote recurring primitives to `wiki/`.

### 3. Circuits *(notebook `03`)*

Disentangle a concrete circuit first, so feature-finding (SAEs) is *motivated*.

- [ ] **Gather** — induction heads, QK/OV, path patching / activation patching.
- [ ] **Build** — isolate a circuit; verify by intervention.
- [ ] **Interrogate** — Socratic; open with cold cues on transformer internals (interleave).
- [ ] **Document** — sidecar (resume bookmark / transcript / retrieval cues / follow-ups) + `## Project Logs` briefing; ungated `learning-record` experience-log entry.

### 4. Sparse Autoencoders *(notebook `04`)*

Decompose the residual stream into interpretable features.

- [ ] **Gather** — superposition, SAE architecture, sparsity penalties, dictionary learning.
- [ ] **Build** — train/load an SAE on captured activations; inspect features.
- [ ] **Interrogate** — Socratic; open with cold cues across circuits + transformer (interleave).
- [ ] **Document** — sidecar (resume bookmark / transcript / retrieval cues / follow-ups) + `## Project Logs` briefing; promote recurring concepts to `wiki/`.

### 5. Evaluations *(notebook `05`)*

Measure whether the interpretability claims actually hold.

- [ ] **Gather** — faithfulness/completeness metrics, ablation studies.
- [ ] **Build** — evaluate a circuit or SAE against a metric.
- [ ] **Interrogate** — Socratic; open with cold cues across all prior topics (interleave).
- [ ] **Document** — sidecar (resume bookmark / transcript / retrieval cues / follow-ups) + `## Project Logs` briefing; ungated `learning-record` experience-log entry.

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

**Topic 1 — loop complete (2026-06-19).** Notebook 01 reads as an **environment smoke test**
across three independent failure surfaces: device plumbing (mismatch fails loud, wrong-choice
fails silent), `.eval()`/determinism (a *mode* bug, not a math bug; greedy decoding cascades
dropout noise into a different sentence), and the HF download/cache pipeline (the only line
exercising network+hub+deserialize; the pretrained weights are topic 2's reference oracle).
Toolchain confirmed wired end-to-end. Fluency, not evidence — topic 2 is the real test.
Deferred: tokenizer round-trip (sidecar Follow-ups). See
[`notebooks/01_gpt2_inference.md`](./notebooks/01_gpt2_inference.md),
[`learning-records/0001`](./learning-records/0001-gpt2-inference-environment.md).

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

### Fix Gemini delegation in the Gather phase — RESOLVED (2026-06-19)

The per-session loop's Gather step is meant to stay *context-cheap* by fanning out one Gemini
worker per source ([`dispatching-parallel-agents`](./CLAUDE.md) + [`delegating-to-gemini`])
to read-and-extract under a strict citation contract, with a second pass reviewing the drafted
page for gaps. During notebook 01's Gather this **failed**: the `agy`/Gemini extraction workers
produced no output, and the drafted-page review pass returned a meta non-answer instead of
engaging. Citation rigor still held — every quote was verified directly against its primary
source — but the labor-saving delegation did not, so the whole Gather ran by hand.

**Root cause:** the `delegating-to-gemini` skill documented the prompt as a *positional arg*.
In `agy` 1.0.10 the prompt must be the **value of `--print`/`-p`**, so `agy --print --model "…"
"<prompt>"` made `--print` swallow `--model` as its value, `--model` silently fell back to
default, and the real prompt was dropped — Gemini answered an empty prompt with a generic
self-identification ("I am running on Gemini 3.5 Flash"), the exact "meta non-answer" seen
above. Nothing was wrong with the network, tools, large prompts, or `--add-dir`.

**Fix:** corrected the invocation throughout `delegating-to-gemini` (prompt as `--print`'s
value, placed last). Validated end-to-end on 2026-06-19:

- [x] Diagnosed `agy --print` — tool use (web search) and `--add-dir` both engage correctly
  once the prompt actually reaches the model; the failure was 100% the dropped-prompt bug.
- [x] Re-validated the worker contract end-to-end: `--add-dir` over `wiki/` returned verbatim
  quotes + section headings under the citation contract.
- [x] Re-validated tool engagement: web-search and file-read calls both return substantive
  results (no `--dangerously-skip-permissions` needed).
- [ ] Confirm the orchestration (`dispatching-parallel-agents`) handles a worker timing out
  (untested — no longer blocking, since workers now produce output).

Topic 2's Gather can run the loop as designed.

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
