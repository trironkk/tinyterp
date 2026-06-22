# nanoGPT & nanochat: Minimal GPT Training in PyTorch

> Sources: Andrej Karpathy, 2022-12-29; Andrej Karpathy, 2025-10-13
> Raw: [nanoGPT](../../raw/llm-from-scratch/2022-12-29-nanogpt.md); [nanochat](../../raw/llm-from-scratch/2025-10-13-nanochat.md)

## Overview

nanoGPT and its successor nanochat are Andrej Karpathy's minimal, hackable PyTorch codebases for
training GPT-style language models on a single GPU node. **nanoGPT** (2022) is a ~600-line GPT-2
reproduction focused purely on pretraining; **nanochat** (2025) extends the same philosophy to the
*entire* LLM pipeline — tokenizer, pretraining, finetuning, RL, evaluation, inference, and a chat
UI — letting you train a GPT-2-grade ChatGPT-like model for roughly the cost of a dinner. Both
prioritize readable, forkable code over framework generality, making them the practical, framework-
backed counterpart to Karpathy's dependency-free [microGPT](microgpt.md).

## nanoGPT: a GPT-2 reproduction in ~600 lines

nanoGPT is "the simplest, fastest repository for training/finetuning medium-sized GPTs," a rewrite
of minGPT that "prioritizes teeth over education." The whole project is essentially two files:
`train.py` (~300-line training loop) and `model.py` (~300-line GPT definition that can load OpenAI's
GPT-2 weights).

- **Reproducing GPT-2 (124M):** tokenize OpenWebText into uint16 BPE token streams, then run
  `torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py` — ~4 days on an 8×A100
  node via PyTorch DDP, down to loss ~2.85. OpenAI's GPT-2 checkpoints provide baselines
  (124M → 350M → 774M → 1558M give val losses 3.12 → 2.84 → 2.67 → 2.54 on OpenWebText). Because
  GPT-2 was trained on the closed WebText, a domain gap means the fair reproduction target is the
  ~2.85 reached by finetuning GPT-2 on OWT.
- **On-ramp:** a character-level GPT on tiny Shakespeare trains in ~3 minutes (val loss 1.47 on one
  A100; runs on CPU/MPS with smaller settings).
- **Finetuning** is just training from a pretrained checkpoint with a smaller learning rate;
  `torch.compile()` (PyTorch 2.0) cut iteration time from ~250ms to ~135ms with one line.

## nanochat: the full pipeline on one dial

nanochat is "the simplest experimental harness for training LLMs" and the successor to nanoGPT
(which "only covered pretraining"). It covers all major stages — tokenization, pretraining, SFT,
RL, evaluation, inference (with KV cache), and a web chat UI — in minimal code designed for a single
GPU node.

- **One complexity dial — `--depth`:** the number of transformer layers. This single integer
  automatically sets width, heads, learning-rate schedule, training horizon, and weight decay so the
  model comes out **compute-optimal**. Sweeping depth yields a "miniseries" of compute-optimal models;
  GPT-2 capability sits around depth 24–26.
- **Cost/speed:** the `runs/speedrun.sh` pipeline trains a GPT-2-grade model (~$43k in 2019) for
  ~$48 (≈2 hours on an 8×H100 node), then serves a ChatGPT-like UI via `python -m scripts.chat_web`.
- **Time-to-GPT-2 leaderboard:** a "speedrun" measuring wall-clock hours to beat GPT-2's DCLM CORE
  score (0.2565). Successive improvements — larger batch sizes, the NVIDIA ClimbMix dataset,
  automated research rounds — have driven the d24/d26 reference from ~3 hours down toward ~1.65.
- **Explicit precision:** no `autocast`; a global `COMPUTE_DTYPE` (bf16 on modern CUDA, fp32 on
  CPU/MPS) with fp32 master weights cast in a custom `Linear` forward.
- Uses `uv` for dependency management — matching this workspace's conventions.

## Philosophy

Both repos embody the same stance: a single, cohesive, minimal, readable, maximally-forkable
"strong baseline" with no giant config objects or model factories. Accessibility is framed as a
matter of *cognitive* complexity as much as cost. nanochat's `--depth` dial is the sharpest
expression of this — the user asks for a bigger or smaller model and "everything just works."

## See Also

- [microGPT](microgpt.md) — the dependency-free, scalar-autograd cousin that packs the same GPT
  ideas into ~200 lines for pedagogy rather than performance.
- [Neural Networks: Zero to Hero](nn-zero-to-hero.md) — Karpathy's course; its "Let's build GPT"
  lecture is the spelled-out walkthrough behind nanoGPT.
- [The Transformer Architecture](transformer-architecture.md) — the architecture these codebases
  implement.
