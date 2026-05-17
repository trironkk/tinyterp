# tinyterp

This repository will hold my personal explorations of the mechanistic interpretability domain. I'll
be re-implementing concepts on a smaller scale to convince myself of my own understanding,
minimizing the dependencies on the critical path of training and inference.

## Development Notes

### Setup

```shell
git clone https://www.github.com/trironkk/tinyterp
cd tinyterp
# Install uv at https://docs.astral.sh/uv/getting-started/installation/
uv sync
uv run nbstripout --install --attributes .gitattributes
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

### Sparse Autoencoders

> TODO

### Circuits

> TODO

### Evaluations

> TODO

## Side Quests

### Automatic Differentiation

Decided to defer this until more precise understanding of linear algebra implementation would be
more valuable.

> TODO

## Troubleshooting

### cudaErrorNoKernelImageForDevice

```markdown
## Troubleshooting

### Torch not compiled with CUDA enabled

```

AssertionError: Torch not compiled with CUDA enabled

````

Raised when code calls `.to("cuda")` (or similar) against a CPU-only torch build.
Make notebooks device-agnostic rather than hardcoding `"cuda"`:

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
inputs = inputs.to(device)
````

### undefined symbol: ncclCommResume

```
ImportError: .../torch/lib/libtorch_cuda.so: undefined symbol: ncclCommResume
```

NCCL ABI mismatch between the installed torch wheel and the NCCL it expects at runtime. Typically
surfaces after torch gets upgraded past a pinned version. Either re-pin to a known-good CUDA wheel
(see below) or switch the project to CPU-only torch.

### cudaErrorNoKernelImageForDevice

```
Search for `cudaErrorNoKernelImageForDevice' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
```

Laptop runs an NVIDIA MX150 (Pascal, compute capability 6.1). Recent torch wheels drop kernel images
for older architectures, so the GPU is detected but no compatible kernels are bundled.

Two fixes, in order of preference:

**Switch to CPU-only torch** (recommended for this project — models are small enough that CPU
training finishes in minutes, and this sidesteps an entire class of CUDA/NCCL/driver issues):

```toml
[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true  # Prevents generic packages (like numpy) from pulling from here

[tool.uv.sources]
torch = { index = "pytorch-cpu" }
```

Then `rm -rf .venv uv.lock && uv sync`.

**Pin to a torch version that still supports the MX150** (cu118 wheels through 2.7.1 include Pascal
kernels):

```toml
[[tool.uv.index]]
name = "pytorch-legacy-cuda"
url = "https://download.pytorch.org/whl/cu118"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-legacy-cuda" }
torchvision = { index = "pytorch-legacy-cuda" }
torchaudio = { index = "pytorch-legacy-cuda" }
```

And pin the version in `[project] dependencies`:

```toml
"torch==2.7.1",
```

Ad-hoc equivalent:

```shell
uv pip install torch==2.7.1 torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu118 --python .venv
```

## References

> TODO
