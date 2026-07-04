# %% [markdown]
# # Hardware detection & verification
#
# What compute does this machine actually have, does torch see it, and does it
# deliver? Defines `get_device()`, the device-selection helper every later
# notebook will initialize from.
#
# **Scope decisions.** The matmul throughput benchmark lives here rather than
# in a future matrix-multiplication notebook: it is essentially a smoke test
# of the detected hardware, and the conceptual ground it covers (what a matmul
# costs) was already familiar, so that session stays free for the operation
# itself. Colab compatibility is by convention rather than machinery: these
# percent-format `.py` files stay the source of truth, converted ad hoc with
# `uvx jupytext --to ipynb` when needed, and the code stays device-agnostic so
# a Colab GPU runtime works unchanged.
#
# **Backlog**
# - Colab bootstrap cell (guarded pip install, `"google.colab" in sys.modules`),
#   deferred until a Colab account/free tier is actually set up.
# - Graduate `get_device()` to `tinyterp/device.py` (follow-up commit).

# %% [markdown]

# **[A] Imports.** Standard library plus torch and matplotlib (both already in
# pyproject). `time.perf_counter` is the benchmark clock in [F]; `platform`
# fingerprints the host, because every number this notebook prints is
# machine-specific and meaningless without knowing which box produced it.

# %% [A] Imports
import platform
import time

import matplotlib.pyplot as plt
import torch

print(f"python {platform.python_version()} on {platform.system()} ({platform.release()})")
print(f"torch {torch.__version__}")

# %% [markdown]
# **[B] Backend detection.** We probe the two accelerator backends that matter
# for this repo's machines, CUDA (this box, WSL2) and MPS (any Apple laptop),
# and fall through to CPU, which is always available. Two distinctions worth
# preserving:
#
# - `is_built()` vs `is_available()` separates "compiled into the wheel" from
#   "usable right now": the diagnostic that matters when a backend
#   unexpectedly vanishes (e.g. driver issues under WSL2).
# - `torch.version.cuda` is the toolkit the wheel was built against, not what
#   the driver supports; the two can disagree.
#
# The backends aren't symmetrical: when we mirrored the CUDA version print for
# MPS, it turned out MPS exposes no toolkit version at all, so its branch
# reports the macOS version instead; on Apple silicon the OS *is* the driver.

# %% [B] Backend detection: cuda / mps / cpu, what's available and why
print(f"cuda: built={torch.backends.cuda.is_built()}, available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  toolkit {torch.version.cuda}, {torch.cuda.device_count()} device(s)")
print(f"mps:  built={torch.backends.mps.is_built()}, available={torch.backends.mps.is_available()}")
if torch.backends.mps.is_available():
    print(f"  macOS {platform.mac_ver()[0]}")  # MPS has no toolkit version; the OS is the driver
print("cpu:  always available")

# %% [markdown]
# **[C] Additional CUDA diagnostic details.** When CUDA is present,
# `torch.cuda.get_device_properties()` gives the full picture: name, VRAM,
# compute capability, and SM count. Compute capability drives the dtype story:
# bf16 and TF32 (tensor-core float32 matmuls) both want Ampere (8.0+).
# TF32 *capability* is a hardware fact, but whether matmuls actually use it is
# a runtime switch (`torch.backends.cuda.matmul.allow_tf32`, off by default
# since torch 1.12). We print both because the gap between them matters:
# [E] measures what flipping that switch does to accuracy.

# %% [C] Additional CUDA diagnostic details
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    cc = (props.major, props.minor)
    print(f"name:               {props.name}")
    print(f"vram:               {props.total_memory / 2**30:.1f} GiB")
    print(f"compute capability: {props.major}.{props.minor}")
    print(f"multiprocessors:    {props.multi_processor_count} SMs")
    print(f"bf16 supported:     {torch.cuda.is_bf16_supported()}")
    print(f"tf32 capable:       {cc >= (8, 0)} (matmul.allow_tf32={torch.backends.cuda.matmul.allow_tf32})")

# %% [markdown]
# **[D] `get_device()`.** The initialization idiom for every notebook in this
# repo: best available backend, fixed priority cuda > mps > cpu. The optional
# `override` hook exists because benchmarking surfaced the need: comparing
# backends means pinning each side explicitly rather than fighting the
# auto-selection ([F] uses it to get a CPU baseline on a CUDA box).
#
# TODO: graduate to `tinyterp/device.py` in a follow-up commit (see backlog).

# %% [D] get_device(): device-agnostic selection helper
def get_device(override: str | None = None) -> torch.device:
    """Best available backend (cuda > mps > cpu), or exactly `override`."""
    if override is not None:
        return torch.device(override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


device = get_device()
print(f"selected: {device}")
print(f"override: {get_device('cpu')}")

# %% [markdown]
# **[E] Smoke test.** One matmul on the selected device, checked for
# correctness against the same matmul on CPU, not just "it didn't crash".
#
# The tolerance question ("why is 1e-4 the right value?") turned out to be the
# interesting part, so instead of assuming an answer the cell measures it:
# worst max-abs-diff across trials in fp32, then again with TF32 forced on.
# Measured here (RTX 5060 Ti, 256×256): fp32 accumulation-order noise between
# cuBLAS and CPU BLAS is ~3e-5 (consistent with the back-of-envelope
# √K·ε·|element| ≈ 16·1.2e-7·16 for K=256 and elements of magnitude √K), while
# TF32 error (~10-bit mantissa) is ~3e-2. Three orders of magnitude apart, so
# the 1e-4 threshold cleanly separates the regimes: it tolerates backend noise
# but fails on a silent TF32 downgrade or real breakage (O(1)). `allclose`
# defaults would be flaky here: near-zero output elements fall back on its
# `atol=1e-8`, which trips on ordinary fp32 noise.

# %% [E] Smoke test: one op on the selected device, verify it computes
def max_matmul_diff(trials: int = 5, n: int = 256) -> float:
    """Worst max-abs-diff between device and CPU matmul across trials."""
    worst = 0.0
    for _ in range(trials):
        a, b = torch.randn(n, n), torch.randn(n, n)
        diff = (a @ b - (a.to(device) @ b.to(device)).cpu()).abs().max().item()
        worst = max(worst, diff)
    return worst


fp32_diff = max_matmul_diff()
print(f"fp32 max abs diff over 5 trials: {fp32_diff:.2e}")

if device.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    print(f"tf32 max abs diff over 5 trials: {max_matmul_diff():.2e}")
    torch.backends.cuda.matmul.allow_tf32 = False

assert fp32_diff < 1e-4, "device result diverges from CPU beyond fp32 noise"
print(f"matmul on {device} matches CPU within fp32 tolerance (1e-4)")

# %% [markdown]
# **[F] Matmul throughput benchmark.** An n×n matmul costs 2n³ FLOPs; timing
# it across sizes gives TFLOP/s per backend. Two things make GPU timing
# honest: a warmup matmul (the first CUDA kernel launch pays one-time JIT and
# allocation costs), and `torch.cuda.synchronize()` before reading the clock
# (launches are async; without the sync you time the launch, not the work).
# Tensors are created on-device so we measure compute, not PCIe transfer.
#
# The plotting evolved under scrutiny of an anomaly: repeated runs showed a
# ~3× throughput dip at n=2048 that other runs didn't reproduce. Min/max error
# bars came first, but they looked suspiciously narrow, and the suspicion was
# justified: a size's reps run back-to-back in a ~1–2 s window, so they share
# whatever GPU state prevails at that moment. Tight per-size spread measures
# short-term jitter only and says nothing about run-to-run variance, which is
# where the dip lives. So the plot shows every sample (small dots) plus the
# mean (large point) instead. A dip that recurs at the same size still isn't
# evidence of a size effect: the sweep's timing is deterministic, so any
# periodic external GPU load (the WDDM scheduler time-slices this GPU between
# CUDA and the Windows desktop under WSL2, and VS Code itself renders through
# it) lands at the same offset, hence the same size, every run. Reading the
# dots disambiguates: all 20 uniformly slow means a sustained contention
# window; bimodal means sporadic preemption. An interleaved sweep (shuffled
# (size, rep) order) would smear temporal effects into noise, but the scatter
# was judged enough for a smoke test.
#
# Measured here (fp32): CPU plateaus near 1 TFLOP/s across the whole sweep;
# the RTX 5060 Ti climbs steeply until n≈1536, then flattens at 15–18 TFLOP/s
# (~20× once saturated). At n=256 the two are nearly tied: launch overhead
# swamps the ~33 μs of actual GPU work, which is why small models don't
# automatically win on GPU.

# %% [F] Matmul throughput benchmark: CPU vs device across sizes, TFLOP/s
def matmul_tflops(dev: torch.device, n: int, reps: int = 20) -> list[float]:
    """Per-rep throughputs of an n×n fp32 matmul on `dev`, in TFLOP/s."""
    a = torch.randn(n, n, device=dev)
    b = torch.randn(n, n, device=dev)
    a @ b  # warmup: JIT/alloc costs land here, not on the clock
    samples = []
    for _ in range(reps):
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        a @ b
        if dev.type == "cuda":
            torch.cuda.synchronize()
        samples.append(2 * n**3 / (time.perf_counter() - t0) / 1e12)
    return samples


sizes = [256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192]
backends = ["cpu"] if device.type == "cpu" else ["cpu", device.type]
results = {be: [matmul_tflops(get_device(be), n) for n in sizes] for be in backends}

for be in backends:
    means = [sum(s) / len(s) for s in results[be]]
    print(f"{be:>4}: " + "  ".join(f"n={n}: {m:7.3f}" for n, m in zip(sizes, means)))

fig, ax = plt.subplots()
for be in backends:
    means = [sum(s) / len(s) for s in results[be]]
    line, = ax.plot(sizes, means, marker="o", markersize=9, label=be)
    for n, s in zip(sizes, results[be]):
        ax.scatter([n] * len(s), s, s=12, alpha=0.35, color=line.get_color(), zorder=3)
ax.set(xlabel="matrix size n", ylabel="TFLOP/s", title="fp32 matmul throughput",
       xscale="log", yscale="log")
ax.set_xticks(sizes, labels=[str(n) for n in sizes], rotation=45, minor=False)
ax.xaxis.set_minor_locator(plt.NullLocator())  # default log ticks would clutter
ax.legend()
plt.show()
