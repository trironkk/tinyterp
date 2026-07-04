"""Device selection: pick the best available torch backend.

References:
    - notebooks/2026-07-03_hardware_detection.py
"""

import torch


def get_device(override: str | None = None) -> torch.device:
    """Best available backend (cuda > cpu), or exactly `override`.

    The `override` flag exists for benchmarking: comparing backends requires
    pinning each side explicitly (e.g. a forced CPU baseline).
    """
    if override is not None:
        return torch.device(override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
