"""Shared on-disk cache for expensive artifacts (tokenizers, token streams, checkpoints).

The cache lives outside any single worktree so multiple checkouts on the same machine reuse
results instead of each re-paying minutes-to-hours of compute. Resolution order for the root:
`$TINYTERP_CACHE_DIR`, else `$XDG_CACHE_HOME/tinyterp`, else `~/.cache/tinyterp`.

Artifacts are addressed by what produces them, not by the notebook they are used in: a tokenizer
by its dataset and vocab size (a stable, semantic name), and a trained model by a short hash of
its training config, so an unrelated edit to the notebook prose never invalidates an hours-long
checkpoint, while a genuine config change lands on a fresh file.
"""

import hashlib
import json
import os
from pathlib import Path


def cache_dir() -> Path:
    """The shared cache root, created if needed."""
    root = os.environ.get("TINYTERP_CACHE_DIR")
    if root:
        base = Path(root)
    else:
        xdg = os.environ.get("XDG_CACHE_HOME")
        base = (Path(xdg) if xdg else Path.home() / ".cache") / "tinyterp"
    base.mkdir(parents=True, exist_ok=True)
    return base


def cache_path(name: str) -> Path:
    """Absolute path for a named artifact in the shared cache."""
    return cache_dir() / name


def config_hash(config: dict, length: int = 12) -> str:
    """Stable short hash of a config dict, for naming config-addressed artifacts. Keys are
    sorted so the hash does not depend on insertion order; values are stringified so anything
    JSON cannot serialize still contributes."""
    payload = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:length]
