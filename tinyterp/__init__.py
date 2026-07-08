"""tinyterp — personal mechanistic interpretability workspace.

Settled code graduates here from experiments/.
"""

from tinyterp.device import get_device
from tinyterp.tokenizer import (
    build_tokenizer,
    decode,
    encode,
    load_tokenizer,
    save_tokenizer,
    show_tokenization,
    train_bpe,
    train_bpe_incremental,
)

__all__ = [
    "get_device",
    "build_tokenizer",
    "decode",
    "encode",
    "load_tokenizer",
    "save_tokenizer",
    "show_tokenization",
    "train_bpe",
    "train_bpe_incremental",
]
