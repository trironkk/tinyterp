"""Decoder-only transformer: config, modules, and checkpoint loading.

Pre-norm GPT-style blocks (attention then MLP, each added into the residual stream), learned
positional embeddings, untied embedding/unembedding. Attention's forward is computed by hand
(`q @ k.T` then softmax, not a fused kernel) so the explicit weights are available to
interpretability probes via `store_attn`/`last_attn`.

References:
    - notebooks/2026-07-07_transformer_forward_pass.py
"""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class Config:
    """Model-sizing knobs only; a training entry point layers training fields on top of this."""

    vocab_size: int
    d_model: int = 128
    n_layers: int = 2
    n_heads: int = 4
    d_mlp: int = 512  # 4x d_model, the standard feedforward expansion
    block_size: int = 128  # context length (max positions)

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def __post_init__(self) -> None:
        assert self.d_model % self.n_heads == 0, "n_heads must divide d_model"


class CausalSelfAttention(nn.Module):
    """Causal multi-head self-attention, computed by hand (not a fused kernel) so the
    softmax weights are available to interpretability probes."""

    def __init__(self, config: Config):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.q_proj = nn.Linear(config.d_model, config.d_model)
        self.k_proj = nn.Linear(config.d_model, config.d_model)
        self.v_proj = nn.Linear(config.d_model, config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)
        # Lower-triangular [1, 1, T, T] mask; broadcast over batch and heads. A buffer, so it is
        # not trained but still follows the module across devices.
        mask = torch.tril(torch.ones(config.block_size, config.block_size)).bool()
        self.register_buffer("causal_mask", mask.view(1, 1, config.block_size, config.block_size))
        # When set, forward stashes the attention weights in `last_attn` for interpretability
        # (attention patterns, induction heads). Off by default so training holds no extra memory;
        # not a parameter or buffer, so it never enters the state_dict.
        self.store_attn = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x [batch, seq, d_model] -> [batch, seq, d_model]. Each position attends over itself and
        earlier positions (causal), across n_heads independent subspaces of width head_dim."""
        batch, seq, d_model = x.shape

        # Project, then split the model dimension into heads: [batch, n_heads, seq, head_dim].
        def to_heads(proj: torch.Tensor) -> torch.Tensor:
            return proj.view(batch, seq, self.n_heads, self.head_dim).transpose(1, 2)

        q = to_heads(self.q_proj(x))
        k = to_heads(self.k_proj(x))
        v = to_heads(self.v_proj(x))

        # Scaled dot-product scores, masked to the causal (lower-triangular) region.
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~self.causal_mask[:, :, :seq, :seq], float("-inf"))
        attn = F.softmax(scores, dim=-1)  # [batch, n_heads, seq, seq]
        if self.store_attn:
            self.last_attn = attn.detach()  # kept for interpretability probes

        # Weighted sum of values, then merge heads back to [batch, seq, d_model].
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(batch, seq, d_model)
        return self.out_proj(out)


class MLP(nn.Module):
    """Position-wise feedforward."""

    def __init__(self, config: Config):
        super().__init__()
        self.fc = nn.Linear(config.d_model, config.d_mlp)
        self.proj = nn.Linear(config.d_mlp, config.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x [batch, seq, d_model] -> [batch, seq, d_model], transformed at each position
        independently (no mixing across positions)."""
        return self.proj(F.gelu(self.fc(x)))


class Block(nn.Module):
    """One pre-norm decoder block: attention then MLP, each added into the residual stream."""

    def __init__(self, config: Config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.mlp = MLP(config)

    def forward(self, residual: torch.Tensor) -> torch.Tensor:
        """residual [batch, seq, d_model] -> [batch, seq, d_model]. Pre-norm: each sublayer reads
        a LayerNorm'd copy of the residual stream and adds its result back."""
        residual = residual + self.attn(self.ln1(residual))  # attention sublayer (across positions)
        residual = residual + self.mlp(self.ln2(residual))  # MLP sublayer (per position)
        return residual


class Transformer(nn.Module):
    """Full decoder-only model: embeddings -> blocks -> final LayerNorm -> unembed -> logits."""

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.block_size, config.d_model)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.n_layers))
        self.final_layernorm = nn.LayerNorm(config.d_model)
        self.unembed = nn.Linear(config.d_model, config.vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """input_ids [batch, seq] (token ids) -> logits [batch, seq, vocab]. Sum token and
        positional embeddings into the residual stream, run the blocks, final LayerNorm, unembed."""
        batch, seq = input_ids.shape
        assert seq <= self.config.block_size, f"sequence length {seq} exceeds block_size"

        positions = torch.arange(seq, device=input_ids.device)
        # Write into the residual stream: token identity + position. pos_emb broadcasts over batch.
        residual = self.token_emb(input_ids) + self.pos_emb(positions)
        for block in self.blocks:
            residual = block(residual)
        residual = self.final_layernorm(residual)
        return self.unembed(residual)


_MODEL_FIELDS = ("vocab_size", "d_model", "n_layers", "n_heads", "d_mlp", "block_size")


def load_model(path: str | Path) -> tuple[Transformer, Config, list]:
    """Load a checkpoint saved by a training entry point: {config, state_dict, history, best_val,
    best_step, samples}. Builds a Config from the checkpoint's model-sizing fields (ignoring any
    training-only fields the saved config also carries), constructs a matching Transformer, loads
    the weights (strict), and returns (model, config, history).

    The contract is load, don't retrain: see tinyterp.cache and the large-run checkpoint in
    notebooks/2026-07-07_transformer_forward_pass.py [K].
    """
    ckpt = torch.load(Path(path), map_location="cpu")
    raw_config: dict[str, Any] = ckpt["config"]
    config = Config(**{field: raw_config[field] for field in _MODEL_FIELDS})
    model = Transformer(config)
    model.load_state_dict(ckpt["state_dict"])
    return model, config, ckpt["history"]
