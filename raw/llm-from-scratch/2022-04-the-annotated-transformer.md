# The Annotated Transformer

> Source: https://nlp.seas.harvard.edu/annotated-transformer/ (code: https://github.com/harvardnlp/annotated-transformer)
> Collected: 2026-06-22
> Published: 2022-04 (v2022 revision; original 2018-04-03)

_Note: Captured from the repository README and the `the_annotated_transformer.py` jupytext source
via GitHub; condensed faithfully, not verbatim. By Harvard NLP — v2022: Austin Huang, Suraj
Subramanian, Jonathan Sum, Khalid Almubarak, et al.; original by Sasha Rush._

## What it is

The Annotated Transformer is a line-by-line PyTorch implementation of the paper
"Attention Is All You Need" (Vaswani et al., 2017), presented as a runnable notebook in which the
paper's text is interleaved with working code. The notebook is authored with
[jupytext](https://github.com/mwouts/jupytext): a python source file (`the_annotated_transformer.py`)
is kept in sync with the `.ipynb`, so cell outputs stay out of version control. It is "a fairly
complete implementation" reproducing the base model end to end.

## Structure

The document is organized as the paper is, code following prose:

### Part 1: Model Architecture

- **Background** — motivation for replacing recurrence/convolution with attention.
- **Encoder and Decoder Stacks** — an encoder-decoder structure; the encoder is a stack of N=6
  identical layers, each with a multi-head self-attention sub-layer and a position-wise
  feed-forward sub-layer, each wrapped with a residual connection and layer normalization
  (`LayerNorm(x + Sublayer(x))`). The decoder adds a third sub-layer performing multi-head
  attention over the encoder output, and masks self-attention so predictions for position *i*
  depend only on positions before *i*.
- **Attention** — *Scaled Dot-Product Attention*: `Attention(Q, K, V) = softmax(QKᵀ/√d_k)V`.
  *Multi-Head Attention* with h=8 heads, each of dimension d_k = d_v = d_model/h = 64. Describes
  the three uses of attention in the model (encoder self-attention, decoder masked
  self-attention, encoder-decoder attention).
- **Position-wise Feed-Forward Networks** — two linear transformations with a ReLU in between
  (inner dimension 2048).
- **Embeddings and Softmax** — learned embeddings; weights shared between embedding layers and
  pre-softmax linear; embeddings scaled by √d_model.
- **Positional Encoding** — fixed sinusoidal encodings of varying frequency added to embeddings.
- **Full Model** — assembled via a `make_model` factory; an **Inference** demo shows untrained
  forward passes.

### Part 2: Model Training

Covers **batching and masking**, the **training loop**, the **optimizer** (Adam with the paper's
warmup/decay learning-rate schedule), and **regularization** via **label smoothing** (using KL
divergence). A toy copy task and then a real **German→English** translation example demonstrate
the full pipeline, with notes on multi-GPU training, BPE/tokenization, averaging checkpoints, and
attention visualization.

## Tooling notes

The repo uses `make notebook` / `make html` (via jupytext + nbconvert) to build artifacts, and
enforces PEP8 with black and flake8.
