# wiki/

Source-grounded **input knowledge** for tinyterp — the notes I read *to learn*, indexed by
concept. This is the "Gather" output of the [per-session loop](../README.md#learning-workflow):
before any notebook is written, the relevant concept gets compiled here **from cited sources**,
never from memory. If a claim isn't traceable to an entry in [RESOURCES.md](../RESOURCES.md),
it doesn't belong on a wiki page.

## Role (and what this is *not*)

- **Input, not output.** The wiki is what I read to understand a topic. The compressed
  *output* — terms I can already use correctly — lives in the glossary. Demonstrated
  understanding lives in `learning-records/`. Don't conflate reading a clean wiki page with
  having learned the thing; that's the fluency illusion the workflow is built to counter.
- **Grounded, not generative.** Every page cites its sources inline. The wiki's value is the
  paper trail from a notebook's prose back to a primary source.
- **Promoted, not pre-written.** A concept earns a standalone page two ways: (a) it's part of
  the **foundation** a notebook needs (compiled during that notebook's Gather phase), or
  (b) it **recurs** across notebooks and graduates from a sidecar into a shared page. Pages
  are not written speculatively ahead of need (YAGNI).

## How pages are organized

- **One concept per page**, filename in kebab-case: `wiki/residual-stream.md`,
  `wiki/attention-qk-ov.md`. Index new pages in the table below.
- **Cite as you write.** Link claims to [RESOURCES.md](../RESOURCES.md) entries; prefer primary
  sources. A page with no citations is a draft, not a wiki page.
- **Trace to a notebook.** A foundation page should name the notebook that motivated it, so the
  prose → wiki → source chain is legible.
- **Follow the curriculum spine** when sequencing depth: Transformer → Circuits → Sparse
  Autoencoders → Evaluations. Foundation first; later topics are deliberately left as
  [`## Gaps` in RESOURCES.md](../RESOURCES.md) until a notebook reaches them.

## Pages

| Concept | Page | Motivated by | Status |
|---|---|---|---|
| Byte-level BPE tokenization | [tokenization-bpe](./tokenization-bpe.md) | notebook 01 | Drafted (cited) |
| The GPT-2 forward pass | [gpt2-forward-pass](./gpt2-forward-pass.md) | notebook 01 | Drafted (cited); re-verified 2026-06-19 |
| HuggingFace inference path | [hf-inference-path](./hf-inference-path.md) | notebook 01 | Drafted (cited) |
| `Conv1D` vs `nn.Linear` (weight layout) | [conv1d-vs-linear](./conv1d-vs-linear.md) | notebook 02 | Drafted (cited) |
| LayerNorm | [layernorm](./layernorm.md) | notebook 02 | Drafted (cited) |
| Attention (fused QKV, multi-head, causal) | [attention](./attention.md) | notebook 02 | Drafted (cited) |
| GELU (`"gelu_new"`) | [gelu](./gelu.md) | notebook 02 | Drafted (cited) |

> First batch authored during notebook 01's Gather phase (transformer foundation at *overview*
> altitude: tokenization → embeddings/residual-stream/attention/MLP → unembedding, plus the HF
> reference path notebook 01 runs). The **per-primitive pages** (`conv1d-vs-linear`, `layernorm`,
> `attention`, `gelu`) were compiled during notebook 02's Gather (2026-06-19) at the
> *linear-algebra* altitude — paper for the *why*, the installed HF source for the *exact tensor
> layout* the `≈ hf_logits` assert runs against. QK/OV circuit decomposition stays deferred to the
> circuits topic.
