# Mechanistic Interpretability Resources

Curated, high-trust sources for tinyterp. Explainers and wiki pages draw from **here**, not from
parametric memory (see [CLAUDE.md](./CLAUDE.md) — "Never teach from parametric memory"). The
split mirrors the [teach RESOURCES format](./.claude/vendor/mattpocock-skills/skills/productivity/teach/RESOURCES-FORMAT.md):
**Knowledge** is what you read to learn; **Wisdom** is the communities/people you learn *from*.

Scope is **foundation-first** ([per the mission](./MISSION.md)): field-level entry points + the
transformer foundation needed for notebooks 01–02. Circuits, SAEs, and evals are deliberately
deferred to [`## Gaps`](#gaps) until a notebook reaches them (YAGNI). All links below were
fetched and verified on 2026-06-14 (the source search); the individual quotes pulled into the
wiki pages were re-verified against these sources on 2026-06-15 (when those pages were drafted).
The topic-2 component sources (GPT-1, GELU, LayerNorm papers; HF modeling source) were added and
their quotes verified on 2026-06-19 during the transformer notebook's Gather.

## Knowledge

### Field entry points (how to get into interp)

- [Neel Nanda — Mechanistic Interpretability hub](https://www.neelnanda.io/mechanistic-interpretability)
  The canonical on-ramp. Indexes his glossary, "Concrete Steps to Get Started", the Quickstart
  Guide, and prerequisite checklist. Use for: orienting in the field, finding the next guide.
- [Neel Nanda — How To Become A Mechanistic Interpretability Researcher (Alignment Forum)](https://www.alignmentforum.org/posts/jP9KDyMkchuv6tHwm/how-to-become-a-mechanistic-interpretability-researcher)
  The current "getting started" target (the older `getting-started` URL now redirects here).
  Three-stage path: learn the ropes → mini-projects → full projects; ≥⅓ of time writing code.
  Use for: pacing the learning arc, deciding when to stop reading and start experimenting.
- [Neel Nanda — A Barebones Guide to Mechanistic Interpretability Prerequisites (LessWrong)](https://www.lesswrong.com/posts/AaABQpuoNC8gpHf2n/a-barebones-guide-to-mechanistic-interpretability)
  The prereq checklist (linear algebra, basic ML, PyTorch). Use for: sanity-checking that a
  foundation gap is real before detouring to shore it up.
- [ARENA — Chapter 1: Transformer Interpretability](https://learn.arena.education/chapter1_transformer_interp/)
  The hands-on curriculum: build a transformer from scratch, load GPT-2 weights, then
  TransformerLens, induction heads, and the IOI circuit. Use for: a structured exercise
  sequence that parallels the tinyterp curriculum spine. ([program overview](https://www.arena.education/))
- [TransformerLens — Getting Started in Mech Interp](https://transformerlensorg.github.io/TransformerLens/content/getting_started_mech_interp.html)
  The library-maintainers' resource hub: recommended reading order, paper list, tooling.
  Use for: cross-checking the entry-point pathway and finding the TransformerLens API when a
  reimplementation needs a reference.
- [Neel Nanda — 200 Concrete Open Problems in Mechanistic Interpretability (LessWrong)](https://www.lesswrong.com/posts/LbrPTJ4fmABEdEnLf/200-concrete-open-problems-in-mechanistic-interpretability)
  Catalogue of tractable research problems by area. Use for (later): picking a mini-project
  once foundations are in place — not needed for the foundation notebooks.

### Transformer foundation (notebooks 01–02)

- [Anthropic — Transformer Circuits Thread](https://transformer-circuits.pub/)
  The primary publication venue for this style of interp ("reverse engineer transformers into
  human-understandable programs"). Use for: the canonical source on residual-stream-centric
  circuit analysis; the index from which the framework paper (below) is the entry point.
- [Elhage et al. (2021) — A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html)
  Primary source for the mental model the whole repo is built on: residual stream as a
  communication channel, attention heads as independent add-to-stream ops, QK vs OV circuits,
  path expansion, induction heads, K/Q/V-composition. Use for: the *why* behind every tensor
  op in the transformer notebooks; reread when a reimplementation feels mechanical.
- [Vaswani et al. (2017) — Attention Is All You Need (arXiv:1706.03762)](https://arxiv.org/abs/1706.03762)
  The original transformer architecture (the primary source the framework reinterprets). Use
  for: the canonical definition of multi-head attention, positional encoding, and the
  encoder/decoder block — the ground truth a reimplementation must match.
- [Radford et al. (2018) — Improving Language Understanding by Generative Pre-Training (GPT-1 paper)](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
  The architecture the GPT-2 paper inherits ("The model largely follows the details of the
  OpenAI GPT model"). Primary source for two choices the GPT-2 paper leaves unstated and that
  were previously read off the HF module printout: the **GELU** activation (*"For the activation
  function, we used the Gaussian Error Linear Unit (GELU)"*, §4.1) and **learned** positional
  embeddings (*"We used learned position embeddings instead of the sinusoidal version proposed in
  the original work"*, §4.1). Use for: re-citing those two claims to a primary source in the
  transformer notebook (topic 2).
- [Radford et al. (2019) — Language Models are Unsupervised Multitask Learners (GPT-2 paper)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
  Primary source for the specific model notebook 01 runs inference on: decoder-only,
  pre-LayerNorm, learned positional embeddings, BPE tokenization. Use for: confirming GPT-2's
  exact architectural choices when reconciling a reimplementation against HF logits.
- [Hendrycks & Gimpel (2016) — Gaussian Error Linear Units (GELUs) (arXiv:1606.08415)](https://arxiv.org/abs/1606.08415)
  Primary source for the activation GPT-1/GPT-2 use. Defines `GELU(x) = x·Φ(x)` (x times the
  standard Gaussian CDF) and the **tanh approximation** GPT-2 actually ships:
  `0.5·x·(1 + tanh[√(2/π)·(x + 0.044715·x³)])` (§2 / Discussion footnote — *"we used the former
  in every experiment in this paper"*). Use for: reimplementing GELU and confirming HF's
  `"gelu_new"` (`NewGELUActivation`) is exactly this tanh form.
- [Ba, Kiros & Hinton (2016) — Layer Normalization (arXiv:1607.06450)](https://arxiv.org/abs/1607.06450)
  Primary source for LayerNorm: normalize each token's activation vector by the **mean and
  std computed over its own features** (Eq. 3, batch-size-independent — *"on a single training
  case"*), then apply a per-feature learnable gain `g` and bias `b`. Use for: the reimplemented
  `ln_1`/`ln_2`/`ln_f` and the mental model that LayerNorm is per-token, not per-batch.
- [Andrej Karpathy — Let's build GPT: from scratch, in code, spelled out (YouTube)](https://www.youtube.com/watch?v=kCc8FmEb1nY)
  1h56m spelled-out build of a GPT to the nanoGPT core, following "Attention Is All You Need".
  Use for: a line-by-line reference when reimplementing attention/MLP blocks at the LA layer —
  the closest match to tinyterp's "reimplement to convince myself" method.
- [HuggingFace — GPT-2 model documentation](https://huggingface.co/docs/transformers/en/model_doc/gpt2)
  Reference for `GPT2LMHeadModel`/`AutoModelForCausalLM`, config, and the
  `openai-community/gpt2` checkpoints. Use for: the HF reference whose logits notebook 01
  asserts against (`assert reimpl_logits ≈ hf_logits`).
- [OpenAI — GPT-2 reference implementation (github.com/openai/gpt-2)](https://github.com/openai/gpt-2)
  The original TensorFlow code and model card for the paper above. Use for: resolving
  ambiguities in the paper against the authors' actual implementation.
- [HuggingFace `transformers` — GPT-2 modeling source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py)
  (`Conv1D` lives in [`pytorch_utils.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/pytorch_utils.py)).
  The exact PyTorch code whose logits the reimplementation asserts against. Primary source for
  **tensor layout** facts the papers don't pin: `Conv1D`'s transposed weight `(nx, nf)`, the
  fused `c_attn` Q/K/V `.split`, the `(batch, heads, seq, head_dim)` reshape, the `head_dim**-0.5`
  scaling, and `"gelu_new"`. **Read the installed version** (`.venv/.../models/gpt2/`) — it's the
  code the `≈ hf_logits` assert actually runs, and it can drift from `main`. Use for: per-primitive
  layout when reimplementing (paper for the *why*, this for the *exact shapes*).

### Interactive / visual

- [Transformer Explainer (Polo Club, Georgia Tech)](https://poloclub.github.io/transformer-explainer/)
  Live GPT-2 (small) in the browser — inspect attention maps, intermediate activations, and
  next-token probabilities while varying the input. Use for: building intuition for a tensor's
  shape/role before (or instead of) printing it; a fast sanity check on "what should attention
  look like here."

## Wisdom (Communities & People)

- [Neel Nanda](https://www.neelnanda.io/) — the field's most active on-ramp author (guides,
  glossary, open-problems list, paper walkthroughs). Reach for: the recommended next step at
  any stage of the learning arc.
- [Alignment Forum — interpretability tag](https://www.alignmentforum.org/tag/interpretability-ml-and-ai)
  Where current interp work and discussion is posted and critiqued. Use for: tracking the
  frontier and seeing how claims get challenged. (LessWrong mirrors much of this.)
- [ARENA (Alignment Research Engineer Accelerator)](https://www.arena.education/) — the program
  and its alumni/Slack community behind Chapter 1. Use for: a peer cohort doing the same
  exercises; structured accountability if self-study stalls.

> **Community preference:** not yet stated. TK has not opted into or out of joining any of the
> above as a participant (vs. reading them). Revisit when a mini-project would benefit from peer
> review; until then these are treated as read-only sources.

## Gaps

Deferred by the foundation-first scope — to be filled when a notebook reaches the topic, each
seeded during that notebook's Gather phase:

- **Notebook 01 forward-pass — two claims sourced to the reference model — RESOLVED (2026-06-19).**
  [`wiki/gpt2-forward-pass.md`](./wiki/gpt2-forward-pass.md) previously backed two facts with the
  HF `GPT2LMHeadModel` *module printout* rather than a paper: (a) **GELU** (the GPT-2 paper §2.3
  doesn't name the activation) and (b) **learned** positional embeddings (§2.3 doesn't say
  "learned"). Both are now re-cited to the GPT-1 paper §4.1 ([Radford et al. 2018](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf),
  which GPT-2 inherits), added as a Knowledge entry above. GELU's exact tanh form is further
  pinned to [Hendrycks & Gimpel (2016)](https://arxiv.org/abs/1606.08415) and HF's `"gelu_new"`.
- **Delegated code-extraction — Gemini refuses verbatim code quotes (found 2026-06-19).** During
  topic 2's Gather the *paper* extraction workers (GPT-1, GELU, LayerNorm) returned clean verbatim
  quotes + locators under the citation contract, but both *code* workers (OpenAI `model.py`, HF
  `modeling_gpt2.py`) refused — *"I cannot output verbatim code … due to copyright guidelines"* —
  and fell back to paraphrase from their own training, the parametric-memory trap CLAUDE.md forbids
  accepting as a source. *Workaround used:* self-fetch code layout from the **local installed**
  `transformers` (in-workspace, authoritative for the assert) via Bash (`grep`/`sed`), since Read
  is gated on `.gitignore` and `.venv/` is ignored. *Takeaway for future Gathers:* delegate **paper
  reading** to Gemini; pull **code** yourself from the local install.
- **Topic-2 wiki review pass run by hand; one residual layout item (2026-06-19).** The prescribed
  second Gemini pass over the four drafted per-primitive pages did not produce output — the `agy`
  call died with empty output (the harness threw "Tool permission stream closed" on both a
  foreground and a backgrounded attempt). Self-reviewed instead (as notebook 01's Gather did):
  fixed two unverified claims (`conv1d-vs-linear` no longer asserts the OpenAI TF code is "a 1×1
  conv"; `gelu` dropped an unsourced "~1e-3" figure for a qualitative note). **Residual:**
  [`attention`](./wiki/attention.md)'s description of *current* HF causal masking (an additive
  mask added before softmax) is **inferred** from the `attn_weights + attention_mask` line, not
  quoted from HF's `create_causal_mask` builder; the page hedges and gives the classic
  `tril`+`-inf` form as the reimplementation target. *Verify the exact current-HF mask
  construction only if the `≈ hf_logits` assert ever disagrees on masking.*
- **Delegated-extraction reliability — RESOLVED (2026-06-19).** The Gather for notebook 01 fell
  back to direct self-verification: the fanned-out `agy`/Gemini extraction workers produced no
  output and the review pass returned a meta non-answer. *Root cause:* `delegating-to-gemini`
  documented the prompt as a positional arg, but `agy` 1.0.10 takes it as the **value of
  `--print`** — the old form dropped the prompt entirely, so Gemini answered a blank prompt with
  a generic self-ID. Fixed the skill and validated tool use + `--add-dir` extraction end-to-end.
  See [README "Fix Gemini delegation"](./README.md). Delegation is usable for topic 2's Gather.
- **Circuits (curriculum topic 3).** Induction heads are *introduced* in the framework paper
  above, but the dedicated treatment is unfilled. Needed: [In-context Learning and Induction
  Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)
  and the IOI / "Interpretability in the Wild" circuit (activation patching, path patching,
  direct logit attribution). *Verify and annotate when notebook on circuits begins.*
- **Sparse Autoencoders (curriculum topic 4).** No source seeded. Needed: Anthropic's
  "Towards Monosemanticity" / "Scaling Monosemanticity" and a canonical SAE-training
  walkthrough. *Gather when circuits motivate feature disentangling.*
- **Evaluations (curriculum topic 5).** No source seeded. Needed: a trusted treatment of
  interp-flavored evals / benchmarks. *Scope is least defined; revisit after SAEs.*
- **Tooling deep-dive.** TransformerLens is referenced as a *reference*, not yet adopted —
  tinyterp reimplements at the LA layer first. Promote to a full Knowledge entry only if/when
  a notebook depends on its hooks. (Autodiff internals stay deferred per the mission.)
- **Math prerequisites.** Assumed in place (the [Barebones Guide](https://www.lesswrong.com/posts/AaABQpuoNC8gpHf2n/a-barebones-guide-to-mechanistic-interpretability)
  covers the checklist). Seed a focused refresher source only if a specific gap surfaces.
