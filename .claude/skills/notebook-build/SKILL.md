---
name: notebook-build
description: Fill a notebook's approved skeleton cell by cell in lockstep. Reached by notebook-arc.
---

**Build.** In lockstep: fill one skeleton cell, update its logbook, narrate one sentence, ask
"Questions?", wait. A bare "continue" advances exactly one cell and never more. The logbook is
the cell's markdown description — what it does, why this approach, and the steering that shaped
it, written as it lands, never backfilled.

## Code trajectory

Default trajectory for any new capability: **naive → demonstrate → optimize.**

1. **Naive first.** Write the obvious version: a single top-to-bottom procedure, one function
   per genuinely reused primitive, inline `#` section-header comments, no cleverness. It's a
   teaching artifact — legibility beats speed.
2. **Demonstrate at small scale.** Run it only at demo scale, to show it works. Do not
   benchmark or scale-calibrate the naive version: its cost is knowable by inspection, so pick
   a small size (a handful of items) by reasoning, not by spawning timed runs.
3. **Optimize on a concrete trigger.** A real corpus, a timing need, or a scale requirement is
   the explicit trigger to write a *separate* optimized version. Keep the naive one as the
   reference, and assert the optimized path reproduces it exactly (byte-identical output across
   the refactor). Benchmark and calibrate only this path — that's where speed matters.

**Front-load known gotchas.** Before optimizing a textbook algorithm, state its subtleties and
design for them, so you don't rediscover them through expensive runs. (BPE: argmax needs a
deterministic tie-break for reproducible merges; greedy-longest-match and merge-replay are
different tokenizers that can diverge.) Bake in, verify once.

**Calibrate cheaply.** Reason about complexity before measuring; estimate, then measure only to
confirm a decision, not to explore. Batch measurements into one run over several sizes. Compare
optimized-vs-naive once, at a size where naive is tolerable, then trust the complexity argument
for larger sizes. Approximate is fine for sizing knobs — don't chase an exact target with
repeated timed runs.

**Minimize expensive re-runs.** Smoke-test new or edited code at tiny scale before any full
run. Compute a parameter sweep from one run where structure allows (BPE: train at the max
vocab; smaller vocabs are merge-order prefixes — truncate, don't retrain). Guard expensive
training with a load-if-exists check; persist intermediate artifacts. Structure the one
expensive run to also produce likely follow-ups (multi-size stats, persistence, analysis).

**Review gate.** The last cell has no next cell for "continue" to name, so do not roll into
shipping. Stop, present the completed notebook, and hold. Build is done only when every cell is
filled, the logbook is current, and the user has explicitly approved the finished notebook — an
approval that names the notebook or the finish, not the next cell. Only then invoke
`notebook-finalize`.

## Outputs

- **Show, don't tell.** The notebook makes its case through output, not prose — its computed
  numbers and real samples carry the persuasion. Demonstrate an effect from both sides, the metric
  and the artifact (ablation damage is a loss delta and a before/after generation), and anchor an
  aggregate in a concrete example before generalising.
- **Record-friendly.** Every cell's output is captured into the committed `runs/` record as
  GitHub-rendered markdown, so prefer outputs that survive that path: matplotlib figures (inline
  PNGs) and plain-text emphasis (markers, monospace alignment). Avoid ANSI/terminal colour and
  inline HTML styling — the record cannot render them and they land as raw escape codes.

## Logbook prose

- Formal lab-notebook register, structured as a research lab notebook: objective, method,
  measured results, conclusions. No conversational narration.
- Lead with the objective: open each cell's note with why the cell exists and what it adds to
  the arc, and where it sharpens name the evidence tier (a hypothesis formed by eye, a per-item
  score, a falsification check, a causal test).
- Falsify: a visualization or a score is a hypothesis, not a result — state the claim it suggests,
  then a check that could refute it (a control, a held-out case, an ablation); successive cells
  raise the standard of evidence on the same claim, so a picture that looks right still owes a test.
- Terse by default: a single tight paragraph usually beats a bulleted exposition; when in doubt,
  prune.
- Document why, not what: non-obvious considerations, measured results, rejected
  alternatives. Never narrate what the next line of code does, and never justify imports.
- No content that duplicates the README; link or omit.
- No em-dashes anywhere, including cell-title separators — use colons, commas, or
  parentheses (en-dashes in numeric ranges are fine). Grep for `—` before finishing.
- Collaboration stays implicit: decisions appear as recorded rationale ("min/max error
  bars were evaluated and rejected"), never as attributed dialogue ("Steering:", "the
  user asked").
- Diagnostic prints use the self-documenting f-string form `print(f"{expr=}")` so every
  logged value is prefixed by the code that generated it; format specs go after the `=`
  (`f"{diff=:.2e}"`). Prose conclusions and tables are exempt.
