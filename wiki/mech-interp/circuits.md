# Mechanistic Interpretability: Features, Circuits, and Superposition

> Sources: Chris Olah, Nick Cammarata, Ludwig Schubert, Gabriel Goh, Michael Petrov, Shan Carter (Distill), 2020-03-10
> Raw: [Zoom In: An Introduction to Circuits](../../raw/mech-interp/2020-03-10-zoom-in-an-introduction-to-circuits.md)

## Overview

Mechanistic interpretability aims to **reverse-engineer the algorithms a trained network learned,
directly from its weights**. The founding statement of this agenda is Olah et al.'s "Zoom In: An
Introduction to Circuits" (Distill, 2020), which argues that instead of explaining a network's
behavior as a whole, we should *zoom in* — study individual neurons and weights, as cellular biology
studied cells through the microscope. At this scale, networks turn out to be surprisingly
approachable: neurons are often meaningful, and the wiring between them ("circuits") implements
readable algorithms. The essay frames three speculative-but-foundational claims — **Features**,
**Circuits**, and **Universality** — and a methodological stance: interpretability as a falsifiable
natural science.

## Claim 1 — Features

> Features are the fundamental unit of neural networks. They correspond to directions. They can be
> rigorously studied and understood.

A **feature** is a scalar function of the input, realized as a **direction** in activation space
(often a single neuron). Early layers hold edge/curve detectors; later layers hold
floppy-ear/wheel/dog-head detectors. Individual understandability is what lets interpretability
escape the curse of dimensionality — going neuron by neuron is a large undertaking but not an
exponential one. Three illustrative InceptionV1 examples:

- **Curve detectors** — respond to oriented curves; come in *families* that tile all orientations.
- **High-low frequency detectors** — a *non-intuitive* feature (low frequency on one side of the
  receptive field, high on the other) that helps detect object boundaries; an example of a useful
  abstraction the researchers did not anticipate.
- **Pose-invariant dog-head detector** — a high-level feature validated by feature visualization,
  dataset examples, and 3D-rendered synthetic images.

To test a claimed feature, the essay offers a **general seven-argument toolkit**: feature
visualization, dataset examples, synthetic examples, joint tuning, and three circuit-based arguments
(reading the implementation off the weights, checking downstream use, and hand-coding the circuit).

**Polysemantic neurons** are the wrinkle: some neurons fire for several unrelated things (the
classic example fires for cat faces, car fronts, and cat legs). They are a major obstacle —
connections between polysemantic neurons can't be reasoned about one at a time — and are
hypothesized to arise from **superposition**.

## Claim 2 — Circuits

> Features are connected by weights, forming circuits. These circuits can also be rigorously studied
> and understood.

A **circuit** is a subgraph: a few tightly linked features plus the weights between them. The
striking finding is that these subgraphs are tractable and meaningful — you can read an algorithm
off the floating-point weights. Examples:

- **Curve-detector circuit** — late curve detectors are built from earlier curve/line detectors via
  positive "tangent curve" weights along the curve and inhibitory weights for the opposite
  orientation; the weight pattern rotates with orientation (an **equivariant circuit**).
- **Oriented dog-head circuit** (four layers) — two mirrored left/right pathways that mutually
  inhibit, then union into a pose-invariant detector ("unioning over cases," with XOR-like
  structure).
- **Cars in superposition** — a pure car detector's feature is deliberately *spread* across later
  neurons that mostly detect dogs.

**Superposition** is the key idea here: a network stores **more features than it has neurons** by
exploiting that high-dimensional space contains exponentially many *almost*-orthogonal directions;
features can be packed together as long as they rarely co-occur. This is the proposed cause of
polysemanticity. **Circuit motifs** — recurring patterns like equivariance, unioning over cases, and
superposition — may, in the long run, matter more than any single circuit.

## Claim 3 — Universality

> Analogous features and circuits form across models and tasks.

Curve detectors and high-low frequency detectors appear across architectures (AlexNet, InceptionV1,
VGG19, ResNet) and datasets. If universality holds strongly, one could build a "periodic table of
features" shared across models; if it fails, research must focus on a few important models. Evidence
is so far suggestive but anecdotal, with intriguing hints of overlap with biological vision.

## Interpretability as a natural science

Borrowing Kuhn's framing, the authors call interpretability **pre-paradigmatic** — no agreed objects
of study, methods, or evaluation. Their proposed footing: treat a network as an object of empirical,
**falsifiable** investigation (predict what editing a circuit's weights will do) rather than relying
on benchmarks or user studies alone. The closing analogy is the microscope, which languished ~50
years until Hooke's *Micrographia* — the qualitative discovery of cells still changed the world.

## Note on lineage

"Zoom In" studies a *vision* model (InceptionV1), but its vocabulary — features as directions,
circuits, superposition, polysemanticity — became the foundation for the **transformer** circuits
program (Anthropic's "A Mathematical Framework for Transformer Circuits," "In-Context Learning and
Induction Heads," "Toy Models of Superposition," "Towards Monosemanticity"). Those papers are not
yet ingested: their host (transformer-circuits.pub) is unreachable under the current GitHub-only
network policy. The [tooling and curriculum](tooling-and-curriculum.md) page covers the practical
stack (TransformerLens, ARENA) that operationalizes these ideas for transformers, including
**induction heads** and **sparse autoencoders** for resolving superposition.

## See Also

- [Mechanistic Interpretability Tooling & Curriculum](tooling-and-curriculum.md) — TransformerLens
  and ARENA, the hands-on stack for doing this on transformers.
- [The Transformer Architecture](../llm-from-scratch/transformer-architecture.md) — the structure
  that transformer-circuits work reverse-engineers.
- [Neural Networks: Zero to Hero](../llm-from-scratch/nn-zero-to-hero.md) — building the models that
  mech interp then takes apart.
