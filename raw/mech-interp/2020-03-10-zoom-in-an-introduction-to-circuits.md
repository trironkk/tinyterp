# Zoom In: An Introduction to Circuits

> Source: https://distill.pub/2020/circuits/zoom-in/
> Collected: 2026-06-22
> Published: 2020-03-10

_Note: Captured from the Distill article's HTML source via GitHub
(distillpub/post--circuits-zoom-in); condensed faithfully, not verbatim. By Chris Olah, Nick
Cammarata, Ludwig Schubert, Gabriel Goh, Michael Petrov, Shan Carter (OpenAI / Distill). The
founding essay of the Circuits thread._

## Thesis

Most interpretability work tries to give simple explanations of a whole network's behavior. This
essay instead proposes **zooming in**: treating individual neurons and even individual weights as
worthy of serious investigation, in the spirit of how microscopes opened up cellular biology.
Studied at this scale, the authors find networks surprisingly approachable: neurons are often
understandable, and the "circuits" of connections between them are meaningful algorithms
corresponding to facts about the world (you can "watch a circle detector be assembled from
curves," see a dog head built from eyes/snout/fur/tongue, and find circuits implementing AND, OR,
XOR over visual features).

In the spirit of Schwann's (partly-wrong) cell theory, the authors offer three deliberately strong,
speculative claims about neural networks.

## Claim 1: Features

> Features are the fundamental unit of neural networks. They correspond to directions. They can be
> rigorously studied and understood.

A **feature** is a scalar function of the input, here a **direction** in activation space (often an
individual neuron). Early layers contain features like edge and curve detectors; later layers have
features like floppy-ear or wheel detectors. The community is divided on whether such meaningful
features exist (a skeptical literature argues networks rely on texture or uninterpretable
patterns), but thousands of hours of study lead the authors to believe the typical case is that
neurons (or directions) are understandable. Understandability is the proposed escape from the
curse of dimensionality. Three illustrative examples (all from InceptionV1):

- **Curve detectors** (layer mixed3b): respond to curves/boundaries of a given orientation; found
  in families that jointly tile all orientations. Seven arguments support the interpretation —
  feature visualization, dataset examples, synthetic examples, joint tuning, and three
  circuit-based arguments (feature implementation, feature use, handwritten circuits). These form a
  general toolkit for testing any feature.
- **High-low frequency detectors**: a *non-intuitive* feature found in early vision — low-frequency
  pattern on one side of the receptive field, high-frequency on the other; a heuristic for object
  boundaries (especially against a blurred background). An example of a useful feature not
  anticipated in advance.
- **Pose-invariant dog head detector**: a high-level feature where feature visualization plus
  dataset examples (and 3D-model synthetic images) already make a strong case.

**Polysemantic neurons** complicate the picture: some neurons respond to multiple unrelated inputs
(e.g., a neuron firing for cat faces, car fronts, and cat legs). They are a major challenge for the
circuits agenda — if a 5-meaning neuron connects to another 5-meaning neuron, that's 25
connections that can't be reasoned about individually. The authors hypothesize they arise from
**superposition**.

## Claim 2: Circuits

> Features are connected by weights, forming circuits. These circuits can also be rigorously
> studied and understood.

A **circuit** is a subgraph of the network: a set of tightly linked features and the weights
between them. Reading the weights, you can literally read meaningful algorithms off the floating
point numbers. Examples:

- **Curve-detector circuit**: late curve detectors are built from earlier curve and line detectors,
  with positive weights arranged along the curve ("tangent curves") and inhibitory weights for the
  opposite orientation. Weights rotate with the curve's orientation — an **equivariant circuit**.
- **Oriented dog-head circuit** (spans four layers): two mirrored left-facing and right-facing
  pathways that inhibit each other, then union into a pose-invariant detector ("unioning over
  cases," with XOR-like properties).
- **Cars in superposition**: a pure car detector (wheels at bottom, windows at top) gets spread
  across later neurons that mostly detect dogs. **Superposition** lets the model store more features
  than it has neurons, exploiting that high-dimensional space has exponentially many *almost*
  orthogonal directions.

**Circuit motifs** — recurring abstract patterns (equivariance, unioning over cases, superposition)
— may ultimately matter more than individual circuits, by analogy to motifs in systems biology.

## Claim 3: Universality

> Analogous features and circuits form across models and tasks.

Curve detectors and high-low frequency detectors appear to form across architectures (AlexNet,
InceptionV1, VGG19, ResNetV2-50) and datasets (Places365 as well as ImageNet). If universality
holds strongly, one could imagine a "periodic table of visual features" catalogued across models;
if weak, research must focus on a few important models. The authors note evidence is so far
anecdotal but suggestive, and speculate about overlap with biological neural networks.

## Interpretability as a natural science

Borrowing from Kuhn, the authors describe interpretability as **pre-paradigmatic** — no consensus
on objects of study, methods, or evaluation. They propose treating networks as objects of empirical,
**falsifiable** investigation (like organisms in biology) rather than relying solely on benchmarks
or user studies. Circuits make this tractable: understand a circuit and you can predict what editing
its weights will do. The closing analogy: microscopes languished ~50 years until Hooke's
*Micrographia*; the qualitative discovery of cells nonetheless changed the world.

## Glossary (selected)

**Circuit** — a subgraph of the network (nodes = neurons/directions, edges = weights).
**Direction** — a linear combination of neurons; a vector in a layer's representation.
**Feature / Meaningful feature** — a scalar function of input that responds to an articulable
property. **Polysemantic feature** — responds to multiple unrelated latent variables.
**Pure feature** — responds to a single latent variable. **Universal feature** — reliably forms
across models and tasks. **Equivariance**, **Family**, **Circuit motif**, **Client feature** —
recurring structural notions defined in the essay.
