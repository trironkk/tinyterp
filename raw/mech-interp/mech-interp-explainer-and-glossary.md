# A Comprehensive Mechanistic Interpretability Explainer & Glossary

> Source: https://www.neelnanda.io/mechanistic-interpretability/glossary
> Collected: 2026-06-24
> Published: Unknown

_Note: By Neel Nanda. The canonical version is a living document hosted on Dynalist; this is the
HTML reference version. Captured now that neelnanda.io is reachable (added to the network
allowlist). Condensed faithfully into a term → definition glossary; framing and intuitions
summarized, definitions preserved._

## Purpose

A canonical reference for the concepts of mechanistic interpretability, written to reduce
"research debt." It aims to explain not just definitions but the underlying intuitions,
implications, and how concepts interconnect. Pair it with the prerequisites guide and the
"Getting Started" / "200 Concrete Open Problems" resources.

## Mechanistic Interpretability: Core Concepts

- **MI / Mech Int / Mechanistic Interpretability**: The field of study of reverse engineering
  neural networks from the learned weights down to human-interpretable algorithms.
- **Interpretability**: The broader subfield studying why AI systems behave as they do and
  explaining this in human-understandable terms.
- **Feature**: A property of an input (or subset thereof) that the model internally represents
  (e.g., an image patch containing a curve, a token for a concept, text in a domain).
- **Interpretable Features**: Features that can be meaningfully understood and described in human
  terms.
- **Circuit**: The sub-part of a model that does some understandable computation to produce some
  interpretable features from prior interpretable features.
- **End-to-End Circuit**: A circuit describing how model inputs convert to outputs through
  interpretable intermediate computations.
- **Intervening On / Editing an Activation**: Running a network, stopping at a computed
  activation, editing or replacing it, then resuming with the modified activation.
- **Pruning**: An intervention where a neuron's activation is set to zero, preventing later layers
  from using its output.
- **Equivariance / Neuron Families**: Distinct but analogous neurons where understanding one
  translates to understanding others (e.g., curve detectors of different orientations).
- **Neuron Splitting**: When a feature in one model decomposes into multiple features in a larger
  model.
- **Universality**: The hypothesis that identical circuits appear across different models trained
  on similar tasks with similar architectures.
- **Motif**: An abstract notion recurring between circuits or features in different models or
  contexts.
- **Localised / Sparse Behavior**: Model behavior determined by only a few components, suggesting
  legible circuits exist.
- **Microscope AI**: The idea that superhuman models could be reverse-engineered to extract
  learned knowledge rather than deployed for use.
- **Curse of Dimensionality**: High-dimensional activations and weights are nearly impossible to
  understand intrinsically.
- **Features as Directions**: The hypothesis that features are represented as directions in
  activation space, allowing independent variation and linear access.
- **Interpretable Basis**: A set of activation-space directions where each corresponds to some
  interpretable feature.
- **Standard / Canonical Basis**: The internal representation basis in code as a tensor of floats
  (e.g., neuron directions post-nonlinearity).
- **Privileged Basis**: A meaningful basis whose coordinates have inherent significance beyond
  arbitrary choices, typically tied to architecture features like ReLUs.
- **Bottleneck Activation / Dimension**: Intermediate activations in low-dimensional spaces
  between higher-dimensional maps, with no privileged basis.
- **Features as Neurons**: The hypothesis that each neuron corresponds to a feature and its
  activation represents feature strength.
- **Enumerative Safety**: The ambitious goal of understanding every feature in a model to identify
  undesirable behavior.
- **Superposition**: When a model represents more than n features in an n-dimensional activation
  space using non-orthogonal directions.
- **Overcomplete Basis**: A set of >n directions (not linearly independent) representing more
  features than the dimensional space allows.
- **Bottleneck Dimension Superposition**: Superposition in bottleneck dimensions like keys,
  queries, or residual streams (storage-focused).
- **Neuron Superposition**: Superposition in neuron activations where computation occurs across >n
  features (computation-focused).
- **Neuron Polysemanticity**: A single neuron activation corresponding to multiple distinct
  features.
- **Monosemantic Neuron**: A neuron corresponding to a single feature.
- **Feature Importance Curve**: An ordering of features by decreasing importance, suggesting
  models always maintain some superposition.
- **Geometric Configurations**: Spatial arrangements of superposed features, including tegum
  products and antipodal pairs.
- **Antipodal Pairs**: A single direction representing one feature positively and another
  negatively.
- **Correlated Features**: Features that co-occur, tending toward orthogonal representation in
  superposition.
- **Anti-correlated Features**: Features that rarely co-occur, tending toward the same tegum
  product clusters.
- **Local Almost-Orthogonal Bases**: Subsets of pairwise-correlated features with approximately
  orthogonal representations despite broader interference.
- **Asymmetric Superposition Motif**: A pattern where one feature interferes more with another
  than vice versa, with cleanup neurons handling residual interference.

## The Broader Interpretability Field

- **Black-Box Interpretability**: Studying models as input-output mappings using differentiable
  properties like saliency maps.
- **White-Box / Inner Interpretability**: Techniques examining internal activations and weights to
  understand represented content and computation.
- **Explainability / XAI**: Explaining model behavior and output decisions, emphasizing user
  utility alongside computational truth.
- **BERTology**: Interpretability subfield specifically studying and understanding BERT models.

## Linear Algebra Fundamentals

- **Basis**: A set of n vectors serving as coordinate axes for an n-dimensional vector space where
  any vector uniquely expresses as a weighted combination.
- **Activation Space**: The vector space containing a model's internal activations, often useful to
  consider as geometric objects.
- **Residual Stream Space**: The d_model-dimensional vector space housing the residual stream,
  where each layer's input and output lives.
- **Orthonormal Basis**: Basis vectors that are mutually orthogonal and unit length.

## Circuits as Computational Subgraphs

- **Computational Graph**: Representation of a transformer where nodes are model components
  (attention heads, neurons) and edges represent information flow.
- **Computational Subgraph**: A subset of nodes and edges in the computational graph sufficient for
  the relevant computation.
- **Causal Scrubbing**: An algorithm testing whether computational subgraphs adequately explain
  model behavior by rewriting computations.

## Machine Learning Fundamentals

### Basic Concepts

- **Tensor**: Generalization of vectors; rank-n tensors are grids with n axes (vectors rank-1,
  images rank-2, image batches rank-3).
- **Activations**: Intermediate values computed when running a network through layers (excluding
  input/output).
- **Weights / Parameters**: Learned numbers determining network function, represented as tensors.
- **MLP / Multi-Layered Perceptron**: Classic architecture where each layer applies a linear
  mapping followed by a nonlinear activation.
- **MLP Layer (in transformers)**: A 2-layer MLP with a single internal activation function.
- **Hidden Layer**: Internal activations between input and output.
- **Bias**: Vectors added to linear-map outputs (both weights and biases are parameters).
- **Activation Function**: Nonlinearity applied post-linear layer, typically elementwise.
- **ReLU / Rectified Linear Unit**: x → max(x, 0).
- **GELU / Gaussian Error Linear Unit**: Smoother ReLU variant, standard in modern models.
- **Neuron**: Single element in an MLP hidden-layer activation tensor.
- **Softmax**: Maps an n-dimensional vector to a probability distribution via x_i → e^{x_i} / Σ_j
  e^{x_j}.
- **Logits**: Pre-softmax values.
- **Log_softmax**: Log of softmax; maps logits to log probabilities.
- **Log Probability Ratio**: Logarithmic representation of probability ratios, enabling Bayesian
  additive composition.
- **Loss Function**: Scores how close model output is to ground-truth labels.
- **Cross-Entropy Loss**: Standard classification loss using model logits and correct-label
  probabilities.
- **Classification Task**: Mapping inputs to one of a fixed finite set of output classes.
- **Regression Task**: Mapping inputs to real-valued outputs.
- **MSE Loss / Mean-Squared Error / Quadratic Loss**: Standard regression loss, squared difference
  between predictions and labels.

### Training Concepts

- **Training Distribution**: The data distribution on which models train.
- **In-Distribution Data**: Data resembling the training distribution (test/validation sets).
- **Out of Distribution / OOD**: Data from a different distribution than the training set.
- **SGD / Stochastic Gradient Descent**: Optimizer updating parameters by gradient times learning
  rate.
- **Batch**: Multiple inputs grouped for parallel efficiency and gradient-noise reduction.
- **Weight Decay / L2 Regularization**: Decreasing weights by a constant factor during updates,
  preferring simplicity.
- **Adam**: SGD variant tracking exponentially weighted moving averages of gradients and squared
  gradients.
- **EWMA / Exponentially Weighted Moving Average**: x_n = b·x_{n-1} + (1−b)·s_n.
- **AdamW**: Adam variant applying weight decay separately from gradient averaging.
- **Gaussian / Normal / Bell Curve**: Standard distributions characterized by mean and variance.
- **Standard Gaussian**: Mean 0, variance 1.
- **Memorization**: Learning the training set well without generalizing — effectively a lookup
  table.
- **Generalisation**: Performing well on both training and test distributions.
- **Overfitting**: Learning noise rather than underlying structure.
- **Phase Transition**: Sudden capability development during training over a brief period (S-shaped
  loss curves).
- **Training-Wise / Data-Wise / Model-Wise**: Phase transitions over training, dataset size, or
  model size respectively.
- **Emergent Phenomena**: Sudden capability jumps as models scale up.
- **Grokking**: Phase transition where models first memorize, then suddenly generalize.
- **Memorisation Circuit**: The learned algorithm for memorizing training data.
- **Circuit Formation**: Slow development of generalizing circuits separate from memorization
  circuits.
- **Clean-Up**: Regularization-driven removal of memorization noise once generalization suffices.
- **Bias-Variance Trade-Off**: Model complexity trades off bias (underfitting) and variance
  (overfitting).
- **Deep Double Descent**: Test loss decreases, then increases, then decreases again with
  increasing model size.
- **Path Dependence / Path Independence**: Whether the final model depends on the specific training
  trajectory, or is a function of the problem setup regardless of trajectory.

### Miscellaneous

- **Log Space / Linear Space**: Log space measures distance by ratio, linear space by difference.
- **Scaling Laws**: Smooth power-law relationships between model/data scale and performance over >7
  orders of magnitude.
- **Emergent Capabilities**: Skills appearing suddenly at certain scales despite smooth scaling
  laws.
- **Chinchilla**: DeepMind model showing compute-optimal training uses smaller models on vastly
  more data than previously believed.

## Transformers

### Transformer Basics

- **Transformer**: The architecture used in modern language models, mapping token sequences to
  next-token logits.
- **Sequence Modelling Model**: Models mapping input sequences to output predictions.
- **Token**: Discrete vocabulary element (roughly sub-words) representing discretized text.
- **Context**: The input sequence of tokens.
- **Transformer Block / Layer**: Contains both attention and MLP layers (terminology is conflated).
- **Embedding Layer**: Converts token integers to d_model-dimensional vectors via lookup.
- **Unembed**: Linear layer mapping the final residual stream to output logits (position ×
  vocabulary).
- **Output Logits**: Pre-softmax next-token scores over the vocabulary.
- **Residual Stream**: The central transformer object; sum of embedding and all prior layer
  outputs, and the input to the next layer.
- **Skip Connection**: Identity mapping enabling information to bypass layers.
- **Shared Bandwidth**: The residual stream as a memory bottleneck through which all information
  flows.
- **Attention Layer**: Moves information between token positions; parallel attention heads.
- **MLP Layer**: Non-linear processing acting independently at each position.
- **Causal / Masked Attention**: Attention restricting flow backward only (GPT-style).
- **Bidirectional Attention**: Attention without directional restriction (BERT-style).
- **Position in Sequence**: Index k where residual streams vary per position.
- **d_model**: Residual stream width (768 in GPT-2 Small); aka embedding_size / hidden_size.
- **d_mlp**: Internal MLP neuron count (3072 in GPT-2 Small), typically 4×d_model.
- **d_head**: Attention-head internal dimension (64 in GPT-2 Small).
- **n_heads**: Heads per layer (12 in GPT-2 Small), where n_heads × d_head = d_model.
- **n_layers**: Transformer layer count (12 in GPT-2 Small).
- **d_vocab**: Vocabulary size.
- **n_ctx**: Maximum context length.

### Tokenization

- **Tokenization**: Converting natural language to fixed-vocabulary discrete sequences via a
  deterministic algorithm.
- **Tokenizer**: The fixed-vocabulary converter implementing that algorithm.
- **De-tokenize**: Recovering input text from token sequences (approximately unique).
- **Byte-Pair Encoding / BPE**: Iterative algorithm identifying common token pairs and
  consolidating them.
- **BOS Token / Beginning of Sequence**: Marks sequence start; provides an attention resting
  position.
- **EOS Token / End of Sequence**: Separates concatenated texts.
- **PAD Token**: Padding token when sequences are padded to uniform length.

### Transformer Components

- **Lookup Table**: Mapping discrete token integers to learned vectors.
- **One-Hot Encoding**: Mapping integer k to a vector with 1 in position k, 0 elsewhere.
- **Positional Information / Embedding / Encoding**: Adding sequence-position information.
- **Learned Absolute Positional Embeddings**: A learned lookup table from positions to vectors
  (GPT-2 approach).
- **Absolute Positional Embeddings**: Position k carries explicit position information.
- **Shortformer Positional Embeddings**: Adding positional embeddings to query/key but not value or
  residual stream.
- **Sinusoidal Embeddings**: Fixed positional encodings using sine/cosine wave frequencies.
- **Rotary / RoPE**: Relative positional encoding rotating query/key by an amount set by position.
- **Relative Positional Encoding**: Position information from position differences rather than
  absolute values.
- **LayerNorm**: Sets residual stream mean 0, norm 1, with learned per-element scales and biases.
- **Centering / Normalising / Scaling / Translation**: Subtracting the mean / dividing by std /
  elementwise multiply by learned scale / adding a learned bias.
- **BatchNorm**: Normalization averaging statistics over the batch dimension.
- **Tied Embeddings / Unembedding**: Setting W_U = W_E^T, sharing embedding and unembedding weights.
- **Transformer Neuron**: A single internal MLP activation with input (W_in column) and output
  (W_out row) weights.

### Attention Heads

- **Attention Head**: Moves information between positions, choosing what/where independently per
  head.
- **Attention Weights**: Per-token-pair scalars determining how much information moves.
- **Attention Pattern**: The attention weights across all position pairs.
- **Source / Destination Token (Position)**: The earlier token providing information / the later
  token receiving it.
- **Query Vector**: "What information am I seeking?" (linear map from the residual stream).
- **Key Vector**: "What information do I offer?" (linear map from the residual stream).
- **W_Q / b_Q, W_K / b_K**: Query and key weight matrices and biases.
- **Attention Score / Logit**: Dot product of source key and destination query.
- **Row-Wise Softmax**: Softmax over a destination's attention scores across all sources.
- **Value Vector**: "The information I share" (linear map from the residual stream).
- **W_V / b_V**: Value weight matrix and bias.
- **Mixed Value**: Attention-pattern-weighted average of source value vectors.
- **Result Vector**: Linear output of the mixed value, added to the residual stream.
- **W_O / b_O**: Output weight mapping head space to the residual stream / shared bias across heads.
- **Information Routing**: Moving features between token positions via attention.
- **Previous Token Head**: Attention head attending to the previous token.
- **Induction Head**: Attention head detecting and continuing repeated subsequences.
