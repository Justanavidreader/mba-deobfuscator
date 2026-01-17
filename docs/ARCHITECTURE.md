# Model Architecture

Complete specification of the MBA Deobfuscator neural architecture.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Input Expression                          │
│                    "(x & y) + (x ^ y)"                          │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    AST Parser & Graph Builder                    │
│  • Parse expression to AST                                       │
│  • Convert to PyTorch Geometric graph                           │
│  • Add edge types (6-type legacy or 8-type optimized)          │
│  • Extract node features (operator types, variable IDs)         │
└────────────────────────┬────────────────────────────────────────┘
                         │
         ┌───────────────┴───────────────┐
         ▼                               ▼
┌────────────────────┐         ┌────────────────────────┐
│   GNN Encoder      │         │ Semantic Fingerprint   │
│                    │         │                        │
│ Choose one:        │         │ Components:            │
│ • GAT+JKNet        │         │ • Symbolic (32)        │
│ • GGNN             │         │ • Corner (256)         │
│ • HGT              │         │ • Random (64)          │
│ • RGCN             │         │ • Truth Table (64)     │
│ • Semantic HGT     │         │ ~~Derivative (32)~~    │
│ • Transformer-only │         │                        │
│ • Hybrid GREAT     │         │ Total: 416 dims (ML)   │
│ • GMN variants     │         │        448 dims (raw)  │
└─────────┬──────────┘         └──────────┬─────────────┘
          │                               │
          │ Node embeddings               │ Global vector
          │ [num_nodes × hidden_dim]      │ [416]
          │                               │
          └───────────────┬───────────────┘
                          ▼
                ┌─────────────────────┐
                │ Fingerprint Fusion  │
                │ • Linear projection │
                │ • Layer norm        │
                │ • Residual add      │
                └──────────┬──────────┘
                           │
                 Combined representation
                 [num_nodes × hidden_dim]
                           │
                           ▼
                ┌─────────────────────────┐
                │ Global Pooling (Mean)   │
                │ [num_nodes × D] → [D]   │
                └──────────┬──────────────┘
                           │
                  Encoder output [D]
                           │
                           ▼
                ┌─────────────────────────┐
                │  Transformer Decoder    │
                │  • 6 layers             │
                │  • 8 heads              │
                │  • 512d (base)          │
                │  • Causal self-attn     │
                │  • Cross-attention      │
                │  • Copy mechanism       │
                └──────────┬──────────────┘
                           │
                 Decoder hidden states
                 [seq_len × decoder_dim]
                           │
         ┌─────────────────┼─────────────────┐
         ▼                 ▼                 ▼
┌────────────────┐ ┌──────────────┐ ┌──────────────┐
│   Token Head   │ │Complexity    │ │  Value Head  │
│   (Vocab)      │ │Head          │ │  (Critic)    │
│                │ │              │ │              │
│ Output:        │ │ Output:      │ │ Output:      │
│ [V] logits     │ │ length/depth │ │ P(simplify)  │
└────────┬───────┘ └──────┬───────┘ └──────┬───────┘
         │                │                │
         └────────────────┼────────────────┘
                          ▼
                   Model Outputs
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Inference Pipeline                          │
│                                                                  │
│  Beam Search (depth < 10)        HTPS (depth ≥ 10)             │
│  • 50 beams                      • UCB search                   │
│  • 4 diversity groups            • 6 tactics                    │
│  • Grammar constraints           • Budget = 500                 │
│                                                                  │
│  ↓                               ↓                              │
│  Candidates [N × seq_len]                                       │
│                                                                  │
│  ↓                                                              │
│  3-Tier Verification:                                           │
│  1. Syntax (~10µs)    → Filter 60%                             │
│  2. Execution (~1ms)  → Filter 35%                             │
│  3. Z3 SMT (~100ms)   → Verify top-10                          │
│                                                                  │
│  ↓                                                              │
│  Reranking (tier, confidence, simplification ratio)            │
└────────────────────────┬────────────────────────────────────────┘
                         ▼
                 Simplified Expression
                    "x | y"
```

---

## Encoder Architectures

All encoders inherit from `BaseEncoder` and implement:
- `forward(data: Data) → Tensor`: Graph → node embeddings
- `get_output_dim() → int`: Output dimensionality

### 1. GAT+JKNet (Homogeneous)

**Parameters**: ~2.8M (256d), ~10M (768d)

**Architecture**:
```python
# 4 GAT layers with Jumping Knowledge
for layer in range(4):
    h = GATConv(h, edge_index, heads=8)
    h = LayerNorm(h)
    h = ReLU(h)
    h = Dropout(h, p=0.1)
    jk_states.append(h)

# Jumping Knowledge aggregation (max pooling over layers)
h = JumpingKnowledge(jk_states, mode='max')
```

**Characteristics**:
- No edge types (homogeneous graph)
- Multi-head attention (8 heads)
- Jumping Knowledge prevents over-smoothing
- Best for depth ≤ 10
- Fast training

**File**: `src/models/encoder.py:GATJKNetEncoder`

---

### 2. GGNN (Gated Graph Neural Network)

**Parameters**: ~3.2M (256d)

**Architecture**:
```python
# 8 timesteps of message passing
for t in range(8):
    for edge_type in edge_types:
        messages[edge_type] = Linear(h[sources])

    # Aggregate messages by edge type
    aggregated = sum([messages[et] for et in edge_types])

    # GRU update
    h = GRU(h, aggregated)
```

**Edge Types** (Legacy 6-type OR Optimized 8-type):
- Supports both edge type systems
- Edge-type-specific transformations
- Recurrent message passing (GRU)

**Characteristics**:
- Iterative refinement (8 timesteps)
- Best for depth 10+
- Handles relational structure well

**File**: `src/models/encoder.py:GGNNEncoder`

---

### 3. HGT (Heterogeneous Graph Transformer)

**Parameters**: ~60M (768d, 12 layers), ~120M (768d, 24 layers)

**Architecture**:
```python
# 12 HGT layers
for layer in range(12):
    # Heterogeneous attention with edge types
    attn = HGTConv(h, edge_index, edge_type,
                   num_heads=16, num_edge_types=8)

    # GCNII over-smoothing mitigation
    h_transformed = attn
    h = alpha * h_initial + (1 - alpha) * h_transformed  # Initial residual
    beta = log(lambda / (layer + 1) + 1)
    h = beta * Identity(h) + (1 - beta) * h  # Identity mapping

    # Optional: Path encoding
    if PATH_ENCODING_ENABLED:
        path_features = aggregate_paths(edge_index, h, max_length=6)
        h = h + path_features

    # Optional: Global attention (every 2 layers)
    if HGT_USE_GLOBAL_ATTENTION and layer % 2 == 0:
        h_global = SelfAttention(h)  # GraphGPS-style
        h = h + h_global

    h = LayerNorm(h)
```

**GCNII Parameters**:
- Alpha = 0.15 (initial residual strength)
- Lambda = 1.0 (identity decay rate)

**Edge Types** (Optimized 8-type, REQUIRED):
- LEFT_OPERAND, RIGHT_OPERAND, UNARY_OPERAND
- LEFT_OPERAND_INV, RIGHT_OPERAND_INV, UNARY_OPERAND_INV
- DOMAIN_BRIDGE_DOWN, DOMAIN_BRIDGE_UP

**Characteristics**:
- Heterogeneous (requires edge types)
- Very deep (12-24 layers without over-smoothing)
- Best for scaled model (360M params)
- Optional path encoding & global attention

**File**: `src/models/encoder.py:HGTEncoder`

---

### 4. RGCN (Relational GCN)

**Parameters**: ~60M (768d, 12 layers)

**Architecture**:
```python
# 12 RGCN layers with edge-specific transforms
for layer in range(12):
    # Edge-type-specific transformations
    for edge_type in range(8):
        h_et = Linear_et(h[sources[edge_type]])
        aggregated += h_et

    h_new = aggregated / sqrt(num_edges)

    # GCNII over-smoothing mitigation (same as HGT)
    h = alpha * h_initial + (1 - alpha) * h_new
    beta = log(lambda / (layer + 1) + 1)
    h = beta * Identity(h) + (1 - beta) * h

    h = LayerNorm(h)
    h = ReLU(h)
```

**Characteristics**:
- Relational (requires 8-type edge types)
- Alternative to HGT
- Edge-specific weight matrices
- GCNII over-smoothing mitigation

**File**: `src/models/encoder.py:RGCNEncoder`

---

### 5. Semantic HGT

**Parameters**: ~68M (768d, 12 layers)

**Architecture**:
```python
# Base: HGT with 12 layers (see above)
for layer in range(12):
    h = HGTConv(...)
    h = GCNII_residuals(...)

    # At layer 8: Inject Walsh-Hadamard spectrum
    if layer == 8:
        wh_spectrum = WalshHadamardTransform(truth_table)  # 64-dim
        wh_features = Linear(wh_spectrum)  # → hidden_dim
        h = h + wh_features.unsqueeze(0)  # Broadcast to all nodes

# Property-aware readout
node_embeddings = h
property_logits = PropertyDetector(global_pool(h))  # [13 properties]

# Enhanced global representation
global_repr = concat([
    mean_pool(h),
    max_pool(h),
    property_logits  # Inject property predictions
])
```

**Property Detection** (13 algebraic properties):
- Commutative, Associative, Distributive, Idempotent
- Identity, Absorbing, Involution, De Morgan
- XOR properties, AND/OR properties

**Walsh-Hadamard Transform**:
- Computes spectral representation of truth table
- Injected at layer 8 (middle of network)
- Captures Boolean function characteristics

**Characteristics**:
- HGT base + property detection + WHT spectrum
- Best for expressions with algebraic structure
- Property-aware training (auxiliary loss)

**File**: `src/models/semantic_hgt.py:SemanticHGTEncoder`

---

### 6. Transformer-Only (Sequence Baseline)

**Parameters**: ~12M (6 layers, 8 heads, 512d)

**Architecture**:
```python
# Tokenize expression to sequence
tokens = tokenizer.encode("(x & y) + (x ^ y)")  # [13, 15, 5, 16, ...]

# Sinusoidal positional encoding
pos_enc = sin_cos_encoding(len(tokens))
h = Embedding(tokens) + pos_enc

# 6 Transformer encoder layers
for layer in range(6):
    h = SelfAttention(h, heads=8)
    h = FFN(h, dim=512)
    h = LayerNorm(h)

# Global pooling
global_repr = mean(h, dim=0)
```

**Characteristics**:
- No graph structure (sequence only)
- Ablation baseline
- Faster than graph encoders
- May miss structural information

**File**: `src/models/encoder.py:TransformerOnlyEncoder`

---

### 7. Hybrid GREAT (Mixed Attention)

**Parameters**: ~25M

**Architecture**:
```python
# Combine graph attention + sequence attention
for layer in range(6):
    # Graph attention on AST structure
    h_graph = GATConv(h, edge_index)

    # Sequence attention on linearized expression
    h_seq = SelfAttention(h)

    # Combine via gating
    gate = Sigmoid(Linear(concat([h_graph, h_seq])))
    h = gate * h_graph + (1 - gate) * h_seq

    h = LayerNorm(h)
```

**Characteristics**:
- Hybrid architecture
- Learns to balance graph vs sequence structure
- Experimental

**File**: `src/models/encoder.py:HybridGREATEncoder`

---

### 8. GMN Variants (Graph Matching Network)

**HGT+GMN Parameters**: ~70M
**GAT+GMN Parameters**: ~15M

**Architecture**:
```python
# Base encoder (HGT or GAT)
h1 = base_encoder(graph1)
h2 = base_encoder(graph2)

# Graph Matching Network cross-attention
# Match corresponding nodes between graphs
attn_logits = (h1 @ h2.T) / sqrt(hidden_dim)  # [N1 × N2]
attn_weights = softmax(attn_logits, dim=-1)

# Cross-graph message passing
h1_matched = attn_weights @ h2  # [N1 × hidden_dim]
h2_matched = attn_weights.T @ h1  # [N2 × hidden_dim]

# Update with matched information
h1 = h1 + Linear(h1_matched)
h2 = h2 + Linear(h2_matched)

# Graph-level matching score
score = MLP(concat([global_pool(h1), global_pool(h2)]))  # Scalar
```

**Training**:
- **Phase 1b**: Frozen encoder, train GMN head only
- **Phase 1c**: End-to-end fine-tuning

**Use Case**: Learn to match equivalent expressions via graph similarity

**File**: `src/models/gmn/`

---

## Encoder Comparison

| Feature | GAT | GGNN | HGT | RGCN | Semantic HGT | Transformer | GREAT |
|---------|-----|------|-----|------|--------------|-------------|-------|
| **Params (256d)** | 2.8M | 3.2M | 60M | 60M | 68M | 12M | 25M |
| **Edge types** | ✗ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ |
| **Depth** | 4 | 8 | 12 | 12 | 12 | 6 | 6 |
| **Heterogeneous** | ✗ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ |
| **Message passing** | Attn | GRU | Attn | Agg | Attn | Self-attn | Mixed |
| **Over-smooth fix** | JK | GRU | GCNII | GCNII | GCNII | N/A | N/A |
| **Property aware** | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ |
| **WHT spectrum** | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ |
| **Best for** | Fast | Deep | Scaled | Alt | Algebra | Baseline | Hybrid |

---

## Semantic Fingerprint

### Full Fingerprint (448 dimensions)

```python
fingerprint = concat([
    symbolic_features,      # 32 dims
    corner_evaluations,     # 256 dims (4 widths × 64 cases)
    random_hash,            # 64 dims (4 widths × 16 inputs)
    derivatives,            # 32 dims (4 widths × 8 orders) - EXCLUDED FOR ML
    truth_table,            # 64 dims (2^6 for 6 vars)
])
# Total: 448 dims
```

### ML Fingerprint (416 dimensions)

**Derivatives excluded** due to C++/Python evaluation differences:
```python
ml_fingerprint = concat([
    fingerprint[:352],   # Symbolic + Corner + Random
    fingerprint[384:]    # Truth table
])
# Total: 416 dims
```

### Component Details

#### 1. Symbolic Features (32 dims)

Structural analysis of AST:
```python
[
    num_nodes,              # Total nodes in AST
    num_leaves,             # Leaf nodes (variables/constants)
    max_depth,              # Maximum depth
    avg_depth,              # Average node depth
    num_operators,          # Total operators
    num_variables,          # Unique variables
    num_constants,          # Unique constants
    # Operator counts (per type)
    count_ADD, count_SUB, count_MUL,
    count_AND, count_OR, count_XOR, count_NOT,
    # Node degree statistics
    max_out_degree, avg_out_degree,
    # Subtree sizes
    max_subtree_size, avg_subtree_size,
    # Additional structural features
    ...
]
```

#### 2. Corner Evaluations (256 dims)

Evaluate at extreme values across 4 bit widths:
```python
bit_widths = [8, 16, 32, 64]
corner_values = [0, 1, -1, MAX, MIN, HALF_MAX, ...]  # 64 values per width

for width in bit_widths:
    for values in corner_values:
        result = evaluate(expression, values, width)
        fingerprint.append(result)

# Total: 4 widths × 64 values = 256 dims
```

**Deterministic**: Same expression always produces same fingerprint

#### 3. Random Hash (64 dims)

Deterministic "random" evaluations:
```python
np.random.seed(42)  # Fixed seed
random_inputs = np.random.randint(0, 2**32, size=(16, 8))  # 16 samples, 8 vars

for width in [8, 16, 32, 64]:
    for sample in random_inputs:
        result = evaluate(expression, sample, width)
        fingerprint.append(result)

# Total: 4 widths × 16 samples = 64 dims
```

#### 4. Derivatives (32 dims) - **EXCLUDED**

**Why excluded**: C++ and Python use different evaluation methods for derivatives, causing inconsistencies.

Original computation (not used for ML):
```python
for width in [8, 16, 32, 64]:
    for order in range(8):
        derivative = compute_derivative(expression, order, width)
        fingerprint.append(derivative)  # NOT USED

# Total: 4 widths × 8 orders = 32 dims (stripped from ML fingerprint)
```

#### 5. Truth Table (64 dims)

Boolean function evaluation:
```python
# For up to 6 variables
num_vars = min(count_variables(expression), 6)

# Enumerate all 2^6 = 64 possible inputs
for input_bits in range(64):
    x0 = (input_bits >> 0) & 1
    x1 = (input_bits >> 1) & 1
    x2 = (input_bits >> 2) & 1
    x3 = (input_bits >> 3) & 1
    x4 = (input_bits >> 4) & 1
    x5 = (input_bits >> 5) & 1

    result = evaluate(expression, [x0, x1, x2, x3, x4, x5])
    truth_table.append(result & 1)  # Boolean output

# Total: 64 dims (covers all 6-variable Boolean functions)
```

**Properties**:
- Complete Boolean function signature
- Enables equivalence detection for Boolean expressions
- Sufficient for most MBA patterns

---

## Fingerprint Fusion

Combine semantic fingerprint with node embeddings:

```python
# Encoder output: [num_nodes × hidden_dim]
node_embeddings = encoder(graph)

# Semantic fingerprint: [416]
fingerprint = SemanticFingerprint().compute(expression)

# Project fingerprint to hidden_dim
fingerprint_projected = Linear(fingerprint)  # [416] → [hidden_dim]
fingerprint_projected = LayerNorm(fingerprint_projected)

# Broadcast and add to all nodes
node_embeddings = node_embeddings + fingerprint_projected.unsqueeze(0)

# Residual connection preserves original embeddings
```

**Purpose**: Inject global semantic information into local node embeddings

---

## Decoder Architecture

### Transformer Decoder with Copy Mechanism

**Base Configuration**:
- Layers: 6 (base) / 8 (scaled)
- Heads: 8 (base) / 24 (scaled)
- Dimension: 512 (base) / 1536 (scaled)
- FFN dim: 2048 (base) / 6144 (scaled)

```python
# Input: target sequence (teacher forcing during training)
target_tokens = tokenizer.encode(simplified_expression)

# Positional encoding
pos_enc = sinusoidal_encoding(len(target_tokens))
h_dec = Embedding(target_tokens) + pos_enc  # [seq_len × decoder_dim]

# Cross-attention encoder representation
encoder_output = global_pool(node_embeddings)  # [hidden_dim]
encoder_output = encoder_output.unsqueeze(0)  # [1 × hidden_dim]

# 6 Transformer decoder layers
for layer in range(6):
    # Causal self-attention (autoregressive)
    h_dec = CausalSelfAttention(h_dec, heads=8, mask=causal_mask)

    # Cross-attention to encoder
    h_dec = CrossAttention(
        query=h_dec,
        key=encoder_output,
        value=encoder_output,
        heads=8
    )

    # FFN
    h_dec = FFN(h_dec, dim=2048)
    h_dec = LayerNorm(h_dec)
    h_dec = Dropout(h_dec, p=0.1)

# Output: [seq_len × decoder_dim]
```

### Copy Mechanism (Pointer-Generator)

Prevents hallucinating non-existent variables:

```python
# Decoder hidden states: [seq_len × decoder_dim]
# Input tokens (source): [src_len]

# Generate probabilities
p_vocab = Softmax(TokenHead(h_dec))  # [seq_len × vocab_size]

# Copy probabilities from source
copy_scores = h_dec @ source_embeddings.T  # [seq_len × src_len]
p_copy = Softmax(copy_scores)

# Copy gate
copy_gate = Sigmoid(Linear(h_dec))  # [seq_len × 1]

# Final probabilities
p_final = (1 - copy_gate) * p_vocab + copy_gate * p_copy_extended
# where p_copy_extended scatters copy probs to vocab positions
```

**Example**:
- Input: `(x3 & x5) + (x3 ^ x5)`
- If decoder tries to generate `x7` (not in input), copy mechanism biases toward `x3` or `x5`

---

## Output Heads

### 1. Token Head (Vocabulary Prediction)

```python
token_logits = Linear(h_dec, out_features=VOCAB_SIZE)  # [seq_len × 300]
token_probs = Softmax(token_logits, dim=-1)
```

**Loss**: Cross-entropy
```python
loss_token = CrossEntropy(token_logits, target_tokens)
```

---

### 2. Complexity Head (Length/Depth Prediction)

Predict properties of simplified expression:

```python
# Global representation
global_repr = h_dec[-1]  # Last decoder state

# Predict length and depth
length_pred = Linear(global_repr)  # Scalar
depth_pred = Linear(global_repr)   # Scalar
```

**Loss**: MSE
```python
loss_complexity = MSE(length_pred, target_length) + MSE(depth_pred, target_depth)
```

**Purpose**:
- Reranking candidates (prefer simpler outputs)
- Early stopping signal
- Curriculum learning progression

---

### 3. Value Head (Critic for HTPS)

Estimate probability that expression is simplifiable:

```python
value = Sigmoid(Linear(global_repr))  # Scalar in [0, 1]
# value ≈ 1: likely simplifiable
# value ≈ 0: likely already simple
```

**Loss** (Phase 3 RL):
```python
# TD error (advantage-based)
advantage = reward + gamma * value_next - value
loss_value = MSE(value, target_value)
```

**Purpose**: Guide HTPS search toward promising subexpressions

---

## Model Dimensions

### Base Model (15M parameters)

```python
# Encoder: GAT+JKNet
encoder_hidden = 256
encoder_layers = 4
encoder_heads = 8
encoder_params = 2.8M

# Fingerprint encoder
fingerprint_dim = 416
fingerprint_encoder_params = 0.2M

# Decoder
decoder_dim = 512
decoder_layers = 6
decoder_heads = 8
decoder_ffn_dim = 2048
decoder_params = 12M

# Total: ~15M parameters
```

### Scaled Model (360M parameters)

```python
# Encoder: HGT
encoder_hidden = 768
encoder_layers = 12
encoder_heads = 16
encoder_params = 60M

# Fingerprint encoder
fingerprint_dim = 416
fingerprint_encoder_params = 0.2M

# Decoder
decoder_dim = 1536
decoder_layers = 8
decoder_heads = 24
decoder_ffn_dim = 6144
decoder_params = 300M

# Total: ~360M parameters
```

---

## Advanced Features

### 1. GCNII Over-Smoothing Mitigation

Prevents information loss in deep GNNs (HGT, RGCN, Semantic HGT):

```python
# Initial residual connection
h_0 = node_features  # Initial embeddings

for layer in range(12):
    # Standard message passing
    h_transformed = MessagePassing(h)

    # Initial residual (connects to layer 0)
    h = alpha * h_0 + (1 - alpha) * h_transformed

    # Identity mapping with decay
    beta_l = log(lambda_param / (layer + 1) + 1)
    h = beta_l * h + (1 - beta_l) * Identity(h)
```

**Hyperparameters**:
- `alpha = 0.15`: Initial residual strength
- `lambda_param = 1.0`: Identity decay rate

**Effect**: Maintains initial information even in very deep networks (12-24 layers)

**Reference**: Chen et al., "Simple and Deep Graph Convolutional Networks" (ICML 2020)

---

### 2. Operation-Aware Aggregation

Respects mathematical semantics of operators:

```python
def aggregate(node, operator_type):
    children = get_children(node)

    if operator_type in [ADD, MUL, AND, OR, XOR]:  # Commutative
        # Order-invariant (sum)
        return sum([h[child] for child in children])

    elif operator_type == SUB:  # Non-commutative
        # Preserve order (concatenate left/right)
        return Linear(concat([h[left_child], h[right_child]]))

    elif operator_type in [NOT, NEG]:  # Unary
        # Single operand
        return h[children[0]]
```

**Rationale**:
- Commutative ops: `x + y == y + x`, order doesn't matter
- Non-commutative: `x - y != y - x`, order matters
- Preserves algebraic structure in learned representations

---

### 3. Path-Based Edge Encoding (Optional)

Captures long-range dependencies via paths:

```python
# For each edge (u, v):
paths = find_paths(u, v, max_length=6)  # All paths ≤ 6 hops

# Aggregate path information
path_features = []
for path in paths:
    path_embedding = sum([h[node] for node in path])
    path_features.append(path_embedding)

# Add to edge features
edge_attr = mean(path_features)
```

**Use Case**: Subexpression sharing detection
- Two nodes may share distant common subexpressions
- Paths connect them through the AST

**Status**: Implemented but disabled by default (`PATH_ENCODING_ENABLED = False`)

---

### 4. Global Attention Blocks (Optional)

GraphGPS-style hybrid local+global attention:

```python
for layer in range(12):
    # Local message passing (HGT)
    h_local = HGTConv(h, edge_index, edge_type)

    # Global self-attention (every 2 layers)
    if layer % 2 == 0:
        h_global = MultiHeadSelfAttention(h, heads=16)
        h = h_local + h_global  # Combine
    else:
        h = h_local

    h = LayerNorm(h)
```

**Memory**: Uses gradient checkpointing for efficiency

**Status**: Implemented but disabled by default (`HGT_USE_GLOBAL_ATTENTION = False`)

**Reference**: Rampášek et al., "Recipe for a General, Powerful, Scalable Graph Transformer" (NeurIPS 2022)

---

## Memory & Compute

### Base Model (15M params)

- **Model size**: ~60 MB
- **Forward pass**: ~50ms (batch_size=32)
- **Training memory**: ~4 GB (batch_size=32)
- **Inference memory**: ~2 GB
- **GPU**: 1× GTX 1080 Ti sufficient

### Scaled Model (360M params)

- **Model size**: ~1.4 GB
- **Forward pass**: ~200ms (batch_size=16)
- **Training memory**: ~16 GB (batch_size=16)
- **Inference memory**: ~8 GB
- **GPU**: 1× RTX 3090 or 4× GTX 1080 Ti

### Fingerprint Computation

- **C++ accelerated**: 1-5ms per expression
- **Pure Python**: 10-50ms per expression
- **Memory**: Negligible (~1 KB per fingerprint)

---

## Implementation Files

| Component | File | Lines |
|-----------|------|-------|
| Encoders | `src/models/encoder.py` | 1097 |
| Base interface | `src/models/encoder_base.py` | 89 |
| Semantic HGT | `src/models/semantic_hgt.py` | 348 |
| Decoder | `src/models/decoder.py` | 188 |
| Full model | `src/models/full_model.py` | 267 |
| Output heads | `src/models/heads.py` | 123 |
| Fingerprint | `src/data/fingerprint.py` | 250 |
| Property detector | `src/models/property_detector.py` | 156 |
| Global attention | `src/models/global_attention.py` | 98 |
| Path encoding | `src/models/path_encoding.py` | 127 |
| GMN modules | `src/models/gmn/*.py` | 300+ |

---

## Configuration

Example model config (YAML):

```yaml
# Base model
model:
  encoder_type: gat_jknet
  hidden_dim: 256
  num_layers: 4
  num_heads: 8
  dropout: 0.1

  decoder_dim: 512
  decoder_layers: 6
  decoder_heads: 8
  decoder_ffn_dim: 2048

  fingerprint_dim: 416
  vocab_size: 300

# Scaled model
model:
  encoder_type: hgt
  hidden_dim: 768
  num_layers: 12
  num_heads: 16

  decoder_dim: 1536
  decoder_layers: 8
  decoder_heads: 24
  decoder_ffn_dim: 6144

  # Enable optional features
  path_encoding: true
  global_attention: true
  global_attention_interval: 2
```

---

## Ablation Studies

Compare encoders via `src/models/encoder_registry.py`:

```python
from src.models.encoder_registry import get_encoder, list_encoders

# List available encoders
encoders = list_encoders()
# ['gat_jknet', 'ggnn', 'hgt', 'rgcn', 'semantic_hgt',
#  'transformer_only', 'hybrid_great', 'hgt_gmn', 'gat_gmn']

# Create encoder
encoder = get_encoder('hgt', hidden_dim=768, num_layers=12)

# Run ablation study
from src.training.ablation_trainer import AblationTrainer
trainer = AblationTrainer(config)
results = trainer.compare_encoders(['gat_jknet', 'ggnn', 'hgt'])
```

See `docs/TRAINING.md` for ablation study details.

---

## References

1. **GAT**: Veličković et al., "Graph Attention Networks" (ICLR 2018)
2. **GGNN**: Li et al., "Gated Graph Sequence Neural Networks" (ICLR 2016)
3. **HGT**: Hu et al., "Heterogeneous Graph Transformer" (WWW 2020)
4. **RGCN**: Schlichtkrull et al., "Modeling Relational Data with Graph Convolutional Networks" (ESWC 2018)
5. **GCNII**: Chen et al., "Simple and Deep Graph Convolutional Networks" (ICML 2020)
6. **Jumping Knowledge**: Xu et al., "Representation Learning on Graphs with Jumping Knowledge Networks" (ICML 2018)
7. **Copy Mechanism**: See et al., "Get To The Point: Summarization with Pointer-Generator Networks" (ACL 2017)
8. **GraphGPS**: Rampášek et al., "Recipe for a General, Powerful, Scalable Graph Transformer" (NeurIPS 2022)
