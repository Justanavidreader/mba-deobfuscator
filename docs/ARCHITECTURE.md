# Model Architecture

Technical specification of the GNN-Transformer MBA deobfuscation model.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         INPUT PROCESSING                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  Obfuscated Expression: (x & y) + (x ^ y) + 2*(a & ~a)              │
│                              │                                        │
│                    ┌─────────┴─────────┐                            │
│                    │                   │                             │
│              ┌─────▼─────┐      ┌─────▼─────┐                       │
│              │ AST Graph │      │Fingerprint│                        │
│              │   Parser  │      │ Generator │                        │
│              └─────┬─────┘      └─────┬─────┘                       │
│                    │                   │                             │
│             [Graph Struct]      [448-dim vector]                     │
│                    │                   │                             │
└────────────────────┼───────────────────┼──────────────────────────────┘
                     │                   │
┌────────────────────┼───────────────────┼──────────────────────────────┐
│                    │   ENCODER         │                              │
├────────────────────┼───────────────────┼──────────────────────────────┤
│                    │                   │                              │
│              ┌─────▼─────┐      ┌─────▼────────┐                     │
│              │    GNN    │      │ Fingerprint  │                      │
│              │  Encoder  │      │   Encoder    │                      │
│              │ (GAT/GGNN)│      │   2-layer    │                      │
│              │    OR     │      │     MLP      │                      │
│              │ (HGT/RGCN)│      │              │                      │
│              └─────┬─────┘      └─────┬────────┘                     │
│                    │                   │                              │
│         [N, hidden_dim]         [B, hidden_dim]                      │
│                    │                   │                              │
│              ┌─────▼─────┐             │                              │
│              │   Graph   │             │                              │
│              │  Readout  │             │                              │
│              │(mean+max) │             │                              │
│              └─────┬─────┘             │                              │
│                    │                   │                              │
│              [B, hidden_dim]     [B, hidden_dim]                     │
│                    │                   │                              │
│              ┌─────▼─────┐       ┌────▼─────┐                        │
│              │Linear Proj│       │Linear Proj│                       │
│              │to d_model │       │to d_model │                       │
│              └─────┬─────┘       └────┬─────┘                        │
│                    │                   │                              │
│              [B, d_model]         [B, d_model]                       │
│                    │                   │                              │
│                    └─────────┬─────────┘                              │
│                              │                                        │
│                        ┌─────▼─────┐                                 │
│                        │   Fusion  │ ← (boolean_domain_only)         │
│                        │   Layer   │   [B] {0: mixed, 1: pure bool}  │
│                        │   concat  │                                  │
│                        └─────┬─────┘                                 │
│                              │                                        │
│                        [B, 1, d_model]                               │
│                              │                                        │
└──────────────────────────────┼───────────────────────────────────────┘
                               │
┌──────────────────────────────┼───────────────────────────────────────┐
│                              │   DECODER                              │
├──────────────────────────────┼───────────────────────────────────────┤
│                              │                                        │
│                        Memory: [B, 1, d_model]                       │
│                              │                                        │
│                       ┌──────▼──────┐                                │
│      Target Seq ─────>│Transformer  │                                │
│       [B, L]          │  Decoder    │                                │
│                       │ 6/8 layers  │                                │
│                       │             │                                │
│                       │ Self-Attn   │ ← causal mask                  │
│                       │ Cross-Attn  │ ← attends to Memory            │
│                       │ Feed-Fwd    │                                │
│                       └──────┬──────┘                                │
│                              │                                        │
│                        [B, L, d_model]                               │
│                              │                                        │
│                    ┌─────────┴─────────┐                             │
│                    │                   │                             │
│              ┌─────▼─────┐       ┌─────▼─────┐                       │
│              │ Decoder   │       │Cross-Attn │                       │
│              │  Output   │       │  Weights  │                       │
│              └─────┬─────┘       └─────┬─────┘                       │
│                    │                   │                             │
│              [B, L, d_model]     [B, L, src_len]                    │
│                    │                   │                             │
└────────────────────┼───────────────────┼──────────────────────────────┘
                     │                   │
┌────────────────────┼───────────────────┼──────────────────────────────┐
│                    │   OUTPUT HEADS    │                              │
├────────────────────┼───────────────────┼──────────────────────────────┤
│                    │                   │                              │
│              ┌─────▼─────┐       ┌─────▼─────┐                       │
│              │  Token    │       │  Copy     │                        │
│              │   Head    │       │   Gate    │                        │
│              │  Linear   │       │ 2-layer   │                        │
│              └─────┬─────┘       │    MLP    │                        │
│                    │             └─────┬─────┘                        │
│              [B, L, vocab]            │                               │
│                    │             [B, L, 1]                            │
│                    │                   │                              │
│                    └─────────┬─────────┘                              │
│                              │                                        │
│                         Final Logits                                  │
│                    p_gen * vocab_dist +                               │
│                    (1-p_gen) * copy_dist                              │
│                              │                                        │
│                        [B, L, vocab]                                 │
│                              │                                        │
│              ┌───────────────┼───────────────┐                       │
│              │               │               │                       │
│        ┌─────▼─────┐   ┌─────▼─────┐   ┌────▼────┐                  │
│        │Complexity │   │   Value   │   │ Greedy/ │                   │
│        │   Head    │   │   Head    │   │  Beam   │                   │
│        │(len/depth)│   │  (critic) │   │ Decode  │                   │
│        └─────┬─────┘   └─────┬─────┘   └────┬────┘                  │
│              │               │               │                       │
│         [B, len/depth]    [B, 1]    Simplified Expression            │
│                                                                       │
└───────────────────────────────────────────────────────────────────────┘
```

**Dimensions**:
- `B`: Batch size
- `N`: Total nodes in batch (variable)
- `L`: Target sequence length
- `hidden_dim`: 256 (standard) or 768 (scaled)
- `d_model`: 512 (standard) or 1536 (scaled)
- `src_len`: 1 (graph-level encoding)

**Data flow**:
1. Parse expression to AST graph + compute semantic fingerprint
2. Encode graph with GNN, encode fingerprint with MLP
3. Fuse embeddings and project to d_model
4. Decode autoregressively with Transformer (cross-attention to fused embedding)
5. Generate token via pointer-generator: mix vocabulary and copy distributions
6. Auxiliary heads predict complexity (reranking) and value (HTPS guidance)

---

## Component Specifications

### 1. GNN Encoder

The encoder converts AST graphs to node embeddings. Two variants for different complexity levels:

#### 1.1 GAT+JKNet (Default, Depth ≤10)

**Architecture**:
- 4 layers of Graph Attention Networks (GAT)
- 8 attention heads per layer (head_dim = hidden_dim / num_heads = 32)
- Jumping Knowledge (JK-Net): concatenate all layer outputs
- Residual connections + LayerNorm per layer
- ELU activation

**Dimensions**:
```python
hidden_dim = 256          # Node embedding dimension
num_layers = 4            # GAT layers
num_heads = 8             # Attention heads per layer
head_dim = 32             # hidden_dim / num_heads
dropout = 0.1
```

**Forward pass**:
```python
x = node_embedding(x)                      # [N, node_dim] -> [N, 256]
x = elu(x)

layer_outputs = []
for gat_layer, layer_norm in layers:
    x_in = x
    x = gat_layer(x, edge_index)          # Multi-head attention
    x = x_in + dropout(x)                 # Residual
    x = layer_norm(x)                     # Normalize
    x = elu(x)
    layer_outputs.append(x)               # Store for JK

jk_concat = cat(layer_outputs, dim=-1)   # [N, 1024]
x = jk_projection(jk_concat)              # [N, 1024] -> [N, 256]
```

**Parameter count**: ~2.8M
- Node embedding: 32 × 256 = 8,192
- 4 GAT layers: ~600K each = 2.4M
- JK projection: 1024 × 256 = 262,144

**Key properties**:
- Homogeneous (edge types ignored)
- Multi-head attention learns diverse relationships
- JK-Net captures both local (early layers) and global (late layers) structure
- Fast convergence for shallow expressions

**When to use**: Expressions with depth ≤10, fast training/inference required.

---

#### 1.2 GGNN (Depth 10+)

**Architecture**:
- Gated Graph Neural Network with explicit edge types
- 8 timesteps of recurrent message passing
- 8 edge types: LEFT/RIGHT/UNARY_OPERAND + inverses + DOMAIN_BRIDGE_DOWN/UP
- GRU-based state update

**Dimensions**:
```python
hidden_dim = 256          # Node state dimension
num_timesteps = 8         # Message passing rounds
num_edge_types = 8        # Heterogeneous edges
```

**Forward pass**:
```python
h = node_embedding(x)                     # [N, node_dim] -> [N, 256]
h = elu(h)

for t in range(num_timesteps):
    messages = zeros(N, hidden_dim)

    for edge_type in range(num_edge_types):
        # Select edges of this type
        mask = (edge_type_tensor == edge_type)
        edge_idx = edge_index[:, mask]
        src, dst = edge_idx[0], edge_idx[1]

        # Type-specific message function
        msg = message_mlps[edge_type](h[src])

        # Aggregate to destinations
        messages.index_add_(0, dst, msg)

    # GRU update
    h = gru(messages, h)                  # Gated update
```

**Parameter count**: ~3.2M
- Node embedding: 32 × 256 = 8,192
- 7 message MLPs: 7 × (256×256 + 256×256) = 917,504
- GRU cell: ~196,608
- Total: ~3.2M

**Key properties**:
- Heterogeneous (respects edge semantics)
- Recurrent propagation enables long-range dependencies
- Edge types distinguish LEFT/RIGHT operands and Boolean↔Arithmetic transitions
- Handles deeper expressions better than GAT

**When to use**: Expressions with depth >10, structured reasoning required.

---

#### 1.3 Graph Readout

Aggregates node embeddings to graph-level representation.

**Standard readout**:
```python
mean_pool = scatter_mean(x, batch)        # [B, 256] - average info
max_pool = scatter_max(x, batch)          # [B, 256] - salient features
cls_token = learnable_param               # [1, 256] - global context

aggregated = cat([mean_pool, max_pool, cls_token], dim=-1)  # [B, 768]
graph_embedding = projection(aggregated)  # [B, 768] -> [B, 256]
```

**Why three aggregations?**:
- Mean pooling: captures average structural properties
- Max pooling: captures most distinctive features (e.g., deepest nesting)
- CLS token: learnable global context (similar to BERT [CLS])

**Scaled readout** (360M model):
- Removes CLS token (redundant at scale)
- Only mean + max pooling: `[B, 768] -> [B, 768]`

---

### 2. Semantic Fingerprint

Fixed 448-dimensional semantic feature vector computed from the expression's behavior.

**Components** (see `src/data/fingerprint.py`):

| Component        | Dims | Description                                                      |
| ---------------- | ---- | ---------------------------------------------------------------- |
| Symbolic         | 32   | AST depth, node counts, operator frequencies, domain flags       |
| Corner evals     | 256  | Evaluate at corner cases (4 widths × 64 values)                  |
| Random hash      | 64   | Evaluate at random inputs (4 widths × 16 samples)                |
| Derivatives      | 32   | Finite difference gradients (4 widths × 8 variables)             |
| Truth table      | 64   | Boolean function for 6 vars (2^6 entries)                        |
| **Total**        | 448  |                                                                  |

**Fingerprint encoder**:
```python
fp_encoder = Sequential(
    Linear(448, hidden_dim),              # 448 -> 256/768
    LayerNorm(hidden_dim),
    ReLU(),
    Linear(hidden_dim, hidden_dim)        # 256 -> 256 / 768 -> 768
)
```

**Why fingerprint?**:
- Provides semantic constraints (graph structure alone is ambiguous)
- Truth table is CRITICAL: two structurally different expressions with same truth table are equivalent (Boolean)
- Corner evaluations catch arithmetic edge cases (overflow, sign bits)
- Independent of syntax (isomorphic to equivalence classes)

**Parameter count**: ~180K (standard), ~1.2M (scaled)

---

### 3. Encoder Fusion

Combines graph structure (GNN) and behavioral semantics (fingerprint).

**Standard model**:
```python
graph_projected = linear(graph_embedding)     # [B, 256] -> [B, 512]
fp_projected = linear(fp_embedding)           # [B, 256] -> [B, 512]

fused = cat([graph_projected, fp_projected], dim=-1)  # [B, 1024]
context = fusion_projection(fused)            # [B, 1024] -> [B, 512]
context = context.unsqueeze(1)                # [B, 1, 512] - add seq dim
```

**Scaled model** (adds Boolean domain conditioning):
```python
graph_projected = linear(graph_embedding)     # [B, 768] -> [B, 1536]
fp_projected = linear(fp_embedding)           # [B, 768] -> [B, 1536]
domain_embed = embedding(boolean_domain_only) # [B] -> [B, 1536]

fused = cat([graph_projected, fp_projected, domain_embed], dim=-1)  # [B, 4608]
context = fusion_projection(fused)            # [B, 4608] -> [B, 1536]
context = context.unsqueeze(1)                # [B, 1, 1536]
```

**Boolean domain flag**:
- `boolean_domain_only = 0`: Mixed Boolean-Arithmetic (MBA obfuscation)
- `boolean_domain_only = 1`: Pure Boolean expression (no arithmetic)
- Helps model specialize tactics (e.g., Boolean algebra laws vs. arithmetic identities)

**Why project then fuse?**:
- GNN outputs `hidden_dim` (256/768)
- Decoder expects `d_model` (512/1536)
- Projecting before fusion ensures dimensional consistency
- Fusion layer learns interaction between structure and semantics

**Parameter count**: ~1M (standard), ~11M (scaled)

---

### 4. Transformer Decoder

Autoregressive sequence decoder with copy mechanism.

**Standard configuration**:
```python
d_model = 512             # Model dimension
num_layers = 6            # Decoder layers
num_heads = 8             # Attention heads
d_ff = 2048               # Feed-forward inner dimension
dropout = 0.1
max_seq_len = 64          # Max output length
```

**Scaled configuration**:
```python
d_model = 1536
num_layers = 8
num_heads = 24
d_ff = 6144
max_seq_len = 2048        # For depth-14 expressions
```

**Decoder layer**:
```python
class TransformerDecoderLayer:
    def forward(x, memory, tgt_mask, memory_mask):
        # 1. Causal self-attention (target sequence)
        x_norm = layer_norm1(x)
        self_attn_out = self_attn(x_norm, x_norm, x_norm, mask=tgt_mask)
        x = x + dropout(self_attn_out)

        # 2. Cross-attention (encoder context)
        x_norm = layer_norm2(x)
        cross_attn_out, cross_attn_weights = cross_attn(x_norm, memory, memory)
        x = x + dropout(cross_attn_out)

        # 3. Feed-forward
        x_norm = layer_norm3(x)
        ff_out = feed_forward(x_norm)
        x = x + dropout(ff_out)

        return x, cross_attn_weights
```

**Full decoder**:
```python
def forward(tgt, memory, tgt_mask):
    # Token embedding + positional encoding
    x = token_embedding(tgt) * sqrt(d_model)  # [B, L] -> [B, L, 512]
    x = pos_encoding(x)                       # Add sinusoidal positions

    # Causal mask (prevents attending to future tokens)
    if tgt_mask is None:
        tgt_mask = causal_mask(L)             # [L, L] upper triangular

    # Stack of decoder layers
    for layer in layers:
        x, cross_attn_weights = layer(x, memory, tgt_mask)

    # Last layer's cross-attention used for copy mechanism
    return x, cross_attn_weights
```

**Parameter count**:
- Standard (6 layers × ~2M/layer): ~12M
- Scaled (8 layers × ~38M/layer): ~302M

**Key properties**:
- Causal self-attention: token at position i can only attend to positions ≤i
- Cross-attention to encoder: conditions generation on input expression
- Residual + LayerNorm: Pre-LN architecture for stable training
- Sinusoidal positional encoding: generalizes to longer sequences than seen during training

---

### 5. Copy Mechanism (Pointer-Generator)

Combines vocabulary generation with copying from input variables.

**Architecture**:
```python
# Decoder output and cross-attention
decoder_out = [B, L, d_model]             # From Transformer
cross_attn = [B, L, src_len]              # Attention weights to encoder

# Compute context vector from attention
context = bmm(cross_attn, memory)         # [B, L, d_model]

# Copy gate: decides vocab vs. copy
combined = cat([decoder_out, context], dim=-1)  # [B, L, 2*d_model]
p_gen = copy_gate(combined)               # [B, L, 1] in [0,1]
                                          # 2-layer MLP with sigmoid

# Vocabulary distribution
vocab_logits = token_head(decoder_out)    # [B, L, vocab_size]
vocab_dist = softmax(vocab_logits)

# Copy distribution (attention weights)
copy_dist = cross_attn                    # [B, L, src_len]

# Final distribution
final_dist = p_gen * vocab_dist + (1 - p_gen) * copy_dist
```

**Why copy mechanism?**:
- Variable names in output must match input (e.g., if input has `x`, `y`, output must use same)
- Vocabulary-only model can hallucinate variable names
- Copy mechanism guarantees variable preservation
- Especially important for expressions with >3 variables

**Training**:
- During training, if target token is a variable, encourage low `p_gen` (copy)
- If target token is operator/constant, encourage high `p_gen` (generate)
- Copy loss: `L_copy = BCE(p_gen, is_copyable_token)`

**Inference**:
- For each position, sample from `final_dist`
- If `p_gen > 0.5` and sampled token is operator, use from vocabulary
- If `p_gen < 0.5` or sampled token is variable, use from copy distribution

**Parameter count**: ~1.5M (standard), ~7M (scaled)
- Copy gate: 2 × d_model → d_model → 1

---

### 6. Output Heads

#### 6.1 Token Head

Linear projection to vocabulary.

```python
token_head = Linear(d_model, vocab_size)  # [B, L, 512] -> [B, L, 300]
```

**Parameter count**: 512 × 300 = 153,600 (standard), 1536 × 300 = 460,800 (scaled)

---

#### 6.2 Complexity Head

Predicts output sequence length and AST depth for reranking.

```python
class ComplexityHead:
    length_head = Linear(d_model, max_output_length)  # [B, 512] -> [B, 64]
    depth_head = Linear(d_model, max_output_depth)    # [B, 512] -> [B, 16]

    def forward(decoder_final):
        # Use final decoder token (or [CLS]-like aggregate)
        length_logits = length_head(decoder_final)
        depth_logits = depth_head(decoder_final)
        return length_logits, depth_logits
```

**Usage**:
- During beam search, generate K candidates
- Rerank by: `score = log_prob - λ_len * length_pred - λ_depth * depth_pred`
- Prioritizes shorter, simpler outputs (aligned with simplification goal)

**Training**:
- Cross-entropy loss on true output length/depth
- `L_complexity = CE(length_pred, true_length) + CE(depth_pred, true_depth)`

**Parameter count**: ~40K (standard), ~120K (scaled)

---

#### 6.3 Value Head (Critic for HTPS)

Estimates probability of successful simplification (used in RL phase and HTPS).

```python
class ValueHead:
    mlp = Sequential(
        Linear(d_model, d_model // 2),    # 512 -> 256
        ReLU(),
        Linear(d_model // 2, 1),          # 256 -> 1
        Sigmoid()                         # Output in [0, 1]
    )

    def forward(graph_embedding):
        return mlp(graph_embedding)       # [B, 512] -> [B, 1]
```

**Usage**:
- **Training (Phase 3)**: PPO value loss, predicts expected reward
- **Inference (HTPS)**: Guides tree search (higher value = more promising subexpression)

**Training**:
- MSE loss: `L_value = (value_pred - discounted_reward)^2`
- Updated during PPO rollouts

**Parameter count**: ~133K (standard), ~1.2M (scaled)

---

## Model Configurations

### Standard Model (~15M parameters)

Target: 10M training samples, depth ≤10 expressions.

| Component               | Configuration         | Parameters  |
| ----------------------- | --------------------- | ----------- |
| **Encoder**             | GAT+JKNet             |             |
| - Layers                | 4 GAT layers          |             |
| - Hidden dim            | 256                   |             |
| - Attention heads       | 8                     |             |
| - Parameters            |                       | ~2.8M       |
| **Graph Readout**       | mean + max + CLS      | ~200K       |
| **Fingerprint Encoder** | 2-layer MLP (448→256) | ~180K       |
| **Fusion**              | Project + concat      | ~1M         |
| **Decoder**             | Transformer           |             |
| - Layers                | 6                     |             |
| - d_model               | 512                   |             |
| - Heads                 | 8                     |             |
| - d_ff                  | 2048                  |             |
| - Max seq len           | 64                    |             |
| - Parameters            |                       | ~12M        |
| **Token Head**          | Linear 512→300        | ~154K       |
| **Complexity Head**     | 2 × Linear            | ~40K        |
| **Value Head**          | 2-layer MLP           | ~133K       |
| **Total**               |                       | **~16.5M**  |

**Training time**: ~3 days on 8×A100 (Phase 1+2+3)

---

### Scaled Model (~360M parameters)

Target: 12M samples (Chinchilla-optimal for 360M params), depth ≤14.

| Component               | Configuration         | Parameters  |
| ----------------------- | --------------------- | ----------- |
| **Encoder**             | HGT (heterogeneous)   |             |
| - Layers                | 12                    |             |
| - Hidden dim            | 768                   |             |
| - Heads                 | 16                    |             |
| - Edge types            | 7 (optimized)         |             |
| - Node types            | 10                    |             |
| - Parameters            |                       | ~60M        |
| **Graph Readout**       | mean + max (no CLS)   | ~1.2M       |
| **Fingerprint Encoder** | 2-layer MLP (448→768) | ~1.2M       |
| **Fusion**              | Project + concat + domain | ~11M    |
| **Decoder**             | Transformer           |             |
| - Layers                | 8                     |             |
| - d_model               | 1536                  |             |
| - Heads                 | 24                    |             |
| - d_ff                  | 6144                  |             |
| - Max seq len           | 2048                  |             |
| - Parameters            |                       | ~302M       |
| **Token Head**          | Linear 1536→300       | ~460K       |
| **Complexity Head**     | 2 × Linear            | ~120K       |
| **Value Head**          | 2-layer MLP           | ~1.2M       |
| **Total**               |                       | **~377M**   |

**Training time**: ~2 weeks on 64×A100 (estimated)

**Key differences**:
- HGT encoder respects heterogeneous edge types (structure-aware message passing)
- Boolean domain conditioning (pure Boolean vs. MBA expressions)
- Larger context window (2048 tokens for depth-14 expressions)
- Chinchilla-optimal: 360M params × 20 tokens/param = 7.2B tokens ≈ 12M samples

---

## Encoder Comparison

### When to Use Each Encoder

| Encoder      | Best For                     | Pros                                      | Cons                          | Params |
| ------------ | ---------------------------- | ----------------------------------------- | ----------------------------- | ------ |
| **GAT+JK**   | Depth ≤10, fast inference    | Fast, homogeneous (simple data)           | Ignores edge semantics        | 2.8M   |
| **GGNN**     | Depth 10+, structured reasoning | Edge types, recurrent propagation       | Slower than GAT               | 3.2M   |
| **HGT**      | Scaled model, depth ≤14      | Heterogeneous, type-specific attention    | Requires edge types, complex  | 60M    |
| **RGCN**     | Scaled model alternative     | Simpler than HGT, relational convolutions | Less expressive than HGT      | 60M    |

**Ablation study results** (see `docs/ML_WORKFLOW.md`):
- GAT+JK: Best for depth ≤6 (95% accuracy)
- GGNN: Best for depth 10-14 (75% accuracy, 10% better than GAT)
- HGT: Best for depth 14 (82% accuracy on scaled data)

### Edge Type Semantics (GGNN/HGT/RGCN)

Defined in `src/models/edge_types.py`:

| Edge Type            | ID  | Direction         | Semantics                              |
| -------------------- | --- | ----------------- | -------------------------------------- |
| `LEFT_OPERAND`       | 0   | operator → left   | Parent to left child                   |
| `RIGHT_OPERAND`      | 1   | operator → right  | Parent to right child                  |
| `UNARY_OPERAND`      | 2   | operator → child  | Unary operator (NEG, NOT) to operand   |
| `LEFT_OPERAND_INV`   | 3   | left → operator   | Left child to parent (inverse)         |
| `RIGHT_OPERAND_INV`  | 4   | right → operator  | Right child to parent (inverse)        |
| `UNARY_OPERAND_INV`  | 5   | child → operator  | Child to unary parent (inverse)        |
| `DOMAIN_BRIDGE_DOWN` | 6   | bool → arith      | Domain transition (parent to child)    |
| `DOMAIN_BRIDGE_UP`   | 7   | arith → bool      | Domain transition (child to parent)    |

**Why bidirectional edges?**:
- Information flows both top-down (parent to children) and bottom-up (children to parent)
- Critical for GNNs: enables O(layers) propagation instead of O(depth) hops
- Example: In `(x & y) + z`, to understand `+` requires knowing `&` (its child), which requires knowing `x` and `y` (its children). Inverse edges enable 2-hop propagation: `x→&→+` instead of 3 hops.

**DOMAIN_BRIDGE edges**:
- Connect Boolean operators to Arithmetic contexts (and vice versa)
- Example: `(x & y) + (x ^ y)` — `&` and `^` are Boolean, `+` is Arithmetic
- `DOMAIN_BRIDGE_DOWN`: Added when Boolean parent has Arithmetic child (e.g., `& → +`)
- `DOMAIN_BRIDGE_UP`: Added when Arithmetic child has Boolean parent (e.g., `+ → &`)
- Having separate UP/DOWN types lets the model learn different transformations for each direction
- Helps model learn MBA transformation rules (Boolean-Arithmetic mixing)

---

## Hyperparameter Effects

### Critical Hyperparameters

| Hyperparameter      | Standard | Scaled | Effect                                           |
| ------------------- | -------- | ------ | ------------------------------------------------ |
| `hidden_dim`        | 256      | 768    | GNN capacity; higher = more structural info      |
| `d_model`           | 512      | 1536   | Decoder capacity; higher = better long sequences |
| `num_encoder_layers`| 4 (GAT)  | 12 (HGT)| Depth of graph propagation                      |
| `num_decoder_layers`| 6        | 8      | Sequence modeling capacity                       |
| `num_heads`         | 8        | 16/24  | Attention diversity                              |
| `d_ff`              | 2048     | 6144   | Feed-forward inner dimension                     |
| `dropout`           | 0.1      | 0.1    | Regularization (higher = less overfitting)       |
| `max_seq_len`       | 64       | 2048   | Max output length (must exceed dataset max)      |

### Tuning Guidelines

**If model underfits** (training loss high):
- Increase `hidden_dim` or `d_model` (more capacity)
- Increase `num_layers` (deeper reasoning)
- Decrease `dropout` (less regularization)

**If model overfits** (train acc >> val acc):
- Increase `dropout`
- Add data augmentation (random subexpression masking)
- Use curriculum learning (start with easy examples)

**If inference is slow**:
- Reduce `num_encoder_layers` (GAT faster than GGNN)
- Use beam search with smaller `beam_width` (50 → 20)
- Use execution pre-filter (see `docs/ML_PIPELINE.md`)

**If outputs are syntactically invalid**:
- Add grammar-constrained decoding (force valid operator-operand structure)
- Increase `complexity_head` weight (penalize invalid lengths)

**If outputs are semantically incorrect**:
- Increase `FINGERPRINT_DIM` contribution (more semantic constraints)
- Use truth table in Phase 1 contrastive loss (align structurally different but equivalent expressions)
- Increase Z3 verification budget (`Z3_TOP_K` in constants)

---

## Forward Pass Example

Input: `(x & y) + (x ^ y)`

### 1. Graph Construction

```
AST:
       +
      / \
     &   ^
    / \ / \
   x  y x  y

Nodes: [+, &, ^, x, y, x, y]  (7 nodes)
Node types: [ADD, AND, XOR, VAR, VAR, VAR, VAR]
Edge index: [[0,0,1,1,2,2],   # src
             [1,2,3,4,5,6]]   # dst
Edge types: [LEFT, RIGHT, LEFT, RIGHT, LEFT, RIGHT]
```

### 2. Fingerprint

```python
fingerprint = compute_fingerprint("(x&y)+(x^y)")
# [448] vector:
# - Symbolic: depth=3, num_ops=3, has_bool=True, ...
# - Corner: eval at (0,0), (0,1), (1,0), (1,1), ...
# - Random: eval at 64 random (x,y) pairs
# - Truth table: [0,1,1,3] (matches x|y)
# - ...
```

### 3. Encoder

```python
# GAT+JKNet
x = node_embedding(node_types)            # [7, 32] -> [7, 256]
layer_outs = []
for layer in gat_layers:
    x = layer(x, edge_index) + x          # Attention + residual
    x = layer_norm(x)
    layer_outs.append(x)
x = jk_projection(cat(layer_outs))        # [7, 1024] -> [7, 256]

# Graph readout
graph_emb = readout(x, batch)             # [7, 256] -> [1, 256]

# Fingerprint
fp_emb = fp_encoder(fingerprint)          # [1, 448] -> [1, 256]

# Fusion
graph_proj = linear(graph_emb)            # [1, 256] -> [1, 512]
fp_proj = linear(fp_emb)                  # [1, 256] -> [1, 512]
fused = cat([graph_proj, fp_proj])        # [1, 1024]
context = fusion_projection(fused)        # [1, 1024] -> [1, 512]
memory = context.unsqueeze(1)             # [1, 1, 512]
```

### 4. Decoder (Greedy)

```python
# Initial: <sos>
tgt = [<sos>]
for t in range(max_seq_len):
    tgt_tensor = tensor(tgt).unsqueeze(0)  # [1, t+1]
    decoder_out, cross_attn, p_gen = decoder(tgt_tensor, memory)

    # Last position
    logits = token_head(decoder_out[:, -1, :])  # [1, vocab]
    next_token = argmax(logits)

    if next_token == <eos>:
        break
    tgt.append(next_token)

# Output: ['<sos>', 'x', '|', 'y', '<eos>']
# Detokenize: "x | y"
```

### 5. Verification

```python
# Execution check
for (x_val, y_val) in random_samples:
    input_result = eval("(x&y)+(x^y)", x=x_val, y=y_val)
    output_result = eval("x|y", x=x_val, y=y_val)
    assert input_result == output_result  # Pass

# Z3 (if execution passes)
solver.add(input_expr != output_expr)
result = solver.check()
assert result == unsat  # Expressions are equivalent
```

---

## Implementation Files

| File                      | Purpose                                               |
| ------------------------- | ----------------------------------------------------- |
| `src/models/encoder_base.py` | Abstract base class for all encoders                  |
| `src/models/encoder.py`   | GAT+JKNet, GGNN, HGT, RGCN implementations            |
| `src/models/decoder.py`   | Transformer decoder with copy mechanism               |
| `src/models/full_model.py`| MBADeobfuscator (encoder + decoder + heads)           |
| `src/models/heads.py`     | TokenHead, ComplexityHead, ValueHead                  |
| `src/models/positional.py`| Sinusoidal and learned positional encodings           |
| `src/models/edge_types.py`| Edge type and node type definitions                   |
| `src/constants.py`        | All hyperparameters and dimensions                    |

---

## Extension Points

### Adding a New Encoder

1. Inherit from `BaseEncoder` in `encoder_base.py`
2. Implement required properties:
   - `requires_edge_types` (bool)
   - `requires_node_features` (bool)
3. Implement `_forward_impl(x, edge_index, batch, edge_type)`
4. Add to `full_model.py` encoder selection

Example skeleton:

```python
class MyEncoder(BaseEncoder):
    def __init__(self, hidden_dim=256, **kwargs):
        super().__init__(hidden_dim=hidden_dim)
        # Initialize layers

    @property
    def requires_edge_types(self) -> bool:
        return False  # or True if heterogeneous

    @property
    def requires_node_features(self) -> bool:
        return True  # [N, node_dim] or False for [N] type IDs

    def _forward_impl(self, x, edge_index, batch, edge_type):
        # Encode graph
        return node_embeddings  # [N, hidden_dim]
```

### Adding a New Output Head

1. Create head class in `heads.py`
2. Initialize in `MBADeobfuscator.__init__`
3. Call in `forward()` method
4. Add loss term in training loop

Example:

```python
class OperatorCountHead(nn.Module):
    """Predicts number of operators in output."""
    def __init__(self, d_model=512, max_ops=20):
        super().__init__()
        self.proj = nn.Linear(d_model, max_ops)

    def forward(self, x):
        return self.proj(x)  # [B, max_ops]
```

### Modifying Attention Mechanism

Decoder uses `nn.MultiheadAttention`. To customize:

1. Implement custom attention in `decoder.py`
2. Replace `self.self_attn` and `self.cross_attn` in `TransformerDecoderLayer`
3. Ensure output format matches: `(output, attention_weights)`

---

## Performance Considerations

**Memory usage** (standard model, batch_size=32):
- Encoder: ~1.5 GB (graph batching + attention)
- Decoder: ~2 GB (sequence length × d_model × num_layers)
- Peak: ~4 GB during backward pass

**Inference speed** (single A100):
- GAT encoder: ~5ms/sample (depth 6)
- GGNN encoder: ~15ms/sample (depth 10)
- Decoder (greedy): ~10ms/sample (length 20)
- Decoder (beam=50): ~200ms/sample
- Total (greedy): ~15-25ms/sample
- Total (beam): ~200-250ms/sample

**Scaling recommendations**:
- For batch inference: Use DataLoader with `batch_size=64-128`
- For low latency: Use greedy decoding + execution pre-filter
- For high accuracy: Use beam search (width=50) + Z3 verification on top-10
- For very deep expressions (depth >10): Use GGNN or HTPS instead of beam search

---

## References

- GAT: Veličković et al., "Graph Attention Networks" (ICLR 2018)
- JK-Net: Xu et al., "Representation Learning on Graphs with Jumping Knowledge Networks" (ICML 2018)
- GGNN: Li et al., "Gated Graph Sequence Neural Networks" (ICLR 2016)
- HGT: Hu et al., "Heterogeneous Graph Transformer" (WWW 2020)
- Pointer-Generator: See et al., "Get To The Point" (ACL 2017)
- Transformer: Vaswani et al., "Attention Is All You Need" (NeurIPS 2017)
