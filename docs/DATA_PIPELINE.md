# Data Pipeline Architecture

> **Purpose**: Convert raw MBA expression pairs into batched graph+fingerprint tensors for GNN+Transformer training.

**Pipeline Flow**: JSONL → Dataset → Tokenize + Parse + Fingerprint → Collate → Batched Tensors

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│ Input: JSONL File                                                   │
│ {"obfuscated": "(x&y)+(x^y)", "simplified": "x|y", "depth": 3}     │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Dataset.__getitem__(idx)                                            │
│   ├─ Load JSON line                                                 │
│   ├─ Apply depth filter (curriculum learning)                       │
│   └─ Split into 3 parallel paths:                                   │
└─┬───────────────────────────┬─────────────────────────┬─────────────┘
  │                           │                         │
  ▼                           ▼                         ▼
┌──────────────────┐  ┌───────────────────┐  ┌────────────────────┐
│ AST Parser       │  │ Tokenizer         │  │ Fingerprint        │
│                  │  │                   │  │                    │
│ expr_to_graph()  │  │ encode()          │  │ compute()          │
│                  │  │                   │  │                    │
│ "x&y" →          │  │ "x|y" →           │  │ "x&y" →            │
│   Graph:         │  │   [1,14,6,15,2]   │  │   [448 floats]     │
│   x [HIDDEN]     │  │   <sos>,x,|,y,<eos>│  │   (5 components)   │
│   ╱ ╲            │  │                   │  │                    │
│  & VAR:y         │  │ get_source_tokens │  │ ├─ Symbolic (32)   │
│  VAR:x           │  │   [14,5,15]       │  │ ├─ Corner (256)    │
│                  │  │   x,&,y           │  │ ├─ Random (64)     │
│ Returns:         │  │                   │  │ ├─ Derivative (32) │
│   Data(          │  │ Returns:          │  │ └─ Truth (64)      │
│     x: [N,32],   │  │   target_ids,     │  │                    │
│     edge_index,  │  │   source_tokens   │  │ Returns:           │
│     edge_type)   │  │                   │  │   np.array[448]    │
└────────┬─────────┘  └─────────┬─────────┘  └─────────┬──────────┘
         │                      │                       │
         └──────────────┬───────┴───────────────────────┘
                        ▼
         ┌─────────────────────────────────────────┐
         │ Dataset.__getitem__ Returns:            │
         │  {                                      │
         │    'graph': Data(...),                  │
         │    'fingerprint': Tensor[448],          │
         │    'target_ids': Tensor[seq_len],       │
         │    'source_tokens': Tensor[src_len],    │
         │    'depth': int,                        │
         │    'obfuscated': str,                   │
         │    'simplified': str                    │
         │  }                                      │
         └──────────────────┬──────────────────────┘
                            │
                (Multiple items batched by DataLoader)
                            │
                            ▼
         ┌─────────────────────────────────────────┐
         │ collate_graphs([item1, item2, ...])     │
         │   ├─ Batch.from_data_list(graphs)       │
         │   ├─ torch.stack(fingerprints)          │
         │   ├─ pad_sequence(target_ids, pad=0)    │
         │   └─ pad_sequence(source_tokens, pad=0) │
         └──────────────────┬──────────────────────┘
                            ▼
         ┌─────────────────────────────────────────┐
         │ Batched Output:                         │
         │  {                                      │
         │    'graph_batch': Data(                 │
         │      x: [total_nodes, 32],              │
         │      edge_index: [2, total_edges],      │
         │      batch: [total_nodes]               │
         │    ),                                   │
         │    'fingerprint': [B, 448],             │
         │    'target_ids': [B, max_len],          │
         │    'target_lengths': [B],               │
         │    'source_tokens': [B, max_src],       │
         │    'source_lengths': [B],               │
         │    'depth': [B]                         │
         │  }                                      │
         └─────────────────────────────────────────┘
                            │
                            ▼
                   ┌──────────────┐
                   │ Model.forward│
                   └──────────────┘
```

---

## 1. Expression Tokenization

**File**: `src/data/tokenizer.py`

### Vocabulary Structure

```
Token ID Range  | Category        | Examples
----------------|-----------------|---------------------------
0-4             | Special         | <pad>, <sos>, <eos>, <unk>, <sep>
5-11            | Operators       | +, -, *, &, |, ^, ~
12-13           | Parentheses     | (, )
14-21           | Variables       | x0-x7 (also x,y,z,a,b,c,d,e)
22-277          | Constants       | 0-255
278-299         | Reserved        | Future expansion
```

**Vocab Size**: 300 (padded for safety)

### Tokenization Process

```python
# Input:  "(x & y) + 1"
# Step 1: Normalize whitespace → "(x & y) + 1"
# Step 2: Add spacing around ops → " ( x & y ) + 1 "
# Step 3: Split and filter → ['(', 'x', '&', 'y', ')', '+', '1']
# Step 4: Map to IDs → [12, 14, 5, 15, 13, 5, 22]
# Step 5: Add special → [1, 12, 14, 5, 15, 13, 5, 22, 2]  # <sos>, ..., <eos>
```

### Key Methods

| Method              | Purpose                                   | Special Tokens |
|---------------------|-------------------------------------------|----------------|
| `tokenize(expr)`    | String → token list                       | No             |
| `encode(expr)`      | String → token IDs                        | Yes (default)  |
| `decode(ids)`       | Token IDs → string                        | Stripped       |
| `get_source_tokens` | For copy mechanism (no SOS/EOS)           | No             |

**Unknown Token Handling**: Out-of-range constants (>255) and variables (x9+) map to `<unk>` (ID 3).

---

## 2. AST Representation and Graph Construction

**File**: `src/data/ast_parser.py`

### AST Node Types

```
Node Type  | ID | Description           | Arity
-----------|----|-----------------------|-------
VAR        | 0  | Variable (x, y, ...)  | Leaf
CONST      | 1  | Constant (0-255)      | Leaf
ADD        | 2  | Addition (+)          | Binary
SUB        | 3  | Subtraction (-)       | Binary
MUL        | 4  | Multiplication (*)    | Binary
AND        | 5  | Bitwise AND (&)       | Binary
OR         | 6  | Bitwise OR (|)        | Binary
XOR        | 7  | Bitwise XOR (^)       | Binary
NOT        | 8  | Bitwise NOT (~)       | Unary
NEG        | 9  | Negation (-)          | Unary
```

### Operator Precedence (High → Low)

1. Variables, constants, parentheses
2. `~` (NOT), unary `-` (NEG)
3. `*` (MUL)
4. `+`, `-` (ADD, SUB)
5. `&` (AND)
6. `^` (XOR)
7. `|` (OR)

### Parsing Algorithm

**Recursive Descent Parser** with precedence climbing:

```
parse_expr → parse_or
parse_or → parse_xor ('|' parse_xor)*
parse_xor → parse_and ('^' parse_and)*
parse_and → parse_additive ('&' parse_additive)*
parse_additive → parse_multiplicative (('+' | '-') parse_multiplicative)*
parse_multiplicative → parse_unary ('*' parse_unary)*
parse_unary → ('~' | '-')? parse_primary
parse_primary → '(' parse_expr ')' | VAR | CONST
```

### Graph Construction: Two Edge Systems

#### Legacy System (6 types)

Used by `ast_to_graph()` for base model:

```
Edge Type     | ID | Direction        | Example
--------------|----|--------------------|------------------
CHILD_LEFT    | 0  | parent → left      | + → x (in x+y)
CHILD_RIGHT   | 1  | parent → right     | + → y (in x+y)
PARENT        | 2  | child → parent     | x → + (in x+y)
SIBLING_NEXT  | 3  | left → right       | x → y (in x+y)
SIBLING_PREV  | 4  | right → left       | y → x (in x+y)
SAME_VAR      | 5  | var_use ↔ var_use  | x[node1] ↔ x[node2]
```

**Node Features**: `[num_nodes, 32]`
- One-hot node type (10 dims)
- Positional encoding: depth/20 (1 dim)
- Variable index (8 dims for x0-x7)
- Constant value normalized to [-1,1] (1 dim)
- Padding (12 dims)

#### Optimized System (7 types)

Used by `ast_to_optimized_graph()` for scaled model with subexpression sharing:

```
Edge Type          | ID | Direction      | Semantic
-------------------|----|-----------------|-----------------------
LEFT_OPERAND       | 0  | parent → left   | Forward dataflow
RIGHT_OPERAND      | 1  | parent → right  | Forward dataflow
UNARY_OPERAND      | 2  | parent → child  | Forward dataflow
PARENT_OF_LEFT     | 3  | left → parent   | Inverse (for GNN symmetry)
PARENT_OF_RIGHT    | 4  | right → parent  | Inverse (for GNN symmetry)
PARENT_OF_UNARY    | 5  | child → parent  | Inverse (for GNN symmetry)
DOMAIN_BRIDGE      | 6  | bool → arith    | Cross-domain edge (e.g., & → +)
```

**Node Features**: `[num_nodes]` (just type IDs for heterogeneous GNN)

**Domain Bridge Edges**: Added when boolean operator has arithmetic child (e.g., `(x+y) & 1` adds bridge from `&` to `+`).

**Subexpression Sharing**: Implemented in `ScaledMBADataset._build_optimized_graph()` via subtree hashing (MD5). Identical subtrees merge to single node with shared incoming edges.

### Example: AST for `(x & y) + (x ^ y)`

```
       +
      / \
     &   ^
    / \ / \
   x  y x  y
```

**Legacy Graph**:
- Nodes: 7 (root +, &, ^, x1, y1, x2, y2)
- Edges: 14 (6 CHILD, 6 PARENT, 2 SIBLING) + 4 SAME_VAR (x1↔x2, y1↔y2)

**Optimized Graph with Subexpression Sharing**:
- Nodes: 5 (root +, &, ^, x_shared, y_shared)
- Edges: 10 (LEFT_OPERAND/RIGHT_OPERAND for each operator + inverses)
- SAME_VAR edges eliminated (redundant with sharing)

---

## 3. Semantic Fingerprint (448 Floats)

**File**: `src/data/fingerprint.py`

### Component Breakdown

```
Component       | Dims | Offset | Purpose
----------------|------|--------|------------------------------------------
Symbolic        | 32   | 0      | Structural features (op counts, depth)
Corner Evals    | 256  | 32     | Outputs at corner cases (4 widths × 64)
Random Hash     | 64   | 288    | Outputs at random inputs (4 widths × 16)
Derivatives     | 32   | 352    | Partial derivatives (4 widths × 8 vars)
Truth Table     | 64   | 384    | LSB of outputs for 2^6 boolean inputs
----------------|------|--------|------------------------------------------
Total           | 448  |        |
```

### 3.1 Symbolic Features (32 dims)

**Structural features extracted via pattern matching**:

```python
features = [
    len(expr) / 200.0,                    # [0] Normalized length
    count('+') / 10.0,                    # [1] Addition count
    count('-') / 10.0,                    # [2] Subtraction count
    count('*') / 10.0,                    # [3] Multiplication count
    count('&') / 10.0,                    # [4] AND count
    count('|') / 10.0,                    # [5] OR count
    count('^') / 10.0,                    # [6] XOR count
    count('~') / 10.0,                    # [7] NOT count
    num_variables / 8.0,                  # [8] Variable count
    num_constants / 10.0,                 # [9] Constant count
    max_paren_depth / 10.0,               # [10] Nesting level
    (total_ops + max_paren_depth) / 20.0, # [11] Total complexity
    # [12-19] Usage counts for x0-x7
    count('x0')/5.0, ..., count('x7')/5.0,
    # [20-23] Constant statistics
    mean(constants)/256.0,
    std(constants)/256.0,
    min(constants)/256.0,
    max(constants)/256.0,
    # [24-31] Reserved/padding
]
```

**Why Symbolic Features?**: Provides coarse structural signal for encoder initialization before learning semantic equivalence.

### 3.2 Corner Evaluations (256 dims)

**Evaluation at edge cases across 4 bit widths**: 8, 16, 32, 64

```python
# For each width (e.g., 8-bit):
corner_values = [
    0, 1, 2, 3,                    # Small values
    255, 254, 253,                 # Near max (for 8-bit)
    128, 127, 129,                 # Around midpoint
    0xAA, 0x55,                    # Alternating bits
    *[2**i for i in range(8)],     # Powers of 2
    *[(2**i)-1 for i in range(8)], # Powers of 2 minus 1
]

# Generate 64 variable assignments per width using corners
# Total: 4 widths × 64 assignments = 256 evaluations
```

**Assignment Strategy**: Cycle through corner values for each variable using `(i + j*7) % len(corners)` pattern for diverse combinations.

**Normalization**: `output / max_value` for each width (e.g., `/255` for 8-bit).

**Why Corner Cases?**: MBA expressions often differ at boundary conditions (0, max, powers of 2). Corner evals distinguish `x & y` from `x | y`.

### 3.3 Random Hash (64 dims)

**Pseudorandom evaluation for collision resistance**:

```python
# Seeded RNG (default seed=42) generates 16 random inputs per width
# For each width: evaluate expression on 16 random variable assignments
# Total: 4 widths × 16 samples = 64 evaluations

# Example 8-bit random inputs (reproducible):
random_inputs[8] = [
    {'x0': 137, 'x1': 42, 'x2': 201, ...},  # Sample 1
    {'x0': 73, 'x1': 189, 'x2': 15, ...},   # Sample 2
    ...  # 16 total
]
```

**Why Random Hash?**: Acts as semantic fingerprint signature. Equivalent expressions produce identical hashes; different expressions unlikely to collide.

### 3.4 Derivatives (32 dims)

**Finite difference approximation of partial derivatives**:

```python
# For each width and variable:
base_point = {f'x{i}': 2^(width-1) for i in range(8)}  # Midpoint
epsilon = 1

# Partial derivative ∂f/∂x_i:
derivative = (f(x0,...,x_i+ε,...,x7) - f(x0,...,x_i,...,x7)) / ε

# Total: 4 widths × 8 variables = 32 derivatives
```

**Normalization**: `derivative / max_value` per width.

**Why Derivatives?**: Captures local behavior. Linear expressions have constant derivatives; nonlinear expressions vary.

### 3.5 Truth Table (64 dims)

**Boolean output signature for first 6 variables**:

```python
# Enumerate all 2^6 = 64 input combinations (6 variables as bits)
for i in range(64):
    assignment = {
        'x0': (i >> 0) & 1,
        'x1': (i >> 1) & 1,
        'x2': (i >> 2) & 1,
        'x3': (i >> 3) & 1,
        'x4': (i >> 4) & 1,
        'x5': (i >> 5) & 1,
    }
    output = evaluate(expr, assignment, width=64)
    truth_table[i] = output & 1  # Extract LSB
```

**Why Truth Table?**: Boolean expressions (AND/OR/XOR) differ in their truth tables. 64 entries fully characterize 6-variable boolean functions.

**Design Choice**: 2^6=64 fits exactly in 64 dims. Larger tables (2^8=256) would dominate fingerprint; smaller (2^4=16) lose expressiveness.

---

## 4. Batch Collation for Variable-Size Graphs

**File**: `src/data/collate.py`

### Challenge: Variable Sizes

- **Graphs**: Different node counts (depth 2: ~5 nodes, depth 14: ~300 nodes)
- **Sequences**: Different token lengths (simplified "x+y": 5 tokens, obfuscated: 50+ tokens)

### Solution: PyG Batching + Sequence Padding

#### Graph Batching (`Batch.from_data_list`)

PyTorch Geometric concatenates graphs along node dimension and adds `batch` tensor:

```python
# Input: 3 graphs with N1=4, N2=3, N3=5 nodes
graph1.x.shape = [4, 32]
graph2.x.shape = [3, 32]
graph3.x.shape = [5, 32]

# After Batch.from_data_list([graph1, graph2, graph3]):
batched.x.shape = [12, 32]  # Concatenated nodes
batched.edge_index.shape = [2, E_total]  # Edge indices offset per graph
batched.batch = [0,0,0,0, 1,1,1, 2,2,2,2,2]  # Graph assignment per node
```

**Graph Pooling**: Use `scatter_mean(batched.x, batched.batch, dim=0)` to get per-graph embeddings.

#### Sequence Padding

```python
# Input: 3 sequences with different lengths
seq1 = [1, 14, 5, 15, 2]       # len=5
seq2 = [1, 14, 6, 15, 2]       # len=5
seq3 = [1, 12, 14, 5, 15, 13, 2]  # len=7

# After pad_sequence([seq1, seq2, seq3], padding_value=0):
padded = [
    [1, 14, 5, 15, 2, 0, 0],   # Padded with 0 (<pad>)
    [1, 14, 6, 15, 2, 0, 0],
    [1, 12, 14, 5, 15, 13, 2],
]
padded.shape = [3, 7]  # [batch_size, max_len]
lengths = [5, 5, 7]
```

**Padding Token**: `PAD_IDX=0` (`<pad>` token).

**Length Tracking**: `target_lengths` and `source_lengths` tensors enable masking in loss computation.

### Collate Functions

#### `collate_graphs(batch)` - Supervised Training

**Input**: List of `MBADataset.__getitem__` outputs
**Output**:
```python
{
    'graph_batch': Data(x=[N_total, 32], edge_index=[2, E_total], batch=[N_total]),
    'fingerprint': Tensor[batch_size, 448],
    'target_ids': Tensor[batch_size, max_target_len],  # Padded with PAD_IDX
    'target_lengths': Tensor[batch_size],
    'source_tokens': Tensor[batch_size, max_source_len],  # For copy mechanism
    'source_lengths': Tensor[batch_size],
    'depth': Tensor[batch_size],
    'obfuscated': List[str],  # For debugging/logging
    'simplified': List[str],
}
```

#### `collate_contrastive(batch)` - Phase 1 Pretraining

**Input**: List of `ContrastiveDataset.__getitem__` outputs
**Output**:
```python
{
    'obf_graph_batch': Data(...),     # Obfuscated expression graphs
    'simp_graph_batch': Data(...),    # Simplified expression graphs
    'obf_fingerprint': Tensor[batch_size, 448],
    'simp_fingerprint': Tensor[batch_size, 448],
    'labels': Tensor[batch_size],     # Index for positive pair matching in InfoNCE
    'obfuscated': List[str],
    'simplified': List[str],
}
```

**InfoNCE Loss**: Uses `labels` to identify positive pairs. Label `i` means `obf_graph[i]` and `simp_graph[i]` are equivalent.

---

## 5. Data Formats

### Input Format (JSONL)

**Base Schema** (for `MBADataset` and `ContrastiveDataset`):
```json
{"obfuscated": "(x & y) + (x ^ y)", "simplified": "x | y", "depth": 3}
{"obfuscated": "(x + y) ^ (x | y)", "simplified": "x & y", "depth": 3}
```

**Fields**:
- `obfuscated` (str, required): Input MBA expression
- `simplified` (str, required): Target simplified expression
- `depth` (int, optional): AST depth for curriculum learning

**Scaled Schema v6** (for `ScaledMBADataset`):
```json
{
  "obfuscated_expr": "(x & y) + (x ^ y)",
  "ground_truth_expr": "x | y",
  "depth": 3,
  "boolean_domain_only": false,
  "complexity_score": 0.42,
  "ast": {
    "nodes": [
      {"id": 0, "type": "ADD"},
      {"id": 1, "type": "AND"},
      {"id": 2, "type": "VAR", "value": "x"},
      ...
    ],
    "edges": [
      {"src": 0, "dst": 1, "type": 0},  // CHILD_LEFT
      ...
    ]
  },
  "fingerprint": {
    "flat": [0.12, 0.45, ..., 0.89]  // 448 floats
  }
}
```

**Additional Fields**:
- `boolean_domain_only` (bool): Conditioning signal for domain-specific rules
- `complexity_score` (float): Pre-computed complexity metric
- `ast` (dict): Pre-built AST (nodes + edges) for subexpression sharing
- `fingerprint.flat` (list): Pre-computed 448-dim fingerprint

**Backward Compatibility**: `ScaledMBADataset` supports both old (`obfuscated`/`simplified`) and new (`obfuscated_expr`/`ground_truth_expr`) field names.

### Output Format (Model Predictions)

**Beam Search Output**:
```json
{
  "input": "(x & y) + (x ^ y)",
  "candidates": [
    {
      "expression": "x | y",
      "score": 12.34,
      "log_prob": -2.1,
      "verified": true,
      "verification_method": "z3",
      "simplification_gain": 0.67
    },
    {
      "expression": "x + y - (x & y)",
      "score": 11.89,
      "log_prob": -2.3,
      "verified": true,
      "verification_method": "execution",
      "simplification_gain": 0.12
    }
  ],
  "best": "x | y"
}
```

**Verification Methods**:
- `syntax`: Parses successfully
- `execution`: Passes random execution tests
- `z3`: Z3 proves equivalence

---

## 6. Dataset Classes

### `MBADataset` - Supervised Seq2Seq Training

**Purpose**: Phase 2 supervised training with cross-entropy loss.

**Key Features**:
- Depth filtering for curriculum learning
- Both source and target tokenization (for copy mechanism)
- Fallback to runtime fingerprint computation

**Usage**:
```python
from src.data.dataset import MBADataset
from src.data.tokenizer import MBATokenizer
from src.data.fingerprint import SemanticFingerprint
from src.data.collate import collate_graphs
from torch.utils.data import DataLoader

tokenizer = MBATokenizer()
fingerprint = SemanticFingerprint(seed=42)

dataset = MBADataset(
    data_path="data/train_depth_5.jsonl",
    tokenizer=tokenizer,
    fingerprint=fingerprint,
    max_depth=5  # Curriculum stage 2
)

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=collate_graphs,
    num_workers=4
)
```

### `ContrastiveDataset` - Phase 1 Pretraining

**Purpose**: Contrastive learning to map equivalent expressions to similar embeddings.

**Key Features**:
- Dual graph encoding (obfuscated + simplified)
- Dual fingerprints
- Labels for InfoNCE positive pair matching

**Loss Functions**:
- **InfoNCE**: Pulls equivalent pairs together, pushes non-equivalent apart
- **MaskLM**: Masked expression modeling (predict masked subexpressions)

### `ScaledMBADataset` - Scaled Model with Optimizations

**Purpose**: 360M parameter model training with efficiency optimizations.

**Key Features**:
- **Subexpression Sharing**: Merges identical subtrees via MD5 hashing
- **Pre-computed Features**: Loads fingerprint from JSONL (avoids runtime computation)
- **Optimized Edge Types**: 7-type system with inverses and domain bridges
- **Conditioning Signals**: `boolean_domain_only` for domain-specific rule learning

**Subexpression Sharing Algorithm**:
```python
def _compute_subtree_hash(node_id):
    """Bottom-up subtree hashing for deduplication."""
    node = nodes[node_id]
    children_hashes = [_compute_subtree_hash(child) for child in node.children]
    subtree_str = f"{node.type}|{node.value}|{','.join(children_hashes)}"
    return md5(subtree_str).hexdigest()

# During graph construction:
if hash in seen_subtrees:
    merge_node(current_node, seen_subtrees[hash])
else:
    seen_subtrees[hash] = current_node
```

**Memory Reduction**: Depth 14 expressions with repeated subexpressions: 300 nodes → 150 nodes (50% reduction).

---

## 7. Performance Characteristics

### Throughput Benchmarks (A100 GPU)

| Operation               | Time per Sample | Batch Size 32 | Bottleneck              |
|-------------------------|-----------------|---------------|-------------------------|
| Tokenization            | ~0.1 ms         | ~3 ms         | Regex matching          |
| AST Parsing             | ~0.5 ms         | ~16 ms        | Recursive descent       |
| Fingerprint Compute     | ~2.0 ms         | ~64 ms        | Expression evaluation   |
| Graph Construction      | ~0.3 ms         | ~10 ms        | Edge list building      |
| Collation (batching)    | ~1.0 ms         | ~1 ms         | PyG batching overhead   |
| **Total (cold)**        | **~3.9 ms**     | **~94 ms**    |                         |
| **Total (pre-computed)**| **~0.9 ms**     | **~30 ms**    | ScaledMBADataset        |

**Recommendation**: For large-scale training (12M samples), pre-compute fingerprints and ASTs offline. `ScaledMBADataset` achieves 3.1× speedup.

### Memory Usage

| Component           | Per Sample (avg) | Batch 32     | Notes                        |
|---------------------|------------------|--------------|------------------------------|
| Graph (depth 6)     | ~2 KB            | ~64 KB       | 20 nodes × 32 floats         |
| Graph (depth 14)    | ~12 KB           | ~384 KB      | 150 nodes (with sharing)     |
| Fingerprint         | 1.75 KB          | 56 KB        | 448 × float32                |
| Target sequence     | ~0.5 KB          | ~16 KB       | Avg 64 tokens × 2 bytes      |
| **Total (depth 6)** | **~4.3 KB**      | **~137 KB**  |                              |
| **Total (depth 14)**| **~14.3 KB**     | **~457 KB**  |                              |

**GPU Memory**: Batch size 32 on A100 (80GB): ~600 MB for data + ~20 GB for model = comfortable fit.

---

## 8. Data Augmentation (Not Implemented)

**Future Work**: Potential augmentation strategies:

1. **Variable Renaming**: `x+y` → `a+b` (preserves semantics)
2. **Commutative Reordering**: `x+y` → `y+x`
3. **Constant Folding**: `3+5` → `8` (lossy but valid)
4. **Subexpression Substitution**: Replace `x|y` with `(x&y)+(x^y)` (increases obfuscation)

**Risk**: Invalid augmentations may introduce incorrect equivalences. Requires Z3 verification.

---

## 9. Engineering Notes

### Adding New Features to Fingerprint

**To add a new feature component** (e.g., "Fourier features"):

1. Update `src/constants.py`:
   ```python
   FOURIER_DIM: int = 32
   FINGERPRINT_DIM = SYMBOLIC_DIM + CORNER_DIM + RANDOM_DIM + DERIVATIVE_DIM + TRUTH_TABLE_DIM + FOURIER_DIM
   ```

2. Implement `_fourier_features()` in `SemanticFingerprint`:
   ```python
   def _fourier_features(self, expr: str, variables: List[str]) -> np.ndarray:
       features = np.zeros(FOURIER_DIM, dtype=np.float32)
       # ... compute Fourier features
       return features
   ```

3. Update `compute()`:
   ```python
   fp[offset:offset + FOURIER_DIM] = self._fourier_features(expr, variables)
   offset += FOURIER_DIM
   ```

4. **Critical**: Regenerate entire dataset with new fingerprints (backward incompatible).

### Debugging Dataset Issues

**Common Issues**:

| Error                          | Cause                          | Fix                                    |
|--------------------------------|--------------------------------|----------------------------------------|
| `No valid data loaded`         | Empty JSONL or wrong path      | Check file path, validate JSONL syntax |
| `Unexpected tokens after pos`  | Invalid expression syntax      | Add validation in data generation      |
| `KeyError: 'obfuscated'`       | Missing required field         | Validate JSONL schema                  |
| `RuntimeError: CUDA OOM`       | Batch size too large           | Reduce batch size or max_depth filter  |
| `AssertionError: fingerprint`  | Dimension mismatch             | Verify FINGERPRINT_DIM in constants.py |

**Validation Script**:
```bash
python scripts/validate_data.py --data data/train.jsonl --check-parse --check-fingerprint
```

---

## 10. Related Documentation

- `docs/ML_PIPELINE.md` - Full model architecture (encoder, decoder, heads)
- `docs/TRAINING.md` - Training phases and curriculum learning
- `scripts/generate_data.py` - Dataset generation logic
- `src/models/encoder.py` - GNN encoder that consumes graph data
- `src/models/decoder.py` - Transformer decoder that consumes fingerprints

---

**Document Version**: v1.0
**Last Updated**: 2026-01-16
**Target Audience**: ML engineers extending data pipeline or debugging preprocessing
