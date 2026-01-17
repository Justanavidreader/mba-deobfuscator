# Data Pipeline

Complete specification of data processing, tokenization, fingerprinting, and batching for the MBA Deobfuscator.

---

## Pipeline Overview

```
Raw Expression String
    ↓
[Normalization] C++ generator format → Internal format
    ↓
[AST Parser] String → Abstract Syntax Tree
    ↓
[Graph Builder] AST → PyTorch Geometric Data
    ↓
[Tokenizer] Expression → Token sequence
    ↓
[Fingerprint] Expression → 448-dim vector → 416-dim ML vector
    ↓
[Augmentation] Variable permutation (80% probability)
    ↓
[Collation] Individual samples → Batched Data
    ↓
Model Input: {graph, tokens, fingerprint}
```

---

## 1. Data Format

### Input Format (JSONL)

```json
{"obfuscated": "(x0 & x1) + (x0 ^ x1)", "simplified": "x0 | x1", "depth": 3}
{"obfuscated": "x2 ^ x2", "simplified": "0", "depth": 1}
{"obfuscated": "(x3 & ~x3) + x4", "simplified": "x4", "depth": 3}
```

**Fields**:
- `obfuscated`: Obfuscated MBA expression (input)
- `simplified`: Simplified equivalent (target)
- `depth`: AST depth (for curriculum learning)

**Variable naming**: `x0` to `x7` (8 variables max)

### C++ Generator Format

External C++ generator produces different format:
```json
{"input": "And(x, y) + Xor(x, y)", "output": "Or(x, y)", "complexity": 3}
```

**Normalization** (handled automatically):
```python
# src/data/dataset.py
def _normalize_cpp_format(sample):
    if 'input' in sample:
        sample['obfuscated'] = sample.pop('input')
    if 'output' in sample:
        sample['simplified'] = sample.pop('output')
    if 'complexity' in sample:
        sample['depth'] = sample.pop('complexity')

    # Function notation → operator notation
    sample['obfuscated'] = sample['obfuscated'].replace('And', '&')
    sample['obfuscated'] = sample['obfuscated'].replace('Or', '|')
    # ... more replacements
```

---

## 2. AST Parser

Converts expression string to Abstract Syntax Tree and PyTorch Geometric graph.

### Expression Grammar

```bnf
<expr>     ::= <term> | <expr> <binop> <term>
<term>     ::= <factor> | <unaryop> <factor>
<factor>   ::= <variable> | <constant> | "(" <expr> ")"
<binop>    ::= "+" | "-" | "*" | "&" | "|" | "^"
<unaryop>  ::= "~" | "-"
<variable> ::= "x0" | "x1" | ... | "x7"
<constant> ::= "0" | "1" | ... | "255"
```

### AST Node Types

```python
class NodeType(Enum):
    VARIABLE = 0
    CONSTANT = 1
    ADD = 2
    SUB = 3
    MUL = 4
    AND = 5
    OR = 6
    XOR = 7
    NOT = 8
    NEG = 9
```

### Example: `(x0 & x1) + (x0 ^ x1)`

**AST**:
```
          ADD
         /   \
       AND    XOR
       / \    / \
      x0 x1  x0 x1
```

**Graph representation**:
```python
# Nodes
nodes = [
    {'type': ADD, 'id': 0},
    {'type': AND, 'id': 1},
    {'type': XOR, 'id': 2},
    {'type': VARIABLE, 'var_id': 0, 'id': 3},  # x0 (left)
    {'type': VARIABLE, 'var_id': 1, 'id': 4},  # x1
    {'type': VARIABLE, 'var_id': 0, 'id': 5},  # x0 (right)
    {'type': VARIABLE, 'var_id': 1, 'id': 6},  # x1
]

# Edges (8-type system)
edges = [
    (0, 1, LEFT_OPERAND),    # ADD → AND
    (0, 2, RIGHT_OPERAND),   # ADD → XOR
    (1, 3, LEFT_OPERAND),    # AND → x0
    (1, 4, RIGHT_OPERAND),   # AND → x1
    (2, 5, LEFT_OPERAND),    # XOR → x0
    (2, 6, RIGHT_OPERAND),   # XOR → x1
    # Inverse edges
    (1, 0, LEFT_OPERAND_INV),
    (2, 0, RIGHT_OPERAND_INV),
    # ... more inverses
]
```

### Edge Type Systems

#### Legacy (6-type)

Used by older datasets and GGNN:
```python
class EdgeType(Enum):
    CHILD_LEFT = 0      # Parent → left child
    CHILD_RIGHT = 1     # Parent → right child
    PARENT = 2          # Child → parent
    SIBLING_NEXT = 3    # Left sibling → right sibling
    SIBLING_PREV = 4    # Right sibling → left sibling
    SAME_VAR = 5        # Same variable occurrences
```

#### Optimized (8-type)

Used by HGT, RGCN, Semantic HGT:
```python
class EdgeType(Enum):
    LEFT_OPERAND = 0           # Operator → left operand
    RIGHT_OPERAND = 1          # Operator → right operand
    UNARY_OPERAND = 2          # Unary operator → operand

    LEFT_OPERAND_INV = 3       # Left operand → operator
    RIGHT_OPERAND_INV = 4      # Right operand → operator
    UNARY_OPERAND_INV = 5      # Operand → unary operator

    DOMAIN_BRIDGE_DOWN = 6     # Connect operator domains downward
    DOMAIN_BRIDGE_UP = 7       # Connect operator domains upward
```

**Rationale**: Optimized system is more expressive for heterogeneous GNNs

### Node Features

```python
# For each node
node_features = {
    'type': one_hot(node_type, num_classes=10),     # [10]
    'var_id': one_hot(var_id, num_classes=8),       # [8] (if variable)
    'const_val': normalized_value,                   # [1] (if constant)
    'depth': node_depth / max_depth,                 # [1] (normalized)
    'subtree_size': subtree_size / total_nodes,      # [1] (normalized)
    'in_degree': in_degree,                          # [1]
    'is_shared': is_shared_subexpression,            # [1] (boolean)
}
# Total: 23 dims per node
```

**DAG Features** (optional):
- Depth: Distance from root
- Subtree size: Number of descendants
- In-degree: Number of parents (>1 if shared subexpression)
- Is shared: Boolean flag for shared nodes

### Implementation

```python
# src/data/ast_parser.py
from src.data.ast_parser import parse_expression, expression_to_graph

# Parse to AST
ast = parse_expression("(x0 & x1) + (x0 ^ x1)")

# Convert to PyG Data
data = expression_to_graph(ast, edge_type_system='optimized')
# data.x: [num_nodes × 23] node features
# data.edge_index: [2 × num_edges] edge connectivity
# data.edge_type: [num_edges] edge type labels
# data.num_nodes: number of nodes
```

---

## 3. Tokenization

### Vocabulary (300 tokens)

```python
# Special tokens (0-4)
PAD = 0
UNK = 1
BOS = 2  # Beginning of sequence
EOS = 3  # End of sequence
MASK = 4  # For masked language modeling

# Operators (5-12)
AND = 5    # &
OR = 6     # |
XOR = 7    # ^
ADD = 8    # +
SUB = 9    # -
MUL = 10   # *
NOT = 11   # ~
NEG = 12   # neg (unary minus)

# Parentheses (13-14)
LPAREN = 13  # (
RPAREN = 14  # )

# Variables (15-22)
X0 = 15, X1 = 16, ..., X7 = 22

# Constants (23-277)
# 0 → 255 mapped to tokens 23-277

# Reserved (278-299)
# For future expansion
```

### Tokenization Process

```python
from src.data.tokenizer import MBATokenizer

tokenizer = MBATokenizer()

# Encode
expression = "(x0 & x1) + (x0 ^ x1)"
tokens = tokenizer.encode(expression)
# tokens = [13, 15, 5, 16, 14, 8, 13, 15, 7, 16, 14]
#           (   x0  &  x1  )   +  (   x0  ^  x1  )

# Decode
decoded = tokenizer.decode(tokens)
# decoded = "(x0 & x1) + (x0 ^ x1)"

# With special tokens
tokens_with_special = tokenizer.encode(expression, add_special_tokens=True)
# tokens_with_special = [2, 13, 15, 5, 16, 14, 8, 13, 15, 7, 16, 14, 3]
#                        BOS (   x0  &  x1  )   +  (   x0  ^  x1  ) EOS
```

### Tokenization Rules

1. **Whitespace**: Ignored
2. **Operators**: Multi-char operators (`<<`, `>>`) tokenized as single tokens (if needed)
3. **Variables**: `x` followed by digit → variable token
4. **Constants**: Decimal integers → constant tokens (0-255 supported)
5. **Parentheses**: `(` and `)` → separate tokens

### Variable Normalization

Automatically renames variables to canonical order:
```python
# Input: (y & z) + (y ^ z)
# Normalized: (x0 & x1) + (x0 ^ x1)

# Variables sorted by first appearance: y→x0, z→x1
```

**Rationale**: Equivalent expressions with different variable names should have identical tokens

---

## 4. Semantic Fingerprint

### Overview

**Purpose**: Capture semantic properties of expressions independent of syntactic form

**Dimensions**: 448 (raw) → 416 (ML)

**Deterministic**: Same expression always produces identical fingerprint

**C++ Acceleration**: Optional 10× speedup via `mba_fingerprint_cpp`

### Computation

```python
from src.data.fingerprint import SemanticFingerprint

fp = SemanticFingerprint()
vector = fp.compute("(x0 & x1) + (x0 ^ x1)")
# vector.shape = (448,)

# For ML (strip derivatives)
ml_vector = fp.compute_ml("(x0 & x1) + (x0 ^ x1)")
# ml_vector.shape = (416,)
```

### Component Breakdown

#### 4.1 Symbolic Features (32 dims)

Structural analysis of AST:
```python
symbolic = [
    num_nodes,              # Total AST nodes
    num_leaves,             # Leaf count (vars + consts)
    max_depth,              # Maximum depth
    avg_depth,              # Average node depth
    num_operators,          # Operator count
    num_unique_vars,        # Unique variables
    num_unique_consts,      # Unique constants

    # Operator counts (per type)
    count_add, count_sub, count_mul,
    count_and, count_or, count_xor,
    count_not, count_neg,

    # Degree statistics
    max_out_degree,
    avg_out_degree,
    max_in_degree,
    avg_in_degree,

    # Subtree statistics
    max_subtree_size,
    avg_subtree_size,

    # Complexity metrics
    cyclomatic_complexity,
    expression_entropy,

    # ... (padded to 32 dims)
]
```

**Example**: `(x0 & x1) + (x0 ^ x1)`
```python
symbolic = [
    7,    # num_nodes (ADD, AND, XOR, 4× variables)
    4,    # num_leaves
    3,    # max_depth
    2.14, # avg_depth
    3,    # num_operators (ADD, AND, XOR)
    2,    # num_unique_vars (x0, x1)
    0,    # num_unique_consts
    1, 0, 0,  # 1× ADD, 0× SUB, 0× MUL
    1, 0, 1,  # 1× AND, 0× OR, 1× XOR
    0, 0,     # 0× NOT, 0× NEG
    # ... more features
]
```

#### 4.2 Corner Evaluations (256 dims)

Evaluate at extreme values across 4 bit widths:

```python
bit_widths = [8, 16, 32, 64]

# Corner values per variable
corner_cases = [
    0, 1, -1,
    MAX_VAL, MIN_VAL,
    MAX_VAL // 2, MIN_VAL // 2,
    # ... (64 total corner value combinations)
]

corner_evals = []
for width in bit_widths:
    for values in corner_cases:
        result = evaluate_expression(expr, values, width)
        corner_evals.append(result % (2 ** width))

# Total: 4 widths × 64 cases = 256 dims
```

**Example**: `x0 & x1` at width=8
```python
evaluate(x0=0, x1=0) = 0
evaluate(x0=0, x1=1) = 0
evaluate(x0=1, x1=0) = 0
evaluate(x0=1, x1=1) = 1
evaluate(x0=255, x1=255) = 255
evaluate(x0=0, x1=255) = 0
# ... all 64 corner cases
```

**Deterministic**: Fixed set of corner values ensures reproducibility

#### 4.3 Random Hash (64 dims)

Pseudo-random evaluations with fixed seed:

```python
np.random.seed(42)  # Fixed for determinism
random_inputs = np.random.randint(0, 2**32, size=(16, 8))  # 16 samples, 8 vars

random_hash = []
for width in [8, 16, 32, 64]:
    for sample in random_inputs:
        result = evaluate_expression(expr, sample[:num_vars], width)
        random_hash.append(result % (2 ** width))

# Total: 4 widths × 16 samples = 64 dims
```

**Purpose**: Probabilistic collision detection
- Different expressions likely produce different hashes
- Complements corner evaluations

#### 4.4 Derivatives (32 dims) - **EXCLUDED FOR ML**

**Original computation** (not used):
```python
for width in [8, 16, 32, 64]:
    for order in range(8):
        derivative = compute_numerical_derivative(expr, order, width)
        # derivative ≈ (f(x+h) - f(x)) / h for various h

# Total: 4 widths × 8 orders = 32 dims
```

**Why excluded**:
- C++ uses forward differences: `(f(x+1) - f(x)) / 1`
- Python uses central differences: `(f(x+h) - f(x-h)) / (2h)`
- Inconsistency causes fingerprint mismatches

**Workaround**:
```python
# src/data/dataset.py
def _strip_derivatives(fingerprint):
    # Remove dims 352:384 (derivatives)
    return np.concatenate([fingerprint[:352], fingerprint[384:]])

# Called automatically in dataset __getitem__
```

#### 4.5 Truth Table (64 dims)

Boolean function evaluation for up to 6 variables:

```python
num_vars = min(count_variables(expr), 6)

truth_table = []
for input_bits in range(64):  # 2^6 = 64
    # Extract 6 bits
    x0 = (input_bits >> 0) & 1
    x1 = (input_bits >> 1) & 1
    x2 = (input_bits >> 2) & 1
    x3 = (input_bits >> 3) & 1
    x4 = (input_bits >> 4) & 1
    x5 = (input_bits >> 5) & 1

    result = evaluate_expression(expr, [x0, x1, x2, x3, x4, x5])
    truth_table.append(result & 1)  # Boolean output (0 or 1)

# Total: 64 dims
```

**Example**: `x0 | x1`
```python
truth_table = [
    0,  # x0=0, x1=0 → 0
    1,  # x0=1, x1=0 → 1
    1,  # x0=0, x1=1 → 1
    1,  # x0=1, x1=1 → 1
    # ... (60 more entries for x2-x5 combinations)
]
```

**Properties**:
- Complete Boolean function signature
- Enables equivalence detection: `f ≡ g ⟺ truth_table(f) == truth_table(g)`
- Sufficient for most MBA patterns

### ML Fingerprint (416 dims)

```python
# Strip derivatives (dims 352:384)
ml_fingerprint = np.concatenate([
    fingerprint[:352],   # Symbolic + Corner + Random
    fingerprint[384:]    # Truth table
])

assert ml_fingerprint.shape == (416,)
```

**Used throughout training and inference** to avoid C++/Python inconsistencies

### C++ Acceleration

```python
try:
    from mba_fingerprint_cpp import compute_fingerprint_cpp
    USE_CPP = True
except ImportError:
    USE_CPP = False

def compute(self, expr):
    if USE_CPP:
        return compute_fingerprint_cpp(expr)  # ~1-5ms
    else:
        return self._compute_python(expr)      # ~10-50ms
```

**Speedup**: 10× faster with C++ (optional dependency)

**Fallback**: Gracefully falls back to pure Python if C++ library unavailable

---

## 5. Dataset Classes

### 5.1 MBADataset (Phase 2 Supervised)

**Usage**: Supervised learning with curriculum

```python
from src.data.dataset import MBADataset

dataset = MBADataset(
    data_path='data/train.json',
    max_depth=10,  # Curriculum stage
    augment=True   # Variable permutation
)

sample = dataset[0]
# sample = {
#     'obfuscated_graph': Data(...),
#     'obfuscated_tokens': Tensor([...]),
#     'simplified_tokens': Tensor([...]),
#     'fingerprint': Tensor([416]),
#     'depth': int,
#     'length': int
# }
```

**Features**:
- Filters by max depth (curriculum learning)
- Strips derivatives from fingerprint
- Variable augmentation (80% probability)
- Normalizes C++ generator format

### 5.2 ContrastiveDataset (Phase 1 Pretraining)

**Usage**: Contrastive learning with positive pairs

```python
from src.data.dataset import ContrastiveDataset

dataset = ContrastiveDataset(
    data_path='data/train.json',
    augment=True
)

sample = dataset[0]
# sample = {
#     'anchor': Data(...),           # Original expression graph
#     'positive': Data(...),         # Equivalent expression graph
#     'fingerprint': Tensor([416])
# }
```

**Positive pairs**:
- Original and simplified expressions (guaranteed equivalent)
- Variable-permuted versions of same expression

**Loss**: InfoNCE (maximize similarity of positive pairs)

### 5.3 ScaledMBADataset (360M Model)

**Usage**: Scaled model with subexpression sharing

```python
from src.data.dataset import ScaledMBADataset

dataset = ScaledMBADataset(
    data_path='data/train_large.json',
    max_seq_len=2048,  # Support depth-14 expressions
    share_subexpressions=True
)
```

**Features**:
- Supports very long sequences (up to 2048 tokens)
- Detects and shares common subexpressions in graph
- Uses 8-type edge system (required for HGT)
- Larger batch fingerprint caching

### 5.4 GMNDataset (Graph Matching)

**Usage**: Graph Matching Network training

```python
from src.data.dataset import GMNDataset

dataset = GMNDataset(
    data_path='data/train.json',
    pair_mode='equivalent'  # or 'random'
)

sample = dataset[0]
# sample = {
#     'graph1': Data(...),
#     'graph2': Data(...),
#     'label': 1,  # 1 = equivalent, 0 = different
#     'fingerprint1': Tensor([416]),
#     'fingerprint2': Tensor([416])
# }
```

**Pair modes**:
- `equivalent`: Positive pairs (obfuscated + simplified)
- `random`: Random pairs with labels (for contrastive training)

---

## 6. Data Augmentation

### Variable Permutation

**Goal**: Increase training data diversity, enforce variable-order invariance

**Method**:
```python
# Original: (x0 & x1) + (x0 ^ x1)
# Permuted: (x1 & x2) + (x1 ^ x2)  (swap x0↔x1, shift all)

def augment_variables(expr, prob=0.8):
    if random.random() > prob:
        return expr  # No augmentation

    # Extract unique variables
    vars = extract_variables(expr)  # ['x0', 'x1']

    # Generate random permutation
    new_vars = random.sample(vars, len(vars))  # ['x1', 'x0']

    # Rename
    for old, new in zip(vars, new_vars):
        expr = expr.replace(old, f'__{new}__')  # Temp names to avoid conflicts

    expr = expr.replace('__', '')  # Remove temp markers

    return expr
```

**Applied**:
- 80% probability during training
- Disabled during validation/testing
- Increases dataset size ~5× effectively

**Example**:
```python
# Original
obf = "(x0 & x1) + (x0 ^ x1)"
sim = "x0 | x1"

# Augmented
obf_aug = "(x1 & x0) + (x1 ^ x0)"  # Variable swap
sim_aug = "x1 | x0"
```

---

## 7. Batch Collation

### Graph Batching

PyTorch Geometric batches graphs by concatenating nodes and offset edge indices:

```python
# Individual graphs
graph1: nodes=[5], edges=[4×2]
graph2: nodes=[7], edges=[6×2]

# Batched
batch: nodes=[12], edges=[10×2]
# graph1 nodes: 0-4
# graph2 nodes: 5-11 (offset by 5)
# graph1 edges: [...]
# graph2 edges: [...] + 5 (indices offset)
```

**Implementation**:
```python
from src.data.collate import collate_mba_batch
from torch.utils.data import DataLoader

loader = DataLoader(
    dataset,
    batch_size=32,
    collate_fn=collate_mba_batch,
    shuffle=True
)

batch = next(iter(loader))
# batch = {
#     'obfuscated_graph': Batch(...),  # PyG Batch
#     'obfuscated_tokens': Tensor([32, max_len]),
#     'simplified_tokens': Tensor([32, max_len]),
#     'fingerprints': Tensor([32, 416]),
#     'depths': Tensor([32]),
#     'lengths': Tensor([32])
# }
```

### Sequence Padding

Token sequences padded to max length in batch:

```python
sequences = [
    [2, 13, 15, 5, 16, 14, 3],           # length=7
    [2, 15, 8, 16, 3],                   # length=5
    [2, 13, 15, 7, 16, 14, 8, 22, 3]    # length=9
]

# Padded to max_len=9
padded = [
    [2, 13, 15, 5, 16, 14, 3, 0, 0],    # + 2 PAD
    [2, 15, 8, 16, 3, 0, 0, 0, 0],      # + 4 PAD
    [2, 13, 15, 7, 16, 14, 8, 22, 3]    # + 0 PAD
]

# Attention mask (1=real token, 0=padding)
mask = [
    [1, 1, 1, 1, 1, 1, 1, 0, 0],
    [1, 1, 1, 1, 1, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1]
]
```

**Used by decoder** to ignore padding during attention

---

## 8. Curriculum Learning

### Depth-Based Curriculum (Phase 2)

```python
# Stage 1: Shallow expressions
dataset_stage1 = MBADataset(data_path, max_depth=2)
train(model, dataset_stage1, epochs=10)

# Stage 2: Medium depth
dataset_stage2 = MBADataset(data_path, max_depth=5)
train(model, dataset_stage2, epochs=15)

# Stage 3: Deep expressions
dataset_stage3 = MBADataset(data_path, max_depth=10)
train(model, dataset_stage3, epochs=15)

# Stage 4: Very deep
dataset_stage4 = MBADataset(data_path, max_depth=14)
train(model, dataset_stage4, epochs=10)
```

**Rationale**: Gradually increase difficulty, prevent overfitting to complex patterns

### Self-Paced Learning

Dynamically adjust difficulty based on model performance:

```python
# After each epoch
accuracy_by_depth = evaluate_per_depth(model, val_set)

# If accuracy on depth-5 > 90%, include depth-6
if accuracy_by_depth[5] > 0.9:
    max_depth = 6
```

**Implementation**: `src/training/phase2_trainer.py`

---

## 9. Data Statistics

### Dataset Sizes (Typical)

| Split | Samples | Depth Range | Size (JSONL) |
|-------|---------|-------------|--------------|
| Train | 10M | 1-14 | ~2 GB |
| Validation | 1M | 1-14 | ~200 MB |
| Test | 100K | 1-14 | ~20 MB |

### Expression Complexity Distribution

| Depth | Avg Nodes | Avg Tokens | Percentage |
|-------|-----------|------------|------------|
| 1-2 | 3-5 | 5-7 | 15% |
| 3-5 | 7-15 | 11-25 | 30% |
| 6-10 | 20-50 | 30-80 | 40% |
| 11-14 | 60-150 | 100-250 | 15% |

### Memory Requirements

| Component | Per Sample | Batch (32) | Dataset (10M) |
|-----------|------------|------------|---------------|
| Graph | ~1 KB | ~32 KB | ~10 GB |
| Tokens | ~200 B | ~6 KB | ~2 GB |
| Fingerprint | ~1.6 KB | ~50 KB | ~16 GB |
| **Total** | ~3 KB | ~100 KB | ~30 GB |

**Streaming**: Use `IterableDataset` for very large datasets (>10M samples)

---

## 10. Data Generation

### Using Provided Script

```bash
python scripts/generate_data.py \
    --depth 1-14 \
    --samples 10000000 \
    --output data/train.json \
    --seed 42
```

**Parameters**:
- `--depth`: Depth range (e.g., `1-14`, `3-10`)
- `--samples`: Number of samples to generate
- `--output`: Output JSONL file
- `--seed`: Random seed for reproducibility

### Manual Generation

```python
from src.data.generator import MBAGenerator

gen = MBAGenerator(seed=42)

# Generate single sample
sample = gen.generate(max_depth=5)
# sample = {
#     'obfuscated': '(x0 & x1) + (x0 ^ x1)',
#     'simplified': 'x0 | x1',
#     'depth': 3
# }

# Generate dataset
samples = gen.generate_dataset(num_samples=10000, max_depth=10)
gen.save_jsonl(samples, 'data/train.json')
```

### MBA Identity Patterns

Common obfuscation patterns:
```python
# Identity laws
x & x → x
x | x → x
x ^ x → 0
x + 0 → x

# MBA patterns
(x & y) + (x ^ y) → x | y
(x | y) - (x ^ y) → x & y
~(x & y) → ~x | ~y  (De Morgan)

# Constant folding
x & 0 → 0
x | 0 → x
x ^ 0 → x
```

**Generator** instantiates these patterns with random variables and depths

---

## 11. Performance Optimization

### Fingerprint Caching

```python
# Cache fingerprints to avoid recomputation
class MBADataset:
    def __init__(self, data_path, cache_fingerprints=True):
        self.fp_cache = {}

    def __getitem__(self, idx):
        expr = self.samples[idx]['obfuscated']

        if expr in self.fp_cache:
            fingerprint = self.fp_cache[expr]
        else:
            fingerprint = self.fingerprint_computer.compute_ml(expr)
            self.fp_cache[expr] = fingerprint

        return {'fingerprint': fingerprint, ...}
```

**Speedup**: ~10× for repeated expressions

### Multiprocessing Data Loading

```python
loader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,      # Parallel data loading
    pin_memory=True,    # Faster GPU transfer
    prefetch_factor=2   # Prefetch 2 batches per worker
)
```

**Speedup**: ~4× with 4 workers (CPU-bound tasks)

### On-the-Fly Graph Construction

For very large datasets, construct graphs during `__getitem__`:

```python
def __getitem__(self, idx):
    expr = self.samples[idx]['obfuscated']

    # Don't precompute graphs (save memory)
    graph = expression_to_graph(parse_expression(expr))

    return {'graph': graph, ...}
```

**Trade-off**: Slower per-sample, but lower memory footprint

---

## 12. Implementation Files

| Component | File | Lines |
|-----------|------|-------|
| AST Parser | `src/data/ast_parser.py` | 350 |
| Tokenizer | `src/data/tokenizer.py` | 150 |
| Fingerprint | `src/data/fingerprint.py` | 250 |
| Datasets | `src/data/dataset.py` | 450 |
| Collation | `src/data/collate.py` | 120 |
| Augmentation | `src/data/augmentation.py` | 80 |
| DAG Features | `src/data/dag_features.py` | 100 |
| Walsh-Hadamard | `src/data/walsh_hadamard.py` | 80 |
| Generator | `scripts/generate_data.py` | 200 |

---

## 13. Testing

```bash
# Test tokenizer
pytest tests/test_data.py::test_tokenizer -v

# Test fingerprint
pytest tests/test_data.py::test_fingerprint -v

# Test AST parser
pytest tests/test_data.py::test_ast_parser -v

# Test datasets
pytest tests/test_data.py::test_mba_dataset -v

# Test C++ compatibility
pytest tests/test_dataset_cpp_compat.py -v

# Validate fingerprint consistency
python scripts/validate_fingerprint_consistency.py
```

---

## 14. Common Issues & Solutions

### Issue: Fingerprint Mismatch (C++ vs Python)

**Symptom**: Different fingerprints for same expression

**Cause**: Derivatives computed differently

**Solution**: Use `compute_ml()` which strips derivatives
```python
fp = SemanticFingerprint()
vector = fp.compute_ml(expr)  # 416-dim, no derivatives
```

### Issue: Out of Memory During Training

**Symptom**: CUDA OOM errors

**Solutions**:
1. Reduce batch size
2. Use gradient accumulation
3. Enable fingerprint caching
4. Use mixed precision training (`torch.amp`)

### Issue: Slow Data Loading

**Symptom**: GPU underutilized, data loading bottleneck

**Solutions**:
1. Increase `num_workers` in DataLoader
2. Enable `pin_memory=True`
3. Use C++ fingerprint acceleration
4. Cache fingerprints to disk

### Issue: Variable Name Inconsistencies

**Symptom**: Model sees `y` and `x0` as different variables

**Solution**: Normalize variable names in tokenizer (already implemented)
```python
# Automatically renames variables by first appearance
tokenizer.normalize_variables("(y & z)") → "(x0 & x1)"
```

---

## 15. Best Practices

1. **Always strip derivatives** from fingerprints for ML
2. **Use C++ acceleration** for large-scale training (10× speedup)
3. **Cache fingerprints** for repeated expressions
4. **Normalize variables** to canonical form
5. **Augment with variable permutations** (80% probability)
6. **Use curriculum learning** (depth 2→5→10→14)
7. **Validate fingerprint consistency** before training
8. **Monitor data loading time** (should be <10% of training time)
9. **Use multiprocessing** for data loading (4-8 workers)
10. **Batch by similar lengths** to minimize padding overhead

---

## References

1. **AST Parsing**: Standard compiler techniques
2. **Graph Construction**: PyTorch Geometric Data format
3. **Fingerprinting**: Inspired by symbolic execution and hash functions
4. **Truth Tables**: Boolean function representation
5. **Variable Augmentation**: Data augmentation for invariance
6. **Curriculum Learning**: Bengio et al., "Curriculum Learning" (ICML 2009)
