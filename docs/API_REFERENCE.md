# API Reference

Technical reference for public APIs in the MBA Deobfuscation ML System.

## Table of Contents

- [Z3 Interface](#z3-interface)
- [Expression Evaluation](#expression-evaluation)
- [Metrics](#metrics)
- [Configuration](#configuration)
- [Logging](#logging)
- [Graph Utilities](#graph-utilities)
- [Ablation Study](#ablation-study)
- [Constants](#constants)

---

## Z3 Interface

**Module**: `src.utils.z3_interface`

Formal verification using Z3 SMT solver. All functions use bitvector theory for fixed-width arithmetic.

### Installation

```bash
pip install z3-solver
```

### Functions

#### `expr_to_z3`

```python
def expr_to_z3(
    expr: str,
    width: int = 64,
    var_cache: Optional[Dict[str, BitVecNumRef]] = None
) -> BitVecNumRef
```

Convert MBA expression string to Z3 bitvector expression.

**Parameters:**
- `expr` (str): Expression in Python syntax, e.g., `"(x&y)+(x^y)"`
- `width` (int): Bit width for bitvector (8, 16, 32, or 64). Default: 64
- `var_cache` (dict, optional): Dictionary for caching Z3 variables. Share across calls for consistency

**Returns:**
- Z3 bitvector expression

**Raises:**
- `ImportError`: If z3-solver not installed
- `ValueError`: If expression contains unsupported operators (e.g., `/`, `%`, `<<`, `>>`)
- `SyntaxError`: If expression is malformed

**Supported Operators:**
- Arithmetic: `+`, `-`, `*` (unary `-`)
- Bitwise: `&`, `|`, `^`, `~`

**Example:**

```python
from src.utils.z3_interface import expr_to_z3
from z3 import Solver, sat

var_cache = {}
z3_expr1 = expr_to_z3("x|y", 64, var_cache)
z3_expr2 = expr_to_z3("(x&y)+(x^y)", 64, var_cache)

solver = Solver()
solver.add(z3_expr1 != z3_expr2)
result = solver.check()  # unsat (expressions are equivalent)
```

---

#### `verify_equivalence`

```python
def verify_equivalence(
    expr1: str,
    expr2: str,
    width: int = 64,
    timeout_ms: int = 1000
) -> bool
```

Verify two expressions are semantically equivalent using Z3.

**Parameters:**
- `expr1` (str): First expression
- `expr2` (str): Second expression
- `width` (int): Bit width for verification. Default: 64
- `timeout_ms` (int): Z3 timeout in milliseconds. Default: 1000ms

**Returns:**
- `True`: Expressions are proven equivalent (unsat result from Z3)
- `False`: Not equivalent, timeout, or verification error

**Example:**

```python
from src.utils.z3_interface import verify_equivalence

# Basic MBA identity
assert verify_equivalence("x|y", "(x&y)+(x^y)")

# Non-equivalent expressions
assert not verify_equivalence("x&y", "x|y")

# Timeout handling (complex expression)
result = verify_equivalence(
    "((x&y)+(x^y))^((x|y)-(x&y))",
    "x",
    timeout_ms=5000
)
```

**Usage Pattern (3-Tier Verification):**

```python
# Tier 1: Syntax check
if not syntax_valid(pred):
    continue

# Tier 2: Execution test (fast)
if not expressions_equal(pred, target, num_samples=100):
    continue

# Tier 3: Z3 verification (formal)
if verify_equivalence(pred, target, timeout_ms=1000):
    return pred
```

---

#### `find_counterexample`

```python
def find_counterexample(
    expr1: str,
    expr2: str,
    width: int = 64,
    timeout_ms: int = 1000
) -> Optional[Dict[str, int]]
```

Find input values where expressions differ.

**Parameters:**
- `expr1` (str): First expression
- `expr2` (str): Second expression
- `width` (int): Bit width. Default: 64
- `timeout_ms` (int): Timeout in milliseconds. Default: 1000ms

**Returns:**
- `Dict[str, int]`: Variable bindings where expressions differ
- `None`: Expressions are equivalent or timeout

**Example:**

```python
from src.utils.z3_interface import find_counterexample

# Find counterexample for non-equivalent expressions
cex = find_counterexample("x&y", "x|y", width=8)
if cex:
    print(f"Counterexample: {cex}")
    # Example output: {'x': 1, 'y': 2}

    # Verify the counterexample
    from src.utils.expr_eval import evaluate_expr
    val1 = evaluate_expr("x&y", cex, width=8)
    val2 = evaluate_expr("x|y", cex, width=8)
    assert val1 != val2

# No counterexample for equivalent expressions
assert find_counterexample("x|y", "(x&y)+(x^y)") is None
```

---

## Expression Evaluation

**Module**: `src.utils.expr_eval`

Pure Python expression evaluation with modular arithmetic. Faster than Z3 for testing.

### Functions

#### `tokenize_expr`

```python
def tokenize_expr(expr: str) -> List[str]
```

Tokenize MBA expression string.

**Parameters:**
- `expr` (str): Expression string, whitespace is ignored

**Returns:**
- List of tokens: operators, variables, constants, parentheses

**Example:**

```python
from src.utils.expr_eval import tokenize_expr

tokens = tokenize_expr("(x & y) + 2")
# ['(', 'x', '&', 'y', ')', '+', '2']

tokens = tokenize_expr("x0+x1*x2")
# ['x0', '+', 'x1', '*', 'x2']
```

---

#### `parse_expr`

```python
def parse_expr(expr: str) -> ast.AST
```

Parse expression to Python AST.

**Parameters:**
- `expr` (str): Expression string

**Returns:**
- Python AST node (ast.Expression)

**Raises:**
- `SyntaxError`: If expression is malformed

**Example:**

```python
from src.utils.expr_eval import parse_expr

tree = parse_expr("x+y")
# ast.Expression(body=ast.BinOp(...))

# Invalid syntax
try:
    parse_expr("x++y")
except SyntaxError:
    print("Malformed expression")
```

---

#### `evaluate_expr`

```python
def evaluate_expr(
    expr: str,
    var_values: Dict[str, int],
    width: int = 64
) -> int
```

Evaluate MBA expression with variable bindings.

**Parameters:**
- `expr` (str): Expression string
- `var_values` (dict): Variable name to integer value mapping
- `width` (int): Bit width for modular arithmetic (8, 16, 32, or 64). Default: 64

**Returns:**
- Integer result in range `[0, 2^width)`

**Raises:**
- `ValueError`: If expression contains undefined variables
- `SyntaxError`: If expression is malformed
- `ZeroDivisionError`: If division by zero occurs

**Example:**

```python
from src.utils.expr_eval import evaluate_expr

# Basic evaluation
result = evaluate_expr("(x&y)+(x^y)", {"x": 5, "y": 3}, width=8)
assert result == 7  # Equivalent to x|y

# Modular arithmetic (overflow wrapping)
result = evaluate_expr("x+y", {"x": 255, "y": 1}, width=8)
assert result == 0  # Wraps at 256

# Undefined variable error
try:
    evaluate_expr("x+z", {"x": 5}, width=8)
except ValueError as e:
    print(e)  # "Undefined variable: z"
```

---

#### `random_inputs`

```python
def random_inputs(
    num_vars: int,
    width: int = 64,
    count: int = 1,
    var_names: Optional[List[str]] = None
) -> List[Dict[str, int]]
```

Generate random input dictionaries for testing.

**Parameters:**
- `num_vars` (int): Number of variables
- `width` (int): Bit width, values in `[0, 2^width)`. Default: 64
- `count` (int): Number of input dictionaries to generate. Default: 1
- `var_names` (list, optional): Variable names. Default: `['x', 'y', 'z']` or `['x0', 'x1', ...]`

**Returns:**
- List of dictionaries mapping variable names to random values

**Example:**

```python
from src.utils.expr_eval import random_inputs

# Generate 100 random test cases for 2 variables
inputs = random_inputs(num_vars=2, width=8, count=100)
# [{'x': 42, 'y': 173}, {'x': 5, 'y': 91}, ...]

# Custom variable names
inputs = random_inputs(
    num_vars=3,
    width=16,
    count=10,
    var_names=['a', 'b', 'c']
)
```

---

#### `expressions_equal`

```python
def expressions_equal(
    expr1: str,
    expr2: str,
    num_samples: int = 100,
    width: int = 64,
    var_names: Optional[List[str]] = None
) -> bool
```

Test if two expressions are equivalent via random sampling.

**Parameters:**
- `expr1` (str): First expression
- `expr2` (str): Second expression
- `num_samples` (int): Number of random test cases. Default: 100
- `width` (int): Bit width. Default: 64
- `var_names` (list, optional): Variable names. If None, extracts from expressions

**Returns:**
- `True`: All samples match (probabilistic equivalence)
- `False`: Found mismatch or evaluation error

**Example:**

```python
from src.utils.expr_eval import expressions_equal

# Fast probabilistic equivalence check
assert expressions_equal("x|y", "(x&y)+(x^y)", num_samples=100)

# More samples for higher confidence
assert expressions_equal(
    "((x&y)+(x^y))^((x|y)-(x&y))",
    "x",
    num_samples=1000
)

# Detect non-equivalence
assert not expressions_equal("x&y", "x|y", num_samples=10)
```

**Trade-off**: Faster than Z3 but probabilistic. Use for pre-filtering before Z3 verification.

---

## Metrics

**Module**: `src.utils.metrics`

Evaluation metrics for simplification quality.

### Functions

#### `exact_match`

```python
def exact_match(pred: str, target: str) -> bool
```

Check if prediction exactly matches target (normalized).

**Parameters:**
- `pred` (str): Predicted expression
- `target` (str): Target expression

**Returns:**
- `True`: Expressions match exactly after whitespace normalization

**Example:**

```python
from src.utils.metrics import exact_match

assert exact_match("x + y", "x+y")
assert exact_match("(x&y)", "(x&y)")
assert not exact_match("x|y", "(x&y)+(x^y)")
```

---

#### `z3_accuracy`

```python
def z3_accuracy(
    preds: List[str],
    targets: List[str],
    inputs: List[str],
    width: int = 64,
    timeout_ms: int = 1000
) -> float
```

Compute percentage of predictions that are Z3-verified equivalent to inputs.

**Parameters:**
- `preds` (list): Predicted simplified expressions
- `targets` (list): Target simplified expressions (not used, included for API consistency)
- `inputs` (list): Original obfuscated expressions
- `width` (int): Bit width. Default: 64
- `timeout_ms` (int): Z3 timeout per verification. Default: 1000ms

**Returns:**
- Float in `[0.0, 1.0]`: Fraction of predictions verified equivalent to inputs

**Example:**

```python
from src.utils.metrics import z3_accuracy

preds = ["x|y", "x&y", "x^y"]
inputs = ["(x&y)+(x^y)", "(x|y)-(x^y)", "x^y"]
targets = ["x|y", "x&y", "x^y"]  # Not used by this metric

accuracy = z3_accuracy(preds, targets, inputs)
# 0.667 (2 out of 3 correct)
```

**Usage Pattern (Evaluation Loop):**

```python
from src.utils.metrics import z3_accuracy, syntax_accuracy, avg_simplification_ratio

# Collect predictions
preds, targets, inputs = [], [], []
for batch in test_loader:
    pred = model.generate(batch)
    preds.extend(pred)
    targets.extend(batch['target'])
    inputs.extend(batch['input'])

# Compute metrics
z3_acc = z3_accuracy(preds, targets, inputs, timeout_ms=1000)
syntax_acc = syntax_accuracy(preds)
avg_ratio = avg_simplification_ratio(inputs, preds)

print(f"Z3 Accuracy: {z3_acc:.3f}")
print(f"Syntax Valid: {syntax_acc:.3f}")
print(f"Avg Simplification: {avg_ratio:.3f}")
```

---

#### `simplification_ratio`

```python
def simplification_ratio(input_expr: str, output_expr: str) -> float
```

Compute token count ratio: `len(output) / len(input)`.

**Parameters:**
- `input_expr` (str): Original expression
- `output_expr` (str): Simplified expression

**Returns:**
- Float: Ratio in `[0.0, inf)`. Lower is better

**Example:**

```python
from src.utils.metrics import simplification_ratio

ratio = simplification_ratio("(x&y)+(x^y)", "x|y")
# 0.2 (3 tokens / 15 tokens)

# No simplification
ratio = simplification_ratio("x+y", "x+y")
# 1.0

# Identity (worst case)
ratio = simplification_ratio("(x&y)+(x^y)", "(x&y)+(x^y)")
# 1.0
```

---

#### `avg_simplification_ratio`

```python
def avg_simplification_ratio(inputs: List[str], outputs: List[str]) -> float
```

Compute average simplification ratio across dataset.

**Parameters:**
- `inputs` (list): Input expressions
- `outputs` (list): Output expressions

**Returns:**
- Float: Average ratio. Lower indicates better simplification

**Example:**

```python
inputs = ["(x&y)+(x^y)", "x+y+z", "x&0"]
outputs = ["x|y", "x+y+z", "0"]

avg_ratio = avg_simplification_ratio(inputs, outputs)
# ~0.53
```

---

#### `syntax_valid`

```python
def syntax_valid(expr: str) -> bool
```

Check if expression parses without error.

**Parameters:**
- `expr` (str): Expression string

**Returns:**
- `True`: Expression is syntactically valid

**Example:**

```python
from src.utils.metrics import syntax_valid

assert syntax_valid("x+y")
assert syntax_valid("(x&y)+(x^y)")
assert not syntax_valid("x++y")
assert not syntax_valid("(x&y")
```

---

#### `syntax_accuracy`

```python
def syntax_accuracy(predictions: List[str]) -> float
```

Compute fraction of syntactically valid predictions.

**Parameters:**
- `predictions` (list): Predicted expressions

**Returns:**
- Float in `[0.0, 1.0]`: Fraction of valid predictions

**Example:**

```python
preds = ["x+y", "x++y", "x|y", "(x&"]
syntax_acc = syntax_accuracy(preds)
# 0.5 (2 out of 4 valid)
```

---

## Configuration

**Module**: `src.utils.config`

YAML configuration loading with dot notation access.

### Class: `Config`

#### Constructor

```python
def __init__(self, path: str)
```

Load configuration from YAML file.

**Parameters:**
- `path` (str): Path to YAML configuration file

**Raises:**
- `FileNotFoundError`: If config file doesn't exist
- `yaml.YAMLError`: If YAML is malformed

**Example:**

```python
from src.utils.config import Config

config = Config("configs/phase2.yaml")
```

---

#### Attribute Access

```python
config.attribute_name
```

Access config values using dot notation.

**Returns:**
- Config value, wrapped in `Config` if value is a dictionary

**Raises:**
- `AttributeError`: If key doesn't exist

**Example:**

```python
# configs/phase2.yaml:
# model:
#   hidden_dim: 256
#   num_layers: 4
# training:
#   batch_size: 32
#   lr: 0.0001

config = Config("configs/phase2.yaml")

# Dot notation access
hidden_dim = config.model.hidden_dim  # 256
batch_size = config.training.batch_size  # 32

# Nested dictionaries are wrapped
model_config = config.model
print(model_config.num_layers)  # 4
```

---

#### `get` Method

```python
def get(self, key: str, default: Any = None) -> Any
```

Get config value with default fallback. Supports nested keys with dot notation.

**Parameters:**
- `key` (str): Configuration key (supports `"parent.child"` syntax)
- `default` (any): Default value if key doesn't exist

**Returns:**
- Configuration value or default

**Example:**

```python
# Safe access with defaults
lr = config.get("training.lr", 0.001)
dropout = config.get("model.dropout", 0.1)

# Missing keys return default
val = config.get("nonexistent.key", None)  # None
```

---

#### `to_dict` Method

```python
def to_dict(self) -> dict
```

Convert config to dictionary.

**Returns:**
- Dictionary copy of configuration

**Example:**

```python
config_dict = config.to_dict()
print(config_dict)
# {'model': {'hidden_dim': 256, ...}, 'training': {...}}
```

---

### Complete Example

```python
from src.utils.config import Config

# Load config
config = Config("configs/phase2.yaml")

# Access training hyperparameters
batch_size = config.training.batch_size
learning_rate = config.training.lr
num_epochs = config.training.epochs

# Access model architecture
hidden_dim = config.model.encoder.hidden_dim
num_layers = config.model.encoder.num_layers

# Safe access with defaults
warmup_steps = config.get("training.warmup_steps", 1000)
gradient_clip = config.get("training.gradient_clip", 1.0)

# Pass to model constructor
model = MBAModel(
    hidden_dim=hidden_dim,
    num_layers=num_layers,
    dropout=config.get("model.dropout", 0.1)
)
```

---

## Logging

**Module**: `src.utils.logging`

Logging setup with optional Weights & Biases integration.

### Functions

#### `setup_logging`

```python
def setup_logging(name: str, level: str = "INFO") -> logging.Logger
```

Set up logging with consistent formatting.

**Parameters:**
- `name` (str): Logger name (typically `__name__`)
- `level` (str): Logging level. Options: `"DEBUG"`, `"INFO"`, `"WARNING"`, `"ERROR"`, `"CRITICAL"`. Default: `"INFO"`

**Returns:**
- Configured `logging.Logger` instance

**Example:**

```python
from src.utils.logging import setup_logging

logger = setup_logging(__name__, level="DEBUG")

logger.debug("Detailed debug information")
logger.info("Training started")
logger.warning("Learning rate decay triggered")
logger.error("Validation loss diverged")
```

**Output Format:**

```
2025-01-16 10:30:45 - src.training.trainer - INFO - Training started
```

---

#### `setup_wandb`

```python
def setup_wandb(
    project: str,
    config: Dict[str, Any],
    enabled: bool = True,
    name: Optional[str] = None,
    tags: Optional[List[str]] = None,
    notes: Optional[str] = None
) -> Optional[wandb.Run]
```

Initialize Weights & Biases experiment tracking.

**Parameters:**
- `project` (str): WandB project name
- `config` (dict): Configuration dictionary to log
- `enabled` (bool): Whether to enable WandB. Default: `True`
- `name` (str, optional): Run name. If None, WandB generates one
- `tags` (list, optional): List of tags for the run
- `notes` (str, optional): Text notes for the run

**Returns:**
- `wandb.Run` object if enabled and available
- `None` if disabled or WandB not installed

**Example:**

```python
from src.utils.config import Config
from src.utils.logging import setup_logging, setup_wandb

# Setup
logger = setup_logging(__name__)
config = Config("configs/phase2.yaml")

# Initialize WandB
wandb_run = setup_wandb(
    project="mba-deobfuscator",
    config=config.to_dict(),
    enabled=True,
    name="phase2-supervised-run1",
    tags=["phase2", "supervised", "gat"],
    notes="Supervised training with GAT encoder"
)

# Log metrics during training
if wandb_run:
    import wandb
    wandb.log({"train_loss": 0.45, "epoch": 1})
    wandb.log({"val_z3_accuracy": 0.87, "epoch": 1})
```

**Dry Run (Disable WandB):**

```python
# Disable WandB for local testing
wandb_run = setup_wandb(
    project="mba-deobfuscator",
    config=config.to_dict(),
    enabled=False  # No WandB calls made
)
```

---

### Complete Training Setup

```python
from src.utils.logging import setup_logging, setup_wandb
from src.utils.config import Config

# Initialize logging
logger = setup_logging(__name__, level="INFO")

# Load config
config = Config("configs/phase2.yaml")

# Initialize WandB
wandb_run = setup_wandb(
    project="mba-deobfuscator",
    config=config.to_dict(),
    name=f"phase2-{config.model.encoder.type}",
    tags=["phase2", config.model.encoder.type],
)

# Training loop
for epoch in range(config.training.epochs):
    logger.info(f"Epoch {epoch+1}/{config.training.epochs}")

    train_loss = train_epoch(model, train_loader)
    val_metrics = evaluate(model, val_loader)

    logger.info(f"Train Loss: {train_loss:.4f}")
    logger.info(f"Val Z3 Accuracy: {val_metrics['z3_accuracy']:.4f}")

    # Log to WandB
    if wandb_run:
        import wandb
        wandb.log({
            "train_loss": train_loss,
            "val_z3_accuracy": val_metrics['z3_accuracy'],
            "val_syntax_accuracy": val_metrics['syntax_accuracy'],
            "epoch": epoch + 1
        })
```

---

## Graph Utilities

**Module**: `src.utils.graph_utils`

Graph traversal utilities for AST processing in ablation encoders.

### Functions

#### `compute_dfs_order`

```python
def compute_dfs_order(
    edge_index: torch.Tensor,
    node_mask: torch.Tensor,
    validate_acyclic: bool = True
) -> torch.Tensor
```

Compute DFS traversal order for subgraph.

**Parameters:**
- `edge_index` (Tensor): Shape `[2, num_edges]`, full edge index
- `node_mask` (Tensor): Shape `[total_nodes]`, boolean mask for subgraph
- `validate_acyclic` (bool): If True, raise error on cycles. Default: `True`

**Returns:**
- Tensor of shape `[num_subgraph_nodes]` with local indices in DFS order

**Raises:**
- `ValueError`: If graph has cycles (when `validate_acyclic=True`)
- `ValueError`: If no root node found (all nodes have incoming edges)

**Example:**

```python
import torch
from src.utils.graph_utils import compute_dfs_order

# Full graph with 5 nodes
edge_index = torch.tensor([
    [0, 0, 1, 2],  # source nodes
    [1, 2, 3, 4]   # target nodes
])

# Subgraph mask (include nodes 0, 1, 3)
node_mask = torch.tensor([True, True, False, True, False])

# Compute DFS order
dfs_order = compute_dfs_order(edge_index, node_mask, validate_acyclic=True)
# Tensor([0, 1, 2]) - local indices in DFS order

# Cycle detection
cyclic_edges = torch.tensor([[0, 1], [1, 0]])
node_mask = torch.tensor([True, True])
try:
    compute_dfs_order(cyclic_edges, node_mask, validate_acyclic=True)
except ValueError as e:
    print(e)  # "Cycle detected in graph: [0, 1, 0]"
```

---

#### `compute_bfs_order`

```python
def compute_bfs_order(
    edge_index: torch.Tensor,
    node_mask: torch.Tensor
) -> torch.Tensor
```

Compute BFS traversal order for subgraph. Fallback for cyclic graphs.

**Parameters:**
- `edge_index` (Tensor): Shape `[2, num_edges]`
- `node_mask` (Tensor): Shape `[total_nodes]`, boolean mask

**Returns:**
- Tensor of shape `[num_subgraph_nodes]` with local indices in BFS order

**Example:**

```python
from src.utils.graph_utils import compute_bfs_order

bfs_order = compute_bfs_order(edge_index, node_mask)
# Tensor([0, 1, 2]) - local indices in BFS order
```

---

#### `safe_dfs_order`

```python
def safe_dfs_order(
    edge_index: torch.Tensor,
    node_mask: torch.Tensor
) -> torch.Tensor
```

Compute traversal order with fallback to BFS on cycle detection. **Recommended for production**.

**Parameters:**
- `edge_index` (Tensor): Shape `[2, num_edges]`
- `node_mask` (Tensor): Shape `[total_nodes]`, boolean mask

**Returns:**
- Tensor of shape `[num_subgraph_nodes]` with local indices in traversal order

**Example:**

```python
from src.utils.graph_utils import safe_dfs_order

# Handles both acyclic and cyclic graphs
traversal_order = safe_dfs_order(edge_index, node_mask)
```

---

#### `validate_ast_structure`

```python
def validate_ast_structure(
    edge_index: torch.Tensor,
    node_mask: torch.Tensor
) -> Tuple[bool, Optional[str]]
```

Validate that subgraph is a valid AST (tree structure).

**Parameters:**
- `edge_index` (Tensor): Shape `[2, num_edges]`
- `node_mask` (Tensor): Shape `[total_nodes]`, boolean mask

**Returns:**
- Tuple of `(is_valid, error_message)`
  - `error_message` is `None` if valid

**Validation Checks:**
1. Tree property: `num_edges == num_nodes - 1`
2. Single root: exactly one node with no incoming edges
3. Connectivity: DFS reaches all nodes
4. Acyclic: no cycles detected

**Example:**

```python
from src.utils.graph_utils import validate_ast_structure

# Valid tree
edge_index = torch.tensor([[0, 0], [1, 2]])
node_mask = torch.tensor([True, True, True])
is_valid, error = validate_ast_structure(edge_index, node_mask)
assert is_valid  # True
assert error is None

# Invalid: too many edges
edge_index = torch.tensor([[0, 0, 1], [1, 2, 2]])
is_valid, error = validate_ast_structure(edge_index, node_mask)
assert not is_valid
print(error)  # "Expected 2 edges for tree, got 3"
```

---

## Ablation Study

Metrics collection and statistical analysis for encoder ablation studies.

### Metrics Collection

**Module**: `src.utils.ablation_metrics`

#### Class: `EncoderMetrics`

Dataclass for per-encoder metrics.

**Fields:**

```python
@dataclass
class EncoderMetrics:
    encoder_name: str           # Encoder architecture name
    run_id: int                 # Run number (1-5)
    depth_bucket: str           # "2-4", "5-7", etc.

    # Accuracy
    exact_match: float          # Token-level exact match
    z3_equivalence: float       # Z3 equivalence rate
    syntax_valid: float         # Syntactically valid outputs

    # Simplification quality
    simplification_ratio: float # avg(output_len / input_len)
    avg_output_depth: float     # Average AST depth of outputs
    identity_rate: float        # How often output == input

    # Performance
    inference_latency_ms: float # Mean per-sample latency
    parameter_count: int        # Total encoder parameters
    training_time_hours: float  # Total training time

    # Sample counts
    num_samples: int            # Total samples in bucket
    num_correct: int            # Number of Z3-verified correct
```

---

#### Class: `AblationMetricsCollector`

Collects and aggregates metrics across depth buckets and runs.

**Constructor:**

```python
def __init__(self, depth_buckets: List[Tuple[int, int]])
```

**Parameters:**
- `depth_buckets` (list): List of `(min_depth, max_depth)` tuples

**Example:**

```python
from src.utils.ablation_metrics import AblationMetricsCollector

collector = AblationMetricsCollector(
    depth_buckets=[(2, 4), (5, 7), (8, 10), (11, 14)]
)
```

---

**Method: `collect`**

```python
def collect(
    self,
    encoder_name: str,
    run_id: int,
    predictions: List[str],
    targets: List[str],
    inputs: List[str],
    depths: List[int],
    latencies: List[float],
    encoder_params: int,
    training_time_hours: float
) -> None
```

Collect metrics from evaluation run.

**Parameters:**
- `encoder_name` (str): Encoder architecture name (e.g., `"GAT"`, `"GGNN"`)
- `run_id` (int): Run number (1-5)
- `predictions` (list): Model outputs
- `targets` (list): Ground truth simplified expressions
- `inputs` (list): Input obfuscated expressions
- `depths` (list): Expression depth for each sample
- `latencies` (list): Per-sample inference latency (seconds)
- `encoder_params` (int): Total encoder parameters
- `training_time_hours` (float): Total training time

**Raises:**
- `ValueError`: If input lists have mismatched lengths

**Example:**

```python
# After evaluation
preds, targets, inputs, depths, latencies = evaluate_model(model, test_loader)

collector.collect(
    encoder_name="GAT",
    run_id=1,
    predictions=preds,
    targets=targets,
    inputs=inputs,
    depths=depths,
    latencies=latencies,
    encoder_params=2_800_000,
    training_time_hours=3.5
)
```

---

**Method: `save_results`**

```python
def save_results(self, output_path: str) -> None
```

Save results to JSON.

---

**Method: `load_results`**

```python
def load_results(self, input_path: str) -> None
```

Load results from JSON.

---

**Method: `aggregate_by_encoder`**

```python
def aggregate_by_encoder(self) -> Dict[str, Dict[str, Dict]]
```

Aggregate metrics across runs for each encoder.

**Returns:**
- Dictionary: `{encoder_bucket_key: {metric_name: {mean, std, runs: [values]}}}`

**Example:**

```python
aggregated = collector.aggregate_by_encoder()
# {
#   'GAT_2-4': {
#     'z3_equivalence': {'mean': 0.95, 'std': 0.02, 'runs': [0.94, 0.96, ...]},
#     ...
#   },
#   ...
# }
```

---

**Method: `summary_table`**

```python
def summary_table(self) -> str
```

Generate summary table for all encoders.

**Returns:**
- Formatted string table

**Example:**

```python
print(collector.summary_table())
```

**Output:**

```
Encoder Ablation Study Results
============================================================
Encoder_Bucket                 Z3 Acc       Exact        Simp Ratio
------------------------------------------------------------
GAT_2-4                        0.950±0.020  0.920±0.015  0.350±0.050
GAT_5-7                        0.880±0.030  0.850±0.025  0.450±0.040
GGNN_2-4                       0.930±0.025  0.900±0.020  0.380±0.055
...
```

---

### Statistical Analysis

**Module**: `src.utils.ablation_stats`

#### Functions

#### `paired_t_test`

```python
def paired_t_test(
    encoder_a_results: List[float],
    encoder_b_results: List[float],
    alpha: float = 0.05
) -> Tuple[float, float, bool]
```

Paired t-test between two encoder results.

**Parameters:**
- `encoder_a_results` (list): Metric values from encoder A (typically 5 runs)
- `encoder_b_results` (list): Metric values from encoder B (same length)
- `alpha` (float): Significance level. Default: 0.05

**Returns:**
- Tuple of `(t_statistic, p_value, is_significant)`

**Raises:**
- `ValueError`: If result lists have different lengths or < 2 samples

**Example:**

```python
from src.utils.ablation_stats import paired_t_test

gat_results = [0.94, 0.96, 0.95, 0.93, 0.97]
ggnn_results = [0.89, 0.91, 0.90, 0.88, 0.92]

t_stat, p_value, significant = paired_t_test(gat_results, ggnn_results)
print(f"t={t_stat:.3f}, p={p_value:.4f}, sig={significant}")
# t=5.123, p=0.0067, sig=True
```

---

#### `compute_effect_size`

```python
def compute_effect_size(
    encoder_a_results: List[float],
    encoder_b_results: List[float]
) -> float
```

Compute Cohen's d effect size.

**Parameters:**
- `encoder_a_results` (list): Metric values from encoder A
- `encoder_b_results` (list): Metric values from encoder B

**Returns:**
- Float: Cohen's d (positive means A > B)

**Interpretation:**
- `|d| < 0.2`: negligible
- `0.2 <= |d| < 0.5`: small
- `0.5 <= |d| < 0.8`: medium
- `|d| >= 0.8`: large

**Example:**

```python
from src.utils.ablation_stats import compute_effect_size, interpret_effect_size

effect_size = compute_effect_size(gat_results, ggnn_results)
interpretation = interpret_effect_size(effect_size)
print(f"Cohen's d = {effect_size:.3f} ({interpretation})")
# Cohen's d = 1.234 (large)
```

---

#### `pairwise_comparison`

```python
def pairwise_comparison(
    aggregated_results: Dict[str, Dict],
    metric: str = "z3_equivalence",
    alpha: float = 0.05
) -> Dict[Tuple[str, str], Dict]
```

Perform pairwise statistical tests across all encoder pairs.

**Parameters:**
- `aggregated_results` (dict): Output from `AblationMetricsCollector.aggregate_by_encoder()`
- `metric` (str): Metric to compare. Default: `"z3_equivalence"`
- `alpha` (float): Significance level. Default: 0.05

**Returns:**
- Dictionary: `{(encoder_a, encoder_b): {t_statistic, p_value, significant, effect_size, effect_interpretation, winner}}`

**Example:**

```python
from src.utils.ablation_stats import pairwise_comparison

aggregated = collector.aggregate_by_encoder()
comparisons = pairwise_comparison(aggregated, metric="z3_equivalence")

for (enc_a, enc_b), result in comparisons.items():
    print(f"{enc_a} vs {enc_b}:")
    print(f"  p-value: {result['p_value']:.4f}")
    print(f"  Effect: {result['effect_interpretation']}")
    print(f"  Winner: {result['winner']}")
```

---

#### `rank_encoders`

```python
def rank_encoders(
    aggregated_results: Dict[str, Dict],
    metric: str = "z3_equivalence"
) -> List[Tuple[str, float, float]]
```

Rank encoders by mean performance.

**Parameters:**
- `aggregated_results` (dict): Output from `aggregate_by_encoder()`
- `metric` (str): Metric to rank by

**Returns:**
- List of `(encoder_name, mean, std)` sorted by mean (descending)

**Example:**

```python
from src.utils.ablation_stats import rank_encoders

rankings = rank_encoders(aggregated, metric="z3_equivalence")
for rank, (encoder, mean, std) in enumerate(rankings, 1):
    print(f"{rank}. {encoder}: {mean:.4f} +/- {std:.4f}")
```

---

#### `generate_comparison_report`

```python
def generate_comparison_report(
    aggregated_results: Dict[str, Dict],
    metric: str = "z3_equivalence",
    alpha: float = 0.05
) -> str
```

Generate text report of pairwise comparisons.

**Parameters:**
- `aggregated_results` (dict): Output from `aggregate_by_encoder()`
- `metric` (str): Metric to compare
- `alpha` (float): Significance level

**Returns:**
- Formatted string report

**Example:**

```python
report = generate_comparison_report(aggregated, metric="z3_equivalence")
print(report)
with open("ablation_report.txt", "w") as f:
    f.write(report)
```

---

### Complete Ablation Study Example

```python
from src.utils.ablation_metrics import AblationMetricsCollector
from src.utils.ablation_stats import generate_comparison_report

# Initialize collector
collector = AblationMetricsCollector(
    depth_buckets=[(2, 4), (5, 7), (8, 10), (11, 14)]
)

# Collect results from 5 runs of each encoder
encoders = ["GAT", "GGNN", "GCN", "TreeLSTM"]
for encoder_name in encoders:
    for run_id in range(1, 6):
        # Train and evaluate model
        model = train_encoder(encoder_name, seed=run_id)
        preds, targets, inputs, depths, latencies = evaluate(model, test_loader)

        # Collect metrics
        collector.collect(
            encoder_name=encoder_name,
            run_id=run_id,
            predictions=preds,
            targets=targets,
            inputs=inputs,
            depths=depths,
            latencies=latencies,
            encoder_params=model.count_parameters(),
            training_time_hours=model.training_time
        )

# Save results
collector.save_results("ablation_results.json")

# Generate report
aggregated = collector.aggregate_by_encoder()
report = generate_comparison_report(aggregated, metric="z3_equivalence")
print(report)

# Summary table
print(collector.summary_table())
```

---

## Constants

**Module**: `src.constants`

Global constants for dimension consistency across all modules.

### AST Node Representation

```python
NODE_TYPES: Dict[str, int]        # Node type to index mapping
NODE_TYPE_TO_STR: Dict[int, str]  # Index to node type mapping
NUM_NODE_TYPES: int = 10          # Total node types
NODE_DIM: int = 32                # Node feature dimension
```

**Node Types:**
- `VAR` (0): Variables (x, y, z, ...)
- `CONST` (1): Constants (0, 1, 2, ...)
- `ADD` (2): +
- `SUB` (3): -
- `MUL` (4): *
- `AND` (5): &
- `OR` (6): |
- `XOR` (7): ^
- `NOT` (8): ~
- `NEG` (9): unary -

---

### Edge Types (GGNN)

```python
EDGE_TYPES: Dict[str, int]   # Edge type to index mapping
NUM_EDGE_TYPES: int = 6      # Total edge types
```

**Edge Types:**
- `CHILD_LEFT` (0): parent → left operand
- `CHILD_RIGHT` (1): parent → right operand
- `PARENT` (2): child → parent
- `SIBLING_NEXT` (3): left → right sibling
- `SIBLING_PREV` (4): right → left sibling
- `SAME_VAR` (5): links all uses of same variable

---

### Tokenizer / Vocabulary

```python
SPECIAL_TOKENS: Dict[str, int]   # Special token to index
PAD_IDX: int = 0
SOS_IDX: int = 1
EOS_IDX: int = 2
UNK_IDX: int = 3

OPERATORS: List[str] = ['+', '-', '*', '&', '|', '^', '~']
PARENS: List[str] = ['(', ')']

MAX_VARS: int = 8          # x0 through x7
MAX_SEQ_LEN: int = 64      # Maximum output sequence length
MAX_CONST: int = 256       # Constants 0-255
VOCAB_SIZE: int = 300      # Total vocabulary size
```

---

### Semantic Fingerprint

```python
SYMBOLIC_DIM: int = 32           # Symbolic features
CORNER_DIM: int = 256            # 4 widths × 64 corner cases
RANDOM_DIM: int = 64             # 4 widths × 16 hash values
DERIVATIVE_DIM: int = 32         # 4 widths × 8 derivative orders
TRUTH_TABLE_DIM: int = 64        # 2^6 entries for 6 variables

FINGERPRINT_DIM: int = 448       # Total fingerprint dimension
TRUTH_TABLE_VARS: int = 6        # Variables for truth table
BIT_WIDTHS: List[int] = [8, 16, 32, 64]
```

**Function:**

```python
def get_corner_values(width: int) -> List[int]
```

Get corner case values for a given bit width (0, 1, max, powers of 2, etc.).

---

### Model Dimensions

```python
# Encoder (GNN)
HIDDEN_DIM: int = 256
NUM_ENCODER_LAYERS: int = 4
NUM_ENCODER_HEADS: int = 8
ENCODER_DROPOUT: float = 0.1

# GGNN specific
GGNN_TIMESTEPS: int = 8

# Decoder (Transformer)
D_MODEL: int = 512
NUM_DECODER_LAYERS: int = 6
NUM_DECODER_HEADS: int = 8
D_FF: int = 2048
DECODER_DROPOUT: float = 0.1

# Output heads
MAX_OUTPUT_LENGTH: int = 64
MAX_OUTPUT_DEPTH: int = 16
```

---

### Training Hyperparameters

#### Phase 1: Contrastive

```python
INFONCE_TEMPERATURE: float = 0.07
MASKLM_MASK_RATIO: float = 0.15
MASKLM_WEIGHT: float = 0.5
```

#### Phase 2: Supervised

```python
CE_WEIGHT: float = 1.0
COMPLEXITY_WEIGHT: float = 0.1
COPY_WEIGHT: float = 0.1
```

**Curriculum:**

```python
CURRICULUM_STAGES: List[Dict] = [
    {'max_depth': 2, 'epochs': 10, 'target': 0.95},
    {'max_depth': 5, 'epochs': 15, 'target': 0.90},
    {'max_depth': 10, 'epochs': 15, 'target': 0.80},
    {'max_depth': 14, 'epochs': 10, 'target': 0.70},
]
```

**Self-Paced Learning:**

```python
SELF_PACED_LAMBDA_INIT: float = 0.5
SELF_PACED_LAMBDA_GROWTH: float = 1.1
```

#### Phase 3: RL (PPO)

```python
PPO_EPSILON: float = 0.2
PPO_VALUE_COEF: float = 0.5
PPO_ENTROPY_COEF: float = 0.01

# Reward function
REWARD_EQUIV_BONUS: float = 10.0
REWARD_LEN_PENALTY: float = 0.1
REWARD_DEPTH_PENALTY: float = 0.2
REWARD_IDENTITY_PENALTY: float = 5.0
REWARD_SYNTAX_ERROR_PENALTY: float = 5.0
REWARD_SIMPLIFICATION_BONUS: float = 2.0
REWARD_IDENTITY_THRESHOLD: float = 0.9
```

---

### Inference Parameters

```python
# Beam search
BEAM_WIDTH: int = 50
BEAM_DIVERSITY_GROUPS: int = 4
BEAM_DIVERSITY_PENALTY: float = 0.5
BEAM_TEMPERATURE: float = 0.7

# HTPS
HTPS_BUDGET: int = 500
HTPS_DEPTH_THRESHOLD: int = 10
HTPS_UCB_CONSTANT: float = 1.414

# Verification
EXEC_TEST_SAMPLES: int = 100
Z3_TIMEOUT_MS: int = 1000
Z3_TOP_K: int = 10
```

---

### HTPS Tactics

```python
HTPS_TACTICS: List[str] = [
    'identity_xor_self',    # x ^ x -> 0
    'identity_and_not',     # x & ~x -> 0
    'identity_or_not',      # x | ~x -> -1
    'mba_and_xor',          # (x&y)+(x^y) -> x|y
    'constant_fold',        # 3 + 5 -> 8
    'simplify_subexpr',     # Recurse on subexpression
]
```

---

### Ablation Study Parameters

```python
ABLATION_DEPTH_BUCKETS: List[Tuple[int, int]] = [
    (2, 4),    # Easy
    (5, 7),    # Medium
    (8, 10),   # Hard
    (11, 14),  # Very hard
]

ABLATION_NUM_RUNS: int = 5
ABLATION_SIGNIFICANCE_LEVEL: float = 0.05
```

---

### Scaled Model Dimensions (360M parameters)

For Chinchilla-optimal 12M sample dataset:

```python
# Encoder (~60M params)
SCALED_HIDDEN_DIM: int = 768
SCALED_NUM_ENCODER_LAYERS: int = 12
SCALED_NUM_ENCODER_HEADS: int = 16

# Decoder (~302M params)
SCALED_D_MODEL: int = 1536
SCALED_NUM_DECODER_LAYERS: int = 8
SCALED_NUM_DECODER_HEADS: int = 24
SCALED_D_FF: int = 6144

SCALED_MAX_SEQ_LEN: int = 2048
```

---

### Usage Example

```python
from src.constants import (
    NODE_TYPES, FINGERPRINT_DIM, HIDDEN_DIM, D_MODEL,
    CURRICULUM_STAGES, BEAM_WIDTH, Z3_TIMEOUT_MS
)

# Model construction
encoder = GATEncoder(
    node_types=len(NODE_TYPES),
    hidden_dim=HIDDEN_DIM,
    num_layers=4
)

decoder = TransformerDecoder(
    d_model=D_MODEL,
    encoder_dim=HIDDEN_DIM + FINGERPRINT_DIM
)

# Training loop
for stage in CURRICULUM_STAGES:
    train_subset = filter_by_depth(dataset, max_depth=stage['max_depth'])
    train_epochs(model, train_subset, num_epochs=stage['epochs'])

# Inference
candidates = beam_search(model, input_expr, beam_width=BEAM_WIDTH)
for candidate in candidates[:10]:
    if verify_equivalence(input_expr, candidate, timeout_ms=Z3_TIMEOUT_MS):
        return candidate
```

---

## Common Usage Patterns

### 3-Tier Verification Pipeline

```python
from src.utils.metrics import syntax_valid
from src.utils.expr_eval import expressions_equal
from src.utils.z3_interface import verify_equivalence

def verify_prediction(pred: str, target: str, input_expr: str) -> bool:
    """3-tier verification: syntax → execution → Z3."""

    # Tier 1: Syntax check (fast)
    if not syntax_valid(pred):
        return False

    # Tier 2: Execution test (100 random samples, fast)
    if not expressions_equal(pred, input_expr, num_samples=100):
        return False

    # Tier 3: Z3 verification (formal, slower)
    return verify_equivalence(pred, input_expr, timeout_ms=1000)
```

---

### Beam Search + Verification

```python
from src.inference.beam_search import beam_search
from src.utils.z3_interface import verify_equivalence

def simplify_with_beam(model, input_expr: str, beam_width: int = 50):
    """Beam search with Z3 reranking."""

    # Generate candidates
    candidates = beam_search(model, input_expr, beam_width=beam_width)

    # Verify top-10 candidates
    for candidate in candidates[:10]:
        if verify_equivalence(input_expr, candidate, timeout_ms=1000):
            return candidate

    # Fallback to top beam result
    return candidates[0] if candidates else input_expr
```

---

### Training with WandB + Config

```python
from src.utils.config import Config
from src.utils.logging import setup_logging, setup_wandb

def train_phase2():
    # Setup
    logger = setup_logging(__name__)
    config = Config("configs/phase2.yaml")

    wandb_run = setup_wandb(
        project="mba-deobfuscator",
        config=config.to_dict(),
        name="phase2-gat-supervised"
    )

    # Training loop
    for epoch in range(config.training.epochs):
        train_loss = train_epoch(model, train_loader, config)
        val_metrics = evaluate(model, val_loader)

        logger.info(f"Epoch {epoch+1}: loss={train_loss:.4f}, z3_acc={val_metrics['z3_accuracy']:.3f}")

        if wandb_run:
            import wandb
            wandb.log({
                "train_loss": train_loss,
                "val_z3_accuracy": val_metrics['z3_accuracy'],
                "epoch": epoch + 1
            })
```

---

### Ablation Study Pipeline

```python
from src.utils.ablation_metrics import AblationMetricsCollector
from src.utils.ablation_stats import generate_comparison_report

# Initialize
collector = AblationMetricsCollector(
    depth_buckets=[(2, 4), (5, 7), (8, 10), (11, 14)]
)

# Collect from 5 runs per encoder
for encoder_name in ["GAT", "GGNN", "GCN"]:
    for run_id in range(1, 6):
        model = train_and_evaluate(encoder_name, seed=run_id)
        preds, targets, inputs, depths, latencies = get_predictions(model)

        collector.collect(
            encoder_name=encoder_name,
            run_id=run_id,
            predictions=preds,
            targets=targets,
            inputs=inputs,
            depths=depths,
            latencies=latencies,
            encoder_params=model.count_parameters(),
            training_time_hours=model.training_time
        )

# Generate report
aggregated = collector.aggregate_by_encoder()
report = generate_comparison_report(aggregated)
print(report)
```

---

## Installation Requirements

```bash
# Core dependencies
pip install torch torch-geometric numpy pyyaml

# Z3 solver (optional, for formal verification)
pip install z3-solver

# Weights & Biases (optional, for experiment tracking)
pip install wandb

# Statistical analysis (optional, scipy improves accuracy)
pip install scipy
```

---

## Error Handling

### Z3 Not Available

```python
from src.utils.z3_interface import verify_equivalence, Z3_AVAILABLE

if not Z3_AVAILABLE:
    print("Warning: Z3 not installed. Using probabilistic verification.")
    result = expressions_equal(expr1, expr2, num_samples=1000)
else:
    result = verify_equivalence(expr1, expr2)
```

### WandB Not Available

```python
from src.utils.logging import setup_wandb, WANDB_AVAILABLE

wandb_run = setup_wandb(
    project="mba-deobfuscator",
    config=config.to_dict(),
    enabled=WANDB_AVAILABLE  # Auto-disable if not installed
)
```

### Configuration File Missing

```python
from src.utils.config import Config

try:
    config = Config("configs/phase2.yaml")
except FileNotFoundError:
    print("Config not found, using defaults")
    config = get_default_config()
```

---

## Performance Tips

1. **Z3 Timeout**: Set `timeout_ms=1000` for interactive use, `timeout_ms=5000` for batch evaluation
2. **Execution Tests**: Use `num_samples=100` for fast pre-filtering, `num_samples=1000` for higher confidence
3. **Beam Search**: Use `beam_width=50` for depth ≤6, `beam_width=10` for depth ≥10
4. **Verification Order**: Always run syntax → execution → Z3 to minimize expensive Z3 calls
5. **Graph Traversal**: Use `safe_dfs_order()` for robustness, `compute_dfs_order()` for strict validation

---

## See Also

- [ML Pipeline Documentation](ML_PIPELINE.md) - Detailed component specifications
- [ML Workflow Guide](ML_WORKFLOW.md) - Training pipeline overview
- [Project Structure](DIRECTORY.md) - Full directory layout
- [Main README](../README.md) - Quick start and commands
