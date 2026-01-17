# API Reference

Complete public API documentation for the MBA Deobfuscator system.

---

## Table of Contents

1. [Model Creation](#model-creation)
2. [Inference](#inference)
3. [Data Processing](#data-processing)
4. [Training](#training)
5. [Verification](#verification)
6. [Utilities](#utilities)
7. [Configuration](#configuration)

---

## Model Creation

### `MBADeobfuscator`

Complete end-to-end model for expression simplification.

```python
from src.models.full_model import MBADeobfuscator

model = MBADeobfuscator(
    encoder_type='gat_jknet',
    hidden_dim=256,
    num_layers=4,
    num_heads=8,
    dropout=0.1,
    decoder_dim=512,
    decoder_layers=6,
    decoder_heads=8,
    decoder_ffn_dim=2048,
    fingerprint_dim=416,
    vocab_size=300,
    max_seq_len=256,
    use_copy_mechanism=True
)
```

**Parameters**:
- `encoder_type` (str): Encoder architecture
  - Options: `'gat_jknet'`, `'ggnn'`, `'hgt'`, `'rgcn'`, `'semantic_hgt'`, `'transformer_only'`, `'hybrid_great'`, `'hgt_gmn'`, `'gat_gmn'`
- `hidden_dim` (int): Encoder hidden dimension (default: 256)
- `num_layers` (int): Number of encoder layers (default: 4)
- `num_heads` (int): Number of attention heads (default: 8)
- `dropout` (float): Dropout probability (default: 0.1)
- `decoder_dim` (int): Decoder hidden dimension (default: 512)
- `decoder_layers` (int): Number of decoder layers (default: 6)
- `decoder_heads` (int): Number of decoder attention heads (default: 8)
- `decoder_ffn_dim` (int): Decoder FFN dimension (default: 2048)
- `fingerprint_dim` (int): Semantic fingerprint dimension (default: 416)
- `vocab_size` (int): Vocabulary size (default: 300)
- `max_seq_len` (int): Maximum sequence length (default: 256)
- `use_copy_mechanism` (bool): Enable copy mechanism (default: True)

**Methods**:

```python
# Forward pass
output = model(
    graph: Data,              # PyG graph
    fingerprint: Tensor,      # [416]
    target_tokens: Tensor     # [seq_len] (optional, for teacher forcing)
)
# Returns: {
#     'token_logits': Tensor,      # [seq_len × vocab_size]
#     'complexity_pred': Tensor,   # [2] (length, depth)
#     'value': Tensor,             # [1] (critic value)
#     'copy_gate': Tensor          # [seq_len] (copy probabilities)
# }

# Load checkpoint
model.load_checkpoint(path: str)

# Save checkpoint
model.save_checkpoint(path: str)
```

---

### `get_encoder`

Factory function for creating encoders.

```python
from src.models.encoder_registry import get_encoder

encoder = get_encoder(
    encoder_type='hgt',
    hidden_dim=768,
    num_layers=12,
    num_heads=16,
    dropout=0.1,
    edge_types=8,
    num_node_types=10
)
```

**Parameters**:
- `encoder_type` (str): Encoder architecture name
- `hidden_dim` (int): Hidden dimension
- `num_layers` (int): Number of layers
- `num_heads` (int): Number of attention heads
- `dropout` (float): Dropout probability
- `edge_types` (int): Number of edge types (for relational encoders)
- `num_node_types` (int): Number of node types (default: 10)

**Returns**: `BaseEncoder` instance

**Methods**:

```python
# Forward pass
node_embeddings = encoder(data: Data)
# Returns: Tensor [num_nodes × hidden_dim]

# Get output dimension
output_dim = encoder.get_output_dim()
# Returns: int
```

---

### `list_encoders`

List all available encoder architectures.

```python
from src.models.encoder_registry import list_encoders

encoders = list_encoders()
# Returns: ['gat_jknet', 'ggnn', 'hgt', 'rgcn', 'semantic_hgt',
#           'transformer_only', 'hybrid_great', 'hgt_gmn', 'gat_gmn']
```

---

## Inference

### `InferencePipeline`

End-to-end inference with beam search/HTPS and verification.

```python
from src.inference.pipeline import InferencePipeline

pipeline = InferencePipeline(
    model: MBADeobfuscator,
    mode='auto',            # 'auto' | 'beam' | 'htps'
    beam_width=50,
    num_diversity_groups=4,
    temperature=0.7,
    htps_budget=500,
    num_execution_samples=100,
    z3_timeout=1000,
    device='cuda'
)

# Simplify expression
result = pipeline.simplify(expression: str, mode: Optional[str] = None)
```

**Parameters**:
- `model`: Trained MBADeobfuscator model
- `mode` (str): Inference mode
  - `'auto'`: Auto-select based on depth (default)
  - `'beam'`: Beam search
  - `'htps'`: HyperTree Proof Search
- `beam_width` (int): Beam search width (default: 50)
- `num_diversity_groups` (int): Diversity groups (default: 4)
- `temperature` (float): Sampling temperature (default: 0.7)
- `htps_budget` (int): HTPS node budget (default: 500)
- `num_execution_samples` (int): Execution test samples (default: 100)
- `z3_timeout` (int): Z3 timeout in ms (default: 1000)
- `device` (str): Device ('cuda' or 'cpu')

**Returns**: `InferenceResult`

```python
@dataclass
class InferenceResult:
    simplified: str                 # Simplified expression
    original: str                   # Original expression
    verification_tier: str          # 'z3' | 'execution' | 'syntax' | 'none'
    proof: Optional[str]            # Z3 proof (if tier='z3')
    inference_time: float           # Total inference time (seconds)
    num_candidates: int             # Number of candidates generated
    num_verified: int               # Number of verified candidates
    all_candidates: List[str]       # All verified candidates
```

**Example**:

```python
result = pipeline.simplify("(x0 & x1) + (x0 ^ x1)")

print(result.simplified)          # "x0 | x1"
print(result.verification_tier)   # "z3"
print(result.inference_time)      # 0.312
print(result.num_verified)        # 3
```

---

### `BeamSearchDecoder`

Diverse beam search decoder.

```python
from src.inference.beam_search import BeamSearchDecoder

decoder = BeamSearchDecoder(
    model: MBADeobfuscator,
    beam_width=50,
    num_groups=4,
    temperature=0.7,
    length_penalty_alpha=0.6,
    diversity_penalty=0.5,
    max_length=256,
    use_grammar=True
)

# Generate candidates
candidates = decoder.search(
    expression: str,
    num_return=50
)
# Returns: List[str] (sorted by score)
```

---

### `HyperTreeProofSearch`

Compositional proof search for deep expressions.

```python
from src.inference.htps import HyperTreeProofSearch

htps = HyperTreeProofSearch(
    tactics: List[Callable],
    budget=500,
    ucb_c=1.414,  # sqrt(2)
    max_depth=20
)

# Search for simplification
simplified = htps.search(expression: str)
# Returns: str
```

---

### `ThreeTierVerifier`

Three-tier verification cascade.

```python
from src.inference.verify import ThreeTierVerifier

verifier = ThreeTierVerifier(
    num_execution_samples=100,
    z3_timeout=1000
)

# Verify equivalence
result = verifier.verify(
    original: str,
    simplified: str,
    max_tier='z3'  # 'syntax' | 'execution' | 'z3'
)
```

**Returns**: `VerificationResult`

```python
@dataclass
class VerificationResult:
    is_verified: bool               # Whether expressions are equivalent
    tier: str                       # Highest tier reached
    proof: Optional[str]            # Z3 proof (if tier='z3')
    counterexample: Optional[Dict]  # Counterexample (if not verified)
    verification_time: float        # Verification time (seconds)
```

---

## Data Processing

### `MBATokenizer`

Expression tokenizer with 300-token vocabulary.

```python
from src.data.tokenizer import MBATokenizer

tokenizer = MBATokenizer()

# Encode expression
tokens = tokenizer.encode(
    expression: str,
    add_special_tokens=True,
    max_length=256,
    padding='max_length'
)
# Returns: List[int] or Tensor

# Decode tokens
expression = tokenizer.decode(
    tokens: Union[List[int], Tensor],
    skip_special_tokens=True
)
# Returns: str

# Get vocabulary
vocab = tokenizer.get_vocab()
# Returns: Dict[str, int]

# Vocabulary size
vocab_size = len(tokenizer)
# Returns: 300
```

**Special tokens**:
- `PAD = 0`
- `UNK = 1`
- `BOS = 2`
- `EOS = 3`
- `MASK = 4`

---

### `SemanticFingerprint`

Semantic fingerprint computation.

```python
from src.data.fingerprint import SemanticFingerprint

fp = SemanticFingerprint(use_cpp=True)

# Compute full fingerprint (448 dims)
vector = fp.compute(expression: str)
# Returns: ndarray [448]

# Compute ML fingerprint (416 dims, derivatives stripped)
ml_vector = fp.compute_ml(expression: str)
# Returns: ndarray [416]

# Component breakdown
components = fp.compute_components(expression: str)
# Returns: {
#     'symbolic': ndarray [32],
#     'corner': ndarray [256],
#     'random': ndarray [64],
#     'derivative': ndarray [32],  # Not used for ML
#     'truth_table': ndarray [64]
# }
```

**Parameters**:
- `use_cpp` (bool): Use C++ acceleration if available (default: True)

---

### `parse_expression`

Parse expression to AST.

```python
from src.data.ast_parser import parse_expression

ast = parse_expression(expression: str)
# Returns: ASTNode
```

**ASTNode**:

```python
class ASTNode:
    type: NodeType          # VARIABLE | CONSTANT | ADD | SUB | ...
    value: Optional[int]    # For CONSTANT
    var_id: Optional[int]   # For VARIABLE (0-7)
    left: Optional[ASTNode]
    right: Optional[ASTNode]
    depth: int
    subtree_size: int
```

---

### `expression_to_graph`

Convert expression to PyTorch Geometric graph.

```python
from src.data.ast_parser import expression_to_graph

data = expression_to_graph(
    expression: str,
    edge_type_system='optimized'  # 'legacy' | 'optimized'
)
# Returns: Data (PyG graph)
```

**Data fields**:
- `x`: Node features `[num_nodes × 23]`
- `edge_index`: Edge connectivity `[2 × num_edges]`
- `edge_type`: Edge type labels `[num_edges]`
- `num_nodes`: Number of nodes

---

### Dataset Classes

#### `MBADataset`

Supervised learning dataset.

```python
from src.data.dataset import MBADataset

dataset = MBADataset(
    data_path: str,
    max_depth: int = 14,
    augment: bool = True,
    cache_fingerprints: bool = True
)

sample = dataset[idx]
# Returns: {
#     'obfuscated_graph': Data,
#     'obfuscated_tokens': Tensor,
#     'simplified_tokens': Tensor,
#     'fingerprint': Tensor [416],
#     'depth': int,
#     'length': int
# }
```

#### `ContrastiveDataset`

Contrastive learning dataset.

```python
from src.data.dataset import ContrastiveDataset

dataset = ContrastiveDataset(
    data_path: str,
    augment: bool = True
)

sample = dataset[idx]
# Returns: {
#     'anchor': Data,
#     'positive': Data,
#     'fingerprint': Tensor [416]
# }
```

#### `ScaledMBADataset`

Scaled model dataset (360M params).

```python
from src.data.dataset import ScaledMBADataset

dataset = ScaledMBADataset(
    data_path: str,
    max_seq_len: int = 2048,
    share_subexpressions: bool = True
)
```

#### `GMNDataset`

Graph Matching Network dataset.

```python
from src.data.dataset import GMNDataset

dataset = GMNDataset(
    data_path: str,
    pair_mode: str = 'equivalent'  # 'equivalent' | 'random'
)

sample = dataset[idx]
# Returns: {
#     'graph1': Data,
#     'graph2': Data,
#     'label': int,  # 1=equivalent, 0=different
#     'fingerprint1': Tensor [416],
#     'fingerprint2': Tensor [416]
# }
```

---

## Training

### Trainers

#### `Phase1Trainer`

Contrastive pretraining trainer.

```python
from src.training.phase1_trainer import Phase1Trainer

trainer = Phase1Trainer(
    model: MBADeobfuscator,
    config: Dict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str = 'cuda'
)

# Train
trainer.train(num_epochs: int = 20)

# Evaluate
metrics = trainer.evaluate()
# Returns: {
#     'loss': float,
#     'info_nce_loss': float,
#     'mlm_loss': float
# }
```

#### `Phase2Trainer`

Supervised learning trainer with curriculum.

```python
from src.training.phase2_trainer import Phase2Trainer

trainer = Phase2Trainer(
    model: MBADeobfuscator,
    config: Dict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str = 'cuda'
)

# Train with curriculum
trainer.train_curriculum(
    stages: List[Dict]  # [{max_depth, epochs, lr, target_acc}, ...]
)

# Evaluate
metrics = trainer.evaluate()
# Returns: {
#     'loss': float,
#     'accuracy': float,
#     'accuracy_by_depth': Dict[int, float],
#     'simplification_ratio': float
# }
```

#### `Phase3Trainer`

RL fine-tuning trainer (PPO).

```python
from src.training.phase3_trainer import Phase3Trainer

trainer = Phase3Trainer(
    model: MBADeobfuscator,
    config: Dict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str = 'cuda'
)

# Train with PPO
trainer.train(num_epochs: int = 10)

# Evaluate
metrics = trainer.evaluate()
# Returns: {
#     'equivalence_rate': float,
#     'simplification_ratio': float,
#     'identity_rate': float,
#     'avg_reward': float
# }
```

#### `AblationTrainer`

Encoder ablation study trainer.

```python
from src.training.ablation_trainer import AblationTrainer

trainer = AblationTrainer(
    config: Dict,
    train_loader: DataLoader,
    val_loader: DataLoader
)

# Compare encoders
results = trainer.compare_encoders(
    encoder_types: List[str],
    num_runs: int = 5
)
# Returns: Dict[str, AblationMetrics]
```

---

### Loss Functions

```python
from src.training.losses import (
    info_nce_loss,
    masked_lm_loss,
    token_cross_entropy_loss,
    complexity_loss,
    copy_loss,
    property_loss,
    ppo_loss
)

# InfoNCE
loss = info_nce_loss(
    anchor: Tensor,
    positive: Tensor,
    negatives: Tensor,
    temperature: float = 0.07
)

# Masked LM
loss = masked_lm_loss(
    predictions: Tensor,
    targets: Tensor,
    mask_indices: Tensor
)

# Token cross-entropy
loss = token_cross_entropy_loss(
    logits: Tensor,
    targets: Tensor,
    ignore_index: int = 0  # PAD
)

# Complexity
loss = complexity_loss(
    length_pred: Tensor,
    depth_pred: Tensor,
    length_target: Tensor,
    depth_target: Tensor
)

# Copy mechanism
loss = copy_loss(
    copy_gate: Tensor,
    copy_labels: Tensor
)

# PPO
policy_loss, value_loss = ppo_loss(
    log_probs: Tensor,
    old_log_probs: Tensor,
    values: Tensor,
    rewards: Tensor,
    advantages: Tensor,
    epsilon: float = 0.2
)
```

---

## Verification

### `verify_equivalence`

Z3 SMT-based equivalence verification.

```python
from src.utils.z3_interface import verify_equivalence

is_equiv, proof = verify_equivalence(
    expr1: str,
    expr2: str,
    timeout: int = 1000,  # milliseconds
    bit_width: int = 64
)
# Returns: (bool, Optional[str])
```

**Parameters**:
- `expr1`, `expr2`: Expressions to verify
- `timeout`: Z3 timeout in milliseconds
- `bit_width`: Bitvector width (8, 16, 32, or 64)

**Returns**:
- `is_equiv`: True if proven equivalent, False otherwise
- `proof`: Z3 proof string (if equivalent), None otherwise

---

### `find_counterexample`

Find counterexample if expressions are not equivalent.

```python
from src.utils.z3_interface import find_counterexample

counterexample = find_counterexample(
    expr1: str,
    expr2: str,
    timeout: int = 1000
)
# Returns: Optional[Dict[str, int]]
```

**Example**:

```python
cex = find_counterexample("x0 & x1", "x0 | x1")
# cex = {'x0': 1, 'x1': 0}  # (1 & 0) = 0 != (1 | 0) = 1
```

---

### `expr_to_z3`

Convert expression string to Z3 formula.

```python
from src.utils.z3_interface import expr_to_z3

z3_formula = expr_to_z3(
    expression: str,
    bit_width: int = 64
)
# Returns: z3.ExprRef
```

---

### `evaluate_expression`

Safely evaluate expression on inputs.

```python
from src.utils.expr_eval import evaluate_expression

result = evaluate_expression(
    expression: str,
    inputs: Union[List[int], Dict[str, int]],
    bit_width: int = 64
)
# Returns: int
```

**Example**:

```python
result = evaluate_expression("(x0 & x1) + 5", {'x0': 3, 'x1': 7}, bit_width=8)
# result = (3 & 7) + 5 = 3 + 5 = 8
```

---

## Utilities

### Metrics

```python
from src.utils.metrics import (
    accuracy,
    semantic_accuracy,
    simplification_ratio,
    depth_reduction,
    f1_score
)

# Accuracy
acc = accuracy(predictions: List[str], targets: List[str])
# Returns: float

# Semantic accuracy (equivalence-based)
sem_acc = semantic_accuracy(
    predictions: List[str],
    originals: List[str],
    verifier: ThreeTierVerifier
)
# Returns: float

# Simplification ratio
ratio = simplification_ratio(
    predictions: List[str],
    originals: List[str]
)
# Returns: float

# Depth reduction
reduction = depth_reduction(
    predictions: List[str],
    originals: List[str]
)
# Returns: float

# F1 score
f1 = f1_score(
    predictions: List[str],
    targets: List[str]
)
# Returns: float
```

---

### Logging

```python
from src.utils.logging import setup_logger

logger = setup_logger(
    name: str = 'mba_deobfuscator',
    log_file: Optional[str] = None,
    level: str = 'INFO'
)

logger.info("Training started")
logger.debug("Batch size: 32")
logger.warning("Learning rate may be too high")
logger.error("CUDA out of memory")
```

---

### Configuration

```python
from src.utils.config import load_config, save_config

# Load YAML config
config = load_config(path: str)
# Returns: Dict

# Save config
save_config(config: Dict, path: str)

# Merge configs
merged = merge_configs(base: Dict, override: Dict)
# Returns: Dict
```

---

## Configuration

### Config Structure

```yaml
# Example config (configs/phase2.yaml)
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
  max_seq_len: 256
  use_copy_mechanism: true

training:
  phase: 2
  epochs: 50
  batch_size: 32
  learning_rate: 3e-4
  weight_decay: 1e-4

  optimizer: adamw
  scheduler: cosine
  warmup_epochs: 2

  gradient_clip: 1.0
  accumulation_steps: 2

  loss_weights:
    complexity: 0.1
    copy: 0.1
    property: 0.05

  curriculum:
    - {max_depth: 2, epochs: 10, lr: 3e-4, target_acc: 0.95}
    - {max_depth: 5, epochs: 15, lr: 2e-4, target_acc: 0.90}
    - {max_depth: 10, epochs: 15, lr: 1e-4, target_acc: 0.80}
    - {max_depth: 14, epochs: 10, lr: 5e-5, target_acc: 0.70}

data:
  train_path: data/train.json
  val_path: data/val.json
  test_path: data/test.json

  dataset: mba  # 'mba' | 'contrastive' | 'scaled' | 'gmn'
  augment: true
  num_workers: 4

  max_depth: 14

inference:
  mode: auto  # 'auto' | 'beam' | 'htps'
  beam_width: 50
  num_diversity_groups: 4
  temperature: 0.7
  htps_budget: 500
  num_execution_samples: 100
  z3_timeout: 1000

logging:
  log_dir: logs/
  tensorboard: true
  wandb: true
  wandb_project: mba-deobfuscator

checkpoints:
  save_dir: checkpoints/
  save_freq: 1  # Save every N epochs
  keep_best: true
```

---

## Constants

All hyperparameters and constants are centralized in `src/constants.py`:

```python
from src.constants import (
    # Vocabulary
    VOCAB_SIZE,
    PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN, MASK_TOKEN,

    # Dimensions
    FINGERPRINT_DIM, FINGERPRINT_DIM_ML,
    SYMBOLIC_DIM, CORNER_DIM, RANDOM_DIM, DERIVATIVE_DIM, TRUTH_TABLE_DIM,

    # Model architecture
    HIDDEN_DIM, NUM_LAYERS, NUM_HEADS,
    DECODER_DIM, DECODER_LAYERS, DECODER_HEADS, DECODER_FFN_DIM,

    # Training
    LEARNING_RATE, BATCH_SIZE, NUM_EPOCHS,
    GRADIENT_CLIP, WEIGHT_DECAY,

    # Inference
    BEAM_WIDTH, NUM_DIVERSITY_GROUPS, TEMPERATURE,
    HTPS_BUDGET, HTPS_UCB_C,

    # Verification
    NUM_EXECUTION_SAMPLES, Z3_TIMEOUT,

    # Paths
    DATA_DIR, CHECKPOINT_DIR, LOG_DIR
)
```

---

## Error Handling

### Custom Exceptions

```python
from src.utils.exceptions import (
    MBADeobfuscatorError,
    ParseError,
    VerificationError,
    ModelLoadError,
    ConfigError
)

try:
    ast = parse_expression("invalid expression")
except ParseError as e:
    print(f"Parse error: {e}")

try:
    result = verify_equivalence("x0", "y0")  # Unknown variable
except VerificationError as e:
    print(f"Verification error: {e}")
```

---

## Type Hints

All public APIs use type hints:

```python
from typing import List, Dict, Optional, Union, Tuple
import torch
from torch import Tensor
from torch_geometric.data import Data

def simplify(
    model: MBADeobfuscator,
    expression: str,
    mode: str = 'auto',
    device: str = 'cuda'
) -> InferenceResult:
    """
    Simplify MBA expression.

    Args:
        model: Trained deobfuscator model
        expression: Input expression string
        mode: Inference mode ('auto', 'beam', 'htps')
        device: Device to run on ('cuda' or 'cpu')

    Returns:
        InferenceResult with simplified expression and metadata

    Raises:
        ParseError: If expression is syntactically invalid
        ModelLoadError: If model checkpoint is corrupted
    """
    ...
```

---

## Examples

### End-to-End Training

```python
from src.models.full_model import MBADeobfuscator
from src.data.dataset import MBADataset
from src.training.phase2_trainer import Phase2Trainer
from src.utils.config import load_config
from torch.utils.data import DataLoader

# Load config
config = load_config('configs/phase2.yaml')

# Create model
model = MBADeobfuscator(
    encoder_type=config['model']['encoder_type'],
    hidden_dim=config['model']['hidden_dim'],
    # ... more params from config
)

# Load pretrained encoder (Phase 1)
model.load_checkpoint('checkpoints/phase1_best.pt', strict=False)

# Create datasets
train_dataset = MBADataset(config['data']['train_path'], augment=True)
val_dataset = MBADataset(config['data']['val_path'], augment=False)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Create trainer
trainer = Phase2Trainer(model, config, train_loader, val_loader)

# Train with curriculum
trainer.train_curriculum(config['training']['curriculum'])

# Save best model
model.save_checkpoint('checkpoints/phase2_best.pt')
```

---

### End-to-End Inference

```python
from src.models.full_model import MBADeobfuscator
from src.inference.pipeline import InferencePipeline

# Load model
model = MBADeobfuscator.load_checkpoint('checkpoints/phase3_best.pt')
model.eval()

# Create pipeline
pipeline = InferencePipeline(model, mode='auto', device='cuda')

# Simplify expressions
expressions = [
    "(x0 & x1) + (x0 ^ x1)",
    "x2 ^ x2",
    "~(x3 & x4)"
]

for expr in expressions:
    result = pipeline.simplify(expr)

    print(f"Original:   {result.original}")
    print(f"Simplified: {result.simplified}")
    print(f"Verified:   {result.verification_tier}")
    print(f"Time:       {result.inference_time:.3f}s")
    print()
```

**Output**:
```
Original:   (x0 & x1) + (x0 ^ x1)
Simplified: x0 | x1
Verified:   z3
Time:       0.312s

Original:   x2 ^ x2
Simplified: 0
Verified:   z3
Time:       0.102s

Original:   ~(x3 & x4)
Simplified: ~x3 | ~x4
Verified:   z3
Time:       0.198s
```

---

### Batch Inference

```python
from concurrent.futures import ThreadPoolExecutor

def simplify_batch(expressions: List[str], pipeline: InferencePipeline, num_workers: int = 4):
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(pipeline.simplify, expressions))

    return results

# Usage
expressions = load_expressions('data/test.json')
results = simplify_batch(expressions, pipeline, num_workers=4)
```

---

### Custom Encoder

```python
from src.models.encoder_base import BaseEncoder
from torch_geometric.nn import GCNConv

class CustomGCNEncoder(BaseEncoder):
    def __init__(self, hidden_dim=256, num_layers=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.convs = nn.ModuleList([
            GCNConv(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])

    def forward(self, data):
        h = data.x

        for conv in self.convs:
            h = conv(h, data.edge_index)
            h = F.relu(h)

        return h  # [num_nodes × hidden_dim]

    def get_output_dim(self):
        return self.hidden_dim

# Register encoder
from src.models.encoder_registry import register_encoder

register_encoder('custom_gcn', CustomGCNEncoder)

# Use encoder
model = MBADeobfuscator(encoder_type='custom_gcn', hidden_dim=256)
```

---

## Performance Tips

1. **Use mixed precision**: `torch.cuda.amp` for 2× speedup
2. **Batch inference**: Process multiple expressions in parallel
3. **Cache fingerprints**: Avoid recomputation
4. **Use C++ fingerprint**: 10× faster than Python
5. **Enable gradient checkpointing**: Reduce memory for large models
6. **Increase num_workers**: Parallelize data loading (4-8 workers)
7. **Pin memory**: Faster CPU→GPU transfer (`pin_memory=True`)
8. **Compile model**: Use `torch.compile()` (PyTorch 2.0+)

---

## Troubleshooting

### Common Issues

**Issue**: `CUDA out of memory`
```python
# Solution 1: Reduce batch size
train_loader = DataLoader(dataset, batch_size=16)  # Was 32

# Solution 2: Gradient accumulation
config['training']['accumulation_steps'] = 4

# Solution 3: Gradient checkpointing
model.encoder.use_checkpoint = True
```

**Issue**: `ModuleNotFoundError: No module named 'mba_fingerprint_cpp'`
```python
# Solution: C++ module is optional, falls back to Python
# To install C++ module:
# cd cpp/
# python setup.py install
```

**Issue**: `Z3Exception: timeout`
```python
# Solution: Increase timeout or use execution-only verification
verifier = ThreeTierVerifier(z3_timeout=5000)  # 5 seconds
# Or
result = verifier.verify(expr1, expr2, max_tier='execution')
```

---

## Version Compatibility

| Component | Minimum Version | Recommended |
|-----------|----------------|-------------|
| Python | 3.10 | 3.11 |
| PyTorch | 2.0.0 | 2.2.0 |
| PyG | 2.2.0 | 2.4.0 |
| Z3 | 4.12.0 | 4.12.2 |
| CUDA | 11.7 | 12.1 |

---

## License

See `LICENSE` file for details.

---

## Support

For issues and questions:
- GitHub Issues: https://github.com/yourusername/mba-deobfuscator/issues
- Documentation: https://mba-deobfuscator.readthedocs.io
- Email: support@example.com
