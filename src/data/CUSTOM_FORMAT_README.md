# Custom Format Dataset Loader

Dataset loader for custom text format files with manual fingerprint and AST computation.

## Format Specification

The custom format supports section-based organization with comma-separated expression columns:

```
#section_header, field1, field2, ...
expression1_field1, expression1_field2, ...
expression2_field1, expression2_field2, ...

#another_section
...
```

### Example File

```
#obfuscated, groundtruth
2*(z^(x|(~y|z))) - (~(x&y)&(x^(y^z))), x + y
(x&y)+(x^y), x|y

#linear,groundtruth,poly
-1*y+1*~(x|y)+1*(x&y), 1*~y-1*(x^y), -8*~y*(x&y)-8*~y*(x&~y)+10*~y*x
3*x+1*~x-2*(x&~y), 3*(x&y)+1*~(x&y), 3*(x&y)*(x&y)-3*(x&y)*(x&~y)

#linear,groundtruth,nonpoly
-1*y+1*~(x|y)+1*(x&y), 1*~y-1*(x^y), -1*(-1*(x&y)+2*(x|~y))
```

### Column Interpretation

- **First column**: Treated as "obfuscated" expression
- **Second column**: Treated as "simplified" (ground truth) expression
- **Third+ columns**: Stored as metadata in `additional` field (optional)

All sections are loaded into a single dataset. Use `filter_by_section()` to extract specific sections.

---

## Usage

### Basic Loading

```python
from src.data import CustomFormatDataset, collate_custom_format
from src.data import MBATokenizer, SemanticFingerprint

# Initialize components
tokenizer = MBATokenizer()
fingerprint = SemanticFingerprint()

# Load dataset
dataset = CustomFormatDataset(
    data_path="path/to/custom_format.txt",
    tokenizer=tokenizer,
    fingerprint=fingerprint,
    max_depth=14,  # Optional depth filtering
    use_dag_features=True,
    edge_type_mode="optimized",  # or "legacy"
    skip_invalid=True,  # Skip unparseable expressions
)

print(f"Loaded {len(dataset)} samples")
```

### DataLoader Integration

```python
from torch.utils.data import DataLoader

dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=collate_custom_format,
    num_workers=4,
)

for batch in dataloader:
    # batch['graph_batch'] - Batched PyG graphs
    # batch['fingerprint'] - [batch_size, 416]
    # batch['obfuscated_tokens'] - [batch_size, max_len]
    # batch['simplified_tokens'] - [batch_size, max_len]
    # batch['depth'] - [batch_size]
    # batch['section'] - List[str]
    # batch['additional'] - List[str or None]
    pass
```

### Section Filtering

```python
# Get section distribution
section_dist = dataset.get_section_distribution()
print(section_dist)
# {'obfuscated_groundtruth': 100, 'linear_groundtruth_poly': 50, ...}

# Filter to specific section
obfuscated_only = dataset.filter_by_section('obfuscated_groundtruth')
print(f"Obfuscated section: {len(obfuscated_only)} samples")
```

### Depth Analysis

```python
# Get depth distribution
depth_dist = dataset.get_depth_distribution()
print(depth_dist)
# {2: 10, 3: 25, 4: 30, 5: 20, ...}
```

---

## Sample Structure

Each sample contains:

```python
{
    'obfuscated': str,           # Obfuscated expression
    'simplified': str,           # Simplified (ground truth)
    'depth': int,                # AST depth
    'graph_data': Data,          # PyTorch Geometric graph
    'fingerprint': Tensor,       # [416] semantic fingerprint
    'obfuscated_tokens': Tensor, # Tokenized obfuscated
    'simplified_tokens': Tensor, # Tokenized simplified
    'section': str,              # Section name (e.g., "obfuscated_groundtruth")
    'additional': str or None,   # Third column if present
}
```

---

## Implementation Details

### Automatic Computation

For each expression, the loader automatically:

1. **Parses to AST**: `parse_to_ast(expr)` → ASTNode tree
2. **Computes depth**: `expr_to_ast_depth(expr)` → int
3. **Builds graph**: `expr_to_graph(expr, use_dag_features, edge_type_mode)` → PyG Data
4. **Computes fingerprint**: `SemanticFingerprint.compute(expr)` → 448-dim vector
5. **Strips derivatives**: Reduces to 416-dim (removes indices 352-383)

### Derivative Stripping

The 448-dim raw fingerprint is reduced to 416-dim for ML:

```
Raw (448 dims):
  0-31:   Symbolic (32)
  32-287: Corner (256)
  288-351: Random (64)
  352-383: Derivative (32) ← REMOVED (C++/Python mismatch)
  384-447: Truth table (64)

ML (416 dims):
  0-31:   Symbolic (32)
  32-287: Corner (256)
  288-351: Random (64)
  352-415: Truth table (64)
```

### Edge Type Modes

- **`optimized` (default)**: 8-type system used by ScaledMBADataset
  - LEFT_OPERAND, RIGHT_OPERAND, UNARY_OPERAND
  - LEFT_OPERAND_INV, RIGHT_OPERAND_INV, UNARY_OPERAND_INV
  - DOMAIN_BRIDGE_DOWN, DOMAIN_BRIDGE_UP
  - No SIBLING or SAME_VAR edges (redundant/handled by dataset)

- **`legacy`**: 6-type system (deprecated)
  - CHILD_LEFT, CHILD_RIGHT, PARENT
  - SIBLING_NEXT, SIBLING_PREV, SAME_VAR

---

## Testing

Test the loader with the sample dataset:

```bash
python scripts/test_custom_format_dataset.py \
    --data data/sample_custom_format.txt \
    --batch-size 4 \
    --num-samples 5
```

Expected output:
```
============================================================
Testing CustomFormatDataset
============================================================

✓ Loaded 10 samples

Depth distribution:
  Depth  2:    2 samples
  Depth  3:    1 samples
  ...

Section distribution:
  obfuscated_groundtruth: 5 samples
  linear_groundtruth_poly: 3 samples
  linear_groundtruth_nonpoly: 2 samples

✓ DataLoader test successful!
All tests passed!
```

---

## Training Integration

### Phase 2 Training with Custom Format

```python
from src.training.phase2_trainer import Phase2Trainer
from src.data import CustomFormatDataset, collate_custom_format

# Load training data
train_dataset = CustomFormatDataset(
    data_path="data/custom_train.txt",
    tokenizer=tokenizer,
    fingerprint=fingerprint,
    max_depth=14,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=collate_custom_format,
)

# Initialize trainer (standard Phase2Trainer)
trainer = Phase2Trainer(model, tokenizer, fingerprint, config, device)

# Train
trainer.fit(train_loader, val_loader)
```

**Note**: The collated batch structure is compatible with existing trainers. Key mappings:

- `batch['graph_batch']` → Used by encoder
- `batch['fingerprint']` → Used by fingerprint fusion
- `batch['simplified_tokens']` → Target sequence for decoder

---

## Error Handling

### Invalid Expressions

By default, unparseable expressions are skipped with a warning:

```python
dataset = CustomFormatDataset(
    data_path="data.txt",
    skip_invalid=True,  # Skip invalid (default)
)
```

Set `skip_invalid=False` to raise errors on parsing failures (useful for debugging).

### Depth Filtering

Filter expressions by maximum depth:

```python
dataset = CustomFormatDataset(
    data_path="data.txt",
    max_depth=10,  # Only load depth ≤ 10
)
```

### Empty Sections

Empty lines and lines without enough columns are automatically skipped.

---

## Performance

**Loading time** (sample dataset with 10k expressions):
- Parse + AST: ~5-10ms per expression
- Graph construction: ~3-5ms per expression
- Fingerprint (Python): ~10-50ms per expression
- Fingerprint (C++): ~1-5ms per expression

**Recommendation**: For large datasets (>10k expressions), pre-compute and cache fingerprints.

---

## Example: Multi-Section Training

Train on specific sections:

```python
# Load full dataset
full_dataset = CustomFormatDataset("data/all_data.txt", tokenizer, fingerprint)

# Split by section
obfuscated_data = full_dataset.filter_by_section('obfuscated_groundtruth')
linear_poly_data = full_dataset.filter_by_section('linear_groundtruth_poly')
linear_nonpoly_data = full_dataset.filter_by_section('linear_groundtruth_nonpoly')

# Train on each section separately or combine
from torch.utils.data import ConcatDataset
combined = ConcatDataset([obfuscated_data, linear_poly_data])

dataloader = DataLoader(combined, batch_size=32, collate_fn=collate_custom_format)
```

---

## Limitations

1. **Manual fingerprint computation**: Each expression is fingerprinted at load time (vs pre-computed in JSONL format). Can be slow for large datasets.

2. **No caching**: Fingerprints and graphs are recomputed on each dataset instantiation. For large datasets, consider pre-processing to JSONL format.

3. **Section name constraints**: Section names are derived from header fields joined by underscores (e.g., `#obfuscated, groundtruth` → `"obfuscated_groundtruth"`).

---

## Migration to JSONL

To convert custom format to JSONL for faster loading:

```python
import json
from src.data import CustomFormatDataset, MBATokenizer, SemanticFingerprint

tokenizer = MBATokenizer()
fingerprint = SemanticFingerprint()

# Load custom format
dataset = CustomFormatDataset("data/custom.txt", tokenizer, fingerprint)

# Convert to JSONL
with open("data/converted.jsonl", "w") as f:
    for sample in dataset.data:
        entry = {
            "obfuscated": sample['obfuscated'],
            "simplified": sample['simplified'],
            "depth": sample['depth'],
            # Store fingerprint as list for JSON serialization
            "fingerprint": sample['fingerprint'].tolist(),
            # Optionally store AST as dict
        }
        f.write(json.dumps(entry) + "\n")
```

Then use standard `MBADataset` with pre-computed fingerprints.

---

## API Reference

### CustomFormatDataset

```python
CustomFormatDataset(
    data_path: str,
    tokenizer: MBATokenizer,
    fingerprint: SemanticFingerprint,
    max_depth: Optional[int] = None,
    use_dag_features: bool = True,
    edge_type_mode: str = "optimized",
    skip_invalid: bool = True,
)
```

**Methods**:
- `__len__()` → int
- `__getitem__(idx)` → Dict
- `get_depth_distribution()` → Dict[int, int]
- `get_section_distribution()` → Dict[str, int]
- `filter_by_section(section_name)` → CustomFormatDataset

### collate_custom_format

```python
collate_custom_format(batch: List[Dict]) → Dict
```

Batches samples from CustomFormatDataset for DataLoader.

**Returns**:
- `graph_batch`: Batched PyG Data
- `fingerprint`: [batch_size, 416]
- `obfuscated_tokens`: [batch_size, max_len] (padded)
- `simplified_tokens`: [batch_size, max_len] (padded)
- `depth`: [batch_size]
- `section`: List[str]
- `additional`: List[str or None]
