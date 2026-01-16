# Dataset Schema Versions

This document describes the node type ID schemas used in MBA deobfuscator datasets
and how to migrate between them.

## Schema Versions

### Version 2 (Current, 2026-01-15+)

Node type IDs match `src/constants.py NODE_TYPES`:

| Node Type | ID | Description |
|-----------|-----|-------------|
| VAR | 0 | Variables (x, y, z, ...) |
| CONST | 1 | Constants (0, 1, 2, ...) |
| ADD | 2 | Addition (+) |
| SUB | 3 | Subtraction (-) |
| MUL | 4 | Multiplication (*) |
| AND | 5 | Bitwise AND (&) |
| OR | 6 | Bitwise OR (\|) |
| XOR | 7 | Bitwise XOR (^) |
| NOT | 8 | Bitwise NOT (~) |
| NEG | 9 | Unary negation (-) |

### Version 1 (Legacy, pre-2026-01-15)

Original node type ordering:

| Node Type | ID | Description |
|-----------|-----|-------------|
| ADD | 0 | Addition (+) |
| SUB | 1 | Subtraction (-) |
| MUL | 2 | Multiplication (*) |
| NEG | 3 | Unary negation (-) |
| AND | 4 | Bitwise AND (&) |
| OR | 5 | Bitwise OR (\|) |
| XOR | 6 | Bitwise XOR (^) |
| NOT | 7 | Bitwise NOT (~) |
| VAR | 8 | Variables (x, y, z, ...) |
| CONST | 9 | Constants (0, 1, 2, ...) |

## Dataset Format

JSONL files should include a `schema_version` field in the AST section:

```json
{
  "obfuscated": "(x & y) + (x ^ y)",
  "simplified": "x | y",
  "depth": 3,
  "ast": {
    "schema_version": 2,
    "nodes": [
      {"id": 0, "type": "ADD", "type_id": 2},
      {"id": 1, "type": "AND", "type_id": 5},
      {"id": 2, "type": "VAR", "type_id": 0, "value": "x"},
      {"id": 3, "type": "VAR", "type_id": 0, "value": "y"}
    ],
    "edges": [...]
  }
}
```

## Usage

### Loading Datasets

The `node_type_schema` parameter is **REQUIRED** when creating dataset objects:

```python
from src.data.dataset import MBADataset
from src.data.tokenizer import MBATokenizer
from src.data.fingerprint import SemanticFingerprint

tokenizer = MBATokenizer()
fingerprint = SemanticFingerprint()

# For new datasets (schema_version >= 2)
dataset = MBADataset(
    data_path="data/train.jsonl",
    tokenizer=tokenizer,
    fingerprint=fingerprint,
    node_type_schema="current"  # REQUIRED
)

# For legacy datasets (pre-2026-01-15)
dataset = MBADataset(
    data_path="data/old_train.jsonl",
    tokenizer=tokenizer,
    fingerprint=fingerprint,
    node_type_schema="legacy"  # REQUIRED - will auto-convert node types
)
```

### Migration Script

Convert legacy datasets to current schema:

```bash
python scripts/migrate_legacy_datasets.py \
    --input data/old_train.jsonl \
    --output data/train_v2.jsonl \
    --verbose
```

The migration script:
1. Converts all `type_id` values using the mapping table
2. Adds `schema_version: 2` to the AST section
3. Preserves all other fields unchanged

## C++ Generator Update

If using an external C++ data generator, update it to:

1. Add `schema_version: 2` to AST JSON output
2. Use node type IDs matching current schema (VAR=0, CONST=1, ADD=2, ...)

Example AST output from generator:

```json
{
  "ast": {
    "schema_version": 2,
    "nodes": [
      {"id": 0, "type": "VAR", "type_id": 0, "value": "x"},
      {"id": 1, "type": "CONST", "type_id": 1, "value": 5}
    ]
  }
}
```

## Validation

The dataset loader validates node type IDs on first batch:

1. **Range check**: All IDs must be in [0-9]
2. **Distribution check**: Verifies reasonable mix of operators and terminals

If validation fails, you'll see an error like:

```
ValueError: Node type IDs must be in [0-9] range, got [0, 15].
Dataset schema validation failed. Check:
  1. Dataset file is not corrupted
  2. node_type_schema parameter matches dataset format
  3. Dataset JSON contains 'schema_version' field
```

## Edge Type Systems

In addition to node type schemas, there are two edge type systems:

| System | Types | Encoders | Config |
|--------|-------|----------|--------|
| Legacy (6-type) | CHILD_LEFT, CHILD_RIGHT, PARENT, SIBLING_*, SAME_VAR | GGNN, GAT | `edge_type_mode: legacy` |
| Optimized (8-type) | LEFT/RIGHT/UNARY_OPERAND + inverses + DOMAIN_BRIDGE | HGT, RGCN | `edge_type_mode: optimized` |

Set `edge_type_mode` in your config file to match your dataset format.
