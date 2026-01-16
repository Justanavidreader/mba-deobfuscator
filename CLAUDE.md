# Polynomial MBA Deobfuscation ML System

> **Goal**: Simplify obfuscated Mixed Boolean-Arithmetic expressions using GNN+Transformer architecture with formal verification.

## Quick Reference

```
Input:  (x & y) + (x ^ y) + 2*(a & ~a)
Output: x | y
```

**Architecture**: GNN Encoder → Semantic Fingerprint → Transformer Decoder → Z3 Verification
**Parameters**: ~15M (base) / ~360M (scaled) | **Timeline**: ~16 weeks | **Target**: >85% accuracy

## Project Structure

```
mba-deobfuscator/
├── configs/                 # YAML configurations
├── src/
│   ├── data/               # Dataset, tokenizer, fingerprint
│   ├── models/             # GNN, Transformer, heads
│   ├── training/           # Phase 1-3 trainers
│   ├── inference/          # Beam search, HTPS, verification
│   └── utils/              # Metrics, logging, Z3 interface
├── scripts/                # Training & eval scripts
├── tests/                  # Unit tests
└── docs/                   # Detailed documentation
```

## Key Commands

```bash
# Generate dataset
python scripts/generate_data.py --depth 1-14 --samples 10M

# Train Phase 1 (Contrastive)
python scripts/train.py --phase 1 --config configs/phase1.yaml

# Train Phase 2 (Supervised)
python scripts/train.py --phase 2 --config configs/phase2.yaml

# Train Phase 3 (RL)
python scripts/train.py --phase 3 --config configs/phase3.yaml

# Evaluate
python scripts/evaluate.py --checkpoint best.pt --test-set test.json

# Inference
python scripts/simplify.py --expr "(x&y)+(x^y)" --mode beam
```

## Training Phases

| Phase | Type | Loss | Epochs | Output |
|-------|------|------|--------|--------|
| 1 | Contrastive | InfoNCE + MaskLM | 20 | Pretrained encoder |
| 2 | Supervised | CE + Complexity + Copy | 50 | Full model |
| 3 | RL (PPO) | Equiv + Anti-identity | 10 | Fine-tuned model |

## Curriculum Learning (Phase 2)

| Stage | Max Depth | Epochs | Target Accuracy |
|-------|-----------|--------|-----------------|
| 1     | 2         | 10     | 95%             |
| 2     | 5         | 15     | 90%             |
| 3     | 10        | 15     | 80%             |
| 4     | 14        | 10     | 70%             |

## Architecture Summary

### Encoder (choose one)
- **GAT+JKNet**: 4 layers, 8 heads, 256d (~2.8M params) — *default, depth ≤10*
- **GGNN**: 8 timesteps, 7 edge types, 256d (~3.2M params) — *depth 10+*
- **HGT**: 12 layers, 16 heads, 768d (~60M params) — *scaled model*
- **RGCN**: Relational GCN, alternative to GGNN (~60M params)

### Semantic Fingerprint (448 floats)
- Symbolic features (32) + Corner evals (256) + Random hash (64) + Derivatives (32) + Truth table (64)

### Decoder
- Transformer: 6 layers, 8 heads, 512d (~12M params)
- Copy mechanism for variable preservation
- Complexity head for output length/depth prediction
- Value head for HTPS guidance (critic)

### Scaled Model (360M params)
- Encoder: HGT, 12 layers, 768d
- Decoder: 8 layers, 1536d, 24 heads
- Max seq len: 2048 (for depth-14 expressions)

## Data Format

**Input JSONL**:
```json
{"obfuscated": "(x & y) + (x ^ y)", "simplified": "x | y", "depth": 3}
```

**Tokenizer**: vocab_size=300 (special tokens 0-4, operators 5-11, parens 12-13, vars x0-x7 14-21, constants 0-255 22-277)

## Novel Approaches (Priority Order)

| Priority | Technique | Phase | Status |
|----------|-----------|-------|--------|
| P0 | Truth table (64 entries) | 2 | Implement |
| P0 | Grammar-constrained decoding | Inf | Implement |
| P0 | Execution pre-filter | 3+Inf | Implement |
| P1 | Copy mechanism | 2 | Implement |
| P1 | Masked expression modeling | 1 | Implement |
| P1 | Self-paced curriculum | 2 | Implement |
| P2 | Minimal HTPS | Inf | Implement |
| P3 | Process Reward Model | 3 | Later |
| P4 | HTPS Online Learning | Inf | Research |

## Inference Pipeline

```
Input → Encode → [Beam Search OR HTPS] → Grammar Filter → 3-Tier Verify → Rerank → Output
                      │                                        │
                 (depth <10)                           Syntax (~10µs) → Exec (~1ms) → Z3 (top-10)
                 (depth ≥10)
```

**Beam Search**: beam_width=50, 4 diversity groups, grammar-constrained decoding
**HTPS**: budget=500, UCB search (c=√2), 6 tactics (identity laws, MBA patterns, constant folding)
**Verification**: 3-tier cascade filters 95% before expensive Z3 calls

## Important Files

| File | Purpose |
|------|---------|
| `src/models/encoder.py` | GAT+JKNet, GGNN, HGT, RGCN implementations |
| `src/models/decoder.py` | Transformer with copy mechanism |
| `src/models/full_model.py` | MBADeobfuscator (encoder + decoder + heads) |
| `src/data/fingerprint.py` | Semantic fingerprint computation (448d) |
| `src/data/tokenizer.py` | Expression tokenizer (vocab size 300) |
| `src/data/ast_parser.py` | AST parsing and graph construction |
| `src/inference/beam_search.py` | Diverse beam search with grammar constraints |
| `src/inference/htps.py` | HyperTree Proof Search (UCB-based) |
| `src/inference/verify.py` | 3-tier verification interface |
| `src/utils/z3_interface.py` | Z3 SMT solver wrapper |
| `src/constants.py` | All hyperparameters and dimensions |

## Testing

```bash
pytest tests/ -v
pytest tests/test_encoder.py -v  # Specific test
```

## Documentation

- `docs/ARCHITECTURE.md` — Model architecture (GNN, Transformer, heads, dimensions)
- `docs/TRAINING.md` — 3-phase training pipeline, curriculum learning, ablation studies
- `docs/INFERENCE.md` — Beam search, HTPS, verification, reranking
- `docs/DATA_PIPELINE.md` — Tokenizer, AST parsing, fingerprint computation, batching
- `docs/API_REFERENCE.md` — Public APIs: Z3 interface, metrics, config, ablation utilities

## Ablation Studies

Run encoder comparisons with statistical significance testing:

```bash
python scripts/run_ablation.py --encoder gat_jknet --run-id 1
python scripts/run_ablation.py --all-encoders --num-runs 5
```

Encoder registry (`src/models/encoder_registry.py`):
```python
encoder = get_encoder('gat_jknet', hidden_dim=256, num_layers=4)
encoder = get_encoder('ggnn', hidden_dim=256, num_timesteps=8)
```

## Code Style

- Python 3.10+, PyTorch 2.0+
- Type hints required
- Docstrings for public functions
- Black formatting, isort imports
