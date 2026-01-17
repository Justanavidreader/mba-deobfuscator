# GCNII Over-Smoothing Mitigation Ablation Study

This guide explains how to run the GCNII ablation study to evaluate over-smoothing mitigation in deep HGT encoders.

## Overview

The ablation study compares two HGT configurations:
- **Baseline**: Standard HGT encoder without GCNII techniques
- **GCNII**: HGT with initial residuals (α=0.15) + identity mapping (λ=1.0)

Expected outcome: GCNII improves accuracy on deep expressions (depth 11-14) by ~10%.

## Quick Start (Small Synthetic Dataset)

For quick validation without waiting for large dataset generation:

```bash
# 1. Generate small synthetic dataset (1300 samples, ~2 minutes)
python scripts/train_gcnii_ablation.py --mode generate-data --generate-samples 100

# 2. Quick training test (2 epochs per stage, ~30 minutes on GPU)
python scripts/train_gcnii_ablation.py --mode baseline --run-id 1 --quick-mode
python scripts/train_gcnii_ablation.py --mode gcnii --run-id 1 --quick-mode

# 3. Evaluate and compare
python scripts/train_gcnii_ablation.py --mode evaluate \
    --baseline-ckpt checkpoints/gcnii_ablation/baseline_run1/phase2_best.pt \
    --gcnii-ckpt checkpoints/gcnii_ablation/gcnii_run1/phase2_best.pt
```

## Full Experiment (Production Dataset)

For full training with production-quality dataset:

```bash
# 1. Generate full dataset (10M samples, ~6 hours)
python scripts/generate_data.py --output data/train.jsonl --samples 7000000 --min-depth 1 --max-depth 14
python scripts/generate_data.py --output data/val.jsonl --samples 1500000 --min-depth 1 --max-depth 14
python scripts/generate_data.py --output data/test.jsonl --samples 1500000 --min-depth 1 --max-depth 14

# 2. Train baseline HGT
python scripts/train_gcnii_ablation.py --mode baseline --run-id 1

# 3. Train GCNII-HGT
python scripts/train_gcnii_ablation.py --mode gcnii --run-id 1

# 4. Evaluate both models
python scripts/train_gcnii_ablation.py --mode evaluate \
    --baseline-ckpt checkpoints/gcnii_ablation/baseline_run1/phase2_best.pt \
    --gcnii-ckpt checkpoints/gcnii_ablation/gcnii_run1/phase2_best.pt \
    --output results/gcnii_ablation_full.json
```

Expected training time: ~36-48 hours per model on single GPU (RTX 3090 / A100).

## Statistical Significance (Multiple Trials)

For publication-quality results with confidence intervals:

```bash
# Run 3 trials of both baseline and GCNII (will take ~6 days on single GPU)
python scripts/train_gcnii_ablation.py --mode full --num-trials 3
```

This automatically:
- Trains 3 baseline models (run IDs 1, 2, 3)
- Trains 3 GCNII models (run IDs 1, 2, 3)
- Evaluates all on depth buckets
- Computes aggregate statistics (mean ± std)
- Saves results to `results/gcnii_ablation.json`

## Output Files

### Checkpoints
```
checkpoints/gcnii_ablation/
├── baseline_run1/
│   ├── phase2_best.pt          # Best baseline model
│   └── phase2_epoch_*.pt       # Periodic checkpoints
└── gcnii_run1/
    ├── phase2_best.pt          # Best GCNII model
    └── phase2_epoch_*.pt
```

### Logs
```
logs/gcnii_ablation/
├── baseline_run1/
│   └── events.out.tfevents.*   # TensorBoard logs
└── gcnii_run1/
    └── events.out.tfevents.*
```

### Results
```
results/gcnii_ablation.json     # Evaluation metrics by depth bucket
```

## Interpreting Results

### Expected Output Format

```
GCNII ABLATION STUDY RESULTS
================================================================================

Depth Bucket    | Baseline Acc    | GCNII Acc       | Improvement
--------------------------------------------------------------------------------
2-4             |         0.9200 |         0.9250 |        +0.0050
5-7             |         0.8500 |         0.8600 |        +0.0100
8-10            |         0.7200 |         0.7500 |        +0.0300
11-14           |         0.6500 |         0.7500 |        +0.1000
================================================================================

Overall Average:
  Baseline: 0.7850
  GCNII:    0.8213
  Improvement: +0.0363

Deep Expressions (depth 11-14):
  Baseline: 0.6500
  GCNII:    0.7500
  Improvement: +0.1000 (+15.4%)
```

### Key Metrics

1. **Overall Improvement**: GCNII should improve average accuracy by 3-5%
2. **Deep Expression Boost**: Depth 11-14 should see 10-15% improvement
3. **Shallow Expression Stability**: Depth 2-4 should remain within ±1%

### Success Criteria

✅ **Pass**: GCNII improves depth 11-14 accuracy by ≥8%
⚠️ **Marginal**: Improvement between 5-8%
❌ **Fail**: Improvement <5% or regression on shallow expressions

## Monitoring Training

### TensorBoard

```bash
tensorboard --logdir logs/gcnii_ablation
```

Key metrics to watch:
- **train/total_loss**: Should converge smoothly for both models
- **val/exact_match**: GCNII should plateau higher than baseline
- **Stage progression**: Both should advance through curriculum stages at similar epochs

### Loss Curves

Expected behavior:
- **Baseline**: Loss plateaus earlier, struggles with deep stages
- **GCNII**: Loss continues decreasing longer, better deep-stage performance

## Troubleshooting

### OOM (Out of Memory)

Reduce batch size in config:
```bash
# Edit configs/phase2.yaml
training:
  batch_size: 8  # Reduce from 16
```

Or use gradient accumulation:
```bash
training:
  batch_size: 8
  gradient_accumulation_steps: 2  # Effective batch size = 16
```

### Training Too Slow

Use quick mode for testing:
```bash
python scripts/train_gcnii_ablation.py --mode baseline --quick-mode
```

This reduces each curriculum stage from 10-15 epochs to 2 epochs.

### Dataset Generation Takes Too Long

Use smaller dataset for validation:
```bash
# Generate 1000 samples per depth (13k total, ~10 minutes)
python scripts/train_gcnii_ablation.py --mode generate-data --generate-samples 1000
```

### Baseline and GCNII Both Underperforming

Check:
1. **Data quality**: Inspect `data/train.jsonl` for valid MBA pairs
2. **Fingerprint computation**: Ensure C++ fingerprint library is built
3. **Model architecture**: Verify HGT edge types match dataset format

## Configuration Details

### Baseline HGT

```yaml
model:
  encoder_type: hgt
  hidden_dim: 256
  num_encoder_layers: 12
  num_encoder_heads: 16
  use_initial_residual: false   # GCNII disabled
  use_identity_mapping: false   # GCNII disabled
  edge_type_mode: optimized     # 8-type edge system
```

### GCNII-HGT

```yaml
model:
  encoder_type: hgt
  hidden_dim: 256
  num_encoder_layers: 12
  num_encoder_heads: 16
  use_initial_residual: true    # GCNII enabled
  use_identity_mapping: true    # GCNII enabled
  gcnii_alpha: 0.15             # Initial residual strength
  gcnii_lambda: 1.0             # Identity mapping decay
  edge_type_mode: optimized
```

### Training Hyperparameters

```yaml
training:
  learning_rate: 3e-4
  batch_size: 16
  curriculum_stages:
    - max_depth: 2, epochs: 10, target: 0.95
    - max_depth: 5, epochs: 15, target: 0.90
    - max_depth: 10, epochs: 15, target: 0.80
    - max_depth: 14, epochs: 10, target: 0.70
```

Total epochs: 50 (10 + 15 + 15 + 10)

## Next Steps

After confirming GCNII improvements:

1. **Update default config**: Set GCNII as default in `src/constants.py`:
   ```python
   GCNII_USE_INITIAL_RESIDUAL: bool = True
   GCNII_USE_IDENTITY_MAPPING: bool = True
   ```

2. **Hyperparameter tuning**: Experiment with α ∈ [0.1, 0.2] and λ ∈ [0.5, 1.5]

3. **Scaled model**: Test GCNII on 360M parameter model with 12M dataset

4. **Documentation**: Update ARCHITECTURE.md with GCNII findings

5. **Paper**: Write ablation section for publication

## References

- **GCNII Paper**: Li et al., "Simple and Deep Graph Convolutional Networks" (ICML 2020)
- **Over-smoothing**: Chen et al., "Measuring and Relieving the Over-smoothing Problem for Graph Neural Networks" (AAAI 2020)
- **MBA Deobfuscation**: Project ARCHITECTURE.md for model details
