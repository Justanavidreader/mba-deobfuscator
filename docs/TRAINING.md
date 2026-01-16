# Training Guide: MBA Deobfuscator

**Target**: ML engineers training or fine-tuning the GNN+Transformer deobfuscation model.

**Quick Start**: Jump to [Phase Commands](#phase-commands) for training scripts.

---

## Table of Contents

1. [Overview](#overview)
2. [3-Phase Training Pipeline](#3-phase-training-pipeline)
3. [Curriculum Learning](#curriculum-learning)
4. [Ablation Studies](#ablation-studies)
5. [Hyperparameters](#hyperparameters)
6. [Checkpointing](#checkpointing)
7. [Monitoring](#monitoring)
8. [Troubleshooting](#troubleshooting)

---

## Overview

**Architecture**: GNN Encoder → Semantic Fingerprint → Transformer Decoder → Verification

**Training Paradigm**: 3-phase progressive training
1. **Phase 1 (Contrastive)**: Self-supervised pretraining of encoder (20 epochs)
2. **Phase 2 (Supervised)**: End-to-end supervised learning with curriculum (50 epochs)
3. **Phase 3 (RL)**: Policy optimization with equivalence rewards (10 epochs)

**Total Training Time**: ~16 weeks on single A100 (80GB) for scaled model (360M params)

**Dataset Requirements**:
- Base model (15M params): 1M samples, 600M tokens
- Scaled model (360M params): 12M samples, 7.2B tokens (Chinchilla-optimal)

---

## 3-Phase Training Pipeline

### Phase 1: Contrastive Pretraining

**Purpose**: Learn robust graph representations through self-supervised objectives before decoder training.

**Objectives**:
1. **InfoNCE Contrastive Loss**: Equivalent expressions pull together in embedding space, non-equivalent push apart
2. **Masked Expression Modeling (MaskLM)**: Predict masked node types (analogous to BERT's MLM)

**Loss Function**:
```python
L_phase1 = L_infonce + λ_mask * L_masklm

# InfoNCE: Pull equivalent expressions together
L_infonce = -log(exp(sim(z_i, z_j) / τ) / Σ_k exp(sim(z_i, z_k) / τ))

# MaskLM: Predict masked node types
L_masklm = CrossEntropy(pred_masked, true_node_types)
```

**Configuration** (`configs/phase1.yaml`):
```yaml
training:
  phase: 1
  epochs: 20
  batch_size: 64  # Large batch for contrastive learning
  learning_rate: 1e-4
  warmup_steps: 1000

loss:
  infonce_temperature: 0.07  # Lower = harder negatives
  masklm_mask_ratio: 0.15    # 15% nodes masked per graph
  masklm_weight: 0.5         # Balance contrastive vs MLM

optimizer:
  type: AdamW
  betas: [0.9, 0.98]
  weight_decay: 0.01
  gradient_clip: 1.0
```

**Data Augmentation**:
- Apply random structural perturbations preserving equivalence
- Mask random nodes (type prediction target)
- Positive pairs: equivalent expressions via rewrite rules
- Negative pairs: non-equivalent expressions from batch

**Encoder Freezing**: Decoder not used in Phase 1 (encoder-only training).

---

### Phase 2: Supervised Learning

**Purpose**: End-to-end training with full model (encoder + decoder) using ground-truth simplified expressions.

**Objectives**:
1. **Cross-Entropy Loss**: Standard next-token prediction on target sequence
2. **Complexity Loss**: Predict output length and depth (guides model to prefer simpler forms)
3. **Copy Mechanism Loss**: Encourage copying variable names from input AST

**Loss Function**:
```python
L_phase2 = λ_ce * L_ce + λ_complexity * L_complexity + λ_copy * L_copy

# Cross-Entropy: Next-token prediction
L_ce = -Σ_t log P(y_t | y_<t, encoder_output)

# Complexity: MSE on length/depth prediction
L_complexity = MSE(pred_length, true_length) + MSE(pred_depth, true_depth)

# Copy: Weighted CE favoring copying over generation
L_copy = -Σ_t [p_gen * log P_vocab + (1-p_gen) * log P_copy]
```

**Configuration** (`configs/phase2.yaml`):
```yaml
training:
  phase: 2
  epochs: 50  # Varies by curriculum stage
  batch_size: 32
  learning_rate: 5e-5
  warmup_steps: 2000
  gradient_accumulation: 2  # Effective batch 64

loss:
  ce_weight: 1.0
  complexity_weight: 0.1
  copy_weight: 0.1

optimizer:
  type: AdamW
  betas: [0.9, 0.98]
  weight_decay: 0.01
  gradient_clip: 1.0

scheduler:
  type: cosine
  warmup_ratio: 0.1
  min_lr: 1e-6
```

**Curriculum Strategy**: See [Curriculum Learning](#curriculum-learning) below.

**Key Features**:
- **Copy mechanism**: Preserves variable names (x, y, z) from input
- **Complexity head**: Predicts target length/depth before generation → guides beam search
- **Label smoothing**: ε=0.1 to prevent overconfidence on training data

---

### Phase 3: Reinforcement Learning (PPO)

**Purpose**: Fine-tune with non-differentiable reward signals (Z3 equivalence checks, identity detection).

**Algorithm**: Proximal Policy Optimization (PPO) with actor-critic architecture.

**Reward Function**:
```python
R_total = R_equiv + R_simplification - R_length - R_depth - R_identity - R_syntax

# Equivalence (Z3 verification): +10 if equivalent, -5 if not
R_equiv = +10 if Z3_check(pred, target) else -5

# Simplification bonus: +2 if pred simpler than input
R_simplification = +2 if complexity(pred) < complexity(input) else 0

# Length/depth penalties: discourage verbose outputs
R_length = -0.1 * len(pred)
R_depth = -0.2 * depth(pred)

# Identity penalty: -5 if output ≈ input (model failed to simplify)
R_identity = -5 if similarity(pred, input) > 0.9 else 0

# Syntax error: -5 if unparseable
R_syntax = -5 if parse_error(pred) else 0
```

**Configuration** (`configs/phase3.yaml`):
```yaml
training:
  phase: 3
  epochs: 10
  batch_size: 16  # Smaller for RL stability
  learning_rate: 1e-5  # Lower LR for fine-tuning

rl:
  algorithm: ppo
  ppo_epsilon: 0.2        # Clip ratio
  ppo_value_coef: 0.5     # Value loss weight
  ppo_entropy_coef: 0.01  # Entropy regularization
  num_rollouts: 4         # Trajectories per update
  discount_gamma: 0.99

rewards:
  equiv_bonus: 10.0
  simplification_bonus: 2.0
  length_penalty: 0.1
  depth_penalty: 0.2
  identity_penalty: 5.0
  syntax_error_penalty: 5.0
  identity_threshold: 0.9

optimizer:
  type: AdamW
  weight_decay: 0.0  # No L2 in RL fine-tuning
  gradient_clip: 0.5  # Tighter clipping
```

**Training Loop**:
1. Generate samples from current policy (greedy or sample with temperature)
2. Execute verification: syntax check → execution test → Z3 (top-k only)
3. Compute rewards from verification results
4. Update policy via PPO objective (clipped surrogate + value + entropy)

**Verification Tiers** (in order):
1. **Syntax**: Parser check (~0ms, filters 5-10% invalid outputs)
2. **Execution**: Random input eval (1ms, catches 60% non-equiv)
3. **Z3 SMT**: Formal verification (100-1000ms, conclusive for survivors)

Apply Z3 only to top-10 candidates by model score (budget constraint).

---

## Curriculum Learning

**Strategy**: Self-paced curriculum with 4 stages of increasing depth.

**Mechanism**:
- Start with shallow expressions (depth 2-4) to build foundational patterns
- Progress to deeper expressions as accuracy targets are met
- **Self-paced weighting**: Dynamically downweight hard examples early, upweight as model improves

**Stages** (Phase 2 only):

| Stage | Max Depth | Epochs | Target Accuracy | λ_init | λ_growth |
|-------|-----------|--------|-----------------|--------|----------|
| 1     | 2         | 10     | 95%             | 0.5    | 1.1      |
| 2     | 5         | 15     | 90%             | 0.5    | 1.1      |
| 3     | 10        | 15     | 80%             | 0.5    | 1.1      |
| 4     | 14        | 10     | 70%             | 0.5    | 1.1      |

**Self-Paced Loss Weighting**:
```python
# Example weight for sample i at epoch e
w_i = λ_e if loss_i < λ_e else 0

# λ grows over epochs: λ_e = λ_init * (λ_growth)^e
# Early epochs: only easy samples (low loss) get non-zero weight
# Later epochs: hard samples included as λ increases
```

**Scaled Model Curriculum** (360M params): 1.5× epochs per stage for stability.

**Stage Progression**:
- **Automatic**: Move to next stage when validation accuracy ≥ target for 2 consecutive epochs
- **Manual override**: `--force-stage N` flag to skip ahead (for resumption)

**Configuration**:
```yaml
curriculum:
  enabled: true
  stages:
    - {max_depth: 2, epochs: 10, target: 0.95}
    - {max_depth: 5, epochs: 15, target: 0.90}
    - {max_depth: 10, epochs: 15, target: 0.80}
    - {max_depth: 14, epochs: 10, target: 0.70}

  self_paced:
    lambda_init: 0.5      # Initial threshold
    lambda_growth: 1.1    # Growth rate per epoch
    patience: 2           # Epochs before stage transition
```

---

## Ablation Studies

**Purpose**: Compare encoder architectures to validate design choices.

**Encoders Under Test**:

| Encoder          | Type                  | Edge Types | Params | Use Case            |
|------------------|-----------------------|------------|--------|---------------------|
| `gat_jknet`      | GAT + Jumping Knowledge | No       | 2.8M   | Baseline (depth ≤10) |
| `ggnn`           | Gated Graph NN        | Yes (6)    | 3.2M   | Depth 10-14         |
| `hgt`            | Heterogeneous GT      | Yes (7)    | 4.5M   | Scaled model        |
| `rgcn`           | Relational GCN        | Yes (7)    | 3.0M   | Alternative to GGNN |
| `transformer_only` | Transformer (no graph) | No      | 2.5M   | Sequence baseline   |
| `hybrid_great`   | GNN+Transformer hybrid | No       | 4.0M   | Research variant    |

**Registry Usage** (`src/models/encoder_registry.py`):
```python
from src.models.encoder_registry import get_encoder, list_encoders

# List available encoders
encoders = list_encoders()
print(encoders)
# {'gat_jknet': {'requires_edge_types': False, ...}, ...}

# Instantiate encoder by name
encoder = get_encoder('gat_jknet', hidden_dim=256, num_layers=4)
encoder = get_encoder('ggnn', hidden_dim=256, num_timesteps=8)
```

**Running Ablation** (`scripts/run_ablation.py`):
```bash
# Single encoder, single run
python scripts/run_ablation.py --encoder gat_jknet --run-id 1

# All encoders, 5 runs each (statistical significance)
python scripts/run_ablation.py --all-encoders --num-runs 5

# Specific encoder group
python scripts/run_ablation.py --group homogeneous --num-runs 5
```

**Metrics Collected** (`src/utils/ablation_metrics.py`):
- **Accuracy**: Exact match, equivalence (via execution), reduction rate
- **Efficiency**: Training time (hours), inference latency (ms/sample)
- **Per-depth performance**: Accuracy in buckets [2-4], [5-7], [8-10], [11-14]
- **Model size**: Parameter count, memory footprint

**Statistical Testing**:
- Paired t-test: Compare encoder pairs on same test set (5 runs each)
- Significance level: α = 0.05
- Report: mean ± std, p-value, effect size (Cohen's d)

**Trainer Integration** (`src/training/ablation_trainer.py`):
```python
from src.training.ablation_trainer import AblationTrainer

config = {
    'encoder': {'name': 'gat_jknet', 'hidden_dim': 256},
    'decoder': {'d_model': 512, 'num_layers': 6},
    'training': {'learning_rate': 1e-4, 'epochs': 50},
}

trainer = AblationTrainer(config)

# Training loop
for epoch in range(epochs):
    train_loss = trainer.train_epoch(train_loader)
    eval_results = trainer.evaluate(val_loader, tokenizer)

# Save metrics
trainer.metrics_collector.save_results('ablation_results.csv')
```

**Checkpoint Compatibility**: All encoders inherit from `BaseEncoder` → unified `.forward()` interface → same checkpoint format.

---

## Hyperparameters

### Base Model (15M params)

**Encoder (GAT+JKNet)**:
```yaml
node_dim: 32
hidden_dim: 256
num_layers: 4
num_heads: 8
dropout: 0.1
```

**Decoder (Transformer)**:
```yaml
d_model: 512
num_layers: 6
num_heads: 8
d_ff: 2048
dropout: 0.1
max_seq_len: 64
```

**Training**:
```yaml
# Phase 1
batch_size: 64
learning_rate: 1e-4
weight_decay: 0.01

# Phase 2
batch_size: 32
learning_rate: 5e-5
gradient_accumulation: 2
label_smoothing: 0.1

# Phase 3
batch_size: 16
learning_rate: 1e-5
gradient_clip: 0.5
```

---

### Scaled Model (360M params)

**Encoder (HGT or RGCN)**:
```yaml
hidden_dim: 768
num_layers: 12
num_heads: 16
dropout: 0.1
num_edge_types: 8
```

**Decoder (Transformer)**:
```yaml
d_model: 1536
num_layers: 8
num_heads: 24
d_ff: 6144
dropout: 0.1
max_seq_len: 2048
```

**Training**:
```yaml
# Phase 2
batch_size: 16  # Smaller for memory
learning_rate: 3e-5
gradient_accumulation: 8  # Effective batch 128
gradient_checkpointing: true  # Saves ~3× memory
mixed_precision: bf16  # A100 only

# Memory optimization
activation_checkpointing: every_3_layers
cpu_offload: false  # Keep on GPU if 80GB available
```

**Hardware Requirements**:
- **GPU**: A100 80GB or 2× A100 40GB (model parallel)
- **RAM**: 128GB+ (dataset preprocessing)
- **Storage**: 500GB SSD (checkpoints + dataset)
- **Estimated runtime**: 16 weeks single A100

---

### Hyperparameter Tuning Guidelines

**Priority Order** (if compute-limited):

| Rank | Param             | Range          | Impact             |
|------|-------------------|----------------|--------------------|
| 1    | `learning_rate`   | [1e-5, 5e-4]   | Critical           |
| 2    | `batch_size`      | [16, 128]      | High (stability)   |
| 3    | `num_layers`      | [4, 12]        | High (capacity)    |
| 4    | `dropout`         | [0.05, 0.2]    | Medium (overfitting) |
| 5    | `label_smoothing` | [0, 0.2]       | Medium             |
| 6    | `warmup_steps`    | [500, 2000]    | Low (convergence)  |

**Typical Sweep** (Ray Tune compatible):
```yaml
tune:
  learning_rate: loguniform(1e-5, 5e-4)
  batch_size: choice([16, 32, 64])
  dropout: uniform(0.05, 0.2)
  num_layers: choice([4, 6, 8])

  metric: val_accuracy
  mode: max
  num_samples: 20
```

---

## Checkpointing

**Checkpoint Contents**:
```python
checkpoint = {
    'encoder_state': encoder.state_dict(),
    'decoder_state': decoder.state_dict(),
    'vocab_head_state': vocab_head.state_dict(),
    'complexity_head_state': complexity_head.state_dict(),
    'optimizer_state': optimizer.state_dict(),
    'scheduler_state': scheduler.state_dict(),
    'epoch': current_epoch,
    'global_step': global_step,
    'best_val_accuracy': best_acc,
    'config': config_dict,
    'training_time_hours': elapsed_hours,
    'curriculum_stage': current_stage,  # For resumption
}
```

**Save Strategy**:
- **Every epoch**: `checkpoint_epoch_{N}.pt`
- **Best validation**: `checkpoint_best.pt` (overwrites)
- **Phase transitions**: `checkpoint_phase{P}_final.pt` (keep all)
- **Manual**: `checkpoint_manual_{timestamp}.pt` (on SIGINT)

**Resumption**:
```bash
# Resume from specific checkpoint
python scripts/train.py --phase 2 --resume checkpoint_epoch_25.pt

# Resume from best
python scripts/train.py --phase 2 --resume checkpoint_best.pt

# Resume phase 3 from phase 2 checkpoint (loads encoder+decoder, resets optimizer)
python scripts/train.py --phase 3 --resume checkpoint_phase2_final.pt
```

**Cross-Phase Loading**:
- Phase 1→2: Load encoder only, initialize decoder randomly
- Phase 2→3: Load full model, reset optimizer for RL learning rate
- Phase 3→Inference: Load full model, no optimizer needed

**Checkpoint Cleanup**:
```bash
# Keep only last 5 epoch checkpoints
python scripts/cleanup_checkpoints.py --keep-last 5

# Keep best + phase finals
python scripts/cleanup_checkpoints.py --keep-best --keep-phase-finals
```

---

## Monitoring

### Logging Integrations

**Weights & Biases** (recommended):
```yaml
wandb:
  enabled: true
  project: mba-deobfuscator
  entity: your-username
  tags: [phase2, gat_jknet, curriculum]

  log_interval: 100  # Steps
  log_gradients: false  # Expensive, enable for debugging
  log_model: true  # Upload checkpoints
```

**TensorBoard** (local):
```bash
tensorboard --logdir runs/
```

---

### Key Metrics to Track

**Phase 1 (Contrastive)**:
- `train/infonce_loss`: Should decrease steadily (target: <1.0)
- `train/masklm_loss`: Should decrease to ~0.5
- `train/masklm_accuracy`: Should reach 80%+
- `val/embedding_variance`: Should be >1.0 (embeddings not collapsed)

**Phase 2 (Supervised)**:
- `train/ce_loss`: Primary metric (target: <0.5 by end)
- `train/complexity_loss`: Should decrease to <0.1
- `val/accuracy_exact_match`: Strict string match (target: 70%+ on depth 14)
- `val/accuracy_equivalence`: Z3-verified or execution-verified (target: 85%+)
- `val/reduction_rate`: `(input_len - output_len) / input_len` (target: >0.3)
- `train/learning_rate`: Monitor scheduler (cosine decay)

**Phase 3 (RL)**:
- `rl/mean_reward`: Should increase (target: >5.0)
- `rl/equiv_rate`: Fraction of outputs passing Z3 (target: >90%)
- `rl/identity_rate`: Fraction of outputs identical to input (target: <5%)
- `rl/policy_entropy`: Should stay >0.1 (policy not collapsing)
- `rl/value_loss`: Critic learning signal

**Per-Depth Breakdown** (all phases):
- `val/accuracy_depth_2-4`: Easy (target: 95%+)
- `val/accuracy_depth_5-7`: Medium (target: 90%+)
- `val/accuracy_depth_8-10`: Hard (target: 80%+)
- `val/accuracy_depth_11-14`: Very hard (target: 70%+)

---

### Alerts & Thresholds

**Early Stopping Criteria**:
```yaml
early_stopping:
  patience: 5  # Epochs without improvement
  metric: val_accuracy_equivalence
  min_delta: 0.01  # Minimum improvement to reset patience
```

**Gradient Monitoring**:
- **Gradient norm explosion**: >10.0 → reduce LR or check data
- **Gradient norm vanishing**: <0.01 → increase LR or check layer norms
- **NaN gradients**: Stop training, check for invalid inputs (division by zero in fingerprint)

**Loss Anomalies**:
- **CE loss spike**: >5.0 → likely corrupted batch or OOV tokens
- **Reward collapse**: mean reward <-5 for 1000 steps → RL divergence, restore checkpoint

---

### Debugging Tools

**Visualize Predictions**:
```python
# scripts/visualize_predictions.py
python scripts/visualize_predictions.py \
  --checkpoint checkpoint_best.pt \
  --samples 50 \
  --depth-range 8-10 \
  --output predictions.html
```

**Attention Inspection**:
```python
# Export attention weights for GAT/HGT layers
python scripts/export_attention.py \
  --checkpoint checkpoint_best.pt \
  --expr "(x&y)+(x^y)" \
  --output attention_viz/
```

**Embedding Projections** (Phase 1):
```python
# Visualize encoder embeddings in 2D (t-SNE)
python scripts/plot_embeddings.py \
  --checkpoint checkpoint_phase1_final.pt \
  --num-samples 1000 \
  --method tsne
```

---

## Troubleshooting

### Common Issues

#### 1. OOM (Out of Memory)

**Symptoms**: CUDA OOM error during forward/backward pass.

**Solutions**:
```yaml
# Reduce batch size
batch_size: 16  # Was 32

# Enable gradient accumulation
gradient_accumulation: 4  # Effective batch = 16 × 4 = 64

# Enable gradient checkpointing (saves ~3× memory at 20% speed cost)
gradient_checkpointing: true

# Mixed precision (FP16/BF16)
mixed_precision: bf16  # Requires A100 for stability

# Reduce sequence length
max_seq_len: 1024  # Was 2048 (for scaled model)

# Offload optimizer states (slow but works)
cpu_offload: true
```

**Verification**:
```bash
# Check memory usage
nvidia-smi dmon -s u -d 1

# Profile memory
python scripts/profile_memory.py --config configs/scaled_model.yaml
```

---

#### 2. Training Divergence (Loss → NaN)

**Symptoms**: Loss suddenly becomes NaN or explodes to >1000.

**Diagnosis**:
```python
# Enable anomaly detection
torch.autograd.set_detect_anomaly(True)

# Log gradient norms
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: {param.grad.norm()}")
```

**Solutions**:
```yaml
# Tighter gradient clipping
gradient_clip: 0.5  # Was 1.0

# Lower learning rate
learning_rate: 1e-5  # Was 5e-5

# Reduce warmup steps (slow start)
warmup_steps: 500  # Was 2000

# Check for invalid fingerprints
# (division by zero in symbolic features → NaN)
fingerprint_validation: true
```

**Common Causes**:
- Division by zero in fingerprint computation (e.g., derivative of constant)
- Invalid edge indices in graph batch (negative or out-of-bounds)
- Mixed precision underflow (use `bf16` instead of `fp16` on A100)

---

#### 3. Poor Validation Accuracy (<50%)

**Symptoms**: Training loss decreases but validation accuracy plateaus.

**Diagnosis**:
- **Overfitting**: Check `train_acc >> val_acc` (>20% gap)
- **Data leakage**: Verify train/val split has no equivalent expressions across sets
- **Tokenizer mismatch**: Ensure same tokenizer used for train/val

**Solutions**:
```yaml
# Increase regularization
dropout: 0.2  # Was 0.1
weight_decay: 0.05  # Was 0.01
label_smoothing: 0.15  # Was 0.1

# Data augmentation (Phase 2)
augmentation:
  structural_perturbation: 0.3  # Probability
  variable_renaming: 0.2

# More training data
# Re-generate dataset with higher depth diversity
```

**If overfitting is NOT the issue**:
- Check encoder architecture (may need more capacity)
- Verify loss weights (complexity/copy losses may dominate)
- Inspect predictions: are they syntactically valid?

---

#### 4. Identity Outputs (Model copies input)

**Symptoms**: Model outputs identical or near-identical expressions to input (RL Phase 3).

**Diagnosis**:
```python
# Compute similarity
similarity = edit_distance(pred, input) / len(input)
# If similarity > 0.9 for >50% of samples → identity problem
```

**Solutions**:
```yaml
# Increase identity penalty
rewards:
  identity_penalty: 10.0  # Was 5.0
  identity_threshold: 0.85  # Was 0.9 (stricter)

# Add simplification bonus
rewards:
  simplification_bonus: 5.0  # Reward length reduction

# Reduce equivalence bonus (model may be too conservative)
rewards:
  equiv_bonus: 5.0  # Was 10.0
```

**Alternative**: Resume from Phase 2 checkpoint with higher temperature sampling in RL rollouts.

---

#### 5. Slow Training (<50 samples/sec)

**Symptoms**: Training takes >2× expected time.

**Bottlenecks**:
- **CPU**: Data loading (use `num_workers=4` in DataLoader)
- **GPU**: Small batch size (increase if memory allows)
- **Disk I/O**: Dataset on HDD (move to SSD)
- **Verification**: Z3 calls in training loop (only do top-k in Phase 3)

**Profiling**:
```python
# PyTorch profiler
with torch.profiler.profile() as prof:
    for batch in train_loader:
        loss = model(batch)
        loss.backward()
print(prof.key_averages().table(sort_by="cuda_time_total"))
```

**Optimizations**:
```yaml
# DataLoader
num_workers: 4
pin_memory: true
persistent_workers: true

# Compilation (PyTorch 2.0+)
compile: true  # ~20% speedup

# Disable expensive logging
log_interval: 500  # Was 100
log_gradients: false
```

---

#### 6. Verification Failures in RL (Phase 3)

**Symptoms**: Reward signal is noisy, training unstable.

**Diagnosis**:
- Check Z3 timeout rate: if >50%, increase timeout or reduce depth
- Check syntax error rate: if >20%, model is generating invalid code

**Solutions**:
```yaml
# Increase Z3 timeout
z3_timeout_ms: 2000  # Was 1000

# Pre-filter with execution tests (faster)
verification:
  exec_test_samples: 100  # Random input tests before Z3
  exec_test_timeout_ms: 10

# Apply Z3 only to promising candidates
z3_top_k: 5  # Only verify top-5 by model score
```

**3-Tier Verification Order** (Phase 3):
1. Syntax check (0ms, ~10% filtered)
2. Execution test (1ms, ~60% filtered)
3. Z3 SMT (100-1000ms, remaining candidates)

---

### Performance Benchmarks

**Expected Training Speed** (A100 80GB):

| Phase | Model       | Batch Size | Samples/sec | Time/Epoch |
|-------|-------------|------------|-------------|------------|
| 1     | Base (15M)  | 64         | 200         | 1.4 hrs    |
| 2     | Base (15M)  | 32         | 120         | 2.3 hrs    |
| 3     | Base (15M)  | 16         | 40          | 7.0 hrs    |
| 2     | Scaled (360M) | 16       | 15          | 18.5 hrs   |

**Expected Accuracy** (test set, depth 2-14 mixed):

| Phase | Exact Match | Equivalence | Reduction Rate |
|-------|-------------|-------------|----------------|
| 2     | 65%         | 80%         | 0.25           |
| 3     | 70%         | 85%         | 0.30           |

**If below benchmarks**: Check hardware (GPU utilization >90%), data quality, hyperparameters.

---

## Phase Commands

**Phase 1: Contrastive Pretraining**
```bash
python scripts/train.py \
  --phase 1 \
  --config configs/phase1.yaml \
  --data-path data/train.json \
  --output checkpoints/phase1/
```

**Phase 2: Supervised Learning**
```bash
python scripts/train.py \
  --phase 2 \
  --config configs/phase2.yaml \
  --resume checkpoints/phase1/checkpoint_phase1_final.pt \
  --data-path data/train.json \
  --output checkpoints/phase2/
```

**Phase 3: RL Fine-Tuning**
```bash
python scripts/train.py \
  --phase 3 \
  --config configs/phase3.yaml \
  --resume checkpoints/phase2/checkpoint_best.pt \
  --data-path data/train.json \
  --output checkpoints/phase3/
```

**Evaluation**
```bash
python scripts/evaluate.py \
  --checkpoint checkpoints/phase3/checkpoint_best.pt \
  --test-set data/test.json \
  --output results.json
```

**Model Verification** (parameter count, forward pass, memory estimate)
```bash
python scripts/verify_model.py
```

---

## Additional Resources

- **Architecture Details**: `docs/ML_PIPELINE.md`
- **Dataset Generation**: `docs/DATA_GENERATION.md`
- **Inference Pipeline**: `docs/INFERENCE.md`
- **Encoder Ablations**: `src/models/encoder_ablation.py`
- **Loss Functions**: `src/training/losses.py`

---

**Next Steps**: Generate dataset → Run Phase 1 → Monitor loss convergence → Proceed to Phase 2 curriculum → Fine-tune with RL → Evaluate on test set.
