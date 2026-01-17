# Training Infrastructure

Complete specification of the 3-phase training pipeline, curriculum learning, loss functions, and ablation studies.

---

## Training Overview

```
Phase 1: Contrastive Pretraining (20 epochs)
    ↓
Phase 1b: GMN Training (10 epochs, frozen encoder) [Optional]
    ↓
Phase 1c: GMN Fine-tuning (10 epochs, end-to-end) [Optional]
    ↓
Phase 2: Supervised Learning (50 epochs, 4-stage curriculum)
    ↓
Phase 3: RL Fine-Tuning (10 epochs, PPO)
    ↓
Final Model
```

**Total training time** (single GPU):
- Phase 1: 40-60 hours
- Phase 2: 80-120 hours
- Phase 3: 20-30 hours
- **Total**: ~150-210 hours (6-9 days)

---

## Phase 1: Contrastive Pretraining

### Goal

Learn semantic expression representations without labels via contrastive learning.

### Architecture

**Encoder-only** training (no decoder):
```
Expression → AST → Graph → GNN Encoder → Embedding [hidden_dim]
                                             ↓
                                    Contrastive Loss
```

### Loss Functions

#### 1.1 InfoNCE (Normalized Temperature-scaled Cross Entropy)

Maximize similarity of equivalent expressions:

```python
# Anchor: original expression
# Positive: equivalent expression (simplified or augmented)
# Negatives: other expressions in batch

def info_nce_loss(anchor, positive, negatives, temperature=0.07):
    # Compute similarities
    pos_sim = cosine_similarity(anchor, positive) / temperature
    neg_sims = [cosine_similarity(anchor, neg) / temperature for neg in negatives]

    # Softmax over positive + negatives
    logits = torch.cat([pos_sim, torch.stack(neg_sims)])
    labels = torch.zeros(len(logits))  # Positive is index 0

    return cross_entropy(logits, labels)
```

**Temperature**: `τ = 0.07` (controls sharpness of distribution)
- Lower τ: Sharper gradients, harder negatives
- Higher τ: Softer gradients, easier training

**Batch construction**:
```python
batch = [
    (expr1_obf, expr1_simple),  # Positive pair 1
    (expr2_obf, expr2_simple),  # Positive pair 2
    ...
]

# For each pair, others in batch are negatives
```

#### 1.2 Masked Language Modeling (MaskLM)

Predict masked tokens in expression:

```python
def mask_tokens(tokens, mask_ratio=0.15):
    num_mask = int(len(tokens) * mask_ratio)
    mask_indices = random.sample(range(len(tokens)), num_mask)

    masked_tokens = tokens.copy()
    for idx in mask_indices:
        masked_tokens[idx] = MASK_TOKEN  # [MASK]

    return masked_tokens, mask_indices, tokens[mask_indices]

# Example
tokens = [13, 15, 5, 16, 14]  # (x0 & x1)
masked = [13, 4, 5, 16, 14]   # (MASK & x1)
# Predict: token at position 1 should be 15 (x0)
```

**Loss**:
```python
# Encoder predicts tokens at masked positions
predictions = encoder(masked_graph).node_embeddings[masked_nodes]
logits = Linear(predictions)  # → [num_masked × vocab_size]

loss_mlm = cross_entropy(logits, true_tokens)
```

**Purpose**:
- Forces encoder to understand expression structure
- Complements contrastive learning (uses both pairs and structure)

#### 1.3 Combined Loss

```python
loss = loss_info_nce + lambda_mlm * loss_mlm
# lambda_mlm = 1.0 (equal weighting)
```

### Configuration

```yaml
# configs/phase1.yaml
phase: 1
model:
  encoder_type: gat_jknet
  hidden_dim: 256
  num_layers: 4
  num_heads: 8

training:
  epochs: 20
  batch_size: 128
  learning_rate: 1e-3
  temperature: 0.07
  mask_ratio: 0.15

  optimizer: adamw
  weight_decay: 1e-4
  scheduler: cosine
  warmup_epochs: 2

  gradient_clip: 1.0
  accumulation_steps: 1

data:
  dataset: contrastive
  augment: true
  num_workers: 4
```

### Usage

```bash
python scripts/train.py \
    --phase 1 \
    --config configs/phase1.yaml \
    --output checkpoints/phase1_best.pt
```

### Expected Results

- **Embedding quality**: Equivalent expressions cluster together
- **Validation accuracy**: Not directly measured (unsupervised)
- **Downstream benefit**: +5-10% accuracy in Phase 2 vs random init

---

## Phase 1b: GMN Training (Frozen Encoder)

### Goal

Train Graph Matching Network head to predict expression equivalence while keeping encoder frozen.

### Architecture

```
Expression 1 → Encoder (frozen) → Embedding 1
                                        ↓
Expression 2 → Encoder (frozen) → Embedding 2
                                        ↓
                              GMN Cross-Attention
                                        ↓
                              Equivalence Score [0, 1]
```

### Loss Function

```python
# Binary cross-entropy
score = gmn_head(embed1, embed2)  # Scalar in [0, 1]
label = 1 if equivalent else 0

loss = binary_cross_entropy(score, label)
```

### Configuration

```yaml
# configs/phase1b_gmn.yaml
phase: 1b
model:
  encoder_type: hgt
  hidden_dim: 768
  num_layers: 12
  freeze_encoder: true  # Freeze encoder weights

  gmn:
    num_layers: 3
    num_heads: 8

training:
  epochs: 10
  batch_size: 64
  learning_rate: 5e-4
```

### Usage

```bash
python scripts/train.py \
    --phase 1b \
    --config configs/phase1b_gmn.yaml \
    --resume checkpoints/phase1_best.pt \
    --output checkpoints/phase1b_best.pt
```

---

## Phase 1c: GMN Fine-Tuning (End-to-End)

### Goal

Fine-tune both encoder and GMN head together for better matching.

### Configuration

```yaml
# configs/phase1c_gmn_finetune.yaml
phase: 1c
model:
  encoder_type: hgt
  freeze_encoder: false  # Unfreeze encoder

training:
  epochs: 10
  batch_size: 32
  learning_rate: 1e-4  # Lower LR for fine-tuning
```

### Usage

```bash
python scripts/train.py \
    --phase 1c \
    --config configs/phase1c_gmn_finetune.yaml \
    --resume checkpoints/phase1b_best.pt \
    --output checkpoints/phase1c_best.pt
```

---

## Phase 2: Supervised Learning with Curriculum

### Goal

Learn to simplify expressions via supervised seq2seq training with 4-stage depth curriculum.

### Architecture

**Full model** (encoder + decoder + heads):
```
Obfuscated Expression → Encoder + Fingerprint → Decoder → Tokens + Complexity
                                                              ↓
                                                     Simplified Expression
```

### Curriculum Stages

| Stage | Max Depth | Epochs | Target Accuracy | Learning Rate |
|-------|-----------|--------|-----------------|---------------|
| 1 | 2 | 10 | 95% | 3e-4 |
| 2 | 5 | 15 | 90% | 2e-4 |
| 3 | 10 | 15 | 80% | 1e-4 |
| 4 | 14 | 10 | 70% | 5e-5 |

**Total**: 50 epochs

### Loss Functions

#### 2.1 Token Cross-Entropy (Main)

Standard seq2seq loss:

```python
# Teacher forcing: use ground truth as decoder input
decoder_input = simplified_tokens[:-1]  # Shift right
decoder_target = simplified_tokens[1:]

logits = decoder(decoder_input, encoder_output)  # [seq_len × vocab_size]

loss_token = cross_entropy(logits, decoder_target, ignore_index=PAD_TOKEN)
```

**Ignore padding**: Loss only on real tokens, not padding

#### 2.2 Complexity Loss (Auxiliary)

Predict length and depth of simplified expression:

```python
# From complexity head
length_pred = complexity_head(decoder_output)  # Scalar
depth_pred = complexity_head(decoder_output)   # Scalar

loss_complexity = mse(length_pred, target_length) + mse(depth_pred, target_depth)
```

**Weight**: `λ_complexity = 0.1`

**Purpose**:
- Guide decoder to generate simpler outputs
- Provide reranking signal during inference

#### 2.3 Copy Loss (Auxiliary)

Encourage copy mechanism to preserve variables:

```python
# Copy gate predictions
p_copy = copy_gate(decoder_hidden)  # [seq_len × 1]

# Ground truth: 1 if token copied from source, 0 if generated
copy_labels = compute_copy_labels(source_tokens, target_tokens)

loss_copy = binary_cross_entropy(p_copy, copy_labels)
```

**Weight**: `λ_copy = 0.1`

**Purpose**: Prevent hallucinating non-existent variables

#### 2.4 Property Loss (Auxiliary) - **PLACEHOLDER**

**Current status**: Using zeros (marked `RULE 2 SHOULD_FIX`)

**Intended** (not yet fully implemented):
```python
# Semantic HGT property predictions
property_logits = semantic_hgt.property_detector(encoder_output)  # [13]

# Ground truth: algebraic properties of expression
property_labels = detect_properties(expression)  # [13] (binary)

loss_property = binary_cross_entropy(property_logits, property_labels)
```

**Weight**: `λ_property = 0.05`

**Properties** (13 total):
- Commutative, Associative, Distributive
- Idempotent, Identity, Absorbing
- Involution, De Morgan
- XOR/AND/OR specific properties

**TODO**: Implement real property detection (currently zeros)

#### 2.5 Combined Loss

```python
loss = loss_token + \
       0.1 * loss_complexity + \
       0.1 * loss_copy + \
       0.05 * loss_property
```

### Configuration

```yaml
# configs/phase2.yaml
phase: 2
model:
  encoder_type: gat_jknet
  hidden_dim: 256
  decoder_dim: 512
  decoder_layers: 6
  decoder_heads: 8

training:
  epochs: 50
  curriculum:
    - {max_depth: 2, epochs: 10, lr: 3e-4, target_acc: 0.95}
    - {max_depth: 5, epochs: 15, lr: 2e-4, target_acc: 0.90}
    - {max_depth: 10, epochs: 15, lr: 1e-4, target_acc: 0.80}
    - {max_depth: 14, epochs: 10, lr: 5e-5, target_acc: 0.70}

  batch_size: 32
  optimizer: adamw
  weight_decay: 1e-4
  scheduler: cosine  # Per curriculum stage

  loss_weights:
    complexity: 0.1
    copy: 0.1
    property: 0.05  # Currently not used (zeros)

  gradient_clip: 1.0
  accumulation_steps: 2

data:
  dataset: mba
  augment: true
  num_workers: 4
```

### Usage

```bash
python scripts/train.py \
    --phase 2 \
    --config configs/phase2.yaml \
    --resume checkpoints/phase1_best.pt \
    --output checkpoints/phase2_best.pt
```

### Self-Paced Curriculum

Automatically advance curriculum stages based on validation accuracy:

```python
# After each epoch
val_accuracy = evaluate(model, val_loader)

if val_accuracy >= current_stage.target_acc:
    print(f"Stage {stage_id} complete ({val_accuracy:.2%} >= {target})")
    advance_to_next_stage()
else:
    print(f"Continue stage {stage_id} ({val_accuracy:.2%} < {target})")
```

**Adaptive**: If model struggles, stays on current stage longer

---

## Phase 3: RL Fine-Tuning with PPO

### Goal

Optimize for equivalence and simplification via reinforcement learning.

### Architecture

**Policy**: Full model (encoder + decoder)
**Critic**: Value head (predicts expected reward)

```
State: Obfuscated expression
Action: Simplified expression (sampled from policy)
Reward: Equivalence (Z3) + Simplification ratio
```

### Reward Function

```python
def compute_reward(obfuscated, simplified):
    reward = 0.0

    # 1. Equivalence (Z3 verification)
    is_equiv = z3_verify_equivalence(obfuscated, simplified)
    if is_equiv:
        reward += 10.0
    else:
        reward -= 5.0  # Penalty for incorrect simplification

    # 2. Simplification ratio
    if is_equiv:
        ratio = len(tokenize(obfuscated)) / len(tokenize(simplified))
        reward += 2.0 * (ratio - 1.0)  # Bonus for shorter output

    # 3. Identity penalty
    if simplified == obfuscated:
        reward -= 5.0  # Discourage identity transformation

    # 4. Syntax error penalty
    if not is_valid_syntax(simplified):
        reward -= 1.0

    return reward
```

**Reward components**:
- **Equivalence**: +10.0 (correct) or -5.0 (incorrect)
- **Simplification**: +2.0 × (ratio - 1.0) if equivalent
- **Identity**: -5.0 if output == input
- **Syntax**: -1.0 if invalid syntax

### PPO Algorithm

```python
# Proximal Policy Optimization
for iteration in range(num_iterations):
    # Collect trajectories
    for batch in dataloader:
        obfuscated = batch['obfuscated']

        # Sample action from policy
        simplified, log_prob, value = policy.sample(obfuscated)

        # Compute reward
        reward = compute_reward(obfuscated, simplified)

        # Store transition
        buffer.store(obfuscated, simplified, log_prob, value, reward)

    # Update policy
    for epoch in range(ppo_epochs):
        for transitions in buffer:
            # Compute advantage
            advantage = reward - value  # Simplified (no GAE)

            # Compute policy ratio
            new_log_prob = policy.log_prob(obfuscated, simplified)
            ratio = exp(new_log_prob - old_log_prob)

            # Clipped surrogate objective
            clipped_ratio = clip(ratio, 1 - epsilon, 1 + epsilon)
            loss_policy = -min(ratio * advantage, clipped_ratio * advantage)

            # Value loss
            loss_value = mse(value, reward)

            # Entropy bonus (exploration)
            entropy = policy.entropy()
            loss_entropy = -0.01 * entropy

            # Total loss
            loss = loss_policy + 0.5 * loss_value + loss_entropy

            # Update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

**Hyperparameters**:
- Clip epsilon: `ε = 0.2`
- PPO epochs: 4
- GAE lambda: 0.95
- Entropy coefficient: 0.01
- Learning rate: 1e-5

### Tactics for Exploration

Fixed simplification tactics for bootstrapping:

```python
TACTICS = [
    # Identity laws
    lambda x: x.replace("x & x", "x"),
    lambda x: x.replace("x | x", "x"),
    lambda x: x.replace("x ^ x", "0"),

    # MBA patterns
    lambda x: x.replace("(x & y) + (x ^ y)", "x | y"),
    lambda x: x.replace("(x | y) - (x ^ y)", "x & y"),

    # Constant folding
    lambda x: fold_constants(x),

    # Distributive
    lambda x: apply_distributive(x),

    # De Morgan
    lambda x: apply_de_morgan(x),

    # Algebraic simplification
    lambda x: simplify_algebraically(x),
]
```

**Exploration**: 20% of time, apply random tactic instead of model policy

### Configuration

```yaml
# configs/phase3.yaml
phase: 3
model:
  # Load from Phase 2
  encoder_type: gat_jknet
  hidden_dim: 256

training:
  epochs: 10
  ppo_epochs: 4
  batch_size: 16

  learning_rate: 1e-5
  clip_epsilon: 0.2
  gae_lambda: 0.95
  entropy_coeff: 0.01

  reward_weights:
    equivalence: 10.0
    simplification: 2.0
    identity_penalty: -5.0
    syntax_penalty: -1.0

  tactics:
    num_tactics: 6
    exploration_prob: 0.2

data:
  dataset: mba
  max_depth: 14  # All depths
```

### Usage

```bash
python scripts/train.py \
    --phase 3 \
    --config configs/phase3.yaml \
    --resume checkpoints/phase2_best.pt \
    --output checkpoints/phase3_best.pt
```

### Expected Results

- **Equivalence rate**: 85-95% (vs 80-90% in Phase 2)
- **Simplification ratio**: 2.5× average (vs 2.0× in Phase 2)
- **Identity rate**: <5% (vs ~10% in Phase 2)

---

## Ablation Studies

### Encoder Comparison

Compare different encoder architectures systematically:

```bash
# Run ablation for all encoders
python scripts/run_ablation.py --all-encoders --num-runs 5

# Run specific encoder
python scripts/run_ablation.py --encoder hgt --run-id 1
```

### Ablation Metrics

For each encoder, measure:

| Metric | Description |
|--------|-------------|
| **Accuracy** | % of correctly simplified expressions |
| **Accuracy by depth** | Accuracy for depth buckets (1-2, 3-5, 6-10, 11-14) |
| **Simplification ratio** | Average output_length / input_length |
| **Inference time** | Average time per expression |
| **Training time** | Time per epoch |
| **Memory usage** | Peak GPU memory during training |

### Statistical Testing

Use paired t-test to determine significance:

```python
from src.utils.ablation_stats import compare_encoders

results_gat = train_and_evaluate('gat_jknet', num_runs=5)
results_hgt = train_and_evaluate('hgt', num_runs=5)

p_value = compare_encoders(results_gat, results_hgt)

if p_value < 0.05:
    print("Statistically significant difference")
else:
    print("No significant difference")
```

### Example Results

| Encoder | Accuracy | Depth 1-2 | Depth 3-5 | Depth 6-10 | Depth 11-14 | Time/Epoch |
|---------|----------|-----------|-----------|------------|-------------|------------|
| GAT+JKNet | 82.3% | 96.5% | 88.1% | 75.2% | 62.8% | 120s |
| GGNN | 84.1% | 96.8% | 89.7% | 77.8% | 65.2% | 150s |
| HGT | 87.5% | 97.2% | 91.4% | 82.1% | 71.3% | 180s |
| RGCN | 86.9% | 97.0% | 90.8% | 81.3% | 69.8% | 175s |
| Semantic HGT | 88.2% | 97.4% | 92.1% | 83.5% | 72.8% | 195s |

**Conclusion**: HGT variants outperform GAT/GGNN, especially on deep expressions

---

## Training Infrastructure

### Base Trainer

All trainers inherit from `BaseTrainer`:

```python
class BaseTrainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.optimizer = self.setup_optimizer()
        self.scheduler = self.setup_scheduler()

    def setup_optimizer(self):
        return AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

    def setup_scheduler(self):
        if self.config.scheduler == 'cosine':
            return CosineAnnealingLR(self.optimizer, T_max=self.config.epochs)
        elif self.config.scheduler == 'linear':
            return LinearLR(self.optimizer, total_iters=self.config.epochs)
        # ...

    def train_epoch(self):
        # Abstract: implemented by subclasses
        raise NotImplementedError

    def evaluate(self):
        # Abstract: implemented by subclasses
        raise NotImplementedError

    def save_checkpoint(self, path):
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'epoch': self.epoch,
            'config': self.config
        }, path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        # ...
```

### Phase-Specific Trainers

- `Phase1Trainer`: Implements InfoNCE + MaskLM
- `Phase1bGMNTrainer`: Implements GMN training (frozen encoder)
- `Phase1cGMNTrainer`: Implements GMN fine-tuning
- `Phase2Trainer`: Implements supervised learning + curriculum
- `Phase3Trainer`: Implements PPO
- `AblationTrainer`: Implements encoder comparison

### Gradient Accumulation

For large models that don't fit in GPU memory:

```python
accumulation_steps = 4
optimizer.zero_grad()

for i, batch in enumerate(dataloader):
    loss = compute_loss(batch) / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
```

**Effective batch size**: `batch_size × accumulation_steps`

### Mixed Precision Training

For faster training and lower memory:

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    with autocast():
        loss = compute_loss(batch)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

**Speedup**: ~2× faster with minimal accuracy loss

---

## Logging & Monitoring

### TensorBoard

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/phase2')

# Log scalars
writer.add_scalar('Loss/train', loss, epoch)
writer.add_scalar('Accuracy/val', accuracy, epoch)

# Log distributions
writer.add_histogram('Encoder/weights', model.encoder.parameters(), epoch)

# Log graphs
writer.add_graph(model, sample_input)
```

**View**:
```bash
tensorboard --logdir runs/
```

### Weights & Biases

```python
import wandb

wandb.init(project='mba-deobfuscator', config=config)

# Log metrics
wandb.log({
    'train/loss': loss,
    'val/accuracy': accuracy,
    'epoch': epoch
})

# Log model
wandb.save('checkpoints/phase2_best.pt')
```

### Console Logging

```python
from tqdm import tqdm

pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
for batch in pbar:
    loss = train_step(batch)
    pbar.set_postfix({'loss': f'{loss:.4f}'})
```

---

## Checkpointing

### Save Best Model

```python
best_accuracy = 0.0

for epoch in range(num_epochs):
    val_accuracy = evaluate(model, val_loader)

    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        save_checkpoint(model, 'checkpoints/phase2_best.pt')
```

### Resume Training

```bash
python scripts/train.py \
    --phase 2 \
    --config configs/phase2.yaml \
    --resume checkpoints/phase2_epoch10.pt
```

```python
# In trainer
if args.resume:
    self.load_checkpoint(args.resume)
    print(f"Resumed from epoch {self.epoch}")
```

---

## Hyperparameter Tuning

### Grid Search

```bash
for lr in 1e-3 1e-4 1e-5; do
    for batch_size in 16 32 64; do
        python scripts/train.py \
            --phase 2 \
            --learning_rate $lr \
            --batch_size $batch_size
    done
done
```

### Random Search

```python
import random

for trial in range(num_trials):
    config = {
        'learning_rate': 10 ** random.uniform(-5, -3),
        'batch_size': random.choice([16, 32, 64]),
        'dropout': random.uniform(0.0, 0.3),
        'hidden_dim': random.choice([256, 512, 768])
    }

    model = create_model(config)
    accuracy = train_and_evaluate(model, config)

    if accuracy > best_accuracy:
        best_config = config
```

### Bayesian Optimization

```python
from ax.service.ax_client import AxClient

ax_client = AxClient()
ax_client.create_experiment(
    parameters=[
        {'name': 'learning_rate', 'type': 'range', 'bounds': [1e-5, 1e-3], 'log_scale': True},
        {'name': 'batch_size', 'type': 'choice', 'values': [16, 32, 64]},
    ]
)

for trial in range(20):
    parameters, trial_index = ax_client.get_next_trial()
    accuracy = train_and_evaluate(parameters)
    ax_client.complete_trial(trial_index=trial_index, raw_data=accuracy)

best_params, _ = ax_client.get_best_parameters()
```

---

## Common Issues & Solutions

### Issue: Overfitting

**Symptoms**:
- High train accuracy, low val accuracy
- Increasing gap between train/val loss

**Solutions**:
1. Increase dropout (0.1 → 0.3)
2. Add weight decay (1e-4)
3. Reduce model size
4. More data augmentation
5. Early stopping

### Issue: Underfitting

**Symptoms**:
- Low train and val accuracy
- Loss plateaus early

**Solutions**:
1. Increase model capacity (hidden_dim, num_layers)
2. Train longer
3. Increase learning rate
4. Remove regularization
5. Check data quality

### Issue: Training Instability

**Symptoms**:
- Loss spikes or NaN
- Gradients explode

**Solutions**:
1. Gradient clipping (max_norm=1.0)
2. Lower learning rate
3. Use mixed precision carefully
4. Check for bad samples in data

### Issue: Slow Convergence

**Symptoms**:
- Loss decreases very slowly
- Many epochs needed

**Solutions**:
1. Increase learning rate
2. Use learning rate warmup
3. Better optimizer (AdamW vs SGD)
4. Batch normalization / Layer normalization
5. Better initialization

---

## Best Practices

1. **Always use Phase 1 pretraining** - Improves Phase 2 by 5-10%
2. **Curriculum learning is essential** - Random depth training fails for depth 10+
3. **Monitor per-depth accuracy** - Overall accuracy can hide depth-specific issues
4. **Save checkpoints frequently** - Training can be interrupted
5. **Use gradient clipping** - Prevents exploding gradients (max_norm=1.0)
6. **Validate every epoch** - Early stopping prevents overfitting
7. **Log everything** - TensorBoard/W&B for debugging
8. **Run ablation studies** - Compare encoders systematically
9. **Use mixed precision** - 2× speedup on modern GPUs
10. **Start with base model** - Scale up only if needed

---

## Performance Benchmarks

### Base Model (15M params)

| Phase | GPU | Batch Size | Time/Epoch | Total Time |
|-------|-----|------------|------------|------------|
| 1 | GTX 1080 Ti | 128 | 180s | 60h |
| 2 | GTX 1080 Ti | 32 | 240s | 100h |
| 3 | GTX 1080 Ti | 16 | 120s | 20h |

### Scaled Model (360M params)

| Phase | GPU | Batch Size | Time/Epoch | Total Time |
|-------|-----|------------|------------|------------|
| 1 | RTX 3090 | 64 | 300s | 100h |
| 2 | RTX 3090 | 16 | 400s | 160h |
| 3 | RTX 3090 | 8 | 180s | 30h |

**Multi-GPU**: Use `torch.nn.DataParallel` or `DistributedDataParallel` for 4× speedup

---

## Implementation Files

| Component | File | Lines |
|-----------|------|-------|
| Base Trainer | `src/training/base_trainer.py` | 250 |
| Phase 1 Trainer | `src/training/phase1_trainer.py` | 200 |
| Phase 1b GMN Trainer | `src/training/phase1b_gmn_trainer.py` | 180 |
| Phase 1c GMN Trainer | `src/training/phase1c_gmn_trainer.py` | 160 |
| Phase 2 Trainer | `src/training/phase2_trainer.py` | 300 |
| Phase 3 Trainer | `src/training/phase3_trainer.py` | 350 |
| Ablation Trainer | `src/training/ablation_trainer.py` | 220 |
| Loss Functions | `src/training/losses.py` | 280 |
| Negative Sampler | `src/training/negative_sampler.py` | 120 |
| Train Script | `scripts/train.py` | 300 |

---

## References

1. **Contrastive Learning**: Chen et al., "A Simple Framework for Contrastive Learning of Visual Representations" (ICML 2020)
2. **Masked Language Modeling**: Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers" (NAACL 2019)
3. **Curriculum Learning**: Bengio et al., "Curriculum Learning" (ICML 2009)
4. **PPO**: Schulman et al., "Proximal Policy Optimization Algorithms" (arXiv 2017)
5. **Graph Matching Networks**: Li et al., "Graph Matching Networks for Learning the Similarity of Graph Structured Objects" (ICML 2019)
6. **Mixed Precision Training**: Micikevicius et al., "Mixed Precision Training" (ICLR 2018)
