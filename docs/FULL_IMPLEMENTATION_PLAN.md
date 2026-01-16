# MBA Deobfuscator Training Pipeline - Full Implementation Plan

**Goal**: Implement complete training pipeline (Phase 1-3) for MBA deobfuscation model in one day (~8-10 hours).

**Estimated Total Time**: 9.5 hours

---

## Revision History

| Date | Version | Changes |
|------|---------|---------|
| 2026-01-17 | 1.1 | Quality review fixes applied: (1) Fixed encoder parameter order in Phase1, (2) Added explicit L2 normalization to InfoNCE, (3) Capped self-paced lambda at 10.0, (4) Clarified scatter_add for copy loss |
| 2026-01-17 | 1.0 | Initial draft |

---

## Table of Contents

1. [Overview](#overview)
2. [Implementation Order](#implementation-order)
3. [Component Specifications](#component-specifications)
4. [Testing Strategy](#testing-strategy)
5. [Dependencies](#dependencies)

---

## Overview

### What Exists
- Model architecture: `MBADeobfuscator`, encoders, decoder with copy mechanism
- Dataset classes: `MBADataset`, `ContrastiveDataset`, `ScaledMBADataset`
- Verification: 3-tier verification cascade (syntax → execution → Z3)
- Utilities: Tokenizer, fingerprint, metrics, Z3 interface, collate functions
- Inference: Beam search, HTPS (already implemented)
- Config system: YAML-based configuration loading

### What's Missing (16 Components)

**Training Infrastructure (Critical)**
1. Base Trainer - Shared training logic
2. Phase 1 Trainer - Contrastive pretraining
3. Phase 2 Trainer - Supervised with curriculum
4. Phase 3 Trainer - RL/PPO fine-tuning

**Loss Functions**
5. InfoNCE loss
6. MaskLM loss
7. Copy mechanism loss
8. Complexity loss
9. PPO loss components

**Training Scripts**
10. `scripts/train.py` - Main entry point
11. `scripts/generate_data.py` - Dataset generation
12. `scripts/evaluate.py` - Evaluation pipeline
13. `scripts/simplify.py` - CLI inference

**Configuration**
14. `configs/phase1.yaml`
15. `configs/phase2.yaml`
16. `configs/phase3.yaml`

---

## Implementation Order

### Phase A: Loss Functions (1.5 hours)
**Priority**: Must implement first, used by all trainers

1. `src/training/losses.py` - All loss functions
   - Time: 1.5 hours

### Phase B: Base Training Infrastructure (2 hours)
**Priority**: Foundation for all trainers

2. `src/training/base_trainer.py` - Shared trainer logic
   - Time: 2 hours

### Phase C: Phase-Specific Trainers (3 hours)
**Priority**: Core training logic, implement sequentially

3. `src/training/phase1_trainer.py` - Contrastive pretraining
   - Time: 1 hour
4. `src/training/phase2_trainer.py` - Supervised curriculum
   - Time: 1.5 hours
5. `src/training/phase3_trainer.py` - RL fine-tuning
   - Time: 0.5 hours (inherits from base)

### Phase D: Configuration Files (0.5 hours)
**Priority**: Required to run trainers

6. Configuration YAML files
   - Time: 0.5 hours

### Phase E: Training Scripts (2 hours)
**Priority**: User-facing entry points

7. `scripts/train.py` - Main training script
   - Time: 0.5 hours
8. `scripts/generate_data.py` - Dataset generation
   - Time: 0.5 hours
9. `scripts/evaluate.py` - Evaluation script
   - Time: 0.5 hours
10. `scripts/simplify.py` - CLI inference
    - Time: 0.5 hours

### Phase F: Integration Testing (0.5 hours)
**Priority**: Verify everything works end-to-end

11. Run Phase 1 training on small dataset
12. Run Phase 2 training on small dataset
13. Run Phase 3 training on small dataset
    - Time: 0.5 hours total

---

## Component Specifications

---

## 1. Loss Functions (`src/training/losses.py`)

**Time**: 1.5 hours
**Dependencies**: None (imports from constants, utils)

### Purpose
All loss functions used across training phases.

### Functions

#### 1.1 InfoNCE Loss (Contrastive)
```python
def infonce_loss(
    obf_embeddings: torch.Tensor,
    simp_embeddings: torch.Tensor,
    temperature: float = INFONCE_TEMPERATURE
) -> torch.Tensor:
    """
    InfoNCE contrastive loss for equivalent expression pairs.

    Learns to map semantically equivalent expressions (obfuscated and simplified)
    to similar embedding vectors while pushing apart non-equivalent expressions.

    Args:
        obf_embeddings: [batch, hidden_dim] embeddings from obfuscated expressions
        simp_embeddings: [batch, hidden_dim] embeddings from simplified expressions
        temperature: Scaling factor for logits (default: 0.07 from constants)

    Returns:
        Scalar loss value

    Algorithm:
        For each obfuscated expression i:
        - Positive pair: (obf[i], simp[i]) - semantically equivalent
        - Negative pairs: (obf[i], simp[j]) for j != i - not equivalent

        Loss = -log(exp(sim(obf[i], simp[i])/τ) / Σ_j exp(sim(obf[i], simp[j])/τ))

        where sim(a, b) = cosine_similarity(a, b)

    Implementation:
        1. Normalize embeddings to unit vectors (L2 norm)
           CRITICAL: obf = F.normalize(obf, p=2, dim=-1)
                     simp = F.normalize(simp, p=2, dim=-1)
           Without normalization, cosine degenerates to dot product causing
           numerical instability (exp(large_value/τ) → Inf → NaN gradients)
        2. Compute similarity matrix: sim[i,j] = obf[i] · simp[j] (now true cosine)
        3. Scale by temperature: sim = sim / τ
        4. Apply log-softmax across each row
        5. Extract diagonal (positive pair logits)
        6. Return negative mean
    """
```

#### 1.2 MaskLM Loss (Masked Language Modeling)
```python
def masklm_loss(
    node_embeddings: torch.Tensor,
    original_node_features: torch.Tensor,
    mask_indices: torch.Tensor,
    prediction_head: nn.Module
) -> torch.Tensor:
    """
    Masked language modeling loss for expression structure learning.

    Masks random nodes in AST graph and predicts their types. Forces encoder
    to learn structural patterns in MBA expressions.

    Args:
        node_embeddings: [total_nodes, hidden_dim] encoder outputs
        original_node_features: [total_nodes] original node type IDs
        mask_indices: [num_masked] indices of masked nodes
        prediction_head: Linear layer mapping hidden_dim → NUM_NODE_TYPES

    Returns:
        Scalar cross-entropy loss

    Implementation:
        1. Extract embeddings at masked positions
        2. Pass through prediction head to get logits
        3. Compute cross-entropy with original node types
    """
```

#### 1.3 Copy Mechanism Loss
```python
def copy_loss(
    vocab_logits: torch.Tensor,
    copy_attn: torch.Tensor,
    p_gen: torch.Tensor,
    target_ids: torch.Tensor,
    source_tokens: torch.Tensor,
    pad_idx: int = PAD_IDX
) -> torch.Tensor:
    """
    Copy mechanism loss combining generation and copying.

    Learns when to generate from vocabulary vs. copy from source. Critical for
    preserving variable names in simplified expressions.

    Args:
        vocab_logits: [batch, tgt_len, vocab_size] vocabulary distribution
        copy_attn: [batch, tgt_len, src_len] attention over source tokens
        p_gen: [batch, tgt_len, 1] probability of generating vs. copying
        target_ids: [batch, tgt_len] target token IDs
        source_tokens: [batch, src_len] source token IDs
        pad_idx: Padding token ID to ignore

    Returns:
        Scalar loss value

    Algorithm:
        For each target position:
        P(token) = p_gen * P_vocab(token) + (1 - p_gen) * Σ_j copy_attn[j] * 1[src[j] == token]

        Loss = -log P(target_token)

    Implementation:
        1. Compute vocabulary distribution: softmax(vocab_logits)
        2. Compute copy distribution using scatter_add (NOT scatter):
           copy_dist = torch.zeros_like(vocab_dist)
           copy_dist.scatter_add_(dim=-1, index=source_tokens.expand_as(copy_attn), src=copy_attn)
           NOTE: scatter_add accumulates probabilities for duplicate source tokens.
                 scatter() would overwrite, producing incorrect distributions when
                 source has duplicates (e.g., "x+x+y" has two x tokens).
        3. Interpolate: final_dist = p_gen * vocab + (1 - p_gen) * copy
        4. Extract probabilities for target tokens
        5. Compute negative log-likelihood, ignoring pad tokens
        6. Return mean
    """
```

#### 1.4 Complexity Prediction Loss
```python
def complexity_loss(
    length_pred: torch.Tensor,
    depth_pred: torch.Tensor,
    target_ids: torch.Tensor,
    depth_labels: torch.Tensor,
    length_weight: float = 0.5,
    depth_weight: float = 0.5
) -> torch.Tensor:
    """
    Loss for predicting output complexity (length and depth).

    Helps model learn to produce appropriately-sized simplified expressions.

    Args:
        length_pred: [batch, MAX_OUTPUT_LENGTH] length prediction logits
        depth_pred: [batch, MAX_OUTPUT_DEPTH] depth prediction logits
        target_ids: [batch, tgt_len] target token IDs (for computing actual length)
        depth_labels: [batch] target depth labels
        length_weight: Weight for length loss
        depth_weight: Weight for depth loss

    Returns:
        Scalar loss value

    Implementation:
        1. Compute actual target lengths (excluding padding/special tokens)
        2. Length loss: cross-entropy(length_pred, actual_lengths)
        3. Depth loss: cross-entropy(depth_pred, depth_labels)
        4. Return weighted sum
    """
```

#### 1.5 PPO Policy Loss
```python
def ppo_policy_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    epsilon: float = PPO_EPSILON
) -> torch.Tensor:
    """
    Proximal Policy Optimization clipped objective.

    Args:
        log_probs: [batch, seq_len] current policy log probabilities
        old_log_probs: [batch, seq_len] old policy log probabilities
        advantages: [batch] advantage estimates
        epsilon: Clipping parameter (default: 0.2)

    Returns:
        Scalar loss value

    Algorithm:
        ratio = exp(log_probs - old_log_probs)
        clipped_ratio = clip(ratio, 1-ε, 1+ε)
        loss = -min(ratio * advantages, clipped_ratio * advantages)
    """
```

#### 1.6 PPO Value Loss
```python
def ppo_value_loss(
    value_pred: torch.Tensor,
    value_target: torch.Tensor,
    old_value_pred: torch.Tensor,
    epsilon: float = PPO_EPSILON
) -> torch.Tensor:
    """
    PPO value function loss with clipping.

    Args:
        value_pred: [batch, 1] current value predictions
        value_target: [batch, 1] target returns
        old_value_pred: [batch, 1] old value predictions
        epsilon: Clipping parameter

    Returns:
        Scalar MSE loss

    Implementation:
        Clipped value loss prevents value function from changing too quickly:
        loss = max((v - v_target)^2, (clip(v, v_old-ε, v_old+ε) - v_target)^2)
    """
```

#### 1.7 Combined Phase 2 Loss
```python
def phase2_loss(
    vocab_logits: torch.Tensor,
    copy_attn: torch.Tensor,
    p_gen: torch.Tensor,
    length_pred: torch.Tensor,
    depth_pred: torch.Tensor,
    target_ids: torch.Tensor,
    source_tokens: torch.Tensor,
    depth_labels: torch.Tensor,
    ce_weight: float = CE_WEIGHT,
    complexity_weight: float = COMPLEXITY_WEIGHT,
    copy_weight: float = COPY_WEIGHT
) -> Dict[str, torch.Tensor]:
    """
    Combined loss for Phase 2 supervised training.

    Returns:
        Dictionary with 'total', 'ce', 'complexity', 'copy' losses
    """
```

#### 1.8 Reward Function for RL
```python
def compute_reward(
    input_expr: str,
    pred_expr: str,
    verifier: ThreeTierVerifier
) -> float:
    """
    Compute reward for RL training based on equivalence and simplification.

    Args:
        input_expr: Original obfuscated expression
        pred_expr: Predicted simplified expression
        verifier: ThreeTierVerifier instance

    Returns:
        Scalar reward value

    Reward Components:
        +10.0: Equivalent (Z3 verified or execution tested)
        -5.0: Syntax error
        -5.0: Identity (output == input, no simplification)
        +2.0 * (1 - len_ratio): Simplification bonus
        -0.1 * len_ratio: Length penalty
        -0.2 * depth_ratio: Depth penalty

    Implementation:
        1. Check syntax validity (tier 1)
        2. Check execution equivalence (tier 2)
        3. Compute length/depth ratios
        4. Combine weighted components per constants.py
    """
```

### Testing
```python
# Test InfoNCE
batch_size = 32
hidden_dim = 256
obf = torch.randn(batch_size, hidden_dim)
simp = torch.randn(batch_size, hidden_dim)
loss = infonce_loss(obf, simp)
assert loss > 0

# Test MaskLM
num_nodes = 100
mask_idx = torch.randint(0, num_nodes, (15,))
embeddings = torch.randn(num_nodes, hidden_dim)
features = torch.randint(0, NUM_NODE_TYPES, (num_nodes,))
head = nn.Linear(hidden_dim, NUM_NODE_TYPES)
loss = masklm_loss(embeddings, features, mask_idx, head)
assert loss > 0

# Test Copy Loss
vocab_size = 300
tgt_len, src_len = 20, 30
vocab_logits = torch.randn(batch_size, tgt_len, vocab_size)
copy_attn = torch.softmax(torch.randn(batch_size, tgt_len, src_len), dim=-1)
p_gen = torch.sigmoid(torch.randn(batch_size, tgt_len, 1))
target_ids = torch.randint(0, vocab_size, (batch_size, tgt_len))
source_tokens = torch.randint(0, vocab_size, (batch_size, src_len))
loss = copy_loss(vocab_logits, copy_attn, p_gen, target_ids, source_tokens)
assert loss > 0
```

---

## 2. Base Trainer (`src/training/base_trainer.py`)

**Time**: 2 hours
**Dependencies**: `losses.py`, constants, utils

### Purpose
Shared training logic (checkpointing, logging, LR scheduling, gradient clipping) used by all phase trainers.

### Class: `BaseTrainer`

```python
class BaseTrainer:
    """
    Base trainer with shared functionality for all training phases.

    Responsibilities:
    - Model initialization and device management
    - Optimizer and LR scheduler setup
    - Checkpointing (save/load)
    - Logging (console + TensorBoard)
    - Gradient clipping and accumulation
    - Training step boilerplate
    """

    def __init__(
        self,
        model: MBADeobfuscator,
        config: Dict[str, Any],
        run_dir: str,
        device: str = 'cuda'
    ):
        """
        Initialize base trainer.

        Args:
            model: MBADeobfuscator instance
            config: Training configuration dict with keys:
                - learning_rate: float
                - weight_decay: float
                - gradient_clip: float
                - scheduler: str ('cosine', 'linear', 'constant')
                - warmup_steps: int
                - total_steps: int
            run_dir: Directory for saving checkpoints and logs
            device: Device for training ('cuda' or 'cpu')
        """
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Setup optimizer
        self.optimizer = self._create_optimizer()

        # Setup LR scheduler
        self.scheduler = self._create_scheduler()

        # Setup logging
        self.logger = self._setup_logger()
        self.writer = SummaryWriter(self.run_dir / 'tensorboard')

        # Training state
        self.global_step = 0
        self.epoch = 0

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """
        Create AdamW optimizer.

        Returns:
            Configured optimizer
        """
        lr = self.config.get('learning_rate', 1e-4)
        weight_decay = self.config.get('weight_decay', 0.01)

        return torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            betas=(0.9, 0.98),
            eps=1e-9,
            weight_decay=weight_decay
        )

    def _create_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """
        Create learning rate scheduler.

        Returns:
            LR scheduler or None if 'constant'
        """
        scheduler_type = self.config.get('scheduler', 'cosine')
        warmup_steps = self.config.get('warmup_steps', 1000)
        total_steps = self.config.get('total_steps', 100000)

        if scheduler_type == 'constant':
            return None
        elif scheduler_type == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps - warmup_steps
            )
        elif scheduler_type == 'linear':
            return torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.0,
                total_iters=total_steps - warmup_steps
            )
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    def _setup_logger(self) -> logging.Logger:
        """Setup logger for console and file output."""
        logger = logging.getLogger(f'trainer_{id(self)}')
        logger.setLevel(logging.INFO)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # File handler
        fh = logging.FileHandler(self.run_dir / 'train.log')
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        return logger

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        accumulation_steps: int = 1
    ) -> Dict[str, float]:
        """
        Single training step with gradient accumulation.

        Args:
            batch: Training batch
            accumulation_steps: Number of steps to accumulate gradients

        Returns:
            Dict with loss components

        Must be implemented by subclass to compute loss.
        """
        raise NotImplementedError("Subclass must implement train_step")

    def backward(
        self,
        loss: torch.Tensor,
        accumulation_steps: int = 1,
        update: bool = True
    ):
        """
        Backward pass with gradient clipping and accumulation.

        Args:
            loss: Loss tensor
            accumulation_steps: Gradient accumulation steps
            update: Whether to update weights (True every accumulation_steps)
        """
        # Scale loss for accumulation
        loss = loss / accumulation_steps
        loss.backward()

        if update:
            # Gradient clipping
            max_norm = self.config.get('gradient_clip', 1.0)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)

            # Optimizer step
            self.optimizer.step()
            self.optimizer.zero_grad()

            # Scheduler step (if exists)
            if self.scheduler is not None:
                self.scheduler.step()

            self.global_step += 1

    def save_checkpoint(self, filename: str = 'checkpoint.pt'):
        """
        Save training checkpoint.

        Args:
            filename: Checkpoint filename
        """
        checkpoint_path = self.run_dir / filename

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'epoch': self.epoch,
            'config': self.config,
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Saved checkpoint to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load training checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.global_step = checkpoint.get('global_step', 0)
        self.epoch = checkpoint.get('epoch', 0)

        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.logger.info(f"Loaded checkpoint from {checkpoint_path}")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log metrics to console and TensorBoard.

        Args:
            metrics: Dictionary of metric name -> value
            step: Global step (default: self.global_step)
        """
        step = step or self.global_step

        # Console log
        metric_str = ' | '.join(f"{k}: {v:.4f}" for k, v in metrics.items())
        self.logger.info(f"Step {step} | {metric_str}")

        # TensorBoard log
        for name, value in metrics.items():
            self.writer.add_scalar(name, value, step)

    def close(self):
        """Close resources (TensorBoard writer)."""
        self.writer.close()
```

### Testing
```python
# Test base trainer initialization
config = {
    'learning_rate': 1e-4,
    'weight_decay': 0.01,
    'gradient_clip': 1.0,
    'scheduler': 'cosine',
    'warmup_steps': 100,
    'total_steps': 1000,
}
model = MBADeobfuscator(encoder_type='gat')
trainer = BaseTrainer(model, config, run_dir='test_runs/base')
assert trainer.optimizer is not None
assert trainer.scheduler is not None
assert (Path('test_runs/base') / 'train.log').exists()

# Test save/load checkpoint
trainer.save_checkpoint('test.pt')
trainer.load_checkpoint('test_runs/base/test.pt')
assert trainer.global_step == 0
```

---

## 3. Phase 1 Trainer (`src/training/phase1_trainer.py`)

**Time**: 1 hour
**Dependencies**: `base_trainer.py`, `losses.py`, `ContrastiveDataset`

### Purpose
Contrastive pretraining with InfoNCE + MaskLM losses.

### Class: `Phase1Trainer`

```python
class Phase1Trainer(BaseTrainer):
    """
    Phase 1: Contrastive pretraining.

    Objectives:
    - Learn to map equivalent expressions to similar embeddings (InfoNCE)
    - Learn expression structure (MaskLM)

    Training:
    - Dataset: ContrastiveDataset (obfuscated/simplified pairs)
    - Loss: InfoNCE + MaskLM (weighted combination)
    - Epochs: 20 (from constants.CURRICULUM_STAGES)
    - Only trains encoder, decoder not used
    """

    def __init__(
        self,
        model: MBADeobfuscator,
        config: Dict[str, Any],
        run_dir: str,
        device: str = 'cuda'
    ):
        """
        Initialize Phase 1 trainer.

        Args:
            model: MBADeobfuscator instance
            config: Training config (includes infonce_temp, masklm_ratio, masklm_weight)
            run_dir: Run directory
            device: Training device
        """
        super().__init__(model, config, run_dir, device)

        # Phase 1 specific params
        self.infonce_temp = config.get('infonce_temperature', INFONCE_TEMPERATURE)
        self.mask_ratio = config.get('masklm_mask_ratio', MASKLM_MASK_RATIO)
        self.masklm_weight = config.get('masklm_weight', MASKLM_WEIGHT)

        # MaskLM prediction head
        hidden_dim = model.graph_encoder.hidden_dim
        self.mlm_head = nn.Linear(hidden_dim, NUM_NODE_TYPES).to(device)

        # Add mlm_head parameters to optimizer
        self.optimizer.add_param_group({'params': self.mlm_head.parameters()})

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        accumulation_steps: int = 1
    ) -> Dict[str, float]:
        """
        Training step for Phase 1.

        Args:
            batch: Batch from ContrastiveDataset with keys:
                - obf_graph_batch: PyG batch for obfuscated
                - simp_graph_batch: PyG batch for simplified
                - obf_fingerprint: [batch, FINGERPRINT_DIM]
                - simp_fingerprint: [batch, FINGERPRINT_DIM]

        Returns:
            Dict with 'total', 'infonce', 'masklm' losses
        """
        # Move to device
        obf_graph = batch['obf_graph_batch'].to(self.device)
        simp_graph = batch['simp_graph_batch'].to(self.device)
        obf_fp = batch['obf_fingerprint'].to(self.device)
        simp_fp = batch['simp_fingerprint'].to(self.device)

        # Encode both expressions
        obf_context = self.model.encode(obf_graph, obf_fp)  # [batch, 1, d_model]
        simp_context = self.model.encode(simp_graph, simp_fp)

        # Squeeze to [batch, d_model]
        obf_embed = obf_context.squeeze(1)
        simp_embed = simp_context.squeeze(1)

        # InfoNCE loss
        infonce = infonce_loss(obf_embed, simp_embed, temperature=self.infonce_temp)

        # MaskLM loss (only on obfuscated graph)
        masked_loss = self._masklm_step(obf_graph)

        # Combined loss
        total_loss = infonce + self.masklm_weight * masked_loss

        # Backward
        self.backward(total_loss, accumulation_steps, update=True)

        return {
            'total': total_loss.item(),
            'infonce': infonce.item(),
            'masklm': masked_loss.item(),
        }

    def _masklm_step(self, graph_batch) -> torch.Tensor:
        """
        Masked language modeling step.

        Args:
            graph_batch: PyG batch

        Returns:
            MaskLM loss

        Algorithm:
            1. Randomly mask MASKLM_MASK_RATIO of nodes
            2. Encode graph with encoder (encoder handles masked nodes)
            3. Predict original node types for masked positions
            4. Compute cross-entropy loss
        """
        # Get node features
        x = graph_batch.x  # [total_nodes, node_dim] or [total_nodes]
        edge_index = graph_batch.edge_index
        batch_idx = graph_batch.batch
        edge_type = getattr(graph_batch, 'edge_type', None)

        num_nodes = x.shape[0]
        num_masked = int(num_nodes * self.mask_ratio)

        if num_masked == 0:
            return torch.tensor(0.0, device=self.device)

        # Randomly select nodes to mask
        mask_indices = torch.randperm(num_nodes, device=self.device)[:num_masked]

        # Store original features
        if x.dim() == 1:
            # Node type IDs
            original_types = x[mask_indices].clone()
            # Replace with special mask token (use 0 for now)
            x_masked = x.clone()
            x_masked[mask_indices] = 0
        else:
            # One-hot features
            original_types = x[mask_indices].argmax(dim=-1)
            x_masked = x.clone()
            x_masked[mask_indices] = 0

        # Encode with masked nodes
        # NOTE: Parameter order is (x, edge_index, edge_type, batch) for GGNN/HGT/RGCN
        if edge_type is not None:
            node_embeddings = self.model.graph_encoder(x_masked, edge_index, edge_type, batch_idx)
        else:
            node_embeddings = self.model.graph_encoder(x_masked, edge_index, batch_idx)

        # Predict masked node types
        loss = masklm_loss(
            node_embeddings,
            original_types,
            mask_indices,
            self.mlm_head
        )

        return loss

    def train_epoch(
        self,
        dataloader: DataLoader,
        accumulation_steps: int = 1
    ) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            dataloader: ContrastiveDataset DataLoader
            accumulation_steps: Gradient accumulation steps

        Returns:
            Average epoch metrics
        """
        self.model.train()
        self.mlm_head.train()

        epoch_metrics = defaultdict(float)
        num_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            # Training step
            metrics = self.train_step(batch, accumulation_steps)

            # Accumulate metrics
            for key, value in metrics.items():
                epoch_metrics[key] += value
            num_batches += 1

            # Log every 100 steps
            if batch_idx % 100 == 0:
                self.log_metrics(metrics)

        # Average metrics
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches

        self.epoch += 1
        return dict(epoch_metrics)
```

### Testing
```python
# Test Phase 1 trainer
from src.data.dataset import ContrastiveDataset
from src.data.collate import collate_contrastive

dataset = ContrastiveDataset('data/train.jsonl', tokenizer, fingerprint)
dataloader = DataLoader(dataset, batch_size=4, collate_fn=collate_contrastive)

model = MBADeobfuscator(encoder_type='gat')
config = {'learning_rate': 1e-4, 'infonce_temperature': 0.07, 'masklm_weight': 0.5}
trainer = Phase1Trainer(model, config, 'test_runs/phase1')

# Run one epoch
metrics = trainer.train_epoch(dataloader)
assert 'infonce' in metrics
assert 'masklm' in metrics
```

---

## 4. Phase 2 Trainer (`src/training/phase2_trainer.py`)

**Time**: 1.5 hours
**Dependencies**: `base_trainer.py`, `losses.py`, `MBADataset`

### Purpose
Supervised training with curriculum learning (4 stages: depth 2→5→10→14).

### Class: `Phase2Trainer`

```python
class Phase2Trainer(BaseTrainer):
    """
    Phase 2: Supervised seq2seq training with curriculum learning.

    Curriculum Stages (from constants.CURRICULUM_STAGES):
    1. Max depth 2, 10 epochs, target 95% accuracy
    2. Max depth 5, 15 epochs, target 90% accuracy
    3. Max depth 10, 15 epochs, target 80% accuracy
    4. Max depth 14, 10 epochs, target 70% accuracy

    Loss Components:
    - Cross-entropy (generation)
    - Copy mechanism
    - Complexity prediction

    Automatic stage progression when target accuracy reached.
    """

    def __init__(
        self,
        model: MBADeobfuscator,
        config: Dict[str, Any],
        train_data_path: str,
        val_data_path: str,
        tokenizer: MBATokenizer,
        fingerprint: SemanticFingerprint,
        run_dir: str,
        device: str = 'cuda'
    ):
        """
        Initialize Phase 2 trainer.

        Args:
            model: MBADeobfuscator instance
            config: Training config
            train_data_path: Path to training data JSONL
            val_data_path: Path to validation data JSONL
            tokenizer: MBATokenizer instance
            fingerprint: SemanticFingerprint instance
            run_dir: Run directory
            device: Device
        """
        super().__init__(model, config, run_dir, device)

        self.tokenizer = tokenizer
        self.fingerprint = fingerprint
        self.train_data_path = train_data_path
        self.val_data_path = val_data_path

        # Loss weights
        self.ce_weight = config.get('ce_weight', CE_WEIGHT)
        self.complexity_weight = config.get('complexity_weight', COMPLEXITY_WEIGHT)
        self.copy_weight = config.get('copy_weight', COPY_WEIGHT)

        # Curriculum learning
        self.curriculum_stages = config.get('curriculum_stages', CURRICULUM_STAGES)
        self.current_stage = 0
        self.stage_epoch = 0

        # Self-paced learning
        self.use_self_paced = config.get('use_self_paced', True)
        self.sp_lambda = SELF_PACED_LAMBDA_INIT
        self.sp_growth = SELF_PACED_LAMBDA_GROWTH

        # Load datasets for current stage
        self._load_stage_datasets()

    def _load_stage_datasets(self):
        """Load train/val datasets filtered by current stage max_depth."""
        stage = self.curriculum_stages[self.current_stage]
        max_depth = stage['max_depth']

        self.logger.info(f"Loading stage {self.current_stage + 1} datasets (max_depth={max_depth})")

        # Training dataset
        self.train_dataset = MBADataset(
            self.train_data_path,
            self.tokenizer,
            self.fingerprint,
            max_depth=max_depth
        )

        # Validation dataset
        self.val_dataset = MBADataset(
            self.val_data_path,
            self.tokenizer,
            self.fingerprint,
            max_depth=max_depth
        )

        # Create dataloaders
        batch_size = self.config.get('batch_size', 32)

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_graphs,
            num_workers=4
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_graphs,
            num_workers=4
        )

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        accumulation_steps: int = 1
    ) -> Dict[str, float]:
        """
        Training step for Phase 2.

        Args:
            batch: Batch from MBADataset with keys:
                - graph_batch: PyG batch
                - fingerprint: [batch, FINGERPRINT_DIM]
                - target_ids: [batch, tgt_len]
                - source_tokens: [batch, src_len]
                - depth: [batch]

        Returns:
            Dict with loss components
        """
        # Move to device
        graph_batch = batch['graph_batch'].to(self.device)
        fingerprint = batch['fingerprint'].to(self.device)
        target_ids = batch['target_ids'].to(self.device)
        source_tokens = batch['source_tokens'].to(self.device)
        depth = batch['depth'].to(self.device)

        # Forward pass
        outputs = self.model(graph_batch, fingerprint, target_ids[:, :-1])

        # Compute loss
        loss_dict = phase2_loss(
            vocab_logits=outputs['vocab_logits'],
            copy_attn=outputs['copy_attn'],
            p_gen=outputs['p_gen'],
            length_pred=outputs['length_pred'],
            depth_pred=outputs['depth_pred'],
            target_ids=target_ids,
            source_tokens=source_tokens,
            depth_labels=depth,
            ce_weight=self.ce_weight,
            complexity_weight=self.complexity_weight,
            copy_weight=self.copy_weight
        )

        total_loss = loss_dict['total']

        # Self-paced learning: weight by loss
        if self.use_self_paced:
            # Compute sample weights: w_i = 1 if loss_i < λ, else 0
            sample_losses = loss_dict.get('sample_losses', None)
            if sample_losses is not None:
                weights = (sample_losses < self.sp_lambda).float()
                total_loss = (total_loss * weights).mean()

        # Backward
        self.backward(total_loss, accumulation_steps, update=True)

        return {k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}

    def train_epoch(
        self,
        accumulation_steps: int = 1
    ) -> Dict[str, float]:
        """Train for one epoch on current curriculum stage."""
        self.model.train()

        epoch_metrics = defaultdict(float)
        num_batches = 0

        for batch_idx, batch in enumerate(self.train_loader):
            metrics = self.train_step(batch, accumulation_steps)

            for key, value in metrics.items():
                epoch_metrics[key] += value
            num_batches += 1

            if batch_idx % 100 == 0:
                self.log_metrics(metrics)

        # Average metrics
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches

        self.epoch += 1
        self.stage_epoch += 1

        # Update self-paced lambda with upper bound
        # Cap at 10.0 to maintain filtering of outliers (loss > 10 indicates severe failure)
        # Without cap: λ = 0.5 × 1.1^50 ≈ 58.6, making all samples pass (defeats purpose)
        if self.use_self_paced:
            self.sp_lambda = min(self.sp_lambda * self.sp_growth, 10.0)

        return dict(epoch_metrics)

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate on validation set.

        Returns:
            Dict with 'exact_match', 'syntax_valid', 'avg_loss'
        """
        self.model.eval()

        total_loss = 0.0
        exact_matches = 0
        syntax_valid_count = 0
        total_samples = 0

        for batch in self.val_loader:
            # Move to device
            graph_batch = batch['graph_batch'].to(self.device)
            fingerprint = batch['fingerprint'].to(self.device)
            target_ids = batch['target_ids'].to(self.device)
            source_tokens = batch['source_tokens'].to(self.device)
            depth = batch['depth'].to(self.device)

            # Forward pass
            outputs = self.model(graph_batch, fingerprint, target_ids[:, :-1])

            # Compute loss
            loss_dict = phase2_loss(
                vocab_logits=outputs['vocab_logits'],
                copy_attn=outputs['copy_attn'],
                p_gen=outputs['p_gen'],
                length_pred=outputs['length_pred'],
                depth_pred=outputs['depth_pred'],
                target_ids=target_ids,
                source_tokens=source_tokens,
                depth_labels=depth,
                ce_weight=self.ce_weight,
                complexity_weight=self.complexity_weight,
                copy_weight=self.copy_weight
            )

            total_loss += loss_dict['total'].item() * len(batch['simplified'])

            # Greedy decode for accuracy
            predictions = self._greedy_decode_batch(graph_batch, fingerprint)
            targets = batch['simplified']

            for pred, tgt in zip(predictions, targets):
                total_samples += 1
                if exact_match(pred, tgt):
                    exact_matches += 1
                if syntax_valid(pred):
                    syntax_valid_count += 1

        return {
            'exact_match': exact_matches / max(total_samples, 1),
            'syntax_valid': syntax_valid_count / max(total_samples, 1),
            'avg_loss': total_loss / max(total_samples, 1),
        }

    def _greedy_decode_batch(
        self,
        graph_batch,
        fingerprint: torch.Tensor
    ) -> List[str]:
        """Greedy decode batch of expressions."""
        memory = self.model.encode(graph_batch, fingerprint)
        batch_size = memory.shape[0]

        # Start with SOS
        output = torch.full(
            (batch_size, 1), SOS_IDX, dtype=torch.long, device=self.device
        )

        for _ in range(MAX_SEQ_LEN - 1):
            decode_output = self.model.decode(output, memory)
            logits = decode_output['vocab_logits'][:, -1, :]
            next_token = logits.argmax(dim=-1, keepdim=True)
            output = torch.cat([output, next_token], dim=1)

            if (next_token.squeeze(-1) == EOS_IDX).all():
                break

        # Decode to strings
        predictions = []
        for i in range(batch_size):
            seq = output[i].tolist()
            pred = self.tokenizer.decode(seq)
            predictions.append(pred)

        return predictions

    def should_advance_stage(self, val_metrics: Dict[str, float]) -> bool:
        """
        Check if should advance to next curriculum stage.

        Args:
            val_metrics: Validation metrics

        Returns:
            True if should advance
        """
        stage = self.curriculum_stages[self.current_stage]
        target_acc = stage['target']
        max_epochs = stage['epochs']

        current_acc = val_metrics.get('exact_match', 0.0)

        # Advance if target reached OR max epochs exhausted
        if current_acc >= target_acc or self.stage_epoch >= max_epochs:
            return True

        return False

    def advance_stage(self):
        """Advance to next curriculum stage."""
        if self.current_stage < len(self.curriculum_stages) - 1:
            self.current_stage += 1
            self.stage_epoch = 0
            self.sp_lambda = SELF_PACED_LAMBDA_INIT  # Reset self-paced lambda

            self.logger.info(f"Advancing to stage {self.current_stage + 1}")
            self._load_stage_datasets()

            return True
        else:
            self.logger.info("All curriculum stages completed")
            return False

    def train_curriculum(self):
        """
        Train through all curriculum stages.

        Main training loop for Phase 2.
        """
        while self.current_stage < len(self.curriculum_stages):
            stage = self.curriculum_stages[self.current_stage]
            self.logger.info(f"Stage {self.current_stage + 1}/{len(self.curriculum_stages)}: "
                           f"max_depth={stage['max_depth']}, "
                           f"target_acc={stage['target']}")

            # Train until stage completion
            while True:
                # Train epoch
                train_metrics = self.train_epoch()
                self.log_metrics(train_metrics)

                # Evaluate
                val_metrics = self.evaluate()
                self.log_metrics({f'val_{k}': v for k, v in val_metrics.items()})

                # Save checkpoint
                self.save_checkpoint(f'stage{self.current_stage + 1}_epoch{self.stage_epoch}.pt')

                # Check if should advance
                if self.should_advance_stage(val_metrics):
                    if not self.advance_stage():
                        # All stages complete
                        return
                    break
```

### Testing
```python
# Test Phase 2 trainer
model = MBADeobfuscator(encoder_type='gat', vocab_size=300)
config = {
    'batch_size': 4,
    'learning_rate': 1e-4,
    'curriculum_stages': CURRICULUM_STAGES[:2],  # Test first 2 stages
}
trainer = Phase2Trainer(
    model, config,
    train_data_path='data/train.jsonl',
    val_data_path='data/val.jsonl',
    tokenizer=tokenizer,
    fingerprint=fingerprint,
    run_dir='test_runs/phase2'
)

# Run one epoch
metrics = trainer.train_epoch()
assert 'total' in metrics

# Evaluate
val_metrics = trainer.evaluate()
assert 'exact_match' in val_metrics
```

---

## 5. Phase 3 Trainer (`src/training/phase3_trainer.py`)

**Time**: 0.5 hours
**Dependencies**: `base_trainer.py`, `losses.py`, `verify.py`

### Purpose
RL fine-tuning with PPO, equivalence rewards, anti-identity penalty.

### Class: `Phase3Trainer`

```python
class Phase3Trainer(BaseTrainer):
    """
    Phase 3: RL fine-tuning with PPO.

    Reward Function:
    - +10.0: Z3/execution verified equivalent
    - -5.0: Syntax error
    - -5.0: Identity (no simplification)
    - +2.0 * (1 - len_ratio): Simplification bonus
    - -0.1 * len_ratio: Length penalty
    - -0.2 * depth_ratio: Depth penalty

    PPO hyperparameters from constants:
    - epsilon: 0.2
    - value_coef: 0.5
    - entropy_coef: 0.01
    """

    def __init__(
        self,
        model: MBADeobfuscator,
        config: Dict[str, Any],
        train_data_path: str,
        tokenizer: MBATokenizer,
        fingerprint: SemanticFingerprint,
        verifier: ThreeTierVerifier,
        run_dir: str,
        device: str = 'cuda'
    ):
        """
        Initialize Phase 3 trainer.

        Args:
            model: MBADeobfuscator (pretrained from Phase 2)
            config: Training config
            train_data_path: Training data path
            tokenizer: MBATokenizer
            fingerprint: SemanticFingerprint
            verifier: ThreeTierVerifier for reward computation
            run_dir: Run directory
            device: Device
        """
        super().__init__(model, config, run_dir, device)

        self.tokenizer = tokenizer
        self.fingerprint = fingerprint
        self.verifier = verifier

        # PPO hyperparameters
        self.ppo_epsilon = config.get('ppo_epsilon', PPO_EPSILON)
        self.value_coef = config.get('ppo_value_coef', PPO_VALUE_COEF)
        self.entropy_coef = config.get('ppo_entropy_coef', PPO_ENTROPY_COEF)

        # Experience buffer
        self.rollout_steps = config.get('rollout_steps', 10)

        # Load dataset
        self.train_dataset = MBADataset(
            train_data_path, self.tokenizer, self.fingerprint
        )
        batch_size = config.get('batch_size', 16)
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_graphs
        )

    def collect_rollout(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """
        Collect rollout (sample from policy, compute rewards).

        Args:
            batch: Batch from MBADataset

        Returns:
            Dict with sampled_ids, log_probs, values, rewards, advantages
        """
        graph_batch = batch['graph_batch'].to(self.device)
        fingerprint = batch['fingerprint'].to(self.device)
        input_exprs = batch['obfuscated']

        # Encode
        memory = self.model.encode(graph_batch, fingerprint)
        batch_size = memory.shape[0]

        # Sample from policy
        sampled_ids = []
        log_probs = []

        output = torch.full(
            (batch_size, 1), SOS_IDX, dtype=torch.long, device=self.device
        )

        for _ in range(MAX_SEQ_LEN - 1):
            decode_output = self.model.decode(output, memory)
            logits = decode_output['vocab_logits'][:, -1, :]

            # Sample from distribution
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            next_token = dist.sample().unsqueeze(1)
            log_prob = dist.log_prob(next_token.squeeze(1))

            sampled_ids.append(next_token)
            log_probs.append(log_prob)

            output = torch.cat([output, next_token], dim=1)

            if (next_token.squeeze(-1) == EOS_IDX).all():
                break

        # Stack
        sampled_ids = torch.cat(sampled_ids, dim=1)  # [batch, seq_len]
        log_probs = torch.stack(log_probs, dim=1)  # [batch, seq_len]

        # Get values
        values = self.model.get_value(graph_batch, fingerprint)  # [batch, 1]

        # Decode to strings
        pred_exprs = []
        for i in range(batch_size):
            seq = sampled_ids[i].tolist()
            pred = self.tokenizer.decode(seq)
            pred_exprs.append(pred)

        # Compute rewards
        rewards = []
        for input_expr, pred_expr in zip(input_exprs, pred_exprs):
            reward = compute_reward(input_expr, pred_expr, self.verifier)
            rewards.append(reward)

        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)

        # Compute advantages: A = R - V
        advantages = rewards.unsqueeze(1) - values

        return {
            'sampled_ids': sampled_ids,
            'log_probs': log_probs,
            'old_log_probs': log_probs.detach(),
            'values': values,
            'rewards': rewards,
            'advantages': advantages.detach(),
        }

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        accumulation_steps: int = 1
    ) -> Dict[str, float]:
        """
        PPO training step.

        Args:
            batch: Batch from MBADataset
            accumulation_steps: Gradient accumulation

        Returns:
            Dict with policy_loss, value_loss, entropy, reward metrics
        """
        # Collect rollout
        rollout = self.collect_rollout(batch)

        # Re-evaluate with current policy (for PPO ratio)
        graph_batch = batch['graph_batch'].to(self.device)
        fingerprint = batch['fingerprint'].to(self.device)
        sampled_ids = rollout['sampled_ids']

        memory = self.model.encode(graph_batch, fingerprint)

        # Decode to get current log probs
        decode_output = self.model.decode(sampled_ids, memory)
        logits = decode_output['vocab_logits']

        # Compute log probs for sampled actions
        log_probs = torch.log_softmax(logits, dim=-1)
        sampled_log_probs = torch.gather(
            log_probs, dim=-1, index=sampled_ids.unsqueeze(-1)
        ).squeeze(-1)

        # Get current value
        current_values = self.model.get_value(graph_batch, fingerprint)

        # PPO losses
        policy_loss = ppo_policy_loss(
            sampled_log_probs,
            rollout['old_log_probs'],
            rollout['advantages'],
            epsilon=self.ppo_epsilon
        )

        value_loss = ppo_value_loss(
            current_values,
            rollout['rewards'].unsqueeze(1),
            rollout['values'],
            epsilon=self.ppo_epsilon
        )

        # Entropy bonus
        probs = torch.softmax(logits, dim=-1)
        entropy = -(probs * torch.log_softmax(logits, dim=-1)).sum(dim=-1).mean()

        # Total loss
        total_loss = (
            policy_loss
            + self.value_coef * value_loss
            - self.entropy_coef * entropy
        )

        # Backward
        self.backward(total_loss, accumulation_steps, update=True)

        return {
            'total': total_loss.item(),
            'policy': policy_loss.item(),
            'value': value_loss.item(),
            'entropy': entropy.item(),
            'avg_reward': rollout['rewards'].mean().item(),
        }

    def train_epoch(self, accumulation_steps: int = 1) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        epoch_metrics = defaultdict(float)
        num_batches = 0

        for batch_idx, batch in enumerate(self.train_loader):
            metrics = self.train_step(batch, accumulation_steps)

            for key, value in metrics.items():
                epoch_metrics[key] += value
            num_batches += 1

            if batch_idx % 50 == 0:
                self.log_metrics(metrics)

        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches

        self.epoch += 1
        return dict(epoch_metrics)
```

### Testing
```python
# Test Phase 3 trainer
model = MBADeobfuscator(encoder_type='gat', vocab_size=300)
verifier = ThreeTierVerifier(tokenizer)
config = {'batch_size': 4, 'learning_rate': 1e-5}

trainer = Phase3Trainer(
    model, config,
    train_data_path='data/train.jsonl',
    tokenizer=tokenizer,
    fingerprint=fingerprint,
    verifier=verifier,
    run_dir='test_runs/phase3'
)

# Run one epoch
metrics = trainer.train_epoch()
assert 'policy' in metrics
assert 'value' in metrics
assert 'avg_reward' in metrics
```

---

## 6. Configuration Files

**Time**: 0.5 hours
**Dependencies**: None

### 6.1 `configs/phase1.yaml`

```yaml
# Phase 1: Contrastive Pretraining Configuration

model:
  encoder_type: gat  # or ggnn, hgt, rgcn
  hidden_dim: 256
  d_model: 512
  vocab_size: 300

training:
  learning_rate: 1e-4
  weight_decay: 0.01
  gradient_clip: 1.0
  scheduler: cosine
  warmup_steps: 1000
  total_steps: 100000  # 20 epochs * 5000 batches/epoch

  # Phase 1 specific
  infonce_temperature: 0.07
  masklm_mask_ratio: 0.15
  masklm_weight: 0.5

  batch_size: 64
  epochs: 20
  accumulation_steps: 1

data:
  train_path: data/train_contrastive.jsonl
  val_path: data/val_contrastive.jsonl

logging:
  log_interval: 100
  eval_interval: 1000
  save_interval: 5000

run:
  run_dir: runs/phase1
  seed: 42
```

### 6.2 `configs/phase2.yaml`

```yaml
# Phase 2: Supervised Curriculum Training Configuration

model:
  encoder_type: gat
  hidden_dim: 256
  d_model: 512
  vocab_size: 300
  # Load from Phase 1 checkpoint
  pretrained_encoder: runs/phase1/checkpoint_final.pt

training:
  learning_rate: 5e-5
  weight_decay: 0.01
  gradient_clip: 1.0
  scheduler: cosine
  warmup_steps: 2000
  total_steps: 250000  # 50 epochs * 5000 batches/epoch

  # Phase 2 specific
  ce_weight: 1.0
  complexity_weight: 0.1
  copy_weight: 0.1

  # Curriculum learning
  use_curriculum: true
  curriculum_stages:
    - max_depth: 2
      epochs: 10
      target: 0.95
    - max_depth: 5
      epochs: 15
      target: 0.90
    - max_depth: 10
      epochs: 15
      target: 0.80
    - max_depth: 14
      epochs: 10
      target: 0.70

  # Self-paced learning
  use_self_paced: true
  sp_lambda_init: 0.5
  sp_lambda_growth: 1.1

  batch_size: 32
  accumulation_steps: 2

data:
  train_path: data/train_supervised.jsonl
  val_path: data/val_supervised.jsonl

logging:
  log_interval: 100
  eval_interval: 1000
  save_interval: 5000

run:
  run_dir: runs/phase2
  seed: 42
```

### 6.3 `configs/phase3.yaml`

```yaml
# Phase 3: RL Fine-tuning Configuration

model:
  encoder_type: gat
  hidden_dim: 256
  d_model: 512
  vocab_size: 300
  # Load from Phase 2 checkpoint
  pretrained_model: runs/phase2/checkpoint_final.pt

training:
  learning_rate: 1e-5
  weight_decay: 0.01
  gradient_clip: 0.5
  scheduler: constant

  # Phase 3 specific (PPO)
  ppo_epsilon: 0.2
  ppo_value_coef: 0.5
  ppo_entropy_coef: 0.01

  rollout_steps: 10

  batch_size: 16
  epochs: 10
  accumulation_steps: 4

data:
  train_path: data/train_supervised.jsonl  # Reuse Phase 2 data

verification:
  exec_samples: 100
  z3_timeout_ms: 1000

logging:
  log_interval: 50
  eval_interval: 500
  save_interval: 2000

run:
  run_dir: runs/phase3
  seed: 42
```

---

## 7. Main Training Script (`scripts/train.py`)

**Time**: 0.5 hours
**Dependencies**: All trainers, config system

```python
#!/usr/bin/env python3
"""
Main training script for MBA Deobfuscator.

Usage:
    python scripts/train.py --phase 1 --config configs/phase1.yaml
    python scripts/train.py --phase 2 --config configs/phase2.yaml
    python scripts/train.py --phase 3 --config configs/phase3.yaml
"""

import argparse
import random
import numpy as np
import torch
from pathlib import Path

from src.utils.config import Config
from src.models.full_model import MBADeobfuscator
from src.data.tokenizer import MBATokenizer
from src.data.fingerprint import SemanticFingerprint
from src.training.phase1_trainer import Phase1Trainer
from src.training.phase2_trainer import Phase2Trainer
from src.training.phase3_trainer import Phase3Trainer
from src.inference.verify import ThreeTierVerifier


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description='Train MBA Deobfuscator')
    parser.add_argument('--phase', type=int, required=True, choices=[1, 2, 3],
                       help='Training phase (1=contrastive, 2=supervised, 3=RL)')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config YAML file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda or cpu)')

    args = parser.parse_args()

    # Load config
    config = Config(args.config)

    # Set seed
    seed = config.get('run.seed', 42)
    set_seed(seed)

    # Initialize tokenizer and fingerprint
    tokenizer = MBATokenizer()
    fingerprint = SemanticFingerprint(seed=seed)

    # Initialize model
    model_config = config.to_dict().get('model', {})
    encoder_type = model_config.get('encoder_type', 'gat')
    model = MBADeobfuscator(encoder_type=encoder_type, **model_config)

    # Load pretrained weights if specified
    if args.phase == 2 and 'pretrained_encoder' in model_config:
        checkpoint_path = model_config['pretrained_encoder']
        print(f"Loading pretrained encoder from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        # Load only encoder weights
        encoder_state = {k: v for k, v in checkpoint['model_state_dict'].items()
                        if k.startswith('graph_encoder') or k.startswith('fingerprint_encoder')}
        model.load_state_dict(encoder_state, strict=False)

    if args.phase == 3 and 'pretrained_model' in model_config:
        checkpoint_path = model_config['pretrained_model']
        print(f"Loading pretrained model from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])

    # Get training config
    training_config = config.to_dict().get('training', {})
    data_config = config.to_dict().get('data', {})
    run_dir = config.get('run.run_dir', f'runs/phase{args.phase}')

    # Initialize trainer based on phase
    if args.phase == 1:
        trainer = Phase1Trainer(
            model=model,
            config=training_config,
            run_dir=run_dir,
            device=args.device
        )

        # Load datasets for Phase 1
        from src.data.dataset import ContrastiveDataset
        from src.data.collate import collate_contrastive
        from torch.utils.data import DataLoader

        train_dataset = ContrastiveDataset(
            data_config['train_path'], tokenizer, fingerprint
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=training_config.get('batch_size', 64),
            shuffle=True,
            collate_fn=collate_contrastive,
            num_workers=4
        )

        # Train
        epochs = training_config.get('epochs', 20)
        for epoch in range(epochs):
            print(f"\n=== Epoch {epoch + 1}/{epochs} ===")
            metrics = trainer.train_epoch(train_loader)
            print(f"Train metrics: {metrics}")

            # Save checkpoint
            trainer.save_checkpoint(f'checkpoint_epoch{epoch + 1}.pt')

        trainer.save_checkpoint('checkpoint_final.pt')

    elif args.phase == 2:
        trainer = Phase2Trainer(
            model=model,
            config=training_config,
            train_data_path=data_config['train_path'],
            val_data_path=data_config['val_path'],
            tokenizer=tokenizer,
            fingerprint=fingerprint,
            run_dir=run_dir,
            device=args.device
        )

        # Resume if specified
        if args.resume:
            trainer.load_checkpoint(args.resume)

        # Train with curriculum
        trainer.train_curriculum()
        trainer.save_checkpoint('checkpoint_final.pt')

    elif args.phase == 3:
        # Initialize verifier for Phase 3
        verification_config = config.to_dict().get('verification', {})
        verifier = ThreeTierVerifier(
            tokenizer=tokenizer,
            exec_samples=verification_config.get('exec_samples', 100),
            z3_timeout_ms=verification_config.get('z3_timeout_ms', 1000)
        )

        trainer = Phase3Trainer(
            model=model,
            config=training_config,
            train_data_path=data_config['train_path'],
            tokenizer=tokenizer,
            fingerprint=fingerprint,
            verifier=verifier,
            run_dir=run_dir,
            device=args.device
        )

        # Resume if specified
        if args.resume:
            trainer.load_checkpoint(args.resume)

        # Train
        epochs = training_config.get('epochs', 10)
        for epoch in range(epochs):
            print(f"\n=== Epoch {epoch + 1}/{epochs} ===")
            metrics = trainer.train_epoch()
            print(f"Train metrics: {metrics}")

            trainer.save_checkpoint(f'checkpoint_epoch{epoch + 1}.pt')

        trainer.save_checkpoint('checkpoint_final.pt')

    print(f"\n=== Training complete! ===")
    print(f"Checkpoints saved to {run_dir}")
    trainer.close()


if __name__ == '__main__':
    main()
```

### Testing
```bash
# Test Phase 1
python scripts/train.py --phase 1 --config configs/phase1.yaml --device cpu

# Test Phase 2
python scripts/train.py --phase 2 --config configs/phase2.yaml --device cpu

# Test Phase 3
python scripts/train.py --phase 3 --config configs/phase3.yaml --device cpu
```

---

## 8. Dataset Generation Script (`scripts/generate_data.py`)

**Time**: 0.5 hours
**Dependencies**: None (external data generation)

```python
#!/usr/bin/env python3
"""
Generate synthetic MBA dataset for training.

Usage:
    python scripts/generate_data.py --depth 1-14 --samples 10000 --output data/train.jsonl
    python scripts/generate_data.py --depth 1-10 --samples 5000 --output data/val.jsonl
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Tuple


def generate_simple_mba_pair(depth: int, num_vars: int = 3) -> Tuple[str, str]:
    """
    Generate obfuscated/simplified MBA pair.

    This is a placeholder - in practice, use a proper MBA obfuscation tool.

    Args:
        depth: Expression depth
        num_vars: Number of variables

    Returns:
        (obfuscated, simplified) tuple
    """
    # Simple patterns for demonstration
    vars = [f'x{i}' for i in range(num_vars)]

    patterns = [
        # AND-XOR to OR
        (lambda v: f"({v[0]}&{v[1]})+({v[0]}^{v[1]})", lambda v: f"{v[0]}|{v[1]}"),
        # AND-NOT identity
        (lambda v: f"({v[0]}&~{v[0]})", lambda v: "0"),
        # XOR-SELF identity
        (lambda v: f"({v[0]}^{v[0]})", lambda v: "0"),
        # OR-NOT
        (lambda v: f"({v[0]}|~{v[0]})", lambda v: "-1"),
        # Simple expression
        (lambda v: f"{v[0]}+{v[1]}", lambda v: f"{v[0]}+{v[1]}"),
    ]

    pattern_idx = random.randint(0, len(patterns) - 1)
    obf_fn, simp_fn = patterns[pattern_idx]

    # For higher depth, nest expressions
    if depth > 3:
        # Recursively nest
        inner_obf, inner_simp = generate_simple_mba_pair(depth - 2, num_vars)
        obfuscated = f"({inner_obf})+{vars[0]}"
        simplified = f"({inner_simp})+{vars[0]}"
    else:
        obfuscated = obf_fn(vars)
        simplified = simp_fn(vars)

    return obfuscated, simplified


def generate_dataset(
    num_samples: int,
    min_depth: int,
    max_depth: int,
    output_path: str
):
    """
    Generate dataset and save to JSONL.

    Args:
        num_samples: Number of samples to generate
        min_depth: Minimum expression depth
        max_depth: Maximum expression depth
        output_path: Output JSONL path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating {num_samples} samples (depth {min_depth}-{max_depth})...")

    with open(output_path, 'w') as f:
        for i in range(num_samples):
            depth = random.randint(min_depth, max_depth)
            num_vars = random.randint(2, 4)

            obfuscated, simplified = generate_simple_mba_pair(depth, num_vars)

            item = {
                'obfuscated': obfuscated,
                'simplified': simplified,
                'depth': depth,
            }

            f.write(json.dumps(item) + '\n')

            if (i + 1) % 1000 == 0:
                print(f"Generated {i + 1}/{num_samples} samples")

    print(f"Dataset saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate MBA dataset')
    parser.add_argument('--depth', type=str, required=True,
                       help='Depth range (e.g., "1-14")')
    parser.add_argument('--samples', type=int, required=True,
                       help='Number of samples')
    parser.add_argument('--output', type=str, required=True,
                       help='Output JSONL path')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    args = parser.parse_args()

    # Parse depth range
    min_depth, max_depth = map(int, args.depth.split('-'))

    # Set seed
    random.seed(args.seed)

    # Generate dataset
    generate_dataset(args.samples, min_depth, max_depth, args.output)


if __name__ == '__main__':
    main()
```

### Testing
```bash
# Generate train dataset
python scripts/generate_data.py --depth 1-14 --samples 1000 --output data/train.jsonl

# Generate val dataset
python scripts/generate_data.py --depth 1-10 --samples 200 --output data/val.jsonl

# Check output
head -n 5 data/train.jsonl
```

---

## 9. Evaluation Script (`scripts/evaluate.py`)

**Time**: 0.5 hours
**Dependencies**: Model, metrics, verification

```python
#!/usr/bin/env python3
"""
Evaluate trained MBA Deobfuscator model.

Usage:
    python scripts/evaluate.py --checkpoint runs/phase2/checkpoint_final.pt \\
                              --test-set data/test.jsonl \\
                              --output results/eval_phase2.json
"""

import argparse
import json
import torch
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

from src.models.full_model import MBADeobfuscator
from src.data.tokenizer import MBATokenizer
from src.data.fingerprint import SemanticFingerprint
from src.data.dataset import MBADataset
from src.data.collate import collate_graphs
from src.utils.metrics import exact_match, syntax_valid, simplification_ratio
from src.inference.verify import ThreeTierVerifier
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate_model(
    model: MBADeobfuscator,
    dataloader: DataLoader,
    tokenizer: MBATokenizer,
    verifier: ThreeTierVerifier,
    device: str = 'cuda'
) -> dict:
    """
    Evaluate model on dataset.

    Args:
        model: MBADeobfuscator model
        dataloader: Test data loader
        tokenizer: MBATokenizer
        verifier: ThreeTierVerifier
        device: Device

    Returns:
        Dict with evaluation metrics
    """
    model.eval()
    model.to(device)

    # Metrics
    total = 0
    exact_matches = 0
    syntax_valid_count = 0
    exec_verified = 0
    z3_verified = 0
    total_simp_ratio = 0.0

    # Per-depth metrics
    depth_metrics = defaultdict(lambda: {
        'total': 0, 'exact': 0, 'syntax': 0, 'exec': 0, 'z3': 0
    })

    predictions = []

    for batch in tqdm(dataloader, desc='Evaluating'):
        # Move to device
        graph_batch = batch['graph_batch'].to(device)
        fingerprint = batch['fingerprint'].to(device)

        input_exprs = batch['obfuscated']
        target_exprs = batch['simplified']
        depths = batch['depth'].tolist()

        # Greedy decode
        memory = model.encode(graph_batch, fingerprint)
        batch_size = memory.shape[0]

        output = torch.full(
            (batch_size, 1), tokenizer.sos_token_id,
            dtype=torch.long, device=device
        )

        for _ in range(64):
            decode_output = model.decode(output, memory)
            logits = decode_output['vocab_logits'][:, -1, :]
            next_token = logits.argmax(dim=-1, keepdim=True)
            output = torch.cat([output, next_token], dim=1)

            if (next_token.squeeze(-1) == tokenizer.eos_token_id).all():
                break

        # Decode predictions
        pred_exprs = []
        for i in range(batch_size):
            seq = output[i].tolist()
            pred = tokenizer.decode(seq)
            pred_exprs.append(pred)

        # Compute metrics
        for inp, pred, tgt, depth in zip(input_exprs, pred_exprs, target_exprs, depths):
            total += 1

            # Exact match
            if exact_match(pred, tgt):
                exact_matches += 1
                depth_metrics[depth]['exact'] += 1

            # Syntax valid
            if syntax_valid(pred):
                syntax_valid_count += 1
                depth_metrics[depth]['syntax'] += 1

            # Simplification ratio
            simp_ratio = simplification_ratio(inp, pred)
            total_simp_ratio += simp_ratio

            # Verification (only for syntax-valid)
            if syntax_valid(pred):
                results = verifier.verify_batch(inp, [pred])
                if results and results[0].exec_valid:
                    exec_verified += 1
                    depth_metrics[depth]['exec'] += 1
                if results and results[0].z3_verified:
                    z3_verified += 1
                    depth_metrics[depth]['z3'] += 1

            depth_metrics[depth]['total'] += 1

            predictions.append({
                'input': inp,
                'prediction': pred,
                'target': tgt,
                'depth': depth,
                'exact_match': exact_match(pred, tgt),
                'syntax_valid': syntax_valid(pred),
            })

    # Aggregate metrics
    metrics = {
        'total_samples': total,
        'exact_match_acc': exact_matches / max(total, 1),
        'syntax_valid_acc': syntax_valid_count / max(total, 1),
        'exec_verified_acc': exec_verified / max(total, 1),
        'z3_verified_acc': z3_verified / max(total, 1),
        'avg_simplification_ratio': total_simp_ratio / max(total, 1),
        'per_depth_metrics': {},
    }

    # Per-depth breakdown
    for depth, depth_data in sorted(depth_metrics.items()):
        depth_total = depth_data['total']
        if depth_total > 0:
            metrics['per_depth_metrics'][depth] = {
                'total': depth_total,
                'exact_match': depth_data['exact'] / depth_total,
                'syntax_valid': depth_data['syntax'] / depth_total,
                'exec_verified': depth_data['exec'] / depth_total,
                'z3_verified': depth_data['z3'] / depth_total,
            }

    return metrics, predictions


def main():
    parser = argparse.ArgumentParser(description='Evaluate MBA Deobfuscator')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--test-set', type=str, required=True,
                       help='Path to test JSONL file')
    parser.add_argument('--output', type=str, required=True,
                       help='Output JSON path for results')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda or cpu)')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size')

    args = parser.parse_args()

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')

    # Initialize model
    config = checkpoint.get('config', {})
    model_config = config.get('model', {})
    encoder_type = model_config.get('encoder_type', 'gat')

    model = MBADeobfuscator(encoder_type=encoder_type, **model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(args.device)

    # Initialize tokenizer and fingerprint
    tokenizer = MBATokenizer()
    fingerprint = SemanticFingerprint()

    # Load test dataset
    print(f"Loading test set from {args.test_set}")
    test_dataset = MBADataset(args.test_set, tokenizer, fingerprint)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_graphs
    )

    # Initialize verifier
    verifier = ThreeTierVerifier(tokenizer)

    # Evaluate
    print("Evaluating...")
    metrics, predictions = evaluate_model(
        model, test_loader, tokenizer, verifier, args.device
    )

    # Print results
    print("\n=== Evaluation Results ===")
    print(f"Total samples: {metrics['total_samples']}")
    print(f"Exact match accuracy: {metrics['exact_match_acc']:.2%}")
    print(f"Syntax valid accuracy: {metrics['syntax_valid_acc']:.2%}")
    print(f"Execution verified accuracy: {metrics['exec_verified_acc']:.2%}")
    print(f"Z3 verified accuracy: {metrics['z3_verified_acc']:.2%}")
    print(f"Avg simplification ratio: {metrics['avg_simplification_ratio']:.3f}")

    print("\n=== Per-Depth Metrics ===")
    for depth, depth_metrics in sorted(metrics['per_depth_metrics'].items()):
        print(f"Depth {depth}: "
              f"exact={depth_metrics['exact_match']:.2%}, "
              f"syntax={depth_metrics['syntax_valid']:.2%}, "
              f"exec={depth_metrics['exec_verified']:.2%}, "
              f"z3={depth_metrics['z3_verified']:.2%}")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump({
            'metrics': metrics,
            'predictions': predictions,
        }, f, indent=2)

    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
```

### Testing
```bash
# Evaluate model
python scripts/evaluate.py \\
    --checkpoint runs/phase2/checkpoint_final.pt \\
    --test-set data/test.jsonl \\
    --output results/eval.json \\
    --device cpu

# View results
cat results/eval.json | jq '.metrics'
```

---

## 10. CLI Inference Script (`scripts/simplify.py`)

**Time**: 0.5 hours
**Dependencies**: Model, inference pipeline

```python
#!/usr/bin/env python3
"""
CLI tool for simplifying MBA expressions.

Usage:
    python scripts/simplify.py --checkpoint runs/phase2/best.pt \\
                              --expr "(x&y)+(x^y)" \\
                              --mode beam

    python scripts/simplify.py --checkpoint runs/phase2/best.pt \\
                              --expr "(x&y)+(x^y)" \\
                              --mode greedy
"""

import argparse
import torch

from src.models.full_model import MBADeobfuscator
from src.data.tokenizer import MBATokenizer
from src.data.fingerprint import SemanticFingerprint
from src.data.ast_parser import expr_to_graph
from src.inference.beam_search import DiverseBeamSearch
from src.inference.verify import ThreeTierVerifier


def simplify_greedy(
    expr: str,
    model: MBADeobfuscator,
    tokenizer: MBATokenizer,
    fingerprint: SemanticFingerprint,
    device: str
) -> str:
    """
    Simplify expression using greedy decoding.

    Args:
        expr: Input expression
        model: MBADeobfuscator model
        tokenizer: MBATokenizer
        fingerprint: SemanticFingerprint
        device: Device

    Returns:
        Simplified expression
    """
    model.eval()

    # Convert to graph
    graph = expr_to_graph(expr).to(device)
    fp = torch.from_numpy(fingerprint.compute(expr)).unsqueeze(0).to(device)

    # Encode
    memory = model.encode(graph, fp)

    # Greedy decode
    output = torch.full(
        (1, 1), tokenizer.sos_token_id, dtype=torch.long, device=device
    )

    for _ in range(64):
        decode_output = model.decode(output, memory)
        logits = decode_output['vocab_logits'][:, -1, :]
        next_token = logits.argmax(dim=-1, keepdim=True)
        output = torch.cat([output, next_token], dim=1)

        if next_token.item() == tokenizer.eos_token_id:
            break

    # Decode
    result = tokenizer.decode(output[0].tolist())
    return result


def simplify_beam(
    expr: str,
    model: MBADeobfuscator,
    tokenizer: MBATokenizer,
    fingerprint: SemanticFingerprint,
    verifier: ThreeTierVerifier,
    device: str,
    beam_width: int = 50
) -> str:
    """
    Simplify expression using beam search.

    Args:
        expr: Input expression
        model: MBADeobfuscator model
        tokenizer: MBATokenizer
        fingerprint: SemanticFingerprint
        verifier: ThreeTierVerifier
        device: Device
        beam_width: Beam width

    Returns:
        Best verified simplified expression
    """
    model.eval()

    # Convert to graph
    graph = expr_to_graph(expr).to(device)
    fp = torch.from_numpy(fingerprint.compute(expr)).unsqueeze(0).to(device)

    # Initialize beam search
    beam_search = DiverseBeamSearch(
        model=model,
        tokenizer=tokenizer,
        beam_width=beam_width,
        device=device
    )

    # Search
    candidates = beam_search.search(graph, fp)

    # Verify candidates
    results = verifier.verify_batch(expr, candidates)

    # Return best verified result
    for result in results:
        if result.z3_verified or result.exec_valid:
            return result.candidate

    # Fallback to first candidate
    return candidates[0] if candidates else expr


def main():
    parser = argparse.ArgumentParser(description='Simplify MBA expression')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--expr', type=str, required=True,
                       help='Expression to simplify')
    parser.add_argument('--mode', type=str, default='greedy',
                       choices=['greedy', 'beam'],
                       help='Inference mode')
    parser.add_argument('--beam-width', type=int, default=50,
                       help='Beam width (for beam mode)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda or cpu)')

    args = parser.parse_args()

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')

    # Initialize model
    config = checkpoint.get('config', {})
    model_config = config.get('model', {})
    encoder_type = model_config.get('encoder_type', 'gat')

    model = MBADeobfuscator(encoder_type=encoder_type, **model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(args.device)

    # Initialize tokenizer and fingerprint
    tokenizer = MBATokenizer()
    fingerprint = SemanticFingerprint()

    # Simplify
    print(f"\nInput:  {args.expr}")

    if args.mode == 'greedy':
        result = simplify_greedy(
            args.expr, model, tokenizer, fingerprint, args.device
        )
    else:
        verifier = ThreeTierVerifier(tokenizer)
        result = simplify_beam(
            args.expr, model, tokenizer, fingerprint, verifier,
            args.device, args.beam_width
        )

    print(f"Output: {result}")


if __name__ == '__main__':
    main()
```

### Testing
```bash
# Test greedy inference
python scripts/simplify.py \\
    --checkpoint runs/phase2/best.pt \\
    --expr "(x&y)+(x^y)" \\
    --mode greedy \\
    --device cpu

# Test beam search
python scripts/simplify.py \\
    --checkpoint runs/phase2/best.pt \\
    --expr "(x&y)+(x^y)" \\
    --mode beam \\
    --beam-width 10 \\
    --device cpu
```

---

## Testing Strategy

### Unit Tests (per component)

Each component should have basic unit tests:

```python
# tests/test_losses.py
def test_infonce_loss():
    batch = 32
    dim = 256
    obf = torch.randn(batch, dim)
    simp = torch.randn(batch, dim)
    loss = infonce_loss(obf, simp)
    assert loss > 0
    assert not torch.isnan(loss)

# tests/test_phase1_trainer.py
def test_phase1_trainer_init():
    model = MBADeobfuscator(encoder_type='gat')
    config = {'learning_rate': 1e-4}
    trainer = Phase1Trainer(model, config, 'test_runs/phase1')
    assert trainer.optimizer is not None

# tests/test_phase2_trainer.py
def test_curriculum_advancement():
    # Test stage progression logic
    pass
```

### Integration Tests

End-to-end pipeline tests:

```bash
# Generate small dataset
python scripts/generate_data.py --depth 1-5 --samples 100 --output data/test_train.jsonl
python scripts/generate_data.py --depth 1-5 --samples 20 --output data/test_val.jsonl

# Train Phase 1 (1 epoch)
python scripts/train.py --phase 1 --config configs/phase1_test.yaml --device cpu

# Train Phase 2 (1 epoch per stage)
python scripts/train.py --phase 2 --config configs/phase2_test.yaml --device cpu

# Evaluate
python scripts/evaluate.py --checkpoint runs/phase2/checkpoint_final.pt \\
                          --test-set data/test_val.jsonl \\
                          --output results/test_eval.json \\
                          --device cpu

# Inference
python scripts/simplify.py --checkpoint runs/phase2/checkpoint_final.pt \\
                          --expr "x+y" --mode greedy --device cpu
```

---

## Dependencies

### Component Dependency Graph

```
losses.py (1.5h)
  └─> base_trainer.py (2h)
        ├─> phase1_trainer.py (1h)
        ├─> phase2_trainer.py (1.5h)
        └─> phase3_trainer.py (0.5h)

configs/*.yaml (0.5h)

train.py (0.5h)
  └─> Depends on: all trainers, configs

generate_data.py (0.5h)
  └─> Independent

evaluate.py (0.5h)
  └─> Depends on: model, metrics, verify

simplify.py (0.5h)
  └─> Depends on: model, inference

TOTAL: 9.5 hours
```

### Implementation Order (Critical Path)

1. **losses.py** (1.5h) - Must implement first
2. **base_trainer.py** (2h) - Foundation for all trainers
3. **phase1_trainer.py** (1h) - Can implement independently
4. **phase2_trainer.py** (1.5h) - Can implement independently
5. **phase3_trainer.py** (0.5h) - Can implement independently
6. **configs/*.yaml** (0.5h) - Can do in parallel with trainers
7. **Scripts** (2h total) - Can do in parallel
   - train.py (0.5h)
   - generate_data.py (0.5h)
   - evaluate.py (0.5h)
   - simplify.py (0.5h)
8. **Integration testing** (0.5h) - Final verification

### Parallelization Opportunities

After completing base_trainer.py, can parallelize:
- Phase 1/2/3 trainers (3 hours total, can do 1.5h if 2 people)
- Config files (0.5h, can do during trainer implementation)
- Scripts (2h total, can do 1h if 2 people)

**Minimum serial path**: 1.5h + 2h + 1.5h + 0.5h + 0.5h = 6 hours
**With parallelization**: Could complete in ~6-7 hours with 2 people

---

## Success Criteria

After implementation, verify:

1. **Phase 1 runs**: Contrastive loss decreases over epochs
2. **Phase 2 runs**: Curriculum advances automatically, accuracy improves per stage
3. **Phase 3 runs**: PPO rewards increase, model learns to simplify
4. **Scripts work**: All 4 scripts run without errors
5. **Evaluation produces metrics**: Exact match, syntax valid, Z3 verified
6. **Inference works**: CLI tool simplifies expressions

**Minimal viable test**:
```bash
# Generate 1000 samples
python scripts/generate_data.py --depth 1-5 --samples 1000 --output data/mini_train.jsonl

# Train Phase 1 (2 epochs)
python scripts/train.py --phase 1 --config configs/phase1.yaml --device cpu

# Train Phase 2 (stage 1 only)
python scripts/train.py --phase 2 --config configs/phase2_minimal.yaml --device cpu

# Evaluate
python scripts/evaluate.py --checkpoint runs/phase2/best.pt --test-set data/mini_train.jsonl --output results/test.json

# Verify metrics exist
cat results/test.json | jq '.metrics.exact_match_acc'
```

---

## Notes

1. **Self-paced learning**: Implemented in Phase 2 trainer, automatically weights easier samples early
2. **Curriculum advancement**: Automatic based on validation accuracy thresholds
3. **Copy mechanism**: Integrated in losses.py, used in Phase 2
4. **Execution pre-filter**: Used in Phase 3 reward computation via ThreeTierVerifier
5. **Z3 verification**: Used in evaluation and Phase 3 rewards

**Known simplifications**:
- Dataset generation uses simple patterns (replace with proper MBA obfuscator)
- Beam search implementation exists in `src/inference/beam_search.py` (reuse)
- HTPS exists in `src/inference/htps.py` (reuse for depth ≥10)

**Extensions for production**:
- Distributed training (DDP)
- Mixed precision (AMP)
- Gradient checkpointing for memory
- More sophisticated data augmentation
- Process Reward Model (P2 novelty, defer to later)

---

End of Implementation Plan
