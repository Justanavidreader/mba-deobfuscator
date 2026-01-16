"""
Phase 2 Trainer: Supervised Learning with Curriculum.

Trains the full encoder-decoder model using:
1. Cross-entropy loss for sequence generation
2. Copy mechanism loss for variable preservation
3. Complexity prediction loss for output length/depth

Implements curriculum learning with 4 stages:
- Stage 1: depth ≤ 2, target 95% accuracy
- Stage 2: depth ≤ 5, target 90% accuracy
- Stage 3: depth ≤ 10, target 80% accuracy
- Stage 4: depth ≤ 14, target 70% accuracy

Also includes self-paced learning to gradually include harder examples.
"""

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.training.base_trainer import BaseTrainer
from src.training.losses import phase2_loss
from src.data.dataset import MBADataset
from src.data.collate import collate_graphs
from src.data.tokenizer import MBATokenizer
from src.data.fingerprint import SemanticFingerprint
from src.constants import (
    CURRICULUM_STAGES,
    CE_WEIGHT, COMPLEXITY_WEIGHT, COPY_WEIGHT,
    SELF_PACED_LAMBDA_INIT, SELF_PACED_LAMBDA_GROWTH,
    SOS_IDX, EOS_IDX, PAD_IDX,
    MAX_SEQ_LEN,
)

logger = logging.getLogger(__name__)


class Phase2Trainer(BaseTrainer):
    """
    Supervised training with curriculum learning.

    Trains the full model to generate simplified expressions from
    obfuscated inputs. Uses curriculum learning to gradually increase
    expression complexity, and self-paced learning to weight samples.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: MBATokenizer,
        fingerprint: SemanticFingerprint,
        train_path: str,
        val_path: str,
        config: Dict[str, Any],
        device: Optional[torch.device] = None,
        checkpoint_dir: Optional[str] = None,
    ):
        """
        Initialize Phase 2 trainer.

        Args:
            model: MBADeobfuscator model
            tokenizer: MBATokenizer for encoding/decoding
            fingerprint: SemanticFingerprint for computing fingerprints
            train_path: Path to training JSONL file
            val_path: Path to validation JSONL file
            config: Training configuration with:
                - ce_weight, complexity_weight, copy_weight
                - use_self_paced: bool
                - curriculum_stages: List[Dict] (optional override)
                - batch_size: int
                - num_workers: int
            device: Training device
            checkpoint_dir: Directory for checkpoints
        """
        super().__init__(
            model=model,
            config=config,
            device=device,
            checkpoint_dir=checkpoint_dir,
            experiment_name="phase2_supervised",
        )

        self.tokenizer = tokenizer
        self.fingerprint = fingerprint
        self.train_path = train_path
        self.val_path = val_path

        # Loss weights
        self.ce_weight = config.get("ce_weight", CE_WEIGHT)
        self.complexity_weight = config.get("complexity_weight", COMPLEXITY_WEIGHT)
        self.copy_weight = config.get("copy_weight", COPY_WEIGHT)

        # Curriculum learning
        self.curriculum_stages = config.get("curriculum_stages", CURRICULUM_STAGES)
        self.current_stage = 0
        self.stage_epoch = 0

        # Self-paced learning
        self.use_self_paced = config.get("use_self_paced", True)
        self.sp_lambda = config.get("sp_lambda_init", SELF_PACED_LAMBDA_INIT)
        self.sp_growth = config.get("sp_lambda_growth", SELF_PACED_LAMBDA_GROWTH)
        self.sp_lambda_max = config.get("sp_lambda_max", 10.0)

        # DataLoader config
        self.batch_size = config.get("batch_size", 32)
        self.num_workers = config.get("num_workers", 4)

        # Initialize dataloaders for first stage
        self._init_dataloaders()

        logger.info(
            f"Phase2Trainer initialized: "
            f"stages={len(self.curriculum_stages)}, "
            f"self_paced={self.use_self_paced}"
        )

    def _init_dataloaders(self) -> None:
        """Initialize dataloaders for current curriculum stage."""
        stage = self.curriculum_stages[self.current_stage]
        max_depth = stage['max_depth']

        logger.info(f"Loading data for stage {self.current_stage + 1} (max_depth={max_depth})")

        # Create datasets with depth filter
        train_dataset = MBADataset(
            data_path=self.train_path,
            tokenizer=self.tokenizer,
            fingerprint=self.fingerprint,
            max_depth=max_depth,
        )

        val_dataset = MBADataset(
            data_path=self.val_path,
            tokenizer=self.tokenizer,
            fingerprint=self.fingerprint,
            max_depth=max_depth,
        )

        # Create dataloaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_graphs,
            pin_memory=True,
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_graphs,
            pin_memory=True,
        )

        logger.info(
            f"Loaded {len(train_dataset)} training samples, "
            f"{len(val_dataset)} validation samples"
        )

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """
        Single training step.

        Args:
            batch: Collated batch from MBADataset with keys:
                - graph_batch: PyG batch
                - fingerprint: [batch, FINGERPRINT_DIM]
                - target_ids: [batch, seq_len]
                - source_tokens: [batch, src_len]
                - depth: [batch]

        Returns:
            Dict with loss values
        """
        self.model.train()

        # Move to device
        graph_batch = batch['graph_batch'].to(self.device)
        fingerprint = batch['fingerprint'].to(self.device)
        target_ids = batch['target_ids'].to(self.device)
        source_tokens = batch['source_tokens'].to(self.device)
        depth = batch['depth'].to(self.device)

        # Forward pass
        # Model expects target without last token for teacher forcing
        outputs = self.model(graph_batch, fingerprint, target_ids[:, :-1])

        # Compute losses
        loss_dict = phase2_loss(
            vocab_logits=outputs['vocab_logits'],
            copy_attn=outputs.get('copy_attn'),
            p_gen=outputs.get('p_gen'),
            length_pred=outputs['length_pred'],
            depth_pred=outputs['depth_pred'],
            target_ids=target_ids,
            source_tokens=source_tokens,
            depth_labels=depth,
            ce_weight=self.ce_weight,
            complexity_weight=self.complexity_weight,
            copy_weight=self.copy_weight,
        )

        total_loss = loss_dict['total']

        # Self-paced learning: weight samples by difficulty
        if self.use_self_paced:
            # Compute per-sample loss
            with torch.no_grad():
                sample_losses = self._compute_sample_losses(outputs, target_ids)
                # Weight: 1 if loss < lambda, 0 otherwise (hard threshold)
                # Softer version: weight = exp(-loss/lambda)
                weights = (sample_losses < self.sp_lambda).float()
                if weights.sum() > 0:
                    total_loss = (total_loss * weights.mean())

        # Backward pass
        self.backward(total_loss, update=True)

        return {
            'total': total_loss.item(),
            'ce': loss_dict['ce'].item(),
            'complexity': loss_dict['complexity'].item(),
            'copy': loss_dict['copy'].item(),
        }

    def _compute_sample_losses(
        self,
        outputs: Dict[str, torch.Tensor],
        target_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute per-sample CE loss for self-paced weighting.

        Vectorized implementation using reduction='none' for efficiency.
        """
        vocab_logits = outputs['vocab_logits']

        # Shift for autoregressive
        shift_logits = vocab_logits[:, :-1, :].contiguous()
        shift_targets = target_ids[:, 1:].contiguous()

        # Compute per-token loss (reduction='none' returns [batch * seq_len])
        per_token_loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.shape[-1]),
            shift_targets.view(-1),
            ignore_index=PAD_IDX,
            reduction='none'
        ).view(shift_targets.shape)  # [batch, seq_len]

        # Average over sequence for each sample, ignoring PAD tokens
        mask = (shift_targets != PAD_IDX).float()
        sample_losses = (per_token_loss * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        return sample_losses

    def train_epoch(self, dataloader: Optional[DataLoader] = None, log_interval: int = 100) -> Dict[str, float]:
        """
        Train for one epoch with curriculum awareness.

        Args:
            dataloader: Optional override (uses self.train_loader if None)
            log_interval: Steps between logging

        Returns:
            Average epoch metrics
        """
        if dataloader is None:
            dataloader = self.train_loader

        self.model.train()
        epoch_metrics: Dict[str, List[float]] = defaultdict(list)
        num_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            metrics = self.train_step(batch)

            for key, value in metrics.items():
                epoch_metrics[key].append(value)
            num_batches += 1

            if batch_idx % log_interval == 0 and batch_idx > 0:
                self.log_metrics(metrics)

        # Average metrics
        avg_metrics = {
            key: sum(values) / len(values)
            for key, values in epoch_metrics.items()
        }

        self.epoch += 1
        self.stage_epoch += 1

        # Update self-paced lambda with upper bound from config
        # Cap prevents unbounded growth: λ = 0.5 × 1.1^50 ≈ 58.6 without cap
        if self.use_self_paced:
            self.sp_lambda = min(self.sp_lambda * self.sp_growth, self.sp_lambda_max)

        return avg_metrics

    @torch.no_grad()
    def evaluate(self, dataloader: Optional[DataLoader] = None) -> Dict[str, float]:
        """
        Evaluate on validation set.

        Args:
            dataloader: Optional override (uses self.val_loader if None)

        Returns:
            Dict with 'exact_match', 'syntax_valid', 'avg_loss'
        """
        if dataloader is None:
            dataloader = self.val_loader

        self.model.eval()

        total_loss = 0.0
        exact_matches = 0
        syntax_valid_count = 0
        total_samples = 0

        for batch in dataloader:
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
                copy_attn=outputs.get('copy_attn'),
                p_gen=outputs.get('p_gen'),
                length_pred=outputs['length_pred'],
                depth_pred=outputs['depth_pred'],
                target_ids=target_ids,
                source_tokens=source_tokens,
                depth_labels=depth,
                ce_weight=self.ce_weight,
                complexity_weight=self.complexity_weight,
                copy_weight=self.copy_weight,
            )

            batch_size = target_ids.shape[0]
            total_loss += loss_dict['total'].item() * batch_size

            # Greedy decode for accuracy
            predictions = self._greedy_decode_batch(graph_batch, fingerprint)
            targets = batch['simplified']

            for pred, tgt in zip(predictions, targets):
                total_samples += 1
                if self._exact_match(pred, tgt):
                    exact_matches += 1
                if self._syntax_valid(pred):
                    syntax_valid_count += 1

        return {
            'exact_match': exact_matches / max(total_samples, 1),
            'syntax_valid': syntax_valid_count / max(total_samples, 1),
            'avg_loss': total_loss / max(total_samples, 1),
        }

    def _greedy_decode_batch(
        self,
        graph_batch,
        fingerprint: torch.Tensor,
        max_len: int = MAX_SEQ_LEN
    ) -> List[str]:
        """
        Greedy decode a batch of expressions.

        Args:
            graph_batch: PyG batch
            fingerprint: [batch, FINGERPRINT_DIM]
            max_len: Maximum sequence length

        Returns:
            List of decoded strings
        """
        # Encode
        memory = self.model.encode(graph_batch, fingerprint)
        batch_size = memory.shape[0]

        # Start with SOS
        output = torch.full(
            (batch_size, 1), SOS_IDX, dtype=torch.long, device=self.device
        )

        # Autoregressive generation
        for _ in range(max_len - 1):
            # Decode
            decoder_out = self.model.decode(output, memory)

            # Get vocab logits from last position
            if isinstance(decoder_out, dict):
                logits = decoder_out['vocab_logits'][:, -1, :]
            else:
                logits = self.model.vocab_head(decoder_out[:, -1, :])

            # Greedy selection
            next_token = logits.argmax(dim=-1, keepdim=True)
            output = torch.cat([output, next_token], dim=1)

            # Check if all sequences have EOS
            if (next_token == EOS_IDX).all():
                break

        # Decode to strings
        predictions = []
        for i in range(batch_size):
            tokens = output[i].tolist()
            pred_str = self.tokenizer.decode(tokens)
            predictions.append(pred_str)

        return predictions

    def _exact_match(self, pred: str, target: str) -> bool:
        """Check exact match after normalization."""
        pred_norm = pred.replace(' ', '').lower()
        tgt_norm = target.replace(' ', '').lower()
        return pred_norm == tgt_norm

    def _syntax_valid(self, expr: str) -> bool:
        """Check if expression has valid syntax."""
        try:
            # Try to parse with tokenizer
            tokens = self.tokenizer.encode(expr)
            return len(tokens) > 2  # More than just SOS/EOS
        except Exception:
            return False

    def should_advance_stage(self, val_metrics: Dict[str, float]) -> bool:
        """
        Check if should advance to next curriculum stage.

        Advances if:
        1. Target accuracy reached, OR
        2. Max epochs for stage exhausted

        Args:
            val_metrics: Validation metrics with 'exact_match'

        Returns:
            True if should advance
        """
        if self.current_stage >= len(self.curriculum_stages) - 1:
            return False  # Already at final stage

        stage = self.curriculum_stages[self.current_stage]
        target_acc = stage['target']
        max_epochs = stage['epochs']

        current_acc = val_metrics.get('exact_match', 0)

        if current_acc >= target_acc:
            logger.info(
                f"Stage {self.current_stage + 1}: Target accuracy {target_acc} reached "
                f"(current: {current_acc:.3f}). Advancing."
            )
            return True

        if self.stage_epoch >= max_epochs:
            logger.info(
                f"Stage {self.current_stage + 1}: Max epochs {max_epochs} reached. "
                f"Advancing (current acc: {current_acc:.3f})."
            )
            return True

        return False

    def advance_stage(self) -> bool:
        """
        Advance to next curriculum stage.

        Returns:
            True if advanced, False if already at final stage
        """
        if self.current_stage >= len(self.curriculum_stages) - 1:
            logger.info("Already at final curriculum stage")
            return False

        self.current_stage += 1
        self.stage_epoch = 0

        # Reset self-paced lambda for new stage
        self.sp_lambda = SELF_PACED_LAMBDA_INIT

        # Reinitialize dataloaders with new depth filter
        self._init_dataloaders()

        logger.info(
            f"Advanced to stage {self.current_stage + 1}: "
            f"max_depth={self.curriculum_stages[self.current_stage]['max_depth']}"
        )

        return True

    def _should_save_checkpoint(self, epoch: int, save_interval: int) -> bool:
        """Check if checkpoint should be saved at this epoch."""
        return (epoch + 1) % save_interval == 0

    def _update_best_model(
        self,
        val_metrics: Dict[str, float],
        metric_name: str,
        higher_is_better: bool
    ) -> bool:
        """
        Update best model tracking.

        Returns True if this is a new best model.
        """
        current_metric = val_metrics.get(metric_name, 0)
        is_best = (current_metric > self.best_metric) if higher_is_better else (current_metric < self.best_metric)
        if is_best:
            self.best_metric = current_metric
        return is_best

    def _get_stage_info(self) -> Dict[str, Any]:
        """Get current curriculum stage info for logging."""
        return {
            'stage': self.current_stage + 1,
            'max_depth': self.curriculum_stages[self.current_stage]['max_depth'],
            'stage_epoch': self.stage_epoch + 1,
        }

    def fit(
        self,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
        num_epochs: Optional[int] = None,
        eval_interval: int = 1,
        save_interval: int = 1,
        metric_for_best: str = "exact_match",
        higher_is_better: bool = True,
    ) -> Dict[str, Any]:
        """
        Full training loop with curriculum learning.

        Automatically advances through curriculum stages.

        Args:
            train_loader: Override training loader (uses internal if None)
            val_loader: Override validation loader (uses internal if None)
            num_epochs: Total epochs (if None, uses sum of all stage epochs)
            eval_interval: Epochs between evaluation
            save_interval: Epochs between checkpoints
            metric_for_best: Metric for best model tracking
            higher_is_better: Whether higher metric is better

        Returns:
            Training history
        """
        if num_epochs is None:
            num_epochs = sum(s['epochs'] for s in self.curriculum_stages)

        history = {"train": [], "val": [], "stages": []}

        for epoch in range(num_epochs):
            stage_info = self._get_stage_info()
            logger.info(
                f"Epoch {epoch + 1}/{num_epochs} | "
                f"Stage {stage_info['stage']} (depth≤{stage_info['max_depth']}) | "
                f"Stage epoch {stage_info['stage_epoch']}"
            )

            # Train
            train_metrics = self.train_epoch()
            history["train"].append({**train_metrics, **stage_info})
            self.log_metrics(train_metrics, prefix="train")

            # Evaluate
            if (epoch + 1) % eval_interval == 0:
                val_metrics = self.evaluate()
                history["val"].append({**val_metrics, **stage_info})
                self.log_metrics(val_metrics, prefix="val")

                # Check curriculum advancement
                if self.should_advance_stage(val_metrics):
                    history["stages"].append({
                        'stage': self.current_stage + 1,
                        'epoch': epoch + 1,
                        'metrics': val_metrics,
                    })
                    self.advance_stage()

                # Track and save best model
                if self._update_best_model(val_metrics, metric_for_best, higher_is_better):
                    self.save_checkpoint(metrics=val_metrics, is_best=True)

            # Save periodic checkpoint
            if self._should_save_checkpoint(epoch, save_interval):
                self.save_checkpoint(filename=f"phase2_epoch_{epoch + 1}.pt", metrics=train_metrics)

        return history

    def save_checkpoint(
        self,
        filename: Optional[str] = None,
        metrics: Optional[Dict[str, float]] = None,
        is_best: bool = False,
    ) -> str:
        """Save checkpoint with curriculum state."""
        if filename is None:
            filename = f"phase2_checkpoint_{self.global_step}.pt"

        checkpoint_path = self.checkpoint_dir / filename

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_metric": self.best_metric,
            "config": self.config,
            # Curriculum state
            "current_stage": self.current_stage,
            "stage_epoch": self.stage_epoch,
            "sp_lambda": self.sp_lambda,
        }

        if metrics:
            checkpoint["metrics"] = metrics

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved Phase 2 checkpoint to {checkpoint_path}")

        if is_best:
            best_path = self.checkpoint_dir / "phase2_best.pt"
            torch.save(checkpoint, best_path)

        return str(checkpoint_path)

    def load_checkpoint(
        self,
        checkpoint_path: str,
        load_optimizer: bool = True,
        strict: bool = True,
    ) -> Dict[str, Any]:
        """Load checkpoint with curriculum state."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"], strict=strict)

        if load_optimizer and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.global_step = checkpoint.get("global_step", 0)
        self.epoch = checkpoint.get("epoch", 0)
        self.best_metric = checkpoint.get("best_metric", float("-inf"))

        # Restore curriculum state
        self.current_stage = checkpoint.get("current_stage", 0)
        self.stage_epoch = checkpoint.get("stage_epoch", 0)
        self.sp_lambda = checkpoint.get("sp_lambda", SELF_PACED_LAMBDA_INIT)

        # Reinitialize dataloaders for current stage
        self._init_dataloaders()

        logger.info(
            f"Loaded Phase 2 checkpoint from {checkpoint_path} "
            f"(step={self.global_step}, epoch={self.epoch}, stage={self.current_stage + 1})"
        )

        return checkpoint
