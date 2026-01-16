"""
Phase 1b Trainer: GMN Training with Frozen Encoder.

Trains Graph Matching Network cross-attention layers while keeping
the pre-trained encoder frozen. Learns graph pair matching without
disrupting learned representations from Phase 1a.
"""

import logging
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.training.base_trainer import BaseTrainer
from src.training.losses import gmn_bce_loss, gmn_combined_loss
from src.training.negative_sampler import NegativeSampler

logger = logging.getLogger(__name__)


class Phase1bGMNTrainer(BaseTrainer):
    """
    Phase 1b: GMN training with frozen encoder.

    Trains cross-attention layers for graph matching while encoder remains frozen.
    Uses binary classification loss with hard negative mining.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        negative_sampler: NegativeSampler,
        device: Optional[torch.device] = None,
        checkpoint_dir: Optional[str] = None,
    ):
        """
        Initialize Phase 1b GMN trainer.

        Args:
            model: HGTWithGMN or GATWithGMN instance with frozen encoder
            config: Training configuration with:
                - learning_rate: float (default: 3e-5, lower than Phase 1a)
                - weight_decay: float (default: 0.01)
                - bce_pos_weight: float (default: 1.0, balance positive/negative)
                - triplet_loss_margin: Optional[float] (default: None, disable triplet)
                - triplet_loss_weight: float (default: 0.1)
                - gradient_accumulation_steps: int
                - warmup_steps: int
                - max_grad_norm: float
                - scheduler_type: str
            negative_sampler: NegativeSampler for generating hard negatives
            device: Training device
            checkpoint_dir: Checkpoint save directory

        Raises:
            ValueError: If encoder is not frozen
            ValueError: If model is not HGTWithGMN or GATWithGMN
        """
        # Validate model type
        if not hasattr(model, 'is_encoder_frozen') or not hasattr(model, 'forward_pair'):
            raise ValueError("Model must be HGTWithGMN or GATWithGMN with forward_pair method")

        # GMN-specific config
        self.bce_pos_weight = config.get('bce_pos_weight', 1.0)
        self.triplet_margin = config.get('triplet_loss_margin', None)
        self.triplet_weight = config.get('triplet_loss_weight', 0.1)

        # Auto-adjust pos_weight based on negative_ratio if using default
        negative_ratio = config.get('negative_ratio', 1.0)
        if self.bce_pos_weight == 1.0 and negative_ratio != 1.0:
            self.bce_pos_weight = negative_ratio
            logger.warning(
                f"Auto-adjusted bce_pos_weight from 1.0 to {negative_ratio} "
                f"to match negative_ratio. Set bce_pos_weight explicitly to override."
            )

        self.negative_sampler = negative_sampler

        super().__init__(
            model=model,
            config=config,
            device=device,
            checkpoint_dir=checkpoint_dir,
            experiment_name="phase1b_gmn",
        )

        # Verify encoder is frozen after init
        if not model.is_encoder_frozen:
            raise ValueError("Encoder must be frozen for Phase 1b training")

        logger.info(
            f"Phase1bGMNTrainer initialized: "
            f"bce_pos_weight={self.bce_pos_weight}, "
            f"triplet_margin={self.triplet_margin}, "
            f"encoder_frozen={model.is_encoder_frozen}"
        )

    def _verify_encoder_frozen(self):
        """Verify encoder remains frozen (call at start of each step)."""
        if not self.model.is_encoder_frozen:
            raise RuntimeError("Encoder became unfrozen during Phase 1b training")

        # Also check requires_grad flags
        encoder_name = 'hgt_encoder' if hasattr(self.model, 'hgt_encoder') else 'gat_encoder'
        encoder = getattr(self.model, encoder_name, None)
        if encoder is not None:
            for name, param in encoder.named_parameters():
                if param.requires_grad:
                    raise RuntimeError(f"Encoder parameter {name} has requires_grad=True")

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """
        Single training step for GMN.

        Args:
            batch: Collated batch from GMNBatchCollator with keys:
                - graph1_batch: PyG Batch (obfuscated expressions)
                - graph2_batch: PyG Batch (candidate simplified expressions)
                - labels: [batch_size] float tensor (1.0=equivalent, 0.0=not)
                - pair_indices: [batch_size, 2] mapping indices

        Returns:
            Dict with keys:
                - 'total': Total loss
                - 'bce': Binary cross-entropy loss
                - 'triplet': Triplet loss (if enabled)
                - 'accuracy': Binary accuracy
                - 'pos_score': Average score on positive pairs
                - 'neg_score': Average score on negative pairs
        """
        self._verify_encoder_frozen()
        self.model.train()

        # Move batch to device
        graph1_batch = batch['graph1_batch'].to(self.device)
        graph2_batch = batch['graph2_batch'].to(self.device)
        labels = batch['labels'].to(self.device)

        # Forward pass
        match_scores = self.model.forward_pair(graph1_batch, graph2_batch)

        # Compute loss
        loss_dict = gmn_combined_loss(
            match_scores=match_scores,
            labels=labels,
            triplet_data=None,  # Triplet requires separate sampling
            pos_weight=self.bce_pos_weight,
            triplet_weight=self.triplet_weight,
            triplet_margin=self.triplet_margin if self.triplet_margin else 0.2,
        )

        total_loss = loss_dict['total']

        # Check for NaN gradients before backward
        if torch.isnan(total_loss):
            logger.error("NaN loss detected, skipping step")
            self.optimizer.zero_grad()
            return {
                'total': float('nan'),
                'bce': float('nan'),
                'triplet': 0.0,
                'accuracy': 0.0,
                'pos_score': 0.0,
                'neg_score': 0.0,
            }

        # Backward pass
        self.backward(total_loss, update=True)

        # Check for NaN gradients after backward
        has_nan = False
        for name, param in self.model.named_parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                logger.error(f"NaN gradient in {name}")
                has_nan = True

        if has_nan:
            logger.error("NaN gradients detected, skipping update")
            self.optimizer.zero_grad()
            return {
                'total': float('nan'),
                'bce': loss_dict['bce'].item(),
                'triplet': loss_dict['triplet'].item(),
                'accuracy': 0.0,
                'pos_score': 0.0,
                'neg_score': 0.0,
            }

        # Compute metrics
        with torch.no_grad():
            predictions = (match_scores.squeeze(-1) > 0.5).float()
            accuracy = (predictions == labels).float().mean().item()

            # Score statistics by label
            pos_mask = labels == 1.0
            neg_mask = labels == 0.0
            pos_score = match_scores[pos_mask].mean().item() if pos_mask.any() else 0.0
            neg_score = match_scores[neg_mask].mean().item() if neg_mask.any() else 0.0

        return {
            'total': total_loss.item(),
            'bce': loss_dict['bce'].item(),
            'triplet': loss_dict['triplet'].item(),
            'accuracy': accuracy,
            'pos_score': pos_score,
            'neg_score': neg_score,
        }

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Evaluate GMN on validation set.

        Args:
            dataloader: Validation DataLoader with GMNBatchCollator

        Returns:
            Dict with keys:
                - 'bce': Average BCE loss
                - 'accuracy': Binary classification accuracy
                - 'precision': Precision on positive class
                - 'recall': Recall on positive class
                - 'f1': F1 score
                - 'pos_score_mean': Mean score on positive pairs
                - 'pos_score_std': Std score on positive pairs
                - 'neg_score_mean': Mean score on negative pairs
                - 'neg_score_std': Std score on negative pairs
                - 'separation_gap': pos_score_mean - neg_score_mean
        """
        self.model.eval()

        total_loss = 0.0
        all_predictions = []
        all_labels = []
        all_scores = []

        for batch in dataloader:
            graph1_batch = batch['graph1_batch'].to(self.device)
            graph2_batch = batch['graph2_batch'].to(self.device)
            labels = batch['labels'].to(self.device)

            match_scores = self.model.forward_pair(graph1_batch, graph2_batch)

            # Compute loss
            loss = gmn_bce_loss(match_scores, labels, pos_weight=self.bce_pos_weight)
            total_loss += loss.item() * labels.size(0)

            # Collect predictions
            predictions = (match_scores.squeeze(-1) > 0.5).float()
            all_predictions.append(predictions.cpu())
            all_labels.append(labels.cpu())
            all_scores.append(match_scores.squeeze(-1).cpu())

        # Aggregate
        all_predictions = torch.cat(all_predictions)
        all_labels = torch.cat(all_labels)
        all_scores = torch.cat(all_scores)
        num_samples = len(all_labels)

        # Metrics
        avg_loss = total_loss / num_samples
        accuracy = (all_predictions == all_labels).float().mean().item()

        # Precision, Recall, F1
        true_pos = ((all_predictions == 1) & (all_labels == 1)).sum().float()
        false_pos = ((all_predictions == 1) & (all_labels == 0)).sum().float()
        false_neg = ((all_predictions == 0) & (all_labels == 1)).sum().float()

        precision = (true_pos / (true_pos + false_pos + 1e-8)).item()
        recall = (true_pos / (true_pos + false_neg + 1e-8)).item()
        f1 = (2 * precision * recall / (precision + recall + 1e-8))

        # Score statistics by class
        pos_mask = all_labels == 1.0
        neg_mask = all_labels == 0.0

        pos_scores = all_scores[pos_mask]
        neg_scores = all_scores[neg_mask]

        pos_score_mean = pos_scores.mean().item() if len(pos_scores) > 0 else 0.0
        pos_score_std = pos_scores.std().item() if len(pos_scores) > 1 else 0.0
        neg_score_mean = neg_scores.mean().item() if len(neg_scores) > 0 else 0.0
        neg_score_std = neg_scores.std().item() if len(neg_scores) > 1 else 0.0

        separation_gap = pos_score_mean - neg_score_mean

        return {
            'bce': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'pos_score_mean': pos_score_mean,
            'pos_score_std': pos_score_std,
            'neg_score_mean': neg_score_mean,
            'neg_score_std': neg_score_std,
            'separation_gap': separation_gap,
        }

    def save_checkpoint(
        self,
        filename: Optional[str] = None,
        metrics: Optional[Dict[str, float]] = None,
        is_best: bool = False,
    ) -> str:
        """
        Save checkpoint including GMN state.

        Args:
            filename: Optional filename (auto-generated if None)
            metrics: Current evaluation metrics
            is_best: Whether this is the best checkpoint

        Returns:
            Path to saved checkpoint
        """
        if filename is None:
            filename = f"phase1b_gmn_step{self.global_step}.pt"

        filepath = self.checkpoint_dir / filename

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'epoch': self.epoch,
            'best_metric': self.best_metric,
            'config': self.config,
            'encoder_frozen': self.model.is_encoder_frozen,
            'metrics': metrics or {},
        }

        torch.save(checkpoint, filepath)
        logger.info(f"Saved checkpoint to {filepath}")

        if is_best:
            best_path = self.checkpoint_dir / "phase1b_gmn_best.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best checkpoint to {best_path}")

        return str(filepath)

    def load_checkpoint(
        self,
        checkpoint_path: str,
        load_optimizer: bool = True,
        strict: bool = True,
    ) -> Dict[str, Any]:
        """
        Load checkpoint with GMN state.

        Args:
            checkpoint_path: Path to checkpoint
            load_optimizer: Whether to restore optimizer state
            strict: Strict state dict loading

        Returns:
            Checkpoint dict

        Raises:
            RuntimeError: If encoder is not frozen after loading
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'], strict=strict)

        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.global_step = checkpoint.get('global_step', 0)
        self.epoch = checkpoint.get('epoch', 0)
        self.best_metric = checkpoint.get('best_metric', float('-inf'))

        # Verify encoder freeze state
        if not self.model.is_encoder_frozen:
            raise RuntimeError("Encoder must be frozen after loading Phase 1b checkpoint")

        logger.info(
            f"Loaded checkpoint from {checkpoint_path}: "
            f"step={self.global_step}, epoch={self.epoch}"
        )

        return checkpoint
