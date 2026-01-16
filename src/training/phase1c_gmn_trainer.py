"""
Phase 1c Trainer: End-to-End GMN Fine-Tuning.

Fine-tunes entire GMN model (encoder + cross-attention) end-to-end.
Optional phase after Phase 1b for performance improvement on complex expressions.
Uses lower learning rate for encoder to prevent catastrophic forgetting.
"""

import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from src.training.phase1b_gmn_trainer import Phase1bGMNTrainer
from src.training.negative_sampler import NegativeSampler

logger = logging.getLogger(__name__)


class Phase1cGMNTrainer(Phase1bGMNTrainer):
    """
    Phase 1c: End-to-end GMN fine-tuning.

    Inherits from Phase1bGMNTrainer but unfreezes encoder for joint optimization.
    Uses lower learning rate for encoder to prevent catastrophic forgetting.
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
        Initialize Phase 1c GMN trainer.

        Args:
            model: HGTWithGMN or GATWithGMN instance
            config: Training configuration with additional keys:
                - encoder_learning_rate: float (default: 1e-5, 10x lower than GMN)
                - encoder_weight_decay: float (default: 0.001, lower regularization)
                - unfreeze_encoder: bool (default: True)
                - All Phase1bGMNTrainer config keys
            negative_sampler: NegativeSampler instance
            device: Training device
            checkpoint_dir: Checkpoint save directory
        """
        # Phase 1c specific config
        self.encoder_learning_rate = config.get('encoder_learning_rate', 3e-6)
        self.encoder_weight_decay = config.get('encoder_weight_decay', 0.001)
        self.unfreeze_encoder_flag = config.get('unfreeze_encoder', True)

        # Store reference to initial encoder weights for drift monitoring
        self._initial_encoder_state = None

        # Initialize parent (calls _init_optimizer)
        # Note: We temporarily bypass the frozen check by setting a flag
        self._skip_frozen_check = True

        # Call grandparent init to avoid Phase1b's frozen check
        from src.training.base_trainer import BaseTrainer
        BaseTrainer.__init__(
            self,
            model=model,
            config=config,
            device=device,
            checkpoint_dir=checkpoint_dir,
            experiment_name="phase1c_gmn_finetune",
        )

        self._skip_frozen_check = False

        # Store GMN-specific config (from Phase1b)
        self.bce_pos_weight = config.get('bce_pos_weight', 1.0)
        self.triplet_margin = config.get('triplet_loss_margin', None)
        self.triplet_weight = config.get('triplet_loss_weight', 0.1)
        self.negative_sampler = negative_sampler

        # Unfreeze encoder if configured
        if self.unfreeze_encoder_flag:
            self._safe_unfreeze_encoder()

        # Capture initial encoder state for drift monitoring
        self._capture_initial_encoder_state()

        logger.info(
            f"Phase1cGMNTrainer initialized: "
            f"encoder_lr={self.encoder_learning_rate}, "
            f"gmn_lr={self.learning_rate}, "
            f"encoder_frozen={model.is_encoder_frozen}"
        )

    def _safe_unfreeze_encoder(self):
        """
        Safely unfreeze encoder with gradient cleanup.

        CRITICAL: Must clear optimizer state and reset gradient accumulation
        to prevent stale frozen-phase gradients from corrupting updates.
        """
        # CRITICAL: Clear optimizer state before unfreezing
        self.optimizer.zero_grad(set_to_none=True)

        # Get encoder reference
        encoder_name = 'hgt_encoder' if hasattr(self.model, 'hgt_encoder') else 'gat_encoder'
        encoder = getattr(self.model, encoder_name, None)

        if encoder is not None:
            for param in encoder.parameters():
                param.requires_grad = True
                # Ensure no stale gradients exist
                if param.grad is not None:
                    param.grad = None

        # Update model's internal state
        self.model._encoder_frozen = False

        # Reset gradient accumulation counter
        self._accumulated_loss = 0.0
        self._accumulation_count = 0

        logger.info("Encoder unfrozen for fine-tuning (gradients cleared)")

    def _capture_initial_encoder_state(self):
        """Capture encoder state for drift monitoring."""
        encoder_name = 'hgt_encoder' if hasattr(self.model, 'hgt_encoder') else 'gat_encoder'
        encoder = getattr(self.model, encoder_name, None)

        if encoder is not None:
            self._initial_encoder_state = {
                name: param.detach().clone()
                for name, param in encoder.named_parameters()
            }

    def _compute_encoder_drift(self) -> float:
        """
        Compute L2 distance from initial encoder weights.

        Used to monitor catastrophic forgetting.

        Returns:
            L2 drift value
        """
        if self._initial_encoder_state is None:
            return 0.0

        encoder_name = 'hgt_encoder' if hasattr(self.model, 'hgt_encoder') else 'gat_encoder'
        encoder = getattr(self.model, encoder_name, None)

        if encoder is None:
            return 0.0

        total_drift = 0.0
        for name, param in encoder.named_parameters():
            if name in self._initial_encoder_state:
                drift = (param - self._initial_encoder_state[name]).pow(2).sum()
                total_drift += drift.item()

        return total_drift ** 0.5

    def _init_optimizer(self) -> Optimizer:
        """
        Initialize optimizer with separate learning rates for encoder and GMN.

        Returns:
            AdamW optimizer with parameter groups
        """
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]

        # Get encoder and GMN parameters separately
        encoder_name = 'hgt_encoder' if hasattr(self.model, 'hgt_encoder') else 'gat_encoder'
        encoder = getattr(self.model, encoder_name, None)

        encoder_params_decay = []
        encoder_params_no_decay = []
        gmn_params_decay = []
        gmn_params_no_decay = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            is_encoder = encoder_name in name
            is_no_decay = any(nd in name for nd in no_decay)

            if is_encoder:
                if is_no_decay:
                    encoder_params_no_decay.append(param)
                else:
                    encoder_params_decay.append(param)
            else:
                if is_no_decay:
                    gmn_params_no_decay.append(param)
                else:
                    gmn_params_decay.append(param)

        optimizer_grouped_parameters = [
            {
                'params': encoder_params_decay,
                'lr': self.encoder_learning_rate,
                'weight_decay': self.encoder_weight_decay,
            },
            {
                'params': encoder_params_no_decay,
                'lr': self.encoder_learning_rate,
                'weight_decay': 0.0,
            },
            {
                'params': gmn_params_decay,
                'lr': self.learning_rate,
                'weight_decay': self.weight_decay,
            },
            {
                'params': gmn_params_no_decay,
                'lr': self.learning_rate,
                'weight_decay': 0.0,
            },
        ]

        return torch.optim.AdamW(
            optimizer_grouped_parameters,
            betas=(0.9, 0.98),
            eps=1e-9,
        )

    def _verify_encoder_frozen(self):
        """Override: No frozen check in Phase 1c (encoder is unfrozen)."""
        if getattr(self, '_skip_frozen_check', False):
            return
        # In Phase 1c, encoder should NOT be frozen
        if self.model.is_encoder_frozen:
            logger.warning("Encoder is still frozen in Phase 1c - this may limit learning")

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """
        Single training step with unfrozen encoder.

        Extends Phase1bGMNTrainer.train_step() with:
        - Gradient norm tracking for encoder vs GMN layers
        - Periodic encoder drift monitoring

        Returns:
            Extended loss dict with:
                - All Phase1bGMNTrainer keys
                - 'encoder_grad_norm': L2 norm of encoder gradients
                - 'gmn_grad_norm': L2 norm of GMN gradients
                - 'encoder_drift': L2 distance from Phase 1b checkpoint (every 100 steps)
        """
        # Call parent train_step
        result = super().train_step(batch)

        # Add gradient norm tracking
        encoder_name = 'hgt_encoder' if hasattr(self.model, 'hgt_encoder') else 'gat_encoder'

        encoder_grad_norm = 0.0
        gmn_grad_norm = 0.0

        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                if encoder_name in name:
                    encoder_grad_norm += grad_norm ** 2
                else:
                    gmn_grad_norm += grad_norm ** 2

        result['encoder_grad_norm'] = encoder_grad_norm ** 0.5
        result['gmn_grad_norm'] = gmn_grad_norm ** 0.5

        # Compute drift every 100 steps
        if self.global_step % 100 == 0:
            result['encoder_drift'] = self._compute_encoder_drift()
        else:
            result['encoder_drift'] = 0.0

        return result

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Evaluate with drift monitoring.

        Returns:
            Extended metrics dict with 'encoder_drift'
        """
        result = super().evaluate(dataloader)
        result['encoder_drift'] = self._compute_encoder_drift()
        return result

    def save_checkpoint(
        self,
        filename: Optional[str] = None,
        metrics: Optional[Dict[str, float]] = None,
        is_best: bool = False,
    ) -> str:
        """Save checkpoint with Phase 1c metadata."""
        if filename is None:
            filename = f"phase1c_gmn_step{self.global_step}.pt"

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
            'phase': '1c',
            'encoder_drift': self._compute_encoder_drift(),
            'metrics': metrics or {},
        }

        torch.save(checkpoint, filepath)
        logger.info(f"Saved checkpoint to {filepath}")

        if is_best:
            best_path = self.checkpoint_dir / "phase1c_gmn_best.pt"
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
        Load checkpoint for Phase 1c.

        Note: Does NOT enforce frozen encoder (unlike Phase1b).
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

        # Re-capture encoder state for drift monitoring from this checkpoint
        self._capture_initial_encoder_state()

        logger.info(
            f"Loaded checkpoint from {checkpoint_path}: "
            f"step={self.global_step}, epoch={self.epoch}, "
            f"encoder_frozen={self.model.is_encoder_frozen}"
        )

        return checkpoint
