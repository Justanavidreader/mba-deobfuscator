"""
Base trainer with shared functionality for all training phases.

Provides:
- Optimizer and LR scheduler setup
- Gradient clipping and accumulation
- Checkpointing (save/load)
- Logging (console + TensorBoard)
- Device management
"""

import logging
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

logger = logging.getLogger(__name__)


class BaseTrainer(ABC):
    """
    Abstract base trainer with shared training infrastructure.

    Subclasses implement:
    - train_step(): Single training iteration
    - evaluate(): Validation loop
    """

    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        device: Optional[torch.device] = None,
        checkpoint_dir: Optional[str] = None,
        experiment_name: str = "experiment",
    ):
        """
        Initialize base trainer.

        Args:
            model: PyTorch model to train
            config: Training configuration dictionary with keys:
                - learning_rate: float
                - weight_decay: float
                - warmup_steps: int
                - max_grad_norm: float
                - gradient_accumulation_steps: int
                - scheduler_type: str ('cosine', 'linear', 'constant')
                - total_steps: int (for scheduler)
            device: Device to train on (default: auto-detect)
            checkpoint_dir: Directory for saving checkpoints
            experiment_name: Name for logging
        """
        self.model = model
        self.config = config
        self.experiment_name = experiment_name

        # Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.model.to(self.device)

        # Training config
        self.learning_rate = config.get("learning_rate", 1e-4)
        self.weight_decay = config.get("weight_decay", 0.01)
        self.warmup_steps = config.get("warmup_steps", 1000)
        self.max_grad_norm = config.get("max_grad_norm", 1.0)
        self.gradient_accumulation_steps = config.get("gradient_accumulation_steps", 1)

        # Initialize optimizer and scheduler
        self.optimizer = self._init_optimizer()
        self.scheduler = self._init_scheduler(config.get("total_steps", 100000))

        # Checkpointing
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else Path("checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_metric = float("-inf")
        self.training_start_time = None

        # Logging
        self.tensorboard_writer = None
        self._accumulated_loss = 0.0
        self._accumulation_count = 0

    def _init_optimizer(self) -> Optimizer:
        """Initialize AdamW optimizer with weight decay."""
        # Separate parameters that should/shouldn't have weight decay
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": 0.0,
            },
        ]

        return torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.learning_rate,
            betas=(0.9, 0.98),
            eps=1e-9,
        )

    def _init_scheduler(self, total_steps: int) -> _LRScheduler:
        """Initialize learning rate scheduler with warmup."""
        scheduler_type = self.config.get("scheduler_type", "cosine")

        if scheduler_type == "cosine":
            from torch.optim.lr_scheduler import CosineAnnealingLR
            # Cosine annealing after warmup
            return CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps - self.warmup_steps,
                eta_min=self.learning_rate * 0.01,
            )
        elif scheduler_type == "linear":
            from torch.optim.lr_scheduler import LinearLR
            return LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.01,
                total_iters=total_steps,
            )
        else:
            # Constant LR (just warmup, then constant)
            from torch.optim.lr_scheduler import ConstantLR
            return ConstantLR(self.optimizer, factor=1.0, total_iters=total_steps)

    def _get_lr(self) -> float:
        """Get current learning rate with warmup."""
        if self.global_step < self.warmup_steps:
            # Linear warmup
            return self.learning_rate * (self.global_step + 1) / self.warmup_steps
        return self.scheduler.get_last_lr()[0]

    def _apply_warmup(self) -> None:
        """Apply warmup learning rate."""
        if self.global_step < self.warmup_steps:
            lr = self._get_lr()
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

    def backward(
        self,
        loss: torch.Tensor,
        update: bool = True,
    ) -> None:
        """
        Backward pass with gradient accumulation and clipping.

        Args:
            loss: Loss tensor to backprop
            update: Whether to perform optimizer step (set False for accumulation)
        """
        # Scale loss for gradient accumulation
        scaled_loss = loss / self.gradient_accumulation_steps
        scaled_loss.backward()

        self._accumulated_loss += loss.item()
        self._accumulation_count += 1

        if update and self._accumulation_count >= self.gradient_accumulation_steps:
            # Gradient clipping
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm,
                )

            # Optimizer step
            self.optimizer.step()
            self.optimizer.zero_grad()

            # LR scheduler step
            self._apply_warmup()
            if self.global_step >= self.warmup_steps:
                self.scheduler.step()

            self.global_step += 1
            self._accumulated_loss = 0.0
            self._accumulation_count = 0

    def save_checkpoint(
        self,
        filename: Optional[str] = None,
        metrics: Optional[Dict[str, float]] = None,
        is_best: bool = False,
    ) -> str:
        """
        Save training checkpoint.

        Args:
            filename: Checkpoint filename (default: checkpoint_{step}.pt)
            metrics: Optional metrics to save
            is_best: If True, also save as best_model.pt

        Returns:
            Path to saved checkpoint
        """
        if filename is None:
            filename = f"checkpoint_{self.global_step}.pt"

        checkpoint_path = self.checkpoint_dir / filename

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_metric": self.best_metric,
            "config": self.config,
        }

        if metrics:
            checkpoint["metrics"] = metrics

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")

        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model to {best_path}")

        return str(checkpoint_path)

    def load_checkpoint(
        self,
        checkpoint_path: str,
        load_optimizer: bool = True,
        strict: bool = True,
    ) -> Dict[str, Any]:
        """
        Load training checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            load_optimizer: Whether to load optimizer state
            strict: Strict mode for model loading

        Returns:
            Checkpoint dictionary with metadata
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"], strict=strict)

        if load_optimizer and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.global_step = checkpoint.get("global_step", 0)
        self.epoch = checkpoint.get("epoch", 0)
        self.best_metric = checkpoint.get("best_metric", float("-inf"))

        logger.info(
            f"Loaded checkpoint from {checkpoint_path} "
            f"(step={self.global_step}, epoch={self.epoch})"
        )

        return checkpoint

    def init_tensorboard(self, log_dir: Optional[str] = None) -> None:
        """Initialize TensorBoard writer."""
        try:
            from torch.utils.tensorboard import SummaryWriter

            if log_dir is None:
                log_dir = self.checkpoint_dir / "tensorboard"

            self.tensorboard_writer = SummaryWriter(log_dir=str(log_dir))
            logger.info(f"TensorBoard logging to {log_dir}")
        except ImportError:
            logger.warning("TensorBoard not available. Install with: pip install tensorboard")

    def log_metrics(
        self,
        metrics: Dict[str, float],
        prefix: str = "train",
    ) -> None:
        """
        Log metrics to console and TensorBoard.

        Args:
            metrics: Dictionary of metric names and values
            prefix: Prefix for metric names (train/val/test)
        """
        # Console logging
        metric_str = " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
        logger.info(f"[{prefix}] Step {self.global_step} | {metric_str}")

        # TensorBoard logging
        if self.tensorboard_writer is not None:
            for name, value in metrics.items():
                self.tensorboard_writer.add_scalar(
                    f"{prefix}/{name}",
                    value,
                    self.global_step,
                )

    def log_lr(self) -> None:
        """Log current learning rate."""
        lr = self._get_lr()
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.add_scalar("lr", lr, self.global_step)

    def close(self) -> None:
        """Clean up resources."""
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.close()

        # Clean up diagnostics if active
        self.cleanup_diagnostics()

    # ========== Diagnostic Integration ==========

    def enable_activation_monitor(self, track_gradients: bool = True) -> None:
        """
        Enable activation monitoring diagnostics.

        Args:
            track_gradients: Whether to track gradient norms
        """
        try:
            from src.diagnostics import ActivationMonitor

            self.activation_monitor = ActivationMonitor(track_gradients=track_gradients)
            self.activation_monitor.register_hooks(self.model)
            logger.info("Activation monitor enabled")
        except ImportError as e:
            logger.warning(f"Cannot enable activation monitor: {e}")

    def enable_collapse_detector(
        self,
        sample_size: int = 500,
        max_nodes_for_full_distance: int = 5000,
    ) -> None:
        """
        Enable representation collapse detection.

        Args:
            sample_size: Number of nodes to sample for diversity computation
            max_nodes_for_full_distance: Memory budget for O(N²) operations
        """
        try:
            from src.diagnostics import CollapseDetector

            self.collapse_detector = CollapseDetector(
                sample_size=sample_size,
                max_nodes_for_full_distance=max_nodes_for_full_distance,
            )
            self.collapse_detector.register_hooks(self.model)
            logger.info("Collapse detector enabled")
        except ImportError as e:
            logger.warning(f"Cannot enable collapse detector: {e}")

    def enable_attention_analyzer(self) -> None:
        """Enable attention pattern analysis."""
        try:
            from src.diagnostics import AttentionAnalyzer

            self.attention_analyzer = AttentionAnalyzer()
            self.attention_analyzer.register_hooks(self.model)
            logger.info("Attention analyzer enabled")
        except ImportError as e:
            logger.warning(f"Cannot enable attention analyzer: {e}")

    def enable_diversity_metrics(
        self,
        sample_size: int = 500,
        max_nodes_for_full_distance: int = 5000,
    ) -> None:
        """
        Enable node embedding diversity metrics.

        Args:
            sample_size: Number of nodes to sample
            max_nodes_for_full_distance: Memory budget for O(N²) operations
        """
        try:
            from src.diagnostics import DiversityMetrics

            self.diversity_metrics = DiversityMetrics(
                sample_size=sample_size,
                max_nodes_for_full_distance=max_nodes_for_full_distance,
            )
            self.diversity_metrics.register_hooks(self.model)
            logger.info("Diversity metrics enabled")
        except ImportError as e:
            logger.warning(f"Cannot enable diversity metrics: {e}")

    def cleanup_diagnostics(self) -> None:
        """
        Clean up all diagnostic hooks.

        CRITICAL: Call this in finally block or use context manager pattern.
        """
        for attr in ['activation_monitor', 'collapse_detector', 'attention_analyzer', 'diversity_metrics']:
            if hasattr(self, attr):
                diagnostic = getattr(self, attr)
                if hasattr(diagnostic, 'remove_hooks'):
                    diagnostic.remove_hooks()

    def log_diagnostics(self, log_interval: int = 100) -> None:
        """
        Log all active diagnostics to TensorBoard and console.

        CRITICAL FIX: Refactored to avoid god function (was 71 lines, 4 nesting levels).
        Now delegates to helper methods.

        Args:
            log_interval: Steps between diagnostic logging
        """
        if self.global_step % log_interval != 0:
            return

        # Log each diagnostic type
        if hasattr(self, 'activation_monitor'):
            self._log_single_diagnostic(
                self.activation_monitor,
                'diagnostics/activations',
                'detect_pathologies',
            )

        if hasattr(self, 'collapse_detector'):
            self._log_single_diagnostic(
                self.collapse_detector,
                'diagnostics/collapse',
                'detect_collapse',
            )

        if hasattr(self, 'attention_analyzer'):
            self._log_single_diagnostic(
                self.attention_analyzer,
                'diagnostics/attention',
                None,  # No pathology detector
            )

        if hasattr(self, 'diversity_metrics'):
            self._log_single_diagnostic(
                self.diversity_metrics,
                'diagnostics/diversity',
                'detect_over_squashing',
            )

    def _log_single_diagnostic(
        self,
        collector,
        prefix: str,
        pathology_method: Optional[str],
    ) -> None:
        """
        Helper: Log single diagnostic collector.

        CRITICAL FIX: Extracted from log_diagnostics to reduce complexity.

        Args:
            collector: Diagnostic collector instance
            prefix: TensorBoard prefix
            pathology_method: Method name for pathology detection (if any)
        """
        # Log to TensorBoard
        collector.log_to_tensorboard(self.tensorboard_writer, self.global_step, prefix)

        # Detect and log pathologies
        if pathology_method and hasattr(collector, pathology_method):
            pathologies = getattr(collector, pathology_method)()
            if any(pathologies.values()):
                warning_msg = self._format_pathology_warning(pathologies, self.global_step)
                logger.warning(warning_msg)

        # Reset collector
        collector.reset()

    def _format_pathology_warning(
        self,
        pathologies: Dict[str, List[str]],
        step: int,
    ) -> str:
        """
        Helper: Format pathology warning message.

        CRITICAL FIX: Extracted from log_diagnostics to reduce complexity.

        Args:
            pathologies: Dictionary of pathology types to layer names
            step: Current training step

        Returns:
            Formatted warning message
        """
        lines = [f"Pathologies detected at step {step}:"]
        for pathology_type, layers in pathologies.items():
            if layers:
                lines.append(f"  {pathology_type}: {len(layers)} layers")
                for layer in layers[:3]:
                    lines.append(f"    - {layer}")
                if len(layers) > 3:
                    lines.append(f"    ... and {len(layers) - 3} more")

        return "\n".join(lines)

    # ========== End Diagnostic Integration ==========

    @abstractmethod
    def train_step(self, batch: Any) -> Dict[str, float]:
        """
        Single training step.

        Args:
            batch: Batch of training data

        Returns:
            Dictionary of loss values
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, dataloader: Any) -> Dict[str, float]:
        """
        Evaluation loop.

        Args:
            dataloader: Validation dataloader

        Returns:
            Dictionary of evaluation metrics
        """
        raise NotImplementedError

    def train_epoch(
        self,
        dataloader: Any,
        log_interval: int = 100,
    ) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            dataloader: Training dataloader
            log_interval: Steps between logging

        Returns:
            Average epoch metrics
        """
        self.model.train()
        epoch_metrics: Dict[str, List[float]] = {}
        num_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            # Training step
            metrics = self.train_step(batch)

            # Accumulate metrics
            for key, value in metrics.items():
                if key not in epoch_metrics:
                    epoch_metrics[key] = []
                epoch_metrics[key].append(value)

            num_batches += 1

            # Log periodically
            if batch_idx % log_interval == 0 and batch_idx > 0:
                self.log_metrics(metrics)
                self.log_lr()

        # Average metrics
        avg_metrics = {
            key: sum(values) / len(values)
            for key, values in epoch_metrics.items()
        }

        self.epoch += 1
        return avg_metrics

    def fit(
        self,
        train_loader: Any,
        val_loader: Optional[Any] = None,
        num_epochs: int = 10,
        eval_interval: int = 1,
        save_interval: int = 1,
        metric_for_best: str = "loss",
        higher_is_better: bool = False,
    ) -> Dict[str, Any]:
        """
        Full training loop.

        Args:
            train_loader: Training dataloader
            val_loader: Validation dataloader (optional)
            num_epochs: Number of epochs to train
            eval_interval: Epochs between evaluation
            save_interval: Epochs between checkpoints
            metric_for_best: Metric to track for best model
            higher_is_better: Whether higher metric is better

        Returns:
            Training history
        """
        self.training_start_time = time.time()
        history = {"train": [], "val": []}

        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")

            # Train
            train_metrics = self.train_epoch(train_loader)
            history["train"].append(train_metrics)
            self.log_metrics(train_metrics, prefix="train_epoch")

            # Evaluate
            if val_loader is not None and (epoch + 1) % eval_interval == 0:
                val_metrics = self.evaluate(val_loader)
                history["val"].append(val_metrics)
                self.log_metrics(val_metrics, prefix="val")

                # Track best model
                current_metric = val_metrics.get(metric_for_best, 0)
                if higher_is_better:
                    is_best = current_metric > self.best_metric
                else:
                    is_best = current_metric < self.best_metric

                if is_best:
                    self.best_metric = current_metric
                    self.save_checkpoint(metrics=val_metrics, is_best=True)

            # Save checkpoint
            if (epoch + 1) % save_interval == 0:
                self.save_checkpoint(
                    filename=f"checkpoint_epoch_{epoch + 1}.pt",
                    metrics=train_metrics,
                )

        # Training summary
        total_time = (time.time() - self.training_start_time) / 3600
        logger.info(f"Training complete. Total time: {total_time:.2f} hours")

        return history
