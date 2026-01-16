"""
Phase 1 Trainer: Contrastive Pretraining.

Trains the encoder using:
1. InfoNCE loss: Pull together equivalent expressions (obfuscated, simplified pairs)
2. MaskLM loss: Predict masked node types in AST graphs

This phase learns semantic representations before supervised training.
"""

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.training.base_trainer import BaseTrainer
from src.training.losses import infonce_loss, masklm_loss
from src.constants import (
    INFONCE_TEMPERATURE,
    MASKLM_MASK_RATIO,
    MASKLM_WEIGHT,
    NUM_NODE_TYPES,
    HIDDEN_DIM,
)

logger = logging.getLogger(__name__)


class Phase1Trainer(BaseTrainer):
    """
    Contrastive pretraining trainer.

    Uses InfoNCE loss to learn that semantically equivalent expressions
    (obfuscated and simplified) should have similar embeddings, while
    non-equivalent expressions should have different embeddings.

    Additionally uses MaskLM to learn AST structure by predicting masked nodes.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        device: Optional[torch.device] = None,
        checkpoint_dir: Optional[str] = None,
    ):
        """
        Initialize Phase 1 trainer.

        Args:
            model: MBADeobfuscator model (only encoder is trained)
            config: Training configuration with:
                - infonce_temperature: float (default: 0.07)
                - masklm_mask_ratio: float (default: 0.15)
                - masklm_weight: float (default: 0.5)
                - learning_rate, weight_decay, etc.
            device: Training device
            checkpoint_dir: Directory for checkpoints
        """
        # Phase 1 specific config (set before super().__init__ so _init_optimizer can use them)
        self.infonce_temp = config.get("infonce_temperature", INFONCE_TEMPERATURE)
        self.mask_ratio = config.get("masklm_mask_ratio", MASKLM_MASK_RATIO)
        self.masklm_weight = config.get("masklm_weight", MASKLM_WEIGHT)

        # MaskLM prediction head: hidden_dim -> NUM_NODE_TYPES
        # Create BEFORE super().__init__ so it's included in optimizer initialization
        hidden_dim = config.get("hidden_dim", HIDDEN_DIM)
        _device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mlm_head = nn.Linear(hidden_dim, NUM_NODE_TYPES).to(_device)

        super().__init__(
            model=model,
            config=config,
            device=device,
            checkpoint_dir=checkpoint_dir,
            experiment_name="phase1_contrastive",
        )

        logger.info(
            f"Phase1Trainer initialized: "
            f"temp={self.infonce_temp}, mask_ratio={self.mask_ratio}, "
            f"masklm_weight={self.masklm_weight}"
        )

    def _init_optimizer(self) -> torch.optim.Optimizer:
        """
        Initialize optimizer including MaskLM head parameters.

        Overrides base to include mlm_head in optimizer from the start,
        ensuring checkpoint resume works correctly.
        """
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]

        # Model parameters
        model_params_decay = [
            p for n, p in self.model.named_parameters()
            if not any(nd in n for nd in no_decay) and p.requires_grad
        ]
        model_params_no_decay = [
            p for n, p in self.model.named_parameters()
            if any(nd in n for nd in no_decay) and p.requires_grad
        ]

        # MaskLM head parameters
        mlm_params_decay = [
            p for n, p in self.mlm_head.named_parameters()
            if not any(nd in n for nd in no_decay) and p.requires_grad
        ]
        mlm_params_no_decay = [
            p for n, p in self.mlm_head.named_parameters()
            if any(nd in n for nd in no_decay) and p.requires_grad
        ]

        optimizer_grouped_parameters = [
            {"params": model_params_decay, "weight_decay": self.weight_decay},
            {"params": model_params_no_decay, "weight_decay": 0.0},
            {"params": mlm_params_decay, "weight_decay": self.weight_decay},
            {"params": mlm_params_no_decay, "weight_decay": 0.0},
        ]

        return torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.learning_rate,
            betas=(0.9, 0.98),
            eps=1e-9,
        )

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """
        Single training step for contrastive pretraining.

        Args:
            batch: Collated batch from ContrastiveDataset with keys:
                - obf_graph_batch: PyG batch for obfuscated expressions
                - simp_graph_batch: PyG batch for simplified expressions
                - obf_fingerprint: [batch, FINGERPRINT_DIM]
                - simp_fingerprint: [batch, FINGERPRINT_DIM]

        Returns:
            Dict with 'total', 'infonce', 'masklm' losses
        """
        self.model.train()
        self.mlm_head.train()

        # Move to device
        obf_graph = batch['obf_graph_batch'].to(self.device)
        simp_graph = batch['simp_graph_batch'].to(self.device)
        obf_fp = batch['obf_fingerprint'].to(self.device)
        simp_fp = batch['simp_fingerprint'].to(self.device)

        # Encode both expressions
        # model.encode returns [batch, 1, d_model]
        obf_context = self.model.encode(obf_graph, obf_fp)
        simp_context = self.model.encode(simp_graph, simp_fp)

        # Squeeze to [batch, d_model] for InfoNCE
        obf_embed = obf_context.squeeze(1)
        simp_embed = simp_context.squeeze(1)

        # InfoNCE loss
        infonce = infonce_loss(obf_embed, simp_embed, temperature=self.infonce_temp)

        # MaskLM loss (only on obfuscated graph for diversity)
        masked_loss = self._masklm_step(obf_graph)

        # Combined loss
        total_loss = infonce + self.masklm_weight * masked_loss

        # Backward pass with gradient accumulation
        self.backward(total_loss, update=True)

        return {
            'total': total_loss.item(),
            'infonce': infonce.item(),
            'masklm': masked_loss.item(),
        }

    def _masklm_step(self, graph_batch) -> torch.Tensor:
        """
        Masked language modeling step.

        Randomly masks nodes and predicts their original types.

        Args:
            graph_batch: PyG batch

        Returns:
            MaskLM loss
        """
        # Get node features
        x = graph_batch.x  # [total_nodes] node type IDs
        edge_index = graph_batch.edge_index
        batch_idx = graph_batch.batch
        edge_type = getattr(graph_batch, 'edge_type', None)

        num_nodes = x.shape[0]
        num_masked = int(num_nodes * self.mask_ratio)

        if num_masked == 0:
            return torch.tensor(0.0, device=self.device)

        # Randomly select nodes to mask
        mask_indices = torch.randperm(num_nodes, device=self.device)[:num_masked]

        # Store original types
        if x.dim() == 1:
            # Node type IDs
            original_types = x[mask_indices].clone()
            # Replace with mask token (use 0 as mask)
            x_masked = x.clone()
            x_masked[mask_indices] = 0
        else:
            # One-hot or embedding features
            original_types = x[mask_indices].argmax(dim=-1)
            x_masked = x.clone()
            x_masked[mask_indices] = 0

        # Encode with masked nodes
        # Parameter order: (x, edge_index, edge_type, batch) for GGNN/HGT/RGCN
        encoder_type = getattr(self.model, 'encoder_type', 'gat')
        if encoder_type in ('ggnn', 'hgt', 'rgcn') and edge_type is not None:
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

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Evaluate contrastive model.

        Computes:
        - InfoNCE loss on validation set
        - MaskLM accuracy
        - Embedding similarity statistics

        Args:
            dataloader: Validation dataloader

        Returns:
            Evaluation metrics
        """
        self.model.eval()
        self.mlm_head.eval()

        total_infonce = 0.0
        total_masklm = 0.0
        total_masklm_correct = 0
        total_masklm_samples = 0
        total_batches = 0

        # Track embedding similarities
        pos_similarities = []
        neg_similarities = []

        for batch in dataloader:
            # Move to device
            obf_graph = batch['obf_graph_batch'].to(self.device)
            simp_graph = batch['simp_graph_batch'].to(self.device)
            obf_fp = batch['obf_fingerprint'].to(self.device)
            simp_fp = batch['simp_fingerprint'].to(self.device)

            # Encode
            obf_context = self.model.encode(obf_graph, obf_fp)
            simp_context = self.model.encode(simp_graph, simp_fp)

            obf_embed = obf_context.squeeze(1)
            simp_embed = simp_context.squeeze(1)

            # InfoNCE loss
            infonce = infonce_loss(obf_embed, simp_embed, temperature=self.infonce_temp)
            total_infonce += infonce.item()

            # Compute similarity matrix for analysis
            obf_norm = torch.nn.functional.normalize(obf_embed, p=2, dim=-1)
            simp_norm = torch.nn.functional.normalize(simp_embed, p=2, dim=-1)
            sim_matrix = torch.matmul(obf_norm, simp_norm.T)

            # Positive similarities (diagonal)
            pos_sim = sim_matrix.diag().cpu().tolist()
            pos_similarities.extend(pos_sim)

            # Negative similarities (off-diagonal)
            batch_size = sim_matrix.shape[0]
            mask = ~torch.eye(batch_size, dtype=torch.bool, device=self.device)
            neg_sim = sim_matrix[mask].cpu().tolist()
            neg_similarities.extend(neg_sim)

            # MaskLM evaluation
            masklm_loss_val, correct, total = self._eval_masklm(obf_graph)
            total_masklm += masklm_loss_val
            total_masklm_correct += correct
            total_masklm_samples += total

            total_batches += 1

        # Compute averages
        avg_infonce = total_infonce / max(total_batches, 1)
        avg_masklm = total_masklm / max(total_batches, 1)
        masklm_acc = total_masklm_correct / max(total_masklm_samples, 1)

        avg_pos_sim = sum(pos_similarities) / max(len(pos_similarities), 1)
        avg_neg_sim = sum(neg_similarities) / max(len(neg_similarities), 1)

        return {
            'infonce': avg_infonce,
            'masklm': avg_masklm,
            'masklm_acc': masklm_acc,
            'pos_similarity': avg_pos_sim,
            'neg_similarity': avg_neg_sim,
            'sim_gap': avg_pos_sim - avg_neg_sim,  # Should be positive and large
        }

    def _eval_masklm(self, graph_batch) -> tuple:
        """
        Evaluate MaskLM accuracy.

        Returns:
            (loss, num_correct, num_total)
        """
        x = graph_batch.x
        edge_index = graph_batch.edge_index
        batch_idx = graph_batch.batch
        edge_type = getattr(graph_batch, 'edge_type', None)

        num_nodes = x.shape[0]
        num_masked = int(num_nodes * self.mask_ratio)

        if num_masked == 0:
            return 0.0, 0, 0

        # Mask nodes
        mask_indices = torch.randperm(num_nodes, device=self.device)[:num_masked]

        if x.dim() == 1:
            original_types = x[mask_indices].clone()
            x_masked = x.clone()
            x_masked[mask_indices] = 0
        else:
            original_types = x[mask_indices].argmax(dim=-1)
            x_masked = x.clone()
            x_masked[mask_indices] = 0

        # Encode
        encoder_type = getattr(self.model, 'encoder_type', 'gat')
        if encoder_type in ('ggnn', 'hgt', 'rgcn') and edge_type is not None:
            node_embeddings = self.model.graph_encoder(x_masked, edge_index, edge_type, batch_idx)
        else:
            node_embeddings = self.model.graph_encoder(x_masked, edge_index, batch_idx)

        # Predict
        masked_embeddings = node_embeddings[mask_indices]
        logits = self.mlm_head(masked_embeddings)

        # Loss
        loss = torch.nn.functional.cross_entropy(logits, original_types)

        # Accuracy
        predictions = logits.argmax(dim=-1)
        correct = (predictions == original_types).sum().item()

        return loss.item(), correct, num_masked

    def save_checkpoint(
        self,
        filename: Optional[str] = None,
        metrics: Optional[Dict[str, float]] = None,
        is_best: bool = False,
    ) -> str:
        """Save checkpoint including MaskLM head."""
        if filename is None:
            filename = f"phase1_checkpoint_{self.global_step}.pt"

        checkpoint_path = self.checkpoint_dir / filename

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "mlm_head_state_dict": self.mlm_head.state_dict(),
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
        logger.info(f"Saved Phase 1 checkpoint to {checkpoint_path}")

        if is_best:
            best_path = self.checkpoint_dir / "phase1_best.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best Phase 1 model to {best_path}")

        return str(checkpoint_path)

    def load_checkpoint(
        self,
        checkpoint_path: str,
        load_optimizer: bool = True,
        strict: bool = True,
    ) -> Dict[str, Any]:
        """Load checkpoint including MaskLM head."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"], strict=strict)

        if "mlm_head_state_dict" in checkpoint:
            self.mlm_head.load_state_dict(checkpoint["mlm_head_state_dict"], strict=strict)

        if load_optimizer and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.global_step = checkpoint.get("global_step", 0)
        self.epoch = checkpoint.get("epoch", 0)
        self.best_metric = checkpoint.get("best_metric", float("-inf"))

        logger.info(
            f"Loaded Phase 1 checkpoint from {checkpoint_path} "
            f"(step={self.global_step}, epoch={self.epoch})"
        )

        return checkpoint
