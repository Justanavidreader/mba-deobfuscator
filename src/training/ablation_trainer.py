"""
Trainer for ablation study experiments.

Handles encoder instantiation, training, and metric collection.

Addresses critical issue from quality review:
- Uses unified BaseEncoder.forward() interface which validates edge_type internally
- No conditional branching on requires_edge_types - encoder handles this
"""

import logging
import time
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.encoder_registry import get_encoder
from src.utils.ablation_metrics import AblationMetricsCollector

logger = logging.getLogger(__name__)


class AblationTrainer:
    """
    Trainer for ablation study.

    Handles encoder instantiation, training, and metric collection.
    Works with any encoder implementing the BaseEncoder interface.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize trainer from config.

        Args:
            config: Training configuration dictionary with keys:
                - encoder: {name: str, hidden_dim: int, ...}
                - decoder: {d_model: int, num_layers: int, ...}
                - training: {learning_rate: float, epochs: int, ...}
                - evaluation: {depth_buckets: List[List[int]], ...}
                - experiment: {run_id: int, ...}
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Instantiate encoder from config
        encoder_cfg = config["encoder"].copy()
        encoder_name = encoder_cfg.pop("name")
        self.encoder_name = encoder_name
        self.encoder = get_encoder(encoder_name, **encoder_cfg).to(self.device)

        logger.info(f"Initialized encoder: {self.encoder}")

        # Decoder (lazy import to avoid circular dependencies)
        self.decoder = None
        self.vocab_head = None
        self.complexity_head = None

        self._init_decoder(config.get("decoder", {}))

        # Optimizer
        self.optimizer = self._init_optimizer(config.get("training", {}))

        # Metrics collector
        depth_buckets = config.get("evaluation", {}).get(
            "depth_buckets", [[2, 4], [5, 7], [8, 10], [11, 14]]
        )
        # Convert to list of tuples
        depth_buckets = [tuple(b) for b in depth_buckets]
        self.metrics_collector = AblationMetricsCollector(depth_buckets=depth_buckets)

        # Training state
        self.training_time_hours = 0.0
        self.run_id = config.get("experiment", {}).get("run_id", 1)

    def _init_decoder(self, decoder_cfg: Dict[str, Any]) -> None:
        """Initialize decoder and output heads."""
        try:
            from src.models.decoder import TransformerDecoderWithCopy
            from src.models.heads import ComplexityHead, VocabHead

            d_model = decoder_cfg.get("d_model", 512)
            num_layers = decoder_cfg.get("num_layers", 6)
            num_heads = decoder_cfg.get("num_heads", 8)
            d_ff = decoder_cfg.get("d_ff", 2048)
            dropout = decoder_cfg.get("dropout", 0.1)

            self.decoder = TransformerDecoderWithCopy(
                d_model=d_model,
                num_layers=num_layers,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=dropout,
                encoder_dim=self.encoder.hidden_dim,
            ).to(self.device)

            self.vocab_head = VocabHead(d_model=d_model).to(self.device)
            self.complexity_head = ComplexityHead(d_model=d_model).to(self.device)

        except ImportError as e:
            logger.warning(f"Could not initialize decoder: {e}")
            # Decoder is optional for encoder-only ablation

    def _init_optimizer(self, training_cfg: Dict[str, Any]) -> torch.optim.Optimizer:
        """Initialize optimizer."""
        params = list(self.encoder.parameters())
        if self.decoder is not None:
            params += list(self.decoder.parameters())
        if self.vocab_head is not None:
            params += list(self.vocab_head.parameters())
        if self.complexity_head is not None:
            params += list(self.complexity_head.parameters())

        lr = training_cfg.get("learning_rate", 1e-4)
        weight_decay = training_cfg.get("weight_decay", 0.01)

        return torch.optim.AdamW(
            params,
            lr=lr,
            betas=(0.9, 0.98),
            eps=1e-9,
            weight_decay=weight_decay,
        )

    def train_epoch(self, dataloader: DataLoader) -> float:
        """
        Train for one epoch.

        Args:
            dataloader: Training data loader

        Returns:
            Average loss for the epoch
        """
        self.encoder.train()
        if self.decoder is not None:
            self.decoder.train()

        total_loss = 0.0
        num_batches = 0
        start_time = time.time()

        for batch in dataloader:
            loss = self._train_step(batch)
            total_loss += loss
            num_batches += 1

        epoch_time = (time.time() - start_time) / 3600  # Convert to hours
        self.training_time_hours += epoch_time

        return total_loss / max(num_batches, 1)

    def _train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """
        Single training step.

        Args:
            batch: Batch dictionary with keys:
                - x: [total_nodes, node_dim] or [total_nodes] node features/IDs
                - edge_index: [2, num_edges] edge indices
                - batch: [total_nodes] batch assignment
                - edge_type: [num_edges] edge type indices (optional)
                - tgt: [batch_size, seq_len] target token IDs
                - tgt_len: [batch_size] target lengths

        Returns:
            Loss value
        """
        # Move batch to device
        x = batch["x"].to(self.device)
        edge_index = batch["edge_index"].to(self.device)
        batch_idx = batch["batch"].to(self.device)

        # edge_type may or may not be present
        edge_type = batch.get("edge_type")
        if edge_type is not None:
            edge_type = edge_type.to(self.device)

        # Target tokens
        tgt = batch.get("tgt")
        if tgt is not None:
            tgt = tgt.to(self.device)

        # Critical fix: Use unified forward interface
        # BaseEncoder.forward() handles edge_type validation internally
        # If encoder requires edge_type and it's None, forward() raises ValueError
        # This is the correct behavior - the error surfaces clearly
        node_embeddings = self.encoder(x, edge_index, batch_idx, edge_type)

        # If no decoder, return dummy loss (encoder-only ablation)
        if self.decoder is None or tgt is None:
            return 0.0

        # Decoder forward
        decoder_out, copy_attn, p_gen = self.decoder(
            tgt=tgt[:, :-1],  # Remove last token
            memory=node_embeddings,
        )

        # Compute loss
        loss = self._compute_loss(decoder_out, copy_attn, p_gen, tgt)

        # Backward
        self.optimizer.zero_grad()
        loss.backward()

        gradient_clip = self.config.get("training", {}).get("gradient_clip", 1.0)
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), gradient_clip)
        if self.decoder is not None:
            torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), gradient_clip)

        self.optimizer.step()

        return loss.item()

    def _compute_loss(
        self,
        decoder_out: torch.Tensor,
        copy_attn: torch.Tensor,
        p_gen: torch.Tensor,
        tgt: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute training loss (CE + complexity + copy).

        This is a simplified implementation - full version uses
        losses from src/training/losses.py
        """
        # Basic cross-entropy loss
        if self.vocab_head is not None:
            logits = self.vocab_head(decoder_out)
            # Shift targets for next-token prediction
            targets = tgt[:, 1:]  # Remove SOS token

            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=0,  # Ignore PAD
            )
            return loss

        return torch.tensor(0.0, device=self.device)

    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader,
        tokenizer: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate and collect metrics.

        Args:
            dataloader: Evaluation data loader
            tokenizer: Tokenizer for decoding (optional)

        Returns:
            Dictionary with predictions, targets, and metrics
        """
        self.encoder.eval()
        if self.decoder is not None:
            self.decoder.eval()

        predictions: List[str] = []
        targets: List[str] = []
        inputs: List[str] = []
        depths: List[int] = []
        latencies: List[float] = []

        for batch in dataloader:
            start_time = time.time()

            # Move to device
            x = batch["x"].to(self.device)
            edge_index = batch["edge_index"].to(self.device)
            batch_idx = batch["batch"].to(self.device)
            edge_type = batch.get("edge_type")
            if edge_type is not None:
                edge_type = edge_type.to(self.device)

            # Encode
            node_embeddings = self.encoder(x, edge_index, batch_idx, edge_type)

            # Decode (greedy for speed)
            if self.decoder is not None and tokenizer is not None:
                pred_tokens = self._greedy_decode(node_embeddings, batch_idx)
                batch_preds = self._tokens_to_strings(pred_tokens, tokenizer)
            else:
                # No decoder - return empty predictions
                batch_size = batch_idx.max().item() + 1
                batch_preds = [""] * batch_size

            latency = time.time() - start_time

            # Collect data
            predictions.extend(batch_preds)

            if "target_strings" in batch:
                targets.extend(batch["target_strings"])
            else:
                targets.extend([""] * len(batch_preds))

            if "input_strings" in batch:
                inputs.extend(batch["input_strings"])
            else:
                inputs.extend([""] * len(batch_preds))

            if "depth" in batch:
                depths.extend(batch["depth"].tolist())
            else:
                depths.extend([0] * len(batch_preds))

            # Per-sample latency
            per_sample_latency = latency / max(len(batch_preds), 1)
            latencies.extend([per_sample_latency] * len(batch_preds))

        # Collect metrics
        self.metrics_collector.collect(
            encoder_name=self.encoder_name,
            run_id=self.run_id,
            predictions=predictions,
            targets=targets,
            inputs=inputs,
            depths=depths,
            latencies=latencies,
            encoder_params=self.encoder.parameter_count(),
            training_time_hours=self.training_time_hours,
        )

        return {
            "predictions": predictions,
            "targets": targets,
            "inputs": inputs,
            "depths": depths,
        }

    def _greedy_decode(
        self,
        memory: torch.Tensor,
        batch: torch.Tensor,
        max_len: int = 64,
    ) -> torch.Tensor:
        """
        Greedy decoding for fast inference.

        Args:
            memory: [total_nodes, hidden_dim] node embeddings
            batch: [total_nodes] batch assignment
            max_len: Maximum output length

        Returns:
            [batch_size, seq_len] decoded token IDs
        """
        from src.constants import EOS_IDX, SOS_IDX

        batch_size = batch.max().item() + 1
        device = memory.device

        # Start with SOS token
        output = torch.full(
            (batch_size, 1), SOS_IDX, dtype=torch.long, device=device
        )

        for _ in range(max_len - 1):
            decoder_out, _, _ = self.decoder(tgt=output, memory=memory)

            # Get logits for last position
            if self.vocab_head is not None:
                logits = self.vocab_head(decoder_out[:, -1, :])
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                next_token = torch.zeros((batch_size, 1), dtype=torch.long, device=device)

            output = torch.cat([output, next_token], dim=1)

            # Check if all sequences have EOS
            if (next_token.squeeze(-1) == EOS_IDX).all():
                break

        return output

    def _tokens_to_strings(
        self,
        tokens: torch.Tensor,
        tokenizer: Any,
    ) -> List[str]:
        """
        Convert token IDs to expression strings.

        Args:
            tokens: [batch_size, seq_len] token IDs
            tokenizer: Tokenizer with decode method

        Returns:
            List of decoded strings
        """
        results = []
        for i in range(tokens.size(0)):
            seq = tokens[i].tolist()
            decoded = tokenizer.decode(seq)
            results.append(decoded)
        return results

    def save_checkpoint(self, path: str) -> None:
        """Save training checkpoint."""
        checkpoint = {
            "encoder_state": self.encoder.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "config": self.config,
            "training_time_hours": self.training_time_hours,
        }
        if self.decoder is not None:
            checkpoint["decoder_state"] = self.decoder.state_dict()
        if self.vocab_head is not None:
            checkpoint["vocab_head_state"] = self.vocab_head.state_dict()

        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str) -> None:
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.encoder.load_state_dict(checkpoint["encoder_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.training_time_hours = checkpoint.get("training_time_hours", 0.0)

        if self.decoder is not None and "decoder_state" in checkpoint:
            self.decoder.load_state_dict(checkpoint["decoder_state"])
        if self.vocab_head is not None and "vocab_head_state" in checkpoint:
            self.vocab_head.load_state_dict(checkpoint["vocab_head_state"])

        logger.info(f"Loaded checkpoint from {path}")
