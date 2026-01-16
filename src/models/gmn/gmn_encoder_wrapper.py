"""
GMN encoder wrappers that combine pre-trained encoders with Graph Matching Networks.

These wrappers subclass BaseEncoder for compatibility with the encoder registry.
"""

import logging
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from src.models.encoder_base import BaseEncoder
from src.models.gmn.graph_matching import GraphMatchingNetwork

logger = logging.getLogger(__name__)


class HGTWithGMN(BaseEncoder):
    """
    Combines pre-trained HGT encoder with GMN cross-attention.

    Subclasses BaseEncoder for compatibility with encoder_registry.

    Training strategy:
      - Phase 1a: Train HGT alone (existing Phase 1 contrastive)
      - Phase 1b: Freeze HGT, train GMN cross-attention layers
      - Phase 1c: (Optional) Fine-tune entire network end-to-end
    """

    def __init__(
        self,
        hgt_checkpoint_path: Optional[str],
        gmn_config: Dict[str, Any],
        hgt_encoder: Optional[nn.Module] = None,
    ):
        """
        Initialize HGT+GMN wrapper.

        Args:
            hgt_checkpoint_path: Path to pre-trained HGT weights (can be None if hgt_encoder provided)
            gmn_config: Config dict for GMN with keys:
                - hidden_dim: int (must match encoder output)
                - num_attention_layers: int (default: 2)
                - num_heads: int (default: 8)
                - dropout: float (default: 0.1)
                - aggregation: str (default: 'mean_max')
                - freeze_encoder: bool (default: True)
            hgt_encoder: Pre-initialized HGT encoder (alternative to checkpoint)

        Raises:
            ValueError: If HGT encoder hidden_dim doesn't match gmn_config['hidden_dim']
            ValueError: If neither checkpoint nor encoder provided
        """
        hidden_dim = gmn_config.get('hidden_dim', 256)
        super().__init__(hidden_dim=hidden_dim)

        # Load or use provided encoder
        if hgt_encoder is not None:
            self.hgt_encoder = hgt_encoder
            encoder_hidden_dim = hgt_encoder.hidden_dim
        elif hgt_checkpoint_path is not None:
            checkpoint = torch.load(hgt_checkpoint_path, map_location='cpu')
            encoder_config = checkpoint.get('encoder_config', {})
            encoder_hidden_dim = encoder_config.get('hidden_dim', hidden_dim)

            # Lazy import to avoid circular dependency
            from src.models.encoder import HGTEncoder
            self.hgt_encoder = HGTEncoder(**encoder_config)
            self.hgt_encoder.load_state_dict(checkpoint['encoder_state_dict'])
            logger.info(f"Loaded HGT encoder from {hgt_checkpoint_path}")
        else:
            raise ValueError("Must provide either hgt_checkpoint_path or hgt_encoder")

        # Validate dimension compatibility
        gmn_hidden_dim = gmn_config.get('hidden_dim', 256)
        if encoder_hidden_dim != gmn_hidden_dim:
            raise ValueError(
                f"Dimension mismatch: HGT encoder has hidden_dim={encoder_hidden_dim}, "
                f"but gmn_config specifies hidden_dim={gmn_hidden_dim}. "
                f"Set gmn_config['hidden_dim']={encoder_hidden_dim} or retrain HGT."
            )

        # Freeze HGT (optional, controlled by config)
        self._encoder_frozen = gmn_config.get('freeze_encoder', True)
        if self._encoder_frozen:
            for param in self.hgt_encoder.parameters():
                param.requires_grad = False
            logger.info("HGT encoder frozen for GMN training")

        # Initialize GMN
        self.gmn = GraphMatchingNetwork(
            encoder=self.hgt_encoder,
            hidden_dim=gmn_config.get('hidden_dim', 256),
            num_attention_layers=gmn_config.get('num_attention_layers', 2),
            num_heads=gmn_config.get('num_heads', 8),
            dropout=gmn_config.get('dropout', 0.1),
            aggregation=gmn_config.get('aggregation', 'mean_max')
        )

    @property
    def requires_edge_types(self) -> bool:
        """HGT requires edge types."""
        return self.hgt_encoder.requires_edge_types

    @property
    def requires_node_features(self) -> bool:
        """Delegate to HGT encoder."""
        return self.hgt_encoder.requires_node_features

    def _forward_impl(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        edge_type: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        BaseEncoder interface: encode single graph.

        Args:
            x: Node features
            edge_index: Edge indices
            batch: Batch assignment
            edge_type: Edge types

        Returns:
            Node embeddings [num_nodes, hidden_dim]
        """
        return self.hgt_encoder(x, edge_index, batch, edge_type)

    def forward_pair(
        self,
        graph1_batch,
        graph2_batch,
        return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        Forward pass through GMN for graph pair matching.

        Args:
            graph1_batch: PyG Batch for first graph (e.g., obfuscated)
            graph2_batch: PyG Batch for second graph (e.g., simplified)
            return_attention: If True, return attention weights

        Returns:
            match_score: [batch_size, 1] similarity score (0-1)
            attention_dict: (optional) attention weights per layer
        """
        return self.gmn(graph1_batch, graph2_batch, return_attention=return_attention)

    def unfreeze_encoder(self):
        """Unfreeze HGT encoder for end-to-end fine-tuning."""
        for param in self.hgt_encoder.parameters():
            param.requires_grad = True
            # Clear stale gradients from frozen phase to prevent corruption
            if param.grad is not None:
                param.grad.zero_()
        self._encoder_frozen = False
        logger.info("HGT encoder unfrozen for fine-tuning")

    def freeze_encoder(self):
        """Freeze HGT encoder."""
        for param in self.hgt_encoder.parameters():
            param.requires_grad = False
        self._encoder_frozen = True
        logger.info("HGT encoder frozen")

    @property
    def is_encoder_frozen(self) -> bool:
        """Check if encoder is frozen."""
        return self._encoder_frozen

    @classmethod
    def from_pretrained(cls, checkpoint_path: str, device: Optional[torch.device] = None):
        """
        Load pre-trained HGT+GMN model.

        Args:
            checkpoint_path: Path to saved checkpoint
            device: Device to load model to

        Returns:
            HGTWithGMN instance
        """
        checkpoint = torch.load(checkpoint_path, map_location=device or 'cpu')

        # Extract configs
        gmn_config = checkpoint.get('gmn_config', {})
        hgt_checkpoint_path = checkpoint.get('hgt_checkpoint_path')

        # Create model
        model = cls(
            hgt_checkpoint_path=hgt_checkpoint_path,
            gmn_config=gmn_config,
        )

        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])

        if device is not None:
            model = model.to(device)

        return model

    def save_checkpoint(
        self,
        path: str,
        hgt_checkpoint_path: Optional[str] = None,
        gmn_config: Optional[Dict] = None,
        additional_data: Optional[Dict] = None,
    ):
        """
        Save model checkpoint.

        Args:
            path: Save path
            hgt_checkpoint_path: Original HGT checkpoint path (for reference)
            gmn_config: GMN configuration
            additional_data: Additional data to save (e.g., training metrics)
        """
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'hgt_checkpoint_path': hgt_checkpoint_path,
            'gmn_config': gmn_config or {},
        }
        if additional_data:
            checkpoint.update(additional_data)

        torch.save(checkpoint, path)
        logger.info(f"Saved HGT+GMN checkpoint to {path}")


class GATWithGMN(BaseEncoder):
    """
    Combines GAT+JKNet encoder with GMN cross-attention.

    Same pattern as HGTWithGMN but for GAT encoder.
    """

    def __init__(
        self,
        gat_checkpoint_path: Optional[str],
        gmn_config: Dict[str, Any],
        gat_encoder: Optional[nn.Module] = None,
    ):
        """
        Initialize GAT+GMN wrapper.

        Args:
            gat_checkpoint_path: Path to pre-trained GAT weights
            gmn_config: Config dict for GMN
            gat_encoder: Pre-initialized GAT encoder (alternative to checkpoint)
        """
        hidden_dim = gmn_config.get('hidden_dim', 256)
        super().__init__(hidden_dim=hidden_dim)

        # Load or use provided encoder
        if gat_encoder is not None:
            self.gat_encoder = gat_encoder
            encoder_hidden_dim = gat_encoder.hidden_dim
        elif gat_checkpoint_path is not None:
            checkpoint = torch.load(gat_checkpoint_path, map_location='cpu')
            encoder_config = checkpoint.get('encoder_config', {})
            encoder_hidden_dim = encoder_config.get('hidden_dim', hidden_dim)

            from src.models.encoder import GATJKNetEncoder
            self.gat_encoder = GATJKNetEncoder(**encoder_config)
            self.gat_encoder.load_state_dict(checkpoint['encoder_state_dict'])
            logger.info(f"Loaded GAT encoder from {gat_checkpoint_path}")
        else:
            raise ValueError("Must provide either gat_checkpoint_path or gat_encoder")

        # Validate dimension compatibility
        gmn_hidden_dim = gmn_config.get('hidden_dim', 256)
        if encoder_hidden_dim != gmn_hidden_dim:
            raise ValueError(
                f"Dimension mismatch: GAT encoder has hidden_dim={encoder_hidden_dim}, "
                f"but gmn_config specifies hidden_dim={gmn_hidden_dim}."
            )

        # Freeze GAT (optional)
        self._encoder_frozen = gmn_config.get('freeze_encoder', True)
        if self._encoder_frozen:
            for param in self.gat_encoder.parameters():
                param.requires_grad = False

        # Initialize GMN
        self.gmn = GraphMatchingNetwork(
            encoder=self.gat_encoder,
            hidden_dim=gmn_config.get('hidden_dim', 256),
            num_attention_layers=gmn_config.get('num_attention_layers', 2),
            num_heads=gmn_config.get('num_heads', 8),
            dropout=gmn_config.get('dropout', 0.1),
            aggregation=gmn_config.get('aggregation', 'mean_max')
        )

    @property
    def requires_edge_types(self) -> bool:
        """GAT does not require edge types."""
        return False

    @property
    def requires_node_features(self) -> bool:
        """GAT expects node features."""
        return True

    def _forward_impl(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        edge_type: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode single graph using GAT encoder."""
        return self.gat_encoder(x, edge_index, batch, edge_type)

    def forward_pair(
        self,
        graph1_batch,
        graph2_batch,
        return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """Forward pass through GMN for graph pair matching."""
        return self.gmn(graph1_batch, graph2_batch, return_attention=return_attention)

    def unfreeze_encoder(self):
        """Unfreeze GAT encoder for fine-tuning."""
        for param in self.gat_encoder.parameters():
            param.requires_grad = True
            if param.grad is not None:
                param.grad.zero_()
        self._encoder_frozen = False

    def freeze_encoder(self):
        """Freeze GAT encoder."""
        for param in self.gat_encoder.parameters():
            param.requires_grad = False
        self._encoder_frozen = True

    @property
    def is_encoder_frozen(self) -> bool:
        """Check if encoder is frozen."""
        return self._encoder_frozen
