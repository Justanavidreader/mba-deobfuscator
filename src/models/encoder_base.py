"""
Base encoder interface for ablation study.

All encoders must implement this interface for fair comparison.
Addresses critical issues:
- Edge type handling: validates edge_type availability when required
- Consistent forward signature across all encoders
"""

from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn


class BaseEncoder(ABC, nn.Module):
    """
    Abstract base class for all encoder architectures.

    Ensures consistent interface across:
    - GAT+JKNet (homogeneous, no edge types)
    - GGNN (heterogeneous, requires edge types)
    - HGT (heterogeneous, requires edge types)
    - RGCN (heterogeneous, requires edge types)
    - Transformer-only (sequence, no edge types)
    - Hybrid GREAT (mixed, no edge types)
    """

    def __init__(self, hidden_dim: int = 256, **kwargs):
        super().__init__()
        self._hidden_dim = hidden_dim

    @property
    def hidden_dim(self) -> int:
        """Output embedding dimension."""
        return self._hidden_dim

    @property
    @abstractmethod
    def requires_edge_types(self) -> bool:
        """
        Whether this encoder uses edge type information.

        If True, forward() will raise ValueError when edge_type is None.
        """
        pass

    @property
    @abstractmethod
    def requires_node_features(self) -> bool:
        """
        True if encoder expects [total_nodes, node_dim] features.
        False if encoder expects [total_nodes] node type IDs (embeds internally).
        """
        pass

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        edge_type: Optional[torch.Tensor] = None,
        dag_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode graph to node embeddings.

        Args:
            x: Node features
               - [total_nodes, node_dim] if requires_node_features is True
               - [total_nodes] node type IDs if requires_node_features is False
            edge_index: [2, num_edges] edge indices
            batch: [total_nodes] batch assignment
            edge_type: [num_edges] edge type indices (required if requires_edge_types)
            dag_pos: [total_nodes, 4] DAG positional features (optional)
                     Columns: [depth, subtree_size, in_degree, is_shared]

        Returns:
            [total_nodes, hidden_dim] node embeddings

        Raises:
            ValueError: If requires_edge_types is True but edge_type is None
        """
        # Critical fix: validate edge_type before forwarding
        if self.requires_edge_types and edge_type is None:
            raise ValueError(
                f"{self.__class__.__name__} requires edge_type but got None. "
                "Ensure your dataset provides edge_type for heterogeneous encoders."
            )

        return self._forward_impl(x, edge_index, batch, edge_type, dag_pos)

    @abstractmethod
    def _forward_impl(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        edge_type: Optional[torch.Tensor],
        dag_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Implementation-specific forward pass.

        Subclasses implement this instead of forward() to ensure
        edge_type validation always runs first.

        Args:
            dag_pos: [total_nodes, 4] DAG positional features (optional)
                     Columns: [depth, subtree_size, in_degree, is_shared]
                     Encoders can ignore if USE_DAG_FEATURES is False.
        """
        pass

    def parameter_count(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"hidden_dim={self.hidden_dim}, "
            f"requires_edge_types={self.requires_edge_types}, "
            f"params={self.parameter_count():,})"
        )
