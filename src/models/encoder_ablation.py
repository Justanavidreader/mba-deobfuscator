"""
Ablation study encoder implementations.

New encoder architectures for comparative evaluation:
- TransformerOnlyEncoder: Sequence baseline (no graph structure)
- HybridGREATEncoder: GNN+Transformer interleaved layers
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

from src.constants import NODE_DIM
from src.models.encoder_base import BaseEncoder
from src.utils.graph_utils import safe_dfs_order


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for Transformer.

    Injects position information into sequence embeddings.
    """

    def __init__(self, d_model: int, max_len: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Pre-compute positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.

        Args:
            x: [batch, seq_len, d_model] input embeddings

        Returns:
            [batch, seq_len, d_model] with positional encoding added
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class TransformerOnlyEncoder(BaseEncoder):
    """
    Transformer encoder baseline (no graph structure).

    Linearizes AST via DFS traversal, applies positional encoding,
    and processes with standard Transformer encoder. Output is
    mapped back to node-level embeddings for compatibility.

    Research question: Does graph structure provide benefit over sequence?

    Architecture: 6 layers, 8 heads, 256 hidden (~2.5M params)
    """

    def __init__(
        self,
        node_dim: int = NODE_DIM,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        d_ff: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 1024,
        **kwargs,
    ):
        super().__init__(hidden_dim=hidden_dim)
        self.node_dim = node_dim

        self.node_embedding = nn.Linear(node_dim, hidden_dim)
        self.pos_encoding = SinusoidalPositionalEncoding(
            hidden_dim, max_len=max_seq_len, dropout=dropout
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-LN for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.node_embedding.weight)
        nn.init.zeros_(self.node_embedding.bias)

    @property
    def requires_edge_types(self) -> bool:
        """Transformer-only does not use edge types."""
        return False

    @property
    def requires_node_features(self) -> bool:
        """Expects [total_nodes, node_dim] features."""
        return True

    def _forward_impl(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        edge_type: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode AST as sequence via DFS linearization.

        Args:
            x: [total_nodes, node_dim] node features
            edge_index: [2, num_edges] (used only for DFS traversal)
            batch: [total_nodes] batch assignment
            edge_type: Ignored

        Returns:
            [total_nodes, hidden_dim] node embeddings
        """
        batch_size = batch.max().item() + 1
        num_nodes = x.size(0)
        device = x.device

        # Embed all nodes first
        h = self.node_embedding(x)  # [total_nodes, hidden_dim]

        # Output tensor
        node_embeddings = torch.zeros(num_nodes, self.hidden_dim, device=device)

        for b in range(batch_size):
            # Get mask for this graph
            mask = batch == b
            graph_nodes = h[mask]  # [num_nodes_in_graph, hidden_dim]
            num_graph_nodes = graph_nodes.size(0)

            if num_graph_nodes == 0:
                continue

            # DFS traversal to get node order (uses safe fallback)
            dfs_order = safe_dfs_order(edge_index, mask)

            # Reorder nodes according to DFS
            ordered_nodes = graph_nodes[dfs_order]  # [num_graph_nodes, hidden_dim]

            # Apply positional encoding
            seq = ordered_nodes.unsqueeze(0)  # [1, seq_len, hidden_dim]
            seq = self.pos_encoding(seq)

            # Transformer forward
            encoded = self.transformer(seq)  # [1, seq_len, hidden_dim]

            # Map back to original node order
            inverse_order = torch.argsort(dfs_order)
            reordered = encoded[0, inverse_order]  # [num_graph_nodes, hidden_dim]

            # Store in output
            node_embeddings[mask] = reordered

        return node_embeddings


class HybridGREATEncoder(BaseEncoder):
    """
    Hybrid GNN+Transformer encoder (GREAT-style).

    Alternates GNN layers (graph-structured message passing) with
    Transformer layers (global sequence attention). GNN layers
    exploit local graph structure; Transformer layers capture
    long-range dependencies.

    Research question: Can interleaving GNN+Transformer improve over pure GNN?

    Architecture: 3 × (GAT + Transformer) blocks, 256 hidden (~4.0M params)
    """

    def __init__(
        self,
        node_dim: int = NODE_DIM,
        hidden_dim: int = 256,
        num_blocks: int = 3,
        num_heads: int = 8,
        d_ff: int = 1024,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__(hidden_dim=hidden_dim)
        self.node_dim = node_dim
        self.num_blocks = num_blocks

        self.node_embedding = nn.Linear(node_dim, hidden_dim)

        # Alternating GNN + Transformer blocks
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            block = nn.ModuleDict(
                {
                    "gnn": GATConv(
                        hidden_dim,
                        hidden_dim // num_heads,
                        heads=num_heads,
                        concat=True,
                        dropout=dropout,
                    ),
                    "gnn_norm": nn.LayerNorm(hidden_dim),
                    "transformer": nn.TransformerEncoderLayer(
                        d_model=hidden_dim,
                        nhead=num_heads,
                        dim_feedforward=d_ff,
                        dropout=dropout,
                        batch_first=True,
                        norm_first=True,
                    ),
                    "transformer_norm": nn.LayerNorm(hidden_dim),
                }
            )
            self.blocks.append(block)

        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.node_embedding.weight)
        nn.init.zeros_(self.node_embedding.bias)

    @property
    def requires_edge_types(self) -> bool:
        """Hybrid uses GAT which doesn't need edge types."""
        return False

    @property
    def requires_node_features(self) -> bool:
        """Expects [total_nodes, node_dim] features."""
        return True

    def _forward_impl(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        edge_type: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward through hybrid GNN+Transformer blocks.

        Each block:
        1. GNN layer: local message passing on graph structure
        2. Transformer layer: global attention over all nodes in batch

        Args:
            x: [total_nodes, node_dim] node features
            edge_index: [2, num_edges]
            batch: [total_nodes] batch assignment
            edge_type: Ignored

        Returns:
            [total_nodes, hidden_dim] node embeddings
        """
        h = self.node_embedding(x)
        h = F.elu(h)

        batch_size = batch.max().item() + 1

        for block in self.blocks:
            # GNN phase: graph message passing
            h_gnn = block["gnn"](h, edge_index)
            h = h + self.dropout(h_gnn)  # Residual
            h = block["gnn_norm"](h)
            h = F.elu(h)

            # Transformer phase: global attention within each graph
            h_transformer = torch.zeros_like(h)

            for b in range(batch_size):
                mask = batch == b
                num_graph_nodes = mask.sum().item()

                if num_graph_nodes == 0:
                    continue

                graph_nodes = h[mask].unsqueeze(0)  # [1, num_nodes, hidden_dim]

                # Transformer operates on sequence (no position encoding needed
                # since GNN already incorporates structural info)
                transformed = block["transformer"](graph_nodes)
                h_transformer[mask] = transformed[0]

            h = h + self.dropout(h_transformer)  # Residual
            h = block["transformer_norm"](h)
            h = F.elu(h)

        return h
