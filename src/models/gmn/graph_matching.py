"""
Graph Matching Network for MBA equivalence detection.

Combines encoder, cross-graph attention, and graph-level aggregation
to compute matching scores between expression graphs.
"""

from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter, softmax as pyg_softmax

from src.models.gmn.cross_attention import MultiHeadCrossGraphAttention


def scatter_mean(src, index, dim=0, dim_size=None):
    """Scatter mean using PyG's scatter."""
    return scatter(src, index, dim=dim, dim_size=dim_size, reduce='mean')


def scatter_max(src, index, dim=0, dim_size=None):
    """Scatter max using PyG's scatter."""
    return scatter(src, index, dim=dim, dim_size=dim_size, reduce='max'), None


def scatter_softmax(src, index, dim=0):
    """Scatter softmax using PyG's softmax."""
    return pyg_softmax(src, index, dim=dim)


class GraphMatchingNetwork(nn.Module):
    """
    Graph Matching Network with cross-graph attention.

    Architecture:
      1. Encode both graphs independently with base encoder
      2. Apply cross-graph attention (bidirectional)
      3. Aggregate to graph-level embeddings
      4. Compute matching score
    """

    def __init__(
        self,
        encoder: nn.Module,
        hidden_dim: int = 768,
        num_attention_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        aggregation: str = 'mean_max',
    ):
        """
        Args:
            encoder: Pre-trained graph encoder (HGT/GGNN/GAT)
            hidden_dim: Must match encoder output dimension
            num_attention_layers: Number of cross-attention layers (stacked)
            num_heads: Attention heads per layer
            dropout: Dropout probability
            aggregation: Graph-level aggregation method ('mean', 'max', 'mean_max', 'attention')
        """
        super().__init__()
        self.encoder = encoder
        self.hidden_dim = hidden_dim
        self.aggregation = aggregation

        # Stack of cross-attention layers (alternating directions)
        self.cross_attn_layers = nn.ModuleList([
            MultiHeadCrossGraphAttention(hidden_dim, num_heads, dropout)
            for _ in range(num_attention_layers)
        ])

        # Layer norms for residual connections (2 per layer: h1 and h2)
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_attention_layers * 2)
        ])

        # Graph-level aggregation
        if aggregation == 'attention':
            self.graph_attention = nn.Sequential(
                nn.Linear(hidden_dim, 1),
                nn.Tanh()
            )

        # Matching score predictor
        self.match_score = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for module in self.match_score:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        if self.aggregation == 'attention':
            nn.init.xavier_uniform_(self.graph_attention[0].weight)
            nn.init.zeros_(self.graph_attention[0].bias)

    def encode_graph(self, graph_batch) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode graph using base encoder.

        Args:
            graph_batch: PyG Batch with x, edge_index, edge_type, batch

        Returns:
            node_embeddings: [total_nodes, hidden_dim]
            batch_indices: [total_nodes] batch assignment
        """
        x = graph_batch.x
        edge_index = graph_batch.edge_index
        batch = graph_batch.batch

        # edge_type may not be present for homogeneous encoders
        edge_type = getattr(graph_batch, 'edge_type', None)

        # Use encoder's forward method
        node_embeddings = self.encoder(x, edge_index, batch, edge_type)

        return node_embeddings, batch

    def cross_attention_pass(
        self,
        h1: torch.Tensor,
        h2: torch.Tensor,
        batch1: torch.Tensor,
        batch2: torch.Tensor,
        track_attention: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Apply stacked cross-attention layers (bidirectional).

        Creates attention mask to prevent cross-graph attention leak.
        Nodes in graph pair (i) can only attend to nodes in graph pair (i),
        not to nodes from other pairs in the batch.

        Args:
            h1: [N1, hidden_dim] graph 1 node embeddings
            h2: [N2, hidden_dim] graph 2 node embeddings
            batch1: [N1] batch assignment for graph 1
            batch2: [N2] batch assignment for graph 2
            track_attention: If True, store attention weights for visualization

        Returns:
            h1_matched: [N1, hidden_dim] updated embeddings for graph 1
            h2_matched: [N2, hidden_dim] updated embeddings for graph 2
            attention_dict: (optional) attention weights per layer if track_attention=True
        """
        # Create cross-attention mask: nodes in batch1[i] can only attend to nodes in batch2[i]
        # Shape: [N1, 1] == [1, N2] -> [N1, N2] boolean mask (True = allow attention)
        cross_mask_1to2 = batch1.unsqueeze(-1) == batch2.unsqueeze(0)  # [N1, N2]
        cross_mask_2to1 = batch2.unsqueeze(-1) == batch1.unsqueeze(0)  # [N2, N1]

        attention_dict = {} if track_attention else None
        h1_out, h2_out = h1, h2

        for i, cross_attn in enumerate(self.cross_attn_layers):
            # Bidirectional attention with batch-aware masking

            # h1 -> h2 attention
            match_1to2, attn_1to2 = cross_attn(h1_out, h2_out, mask1=None, mask2=cross_mask_1to2)
            h1_out = self.layer_norms[i * 2](h1_out + match_1to2)

            # h2 -> h1 attention (symmetric)
            match_2to1, attn_2to1 = cross_attn(h2_out, h1_out, mask1=None, mask2=cross_mask_2to1)
            h2_out = self.layer_norms[i * 2 + 1](h2_out + match_2to1)

            if track_attention:
                attention_dict[f'layer_{i}'] = {
                    'h1_to_h2': attn_1to2.detach(),
                    'h2_to_h1': attn_2to1.detach()
                }

        return h1_out, h2_out, attention_dict

    def aggregate_graph(
        self,
        node_embeddings: torch.Tensor,
        batch: torch.Tensor
    ) -> torch.Tensor:
        """
        Aggregate node embeddings to graph-level.

        Args:
            node_embeddings: [total_nodes, hidden_dim]
            batch: [total_nodes] batch assignment

        Returns:
            graph_embedding: [batch_size, hidden_dim]
        """
        batch_size = batch.max().item() + 1

        if self.aggregation == 'mean':
            return scatter_mean(node_embeddings, batch, dim=0, dim_size=batch_size)

        elif self.aggregation == 'max':
            graph_emb, _ = scatter_max(node_embeddings, batch, dim=0, dim_size=batch_size)
            return graph_emb

        elif self.aggregation == 'mean_max':
            mean_pool = scatter_mean(node_embeddings, batch, dim=0, dim_size=batch_size)
            max_pool, _ = scatter_max(node_embeddings, batch, dim=0, dim_size=batch_size)
            return mean_pool + max_pool

        elif self.aggregation == 'attention':
            # Learnable attention weights per node
            attn_scores = self.graph_attention(node_embeddings)  # [total_nodes, 1]
            attn_weights = scatter_softmax(attn_scores.squeeze(-1), batch, dim=0)  # [total_nodes]
            weighted = node_embeddings * attn_weights.unsqueeze(-1)
            return scatter_mean(weighted, batch, dim=0, dim_size=batch_size)

        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

    def _compute_match_score(
        self,
        h1_matched: torch.Tensor,
        h2_matched: torch.Tensor,
        batch1: torch.Tensor,
        batch2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute matching score from matched node embeddings.

        Args:
            h1_matched: [N1, hidden_dim] matched embeddings for graph 1
            h2_matched: [N2, hidden_dim] matched embeddings for graph 2
            batch1: [N1] batch assignment for graph 1
            batch2: [N2] batch assignment for graph 2

        Returns:
            match_score: [batch_size, 1] similarity score (0-1)
        """
        # Aggregate to graph-level embeddings
        g1_embedding = self.aggregate_graph(h1_matched, batch1)
        g2_embedding = self.aggregate_graph(h2_matched, batch2)

        # Compute matching score
        combined = torch.cat([g1_embedding, g2_embedding], dim=-1)
        match_score = torch.sigmoid(self.match_score(combined))

        return match_score

    def forward(
        self,
        graph1_batch,
        graph2_batch,
        return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        Forward pass for graph pair.

        Args:
            graph1_batch: PyG Batch for obfuscated expression
            graph2_batch: PyG Batch for simplified expression
            return_attention: If True, return attention weights for visualization

        Returns:
            match_score: [batch_size, 1] similarity score (0-1)
            attention_dict: (optional) attention weights per layer
        """
        # Step 1: Encode both graphs
        h1, batch1 = self.encode_graph(graph1_batch)
        h2, batch2 = self.encode_graph(graph2_batch)

        # Step 2: Cross-graph attention (bidirectional)
        h1_matched, h2_matched, attention_dict = self.cross_attention_pass(
            h1, h2, batch1, batch2, track_attention=return_attention
        )

        # Step 3: Compute matching score
        match_score = self._compute_match_score(h1_matched, h2_matched, batch1, batch2)

        if return_attention:
            return match_score, attention_dict

        return match_score
