"""
Global self-attention module for GraphGPS-style hybrid architecture.

Provides full self-attention across all nodes in a graph, enabling O(1) detection
of repeated subexpressions instead of O(depth) message passing propagation.

Usage:
    from src.models.global_attention import GlobalAttentionBlock

    block = GlobalAttentionBlock(hidden_dim=768, num_heads=8)
    x = block(x, batch)  # batch tensor prevents cross-graph attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Optional


class GlobalSelfAttention(nn.Module):
    """
    Full self-attention across all nodes in a graph.

    For batched graphs, attention is masked to prevent cross-graph attention.
    Each node attends to all other nodes in the same graph.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_bias: bool = True,
    ):
        """
        Args:
            hidden_dim: Node feature dimension
            num_heads: Number of attention heads (must divide hidden_dim evenly)
            dropout: Dropout probability
            use_bias: Whether to use bias in linear projections
        """
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})")

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=use_bias)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=use_bias)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=use_bias)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=use_bias)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

        self._init_weights()

    def _init_weights(self):
        # Xavier initialization for attention projections
        for module in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply global self-attention.

        Args:
            x: Node features [num_nodes, hidden_dim]
            batch: Batch assignment [num_nodes] - which graph each node belongs to.
                   If None, assumes single graph (all nodes attend to all).

        Returns:
            Updated node features [num_nodes, hidden_dim]
        """
        residual = x
        x = self.layer_norm(x)

        num_nodes = x.size(0)

        # Project to Q, K, V: [num_nodes, num_heads, head_dim]
        q = self.q_proj(x).view(num_nodes, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(num_nodes, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(num_nodes, self.num_heads, self.head_dim)

        # Compute attention scores: [num_nodes, num_heads, num_nodes]
        # q[i,h,d] * k[j,h,d] -> attn[i,h,j]
        attn = torch.einsum('ihd,jhd->ihj', q, k) * self.scale

        # Mask cross-graph attention in batched setting
        if batch is not None:
            # mask[i,j] = True where nodes i,j are in the same graph
            same_graph_mask = batch.unsqueeze(0) == batch.unsqueeze(1)  # [num_nodes, num_nodes]
            same_graph_mask = same_graph_mask.unsqueeze(1)  # [num_nodes, 1, num_nodes] for broadcast
            attn = attn.masked_fill(~same_graph_mask, float('-inf'))

        attn = F.softmax(attn, dim=-1)

        # Handle NaN from softmax when entire row is -inf (isolated nodes)
        attn = torch.nan_to_num(attn, nan=0.0)

        attn = self.dropout(attn)

        # Apply attention to values: attn[i,h,j] * v[j,h,d] -> out[i,h,d]
        out = torch.einsum('ihj,jhd->ihd', attn, v)
        out = out.reshape(num_nodes, self.hidden_dim)
        out = self.out_proj(out)
        out = self.dropout(out)

        return residual + out


class GlobalAttentionBlock(nn.Module):
    """
    Complete global attention block with FFN, matching Transformer architecture.

    Includes optional gradient checkpointing to reduce memory usage during training.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        ffn_ratio: float = 4.0,
        dropout: float = 0.1,
        use_checkpoint: bool = True,
    ):
        """
        Args:
            hidden_dim: Node feature dimension
            num_heads: Number of attention heads
            ffn_ratio: FFN hidden dimension multiplier
            dropout: Dropout probability
            use_checkpoint: Enable gradient checkpointing to save memory
        """
        super().__init__()
        self.use_checkpoint = use_checkpoint

        self.attn = GlobalSelfAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, int(hidden_dim * ffn_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(hidden_dim * ffn_ratio), hidden_dim),
            nn.Dropout(dropout),
        )

        self._init_weights()

    def _init_weights(self):
        # Initialize FFN linear layers
        for module in self.ffn:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def _forward_attn(self, x: torch.Tensor, batch: Optional[torch.Tensor]) -> torch.Tensor:
        """Attention forward for checkpointing (must be separate method)."""
        return self.attn(x, batch)

    def _forward_ffn(self, x: torch.Tensor) -> torch.Tensor:
        """FFN forward for checkpointing."""
        return x + self.ffn(x)

    def forward(
        self,
        x: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply global attention block.

        Args:
            x: Node features [num_nodes, hidden_dim]
            batch: Batch assignment [num_nodes]

        Returns:
            Updated node features [num_nodes, hidden_dim]
        """
        if self.use_checkpoint and self.training:
            # Gradient checkpointing: recompute forward during backward to save memory
            # Requires use_reentrant=False for nested checkpointing compatibility
            x = checkpoint(self._forward_attn, x, batch, use_reentrant=False)
            x = checkpoint(self._forward_ffn, x, use_reentrant=False)
        else:
            x = self._forward_attn(x, batch)
            x = self._forward_ffn(x)

        return x
