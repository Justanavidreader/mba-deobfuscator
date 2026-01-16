"""
Cross-graph attention modules for Graph Matching Networks.

Enables nodes in one graph to attend to nodes in another graph,
capturing variable correspondence and cancellation patterns in MBA expressions.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossGraphAttention(nn.Module):
    """
    Cross-graph attention for node correspondence detection.

    Given embeddings from two graphs, computes attention scores and
    matching vectors that capture how nodes in graph 1 correspond to
    nodes in graph 2.

    For MBA: Detects variable correspondence, identifies cancellations,
    and recognizes structural equivalences.
    """

    def __init__(self, hidden_dim: int = 768, dropout: float = 0.1):
        """
        Args:
            hidden_dim: Node embedding dimension (must match encoder output)
            dropout: Dropout probability for attention weights
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.scale = hidden_dim ** -0.5

        # Linear projections for query, key, value
        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)

        self.attn_dropout = nn.Dropout(dropout)

        # Projection for matching vector
        self.match_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for module in [self.W_q, self.W_k, self.W_v]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.match_projection[0].weight)
        nn.init.zeros_(self.match_projection[0].bias)

    def forward(
        self,
        h1: torch.Tensor,
        h2: torch.Tensor,
        mask1: Optional[torch.Tensor] = None,
        mask2: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute cross-graph attention from h1 to h2.

        Args:
            h1: [N1, hidden_dim] node embeddings from graph 1
            h2: [N2, hidden_dim] node embeddings from graph 2
            mask1: [N1] boolean mask for valid nodes in graph 1 (unused, for API consistency)
            mask2: [N2] or [N1, N2] boolean mask for valid attention targets
                   - [N2]: simple padding mask
                   - [N1, N2]: full attention mask (e.g., for batch separation)

        Returns:
            matching_vector: [N1, hidden_dim] matching representation for graph 1
            attention_weights: [N1, N2] attention weights (for visualization)
        """
        # Project to query, key, value
        Q = self.W_q(h1)  # [N1, hidden_dim]
        K = self.W_k(h2)  # [N2, hidden_dim]
        V = self.W_v(h2)  # [N2, hidden_dim]

        # Compute attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [N1, N2]

        # Apply mask (use -1e9 instead of -inf to avoid NaN)
        if mask2 is not None:
            if mask2.dim() == 1:
                # [N2] -> broadcast to [N1, N2]
                attn_scores = attn_scores.masked_fill(~mask2.unsqueeze(0), -1e9)
            else:
                # [N1, N2] full mask
                attn_scores = attn_scores.masked_fill(~mask2, -1e9)

        # Softmax over graph 2 nodes
        attn_weights = F.softmax(attn_scores, dim=-1)  # [N1, N2]

        # Handle all-masked rows (replace NaN with uniform distribution)
        nan_mask = torch.isnan(attn_weights)
        if nan_mask.any():
            uniform_weight = 1.0 / max(attn_weights.size(-1), 1)
            attn_weights = torch.where(
                nan_mask,
                torch.full_like(attn_weights, uniform_weight),
                attn_weights
            )

        attn_weights = self.attn_dropout(attn_weights)

        # Compute attended representation
        h2_attended = torch.matmul(attn_weights, V)  # [N1, hidden_dim]

        # Matching vector: captures difference between h1 and what it attends to
        # If h1[i] strongly attends to h2[j] and they're similar, matching vector is small
        # If h1[i] has no good match, matching vector is large (node unique to g1)
        matching_input = torch.cat([h1, h2_attended], dim=-1)  # [N1, hidden_dim*2]
        matching_vector = self.match_projection(matching_input)  # [N1, hidden_dim]

        return matching_vector, attn_weights


class MultiHeadCrossGraphAttention(nn.Module):
    """
    Multi-head cross-graph attention for diverse matching patterns.

    Different heads can specialize:
      - Head 1: Variable correspondence (x ↔ x)
      - Head 2: Constant matching (5 ↔ 5)
      - Head 3: Structural patterns (AND-subtree ↔ simplified node)
      - Head 4: Cancellation detection (no correspondence)
    """

    def __init__(self, hidden_dim: int = 768, num_heads: int = 8, dropout: float = 0.1):
        """
        Args:
            hidden_dim: Must be divisible by num_heads
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})")

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Projections (shared across heads, split in forward)
        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)

        self.attn_dropout = nn.Dropout(dropout)

        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # Matching projection
        self.match_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for module in [self.W_q, self.W_k, self.W_v, self.out_proj]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.match_projection[0].weight)
        nn.init.zeros_(self.match_projection[0].bias)

    def forward(
        self,
        h1: torch.Tensor,
        h2: torch.Tensor,
        mask1: Optional[torch.Tensor] = None,
        mask2: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Multi-head cross-attention.

        Args:
            h1: [N1, hidden_dim]
            h2: [N2, hidden_dim]
            mask1: [N1] boolean mask (unused, for API consistency)
            mask2: [N2] or [N1, N2] boolean mask for valid attention targets

        Returns:
            matching_vector: [N1, hidden_dim]
            attention_weights: [num_heads, N1, N2] (for visualization)
        """
        N1, N2 = h1.size(0), h2.size(0)

        # Project and reshape for multi-head
        # [N, hidden_dim] -> [N, num_heads, head_dim] -> [num_heads, N, head_dim]
        Q = self.W_q(h1).view(N1, self.num_heads, self.head_dim).transpose(0, 1)
        K = self.W_k(h2).view(N2, self.num_heads, self.head_dim).transpose(0, 1)
        V = self.W_v(h2).view(N2, self.num_heads, self.head_dim).transpose(0, 1)

        # Compute attention: [num_heads, N1, head_dim] @ [num_heads, head_dim, N2]
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [num_heads, N1, N2]

        # Apply mask (use -1e9 instead of -inf to avoid NaN)
        if mask2 is not None:
            if mask2.dim() == 1:
                # [N2] -> [1, 1, N2] -> broadcast to [num_heads, N1, N2]
                attn_scores = attn_scores.masked_fill(~mask2.unsqueeze(0).unsqueeze(1), -1e9)
            else:
                # [N1, N2] -> [1, N1, N2] -> broadcast to [num_heads, N1, N2]
                attn_scores = attn_scores.masked_fill(~mask2.unsqueeze(0), -1e9)

        attn_weights = F.softmax(attn_scores, dim=-1)  # [num_heads, N1, N2]

        # Handle all-masked rows (replace NaN with uniform distribution)
        nan_mask = torch.isnan(attn_weights)
        if nan_mask.any():
            uniform_weight = 1.0 / max(attn_weights.size(-1), 1)
            attn_weights = torch.where(
                nan_mask,
                torch.full_like(attn_weights, uniform_weight),
                attn_weights
            )

        attn_weights = self.attn_dropout(attn_weights)

        # Aggregate: [num_heads, N1, N2] @ [num_heads, N2, head_dim]
        h2_attended = torch.matmul(attn_weights, V)  # [num_heads, N1, head_dim]

        # Concatenate heads: [num_heads, N1, head_dim] -> [N1, num_heads * head_dim]
        h2_attended = h2_attended.transpose(0, 1).contiguous().view(N1, self.hidden_dim)
        h2_attended = self.out_proj(h2_attended)  # [N1, hidden_dim]

        # Matching vector
        matching_input = torch.cat([h1, h2_attended], dim=-1)
        matching_vector = self.match_projection(matching_input)

        return matching_vector, attn_weights
