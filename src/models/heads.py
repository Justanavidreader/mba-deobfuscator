"""
Output heads for MBA deobfuscation model.
"""

import torch
import torch.nn as nn
from typing import Tuple
from src.constants import VOCAB_SIZE, D_MODEL, MAX_OUTPUT_LENGTH, MAX_OUTPUT_DEPTH


class TokenHead(nn.Module):
    """Vocabulary prediction head."""

    def __init__(self, d_model: int = D_MODEL, vocab_size: int = VOCAB_SIZE):
        """
        Initialize token prediction head.

        Args:
            d_model: Model dimension
            vocab_size: Vocabulary size
        """
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict token probabilities.

        Args:
            x: [batch, seq_len, d_model]

        Returns:
            [batch, seq_len, vocab_size] logits
        """
        return self.proj(x)


class ComplexityHead(nn.Module):
    """Predict output length and AST depth for reranking."""

    def __init__(self, d_model: int = D_MODEL, max_length: int = MAX_OUTPUT_LENGTH,
                 max_depth: int = MAX_OUTPUT_DEPTH):
        """
        Initialize complexity prediction head.

        Args:
            d_model: Model dimension
            max_length: Maximum output sequence length
            max_depth: Maximum AST depth
        """
        super().__init__()
        self.length_head = nn.Linear(d_model, max_length)
        self.depth_head = nn.Linear(d_model, max_depth)

        nn.init.xavier_uniform_(self.length_head.weight)
        nn.init.zeros_(self.length_head.bias)
        nn.init.xavier_uniform_(self.depth_head.weight)
        nn.init.zeros_(self.depth_head.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict length and depth distributions.

        Args:
            x: [batch, d_model] - typically final token or [CLS] token embedding

        Returns:
            length_logits: [batch, max_length]
            depth_logits: [batch, max_depth]
        """
        length_logits = self.length_head(x)
        depth_logits = self.depth_head(x)
        return length_logits, depth_logits


class ValueHead(nn.Module):
    """Critic head for HTPS - estimates P(simplifiable)."""

    def __init__(self, d_model: int = D_MODEL):
        """
        Initialize value head for RL critic.

        Args:
            d_model: Model dimension
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )

        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Estimate value (probability of successful simplification).

        Args:
            x: [batch, d_model] graph-level representation

        Returns:
            [batch, 1] value estimate in [0, 1]
        """
        return self.mlp(x)
