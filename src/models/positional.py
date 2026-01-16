"""
Positional encoding implementations for transformer models.
"""

import torch
import torch.nn as nn
import math
from src.constants import D_MODEL, MAX_SEQ_LEN


class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding for Transformer."""

    def __init__(self, d_model: int = D_MODEL, max_len: int = MAX_SEQ_LEN, dropout: float = 0.1):
        """
        Initialize sinusoidal positional encoding.

        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Precompute PE matrix [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.

        Args:
            x: [batch, seq_len, d_model]

        Returns:
            [batch, seq_len, d_model] with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    """Learned positional embeddings."""

    def __init__(self, d_model: int = D_MODEL, max_len: int = MAX_SEQ_LEN, dropout: float = 0.1):
        """
        Initialize learned positional encoding.

        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(p=dropout)

        nn.init.normal_(self.pe.weight, mean=0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add learned positional encoding to input.

        Args:
            x: [batch, seq_len, d_model]

        Returns:
            [batch, seq_len, d_model] with positional encoding added
        """
        batch_size, seq_len, d_model = x.shape
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.pe(positions)
        x = x + pos_emb
        return self.dropout(x)
