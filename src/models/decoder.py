"""
Transformer decoder with copy mechanism for MBA deobfuscation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from src.constants import D_MODEL, NUM_DECODER_LAYERS, NUM_DECODER_HEADS, D_FF, DECODER_DROPOUT, VOCAB_SIZE, MAX_SEQ_LEN
from src.models.positional import SinusoidalPositionalEncoding


class TransformerDecoderLayer(nn.Module):
    """Single decoder layer with causal self-attention and cross-attention."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float):
        """
        Initialize transformer decoder layer.

        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Feed-forward dimension
            dropout: Dropout probability
        """
        super().__init__()

        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for layer in self.feed_forward:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor, memory: torch.Tensor,
                tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through decoder layer.

        Args:
            x: [batch, tgt_len, d_model] target sequence
            memory: [batch, src_len, d_model] encoder output
            tgt_mask: [tgt_len, tgt_len] causal mask
            memory_mask: [batch, src_len] memory mask (optional)

        Returns:
            output: [batch, tgt_len, d_model]
            cross_attn_weights: [batch, tgt_len, src_len]
        """
        x_norm = self.norm1(x)
        self_attn_out, _ = self.self_attn(x_norm, x_norm, x_norm, attn_mask=tgt_mask)
        x = x + self.dropout1(self_attn_out)

        x_norm = self.norm2(x)
        cross_attn_out, cross_attn_weights = self.cross_attn(
            x_norm, memory, memory, key_padding_mask=memory_mask
        )
        x = x + self.dropout2(cross_attn_out)

        x_norm = self.norm3(x)
        ff_out = self.feed_forward(x_norm)
        x = x + self.dropout3(ff_out)

        return x, cross_attn_weights


class TransformerDecoderWithCopy(nn.Module):
    """
    Transformer decoder with pointer-generator copy mechanism.
    """

    def __init__(self, vocab_size: int = VOCAB_SIZE, d_model: int = D_MODEL,
                 num_layers: int = NUM_DECODER_LAYERS, num_heads: int = NUM_DECODER_HEADS,
                 d_ff: int = D_FF, dropout: float = DECODER_DROPOUT,
                 max_seq_len: int = MAX_SEQ_LEN):
        """
        Initialize transformer decoder with copy mechanism.

        Args:
            vocab_size: Vocabulary size
            d_model: Model dimension
            num_layers: Number of decoder layers
            num_heads: Number of attention heads
            d_ff: Feed-forward dimension
            dropout: Dropout probability
            max_seq_len: Maximum sequence length (default 64, scaled model uses 2048)
        """
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_len=max_seq_len, dropout=dropout)

        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.copy_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        nn.init.normal_(self.token_embedding.weight, mean=0, std=0.02)

        for layer in self.copy_gate:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor,
                tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through decoder.

        Args:
            tgt: [batch, tgt_len] target token IDs
            memory: [batch, src_len, d_model] encoder output
            tgt_mask: [tgt_len, tgt_len] causal mask
            memory_mask: [batch, src_len] memory padding mask

        Returns:
            vocab_logits: [batch, tgt_len, vocab_size] vocabulary distribution
            copy_attn: [batch, tgt_len, src_len] attention weights for copying
            p_gen: [batch, tgt_len, 1] generation probability
        """
        batch_size, tgt_len = tgt.shape

        # Scale embeddings by sqrt(d_model) - use scalar to avoid device issues
        x = self.token_embedding(tgt) * (self.d_model ** 0.5)
        x = self.pos_encoding(x)

        if tgt_mask is None:
            tgt_mask = self.generate_square_subsequent_mask(tgt_len).to(tgt.device)

        cross_attn_weights = None
        for layer in self.layers:
            x, cross_attn_weights = layer(x, memory, tgt_mask, memory_mask)

        context = torch.bmm(cross_attn_weights, memory)

        combined = torch.cat([x, context], dim=-1)
        p_gen = self.copy_gate(combined)

        return x, cross_attn_weights, p_gen

    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """
        Create causal mask for self-attention.

        Args:
            sz: Sequence length

        Returns:
            [sz, sz] mask with -inf for future positions
        """
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
