"""
Full MBA deobfuscation model combining encoder, decoder, and output heads.

Supports both standard (~15M) and scaled (~360M) configurations:
- Standard: GAT/GGNN encoder, 6-layer decoder
- Scaled: HGT/RGCN encoder, 8-layer decoder, boolean domain conditioning
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
from src.constants import (
    HIDDEN_DIM, D_MODEL, FINGERPRINT_DIM,
    SCALED_HIDDEN_DIM, SCALED_D_MODEL, SCALED_D_FF,
    SCALED_NUM_DECODER_LAYERS, SCALED_NUM_DECODER_HEADS, SCALED_MAX_SEQ_LEN,
    NUM_NODE_TYPES_HETEROGENEOUS, NUM_OPTIMIZED_EDGE_TYPES
)
from src.models.encoder import (
    GATJKNetEncoder, GGNNEncoder, GraphReadout, FingerprintEncoder,
    HGTEncoder, RGCNEncoder, ScaledGraphReadout
)
from src.models.decoder import TransformerDecoderWithCopy
from src.models.heads import TokenHead, ComplexityHead, ValueHead


class MBADeobfuscator(nn.Module):
    """
    Full MBA deobfuscation model.
    Encoder (GNN) -> Fingerprint fusion -> Decoder (Transformer) -> Output heads
    """

    def __init__(self, encoder_type: str = 'gat', **kwargs):
        """
        Initialize MBA deobfuscation model.

        Args:
            encoder_type: 'gat', 'ggnn', 'hgt', or 'rgcn'
            **kwargs: Additional arguments for encoder/decoder components
        """
        super().__init__()
        self.encoder_type = encoder_type
        self.is_scaled = encoder_type in ('hgt', 'rgcn')

        # Determine dimensions based on model type
        hidden_dim = kwargs.get('hidden_dim', SCALED_HIDDEN_DIM if self.is_scaled else HIDDEN_DIM)
        d_model = kwargs.get('d_model', SCALED_D_MODEL if self.is_scaled else D_MODEL)

        if encoder_type == 'gat':
            self.graph_encoder = GATJKNetEncoder(**{k: v for k, v in kwargs.items()
                                                     if k in ['node_dim', 'hidden_dim', 'num_layers',
                                                              'num_heads', 'dropout']})
            self.graph_readout = GraphReadout(hidden_dim=hidden_dim)
        elif encoder_type == 'ggnn':
            self.graph_encoder = GGNNEncoder(**{k: v for k, v in kwargs.items()
                                                 if k in ['node_dim', 'hidden_dim', 'num_timesteps',
                                                          'num_edge_types']})
            self.graph_readout = GraphReadout(hidden_dim=hidden_dim)
        elif encoder_type == 'hgt':
            self.graph_encoder = HGTEncoder(
                hidden_dim=hidden_dim,
                num_layers=kwargs.get('num_encoder_layers', 12),
                num_heads=kwargs.get('num_encoder_heads', 16),
                num_node_types=kwargs.get('num_node_types', NUM_NODE_TYPES_HETEROGENEOUS),
                num_edge_types=kwargs.get('num_edge_types', NUM_OPTIMIZED_EDGE_TYPES),
                dropout=kwargs.get('encoder_dropout', 0.1)
            )
            self.graph_readout = ScaledGraphReadout(hidden_dim=hidden_dim)
        elif encoder_type == 'rgcn':
            self.graph_encoder = RGCNEncoder(
                hidden_dim=hidden_dim,
                num_layers=kwargs.get('num_encoder_layers', 12),
                num_relations=kwargs.get('num_edge_types', NUM_OPTIMIZED_EDGE_TYPES),
                num_node_types=kwargs.get('num_node_types', NUM_NODE_TYPES_HETEROGENEOUS),
                dropout=kwargs.get('encoder_dropout', 0.1)
            )
            self.graph_readout = ScaledGraphReadout(hidden_dim=hidden_dim)
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")

        self.fingerprint_encoder = FingerprintEncoder(
            hidden_dim=hidden_dim  # Match encoder hidden dim
        )

        # Project embeddings to d_model before fusion (handles dimension mismatch)
        self.graph_proj = nn.Linear(hidden_dim, d_model)
        self.fp_proj = nn.Linear(hidden_dim, d_model)

        # Boolean domain conditioning for scaled model
        if self.is_scaled:
            self.boolean_domain_embed = nn.Embedding(2, d_model)
            fusion_input_dim = d_model * 3  # graph + fingerprint + boolean_domain
        else:
            self.boolean_domain_embed = None
            fusion_input_dim = d_model * 2  # graph + fingerprint

        self.fusion_projection = nn.Sequential(
            nn.Linear(fusion_input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU()
        )

        # Decoder with scaled params if needed
        if self.is_scaled:
            self.decoder = TransformerDecoderWithCopy(
                vocab_size=kwargs.get('vocab_size', 300),
                d_model=d_model,
                num_layers=kwargs.get('num_decoder_layers', SCALED_NUM_DECODER_LAYERS),
                num_heads=kwargs.get('num_decoder_heads', SCALED_NUM_DECODER_HEADS),
                d_ff=kwargs.get('d_ff', SCALED_D_FF),
                dropout=kwargs.get('decoder_dropout', 0.1),
                max_seq_len=kwargs.get('max_seq_len', SCALED_MAX_SEQ_LEN)
            )
        else:
            self.decoder = TransformerDecoderWithCopy(**{k: v for k, v in kwargs.items()
                                                          if k in ['vocab_size', 'd_model', 'num_layers',
                                                                   'num_heads', 'd_ff', 'dropout', 'max_seq_len']})

        self.token_head = TokenHead(d_model=d_model)
        self.complexity_head = ComplexityHead(d_model=d_model)
        self.value_head = ValueHead(d_model=d_model)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for fusion projection and graph projection."""
        for layer in self.fusion_projection:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

        nn.init.xavier_uniform_(self.graph_proj.weight)
        nn.init.zeros_(self.graph_proj.bias)
        nn.init.xavier_uniform_(self.fp_proj.weight)
        nn.init.zeros_(self.fp_proj.bias)

        if self.boolean_domain_embed is not None:
            nn.init.normal_(self.boolean_domain_embed.weight, mean=0, std=0.02)

    def encode(self, graph_batch, fingerprint: torch.Tensor,
               boolean_domain_only: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode input expression.

        Args:
            graph_batch: PyG batch with x, edge_index, batch attributes
                        (and edge_type for GGNN/HGT/RGCN)
            fingerprint: [batch, FINGERPRINT_DIM] semantic fingerprint
            boolean_domain_only: [batch] bool tensor for domain conditioning (scaled model)

        Returns:
            [batch, 1, D_MODEL] context for decoder
        """
        x = graph_batch.x
        edge_index = graph_batch.edge_index
        batch = graph_batch.batch

        # All encoder types except GAT need edge_type
        if self.encoder_type in ('ggnn', 'hgt', 'rgcn'):
            edge_type = graph_batch.edge_type
            node_embeddings = self.graph_encoder(x, edge_index, edge_type, batch)
        else:
            node_embeddings = self.graph_encoder(x, edge_index, batch)

        graph_embedding = self.graph_readout(node_embeddings, batch)

        # Project graph embedding to d_model
        graph_projected = self.graph_proj(graph_embedding)

        # Fingerprint encoder outputs hidden_dim, project to d_model
        fp_embedding = self.fingerprint_encoder(fingerprint)
        fp_projected = self.fp_proj(fp_embedding)

        # Fuse embeddings
        if self.is_scaled and self.boolean_domain_embed is not None:
            # Default to False (mixed domain) if not provided
            if boolean_domain_only is None:
                boolean_domain_only = torch.zeros(
                    graph_projected.shape[0], dtype=torch.long, device=graph_projected.device
                )
            domain_embed = self.boolean_domain_embed(boolean_domain_only.long())
            fused = torch.cat([graph_projected, fp_projected, domain_embed], dim=-1)
        else:
            fused = torch.cat([graph_projected, fp_projected], dim=-1)

        context = self.fusion_projection(fused)
        context = context.unsqueeze(1)

        return context

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor,
               tgt_mask: Optional[torch.Tensor] = None,
               memory_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Decode step.

        Args:
            tgt: [batch, tgt_len] target token IDs
            memory: [batch, src_len, D_MODEL] encoder context
            tgt_mask: [tgt_len, tgt_len] causal mask
            memory_mask: [batch, src_len] memory mask

        Returns:
            Dict with vocab_logits, copy_attn, p_gen, decoder_out
        """
        decoder_out, copy_attn, p_gen = self.decoder(tgt, memory, tgt_mask, memory_mask)

        vocab_logits = self.token_head(decoder_out)

        return {
            'vocab_logits': vocab_logits,
            'copy_attn': copy_attn,
            'p_gen': p_gen,
            'decoder_out': decoder_out  # Include decoder output for ComplexityHead
        }

    def forward(self, graph_batch, fingerprint: torch.Tensor,
                tgt: torch.Tensor,
                boolean_domain_only: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Full forward pass for training.

        Args:
            graph_batch: PyG batch of input graphs
            fingerprint: [batch, FINGERPRINT_DIM] semantic fingerprints
            tgt: [batch, tgt_len] target token IDs
            boolean_domain_only: [batch] bool tensor for domain conditioning (scaled model)

        Returns:
            Dict with: vocab_logits, copy_attn, p_gen, length_pred, depth_pred, value
        """
        memory = self.encode(graph_batch, fingerprint, boolean_domain_only)

        decode_output = self.decode(tgt, memory)

        # Use final token from decoder output for complexity prediction
        # ComplexityHead predicts OUTPUT properties, not INPUT properties
        decoder_final = decode_output['decoder_out'][:, -1, :]  # [batch, d_model]
        length_pred, depth_pred = self.complexity_head(decoder_final)

        # ValueHead uses encoder output (graph embedding) to estimate P(simplifiable)
        encoder_embedding = memory.squeeze(1)  # [batch, d_model]
        value = self.value_head(encoder_embedding)

        return {
            'vocab_logits': decode_output['vocab_logits'],
            'copy_attn': decode_output['copy_attn'],
            'p_gen': decode_output['p_gen'],
            'length_pred': length_pred,
            'depth_pred': depth_pred,
            'value': value
        }

    def get_value(self, graph_batch, fingerprint: torch.Tensor,
                  boolean_domain_only: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get critic value for HTPS.

        Args:
            graph_batch: PyG batch of input graphs
            fingerprint: [batch, FINGERPRINT_DIM] semantic fingerprints
            boolean_domain_only: [batch] bool tensor for domain conditioning (scaled model)

        Returns:
            [batch, 1] value estimate
        """
        memory = self.encode(graph_batch, fingerprint, boolean_domain_only)
        final_embedding = memory.squeeze(1)
        return self.value_head(final_embedding)


class ScaledMBADeobfuscator(MBADeobfuscator):
    """
    Convenience class for 360M parameter model with HGT encoder.

    Uses Chinchilla-optimal configuration for 12M samples (7.2B tokens).
    """

    def __init__(self, encoder_type: str = 'hgt', **kwargs):
        """
        Initialize scaled model with defaults optimized for 360M params.

        Args:
            encoder_type: 'hgt' (default) or 'rgcn'
            **kwargs: Override any default parameters
        """
        defaults = {
            'hidden_dim': SCALED_HIDDEN_DIM,
            'd_model': SCALED_D_MODEL,
            'd_ff': SCALED_D_FF,
            'num_encoder_layers': 12,
            'num_encoder_heads': 16,
            'num_decoder_layers': SCALED_NUM_DECODER_LAYERS,
            'num_decoder_heads': SCALED_NUM_DECODER_HEADS,
            'max_seq_len': SCALED_MAX_SEQ_LEN,
            'encoder_dropout': 0.1,
            'decoder_dropout': 0.1,
            'num_node_types': NUM_NODE_TYPES_HETEROGENEOUS,
            'num_edge_types': NUM_OPTIMIZED_EDGE_TYPES,
        }
        # Allow kwargs to override defaults
        defaults.update(kwargs)
        super().__init__(encoder_type=encoder_type, **defaults)
