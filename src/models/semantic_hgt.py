"""
Semantic HGT Encoder with algebraic invariant detection.

Extends standard HGT with:
1. Property detection heads for algebraic invariants
2. Spectral feature injection from Walsh-Hadamard analysis
3. Property-aware graph readout

Based on research:
- Grokking: Networks discover Fourier representations for modular arithmetic
- SiMBA: Linear MBA equivalence reduces to 2^t corner case verification
- gMBA: Truth table guidance achieves ~90% accuracy
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
from torch_scatter import scatter_mean, scatter_max

from src.constants import (
    FINGERPRINT_DIM,
    NUM_OPTIMIZED_EDGE_TYPES,
    SEMANTIC_HGT_WALSH_OUTPUT_DIM,
    NUM_VAR_PROPERTIES,
)
from src.models.encoder import HGTEncoder
from src.models.property_detector import InvariantDetector


class SemanticHGTEncoder(HGTEncoder):
    """
    HGT encoder augmented with semantic property detection.

    Inherits all HGT functionality and adds:
    - InvariantDetector for property prediction
    - Property-aware readout that weights by detected properties
    - Auxiliary outputs for property supervision

    Property injection occurs at a specified layer (default layer 8 of 12),
    allowing earlier layers to learn local structure and later layers to
    integrate property information before final readout.

    Usage:
        encoder = SemanticHGTEncoder(hidden_dim=768)
        output = encoder(x, edge_index, edge_type, batch, fingerprint=fp)
        # output['embeddings']: node embeddings
        # output['var_properties']: for auxiliary loss
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 16,
        dropout: float = 0.1,
        num_node_types: int = 10,
        num_edge_types: int = NUM_OPTIMIZED_EDGE_TYPES,
        fingerprint_dim: int = FINGERPRINT_DIM,
        # Property detection settings
        enable_property_detection: bool = True,
        property_injection_layer: int = 8,  # Inject after this layer
        **kwargs,
    ):
        """
        Initialize Semantic HGT encoder.

        Args:
            hidden_dim: Hidden dimension (default 768 for scaled model)
            num_layers: Number of HGT layers (default 12)
            num_heads: Attention heads (default 16)
            dropout: Dropout probability
            num_node_types: Number of node types (10 for MBA)
            num_edge_types: Number of edge types (8 for optimized)
            fingerprint_dim: Fingerprint dimension (448)
            enable_property_detection: Enable invariant detection
            property_injection_layer: Layer to inject property features
                Rationale: Early layers learn local structure, middle layers
                learn patterns, late layers integrate for final representation.
                Injecting at layer 8/12 allows 4 layers to integrate property
                information before readout.
            **kwargs: Additional args for HGTEncoder
        """
        # Validate property injection layer bounds (RULE 0 HIGH)
        if property_injection_layer < 0 or property_injection_layer >= num_layers:
            raise ValueError(
                f"property_injection_layer must be in [0, {num_layers-1}], "
                f"got {property_injection_layer}. Set to num_layers-4 or similar "
                f"to inject properties near the end of encoding."
            )

        # Explicitly set edge_type_mode to "optimized" (RULE 0 issue #5)
        super().__init__(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            num_node_types=num_node_types,
            num_edge_types=num_edge_types,
            edge_type_mode="optimized",  # Explicit mode for HGT
            **kwargs,
        )

        self.enable_property_detection = enable_property_detection
        self.property_injection_layer = property_injection_layer
        self.fingerprint_dim = fingerprint_dim

        if enable_property_detection:
            self.invariant_detector = InvariantDetector(
                hidden_dim=hidden_dim,
                fingerprint_dim=fingerprint_dim,
                walsh_output_dim=SEMANTIC_HGT_WALSH_OUTPUT_DIM,
            )

            # Property-conditioned layer norm for post-injection
            self.property_layer_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        batch: torch.Tensor,
        fingerprint: Optional[torch.Tensor] = None,
        return_property_outputs: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with optional property detection.

        Args:
            x: [num_nodes] node type IDs
            edge_index: [2, num_edges] edge indices
            edge_type: [num_edges] edge type IDs
            batch: [num_nodes] batch assignment
            fingerprint: [batch_size, 448] semantic fingerprints (optional)
            return_property_outputs: Return property predictions for aux loss

        Returns:
            Dict with:
                - 'embeddings': [num_nodes, hidden_dim] node embeddings
                - 'var_properties': property predictions (if enabled)
                - 'interactions': interaction predictions (if enabled)
                - 'walsh_features': Walsh features (if enabled)
                - 'walsh_raw': Raw Walsh statistics (if enabled)
        """
        # Store original node types for property detection
        original_node_types = x.clone()

        # Convert to heterogeneous format
        x_dict, edge_index_dict = self._to_heterogeneous(x, edge_index, edge_type)

        # Compute path encoding if enabled (from parent class)
        path_edge_emb = None
        if self.use_path_encoding and self.path_encoder is not None:
            path_edge_emb = self.path_encoder(
                edge_index, edge_type, x, batch
            )

        # Process through HGT layers with property injection
        global_block_idx = 0
        path_inject_idx = 0

        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            # Standard HGT layer
            x_dict_out = conv(x_dict, edge_index_dict)

            # Residual + norm for each node type
            for ntype in x_dict:
                if ntype in x_dict_out:
                    x_dict[ntype] = norm(x_dict[ntype] + x_dict_out[ntype])
                    x_dict[ntype] = self.dropout(x_dict[ntype])

            # Property injection at specified layer
            if (self.enable_property_detection and
                fingerprint is not None and
                i == self.property_injection_layer):

                # Convert to flat tensor for property detection
                h_flat = self._dict_to_flat(x_dict, original_node_types)

                # Detect properties and augment
                prop_output = self.invariant_detector(
                    h_flat, original_node_types, fingerprint, batch
                )

                # Update embeddings with augmented versions
                h_augmented = prop_output['augmented_embeddings']
                h_augmented = self.property_layer_norm(h_augmented)

                # Convert back to heterogeneous dict
                x_dict = self._flat_to_dict(h_augmented, original_node_types)

                # Store for output
                if return_property_outputs:
                    self._last_prop_output = prop_output

            # Global attention (from parent class)
            if (self.use_global_attention and
                self.global_attn_blocks is not None and
                (i + 1) % self.global_attn_interval == 0):

                h_flat = self._dict_to_flat(x_dict, original_node_types)
                h_flat = self.global_attn_blocks[global_block_idx](h_flat, batch)
                x_dict = self._flat_to_dict(h_flat, original_node_types)
                global_block_idx += 1

            # Path injection (from parent class)
            if (self.use_path_encoding and
                path_edge_emb is not None and
                self.path_projectors is not None and
                (i + 1) % self.path_injection_interval == 0 and
                path_inject_idx < len(self.path_projectors)):

                x_dict = self._inject_path_context(
                    x_dict, path_edge_emb, edge_index,
                    original_node_types, path_inject_idx
                )
                path_inject_idx += 1

        # Convert final output to flat tensor
        h_final = self._dict_to_flat(x_dict, original_node_types)

        # Build result
        result = {'embeddings': h_final}

        if return_property_outputs and hasattr(self, '_last_prop_output'):
            result['var_properties'] = self._last_prop_output.get('var_properties', [])
            result['interactions'] = self._last_prop_output.get('interactions', [])
            result['walsh_features'] = self._last_prop_output.get('walsh_features')
            result['walsh_raw'] = self._last_prop_output.get('walsh_raw')

        return result

    def _dict_to_flat(
        self,
        x_dict: Dict[str, torch.Tensor],
        node_types: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert heterogeneous dict to flat tensor.

        Args:
            x_dict: Dict mapping node type strings to embeddings
            node_types: [num_nodes] node type IDs

        Returns:
            [num_nodes, hidden_dim] flat tensor
        """
        num_nodes = node_types.size(0)
        device = node_types.device

        h_flat = torch.zeros(num_nodes, self.hidden_dim, device=device)
        for ntype_str in x_dict:
            ntype_int = int(ntype_str)
            mask = (node_types == ntype_int)
            h_flat[mask] = x_dict[ntype_str]

        return h_flat

    def _flat_to_dict(
        self,
        h_flat: torch.Tensor,
        node_types: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Convert flat tensor to heterogeneous dict.

        Args:
            h_flat: [num_nodes, hidden_dim] flat embeddings
            node_types: [num_nodes] node type IDs

        Returns:
            Dict mapping node type strings to embeddings
        """
        x_dict = {}
        for ntype in range(self.num_node_types):
            mask = (node_types == ntype)
            if mask.any():
                x_dict[str(ntype)] = h_flat[mask]
        return x_dict


class PropertyAwareReadout(nn.Module):
    """
    Graph readout that uses detected properties to weight contributions.

    Variables with LINEAR property get higher weight for simple expressions.
    Variables with CONST_CONTRIB get lower weight (they cancel out).
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Property-to-weight network
        self.weight_net = nn.Sequential(
            nn.Linear(NUM_VAR_PROPERTIES, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid(),
        )

        # Standard readout components
        self.pre_readout = nn.Linear(hidden_dim, hidden_dim)
        self.post_readout = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(
        self,
        node_embeddings: torch.Tensor,
        batch: torch.Tensor,
        var_property_probs: Optional[torch.Tensor] = None,
        node_types: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Property-aware graph readout.

        Args:
            node_embeddings: [num_nodes, hidden_dim]
            batch: [num_nodes] batch assignment
            var_property_probs: [num_vars, num_properties] (optional)
            node_types: [num_nodes] for identifying variables

        Returns:
            [batch_size, hidden_dim] graph-level embeddings
        """
        h = self.pre_readout(node_embeddings)

        # If property weights available, apply them to variable nodes
        if var_property_probs is not None and node_types is not None:
            var_mask = (node_types == 0)
            if var_mask.sum() > 0:
                # Compute importance weights from properties
                weights = self.weight_net(var_property_probs)  # [num_vars, 1]
                # Apply weights to variable embeddings
                h_weighted = h.clone()
                h_weighted[var_mask] = h[var_mask] * weights
                h = h_weighted

        # Standard mean + max pooling
        h_mean = scatter_mean(h, batch, dim=0)
        h_max, _ = scatter_max(h, batch, dim=0)

        # Combine
        h_combined = torch.cat([h_mean, h_max], dim=-1)
        return self.post_readout(h_combined)
