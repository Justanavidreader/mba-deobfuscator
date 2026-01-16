"""
Algebraic property detection heads for Semantic HGT.

Detects variable-level and interaction-level properties that enable
algebraic identity recognition and improved generalization.

Based on research showing:
- Variable properties (linearity, domain) are 85-98% learnable
- Interaction properties enable identity detection across compositions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from src.constants import (
    NUM_VAR_PROPERTIES,
    NUM_INTERACTION_PROPERTIES,
    SEMANTIC_HGT_PROPERTY_DIM,
    SEMANTIC_HGT_INTERACTION_HEADS,
    FINGERPRINT_MODE,
    FINGERPRINT_DIM,
)
from src.data.walsh_hadamard import WalshSpectrumEncoder


class VariablePropertyHead(nn.Module):
    """
    Multi-label classification head for variable-level properties.

    Given node embeddings, identifies which variables exhibit which
    algebraic properties (linear, boolean-only, complementary, etc.)

    Architecture:
        var_embeddings [num_vars, hidden_dim]
        -> MLP -> [num_vars, NUM_VAR_PROPERTIES] logits
        -> Sigmoid for multi-label prediction
    """

    def __init__(self, hidden_dim: int, num_properties: int = NUM_VAR_PROPERTIES):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_properties = num_properties

        # 2-layer MLP with residual
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_properties),
        )

        # Property embeddings (for augmenting variable representations)
        self.property_embed = nn.Embedding(num_properties, SEMANTIC_HGT_PROPERTY_DIM)

        self._init_weights()

    def _init_weights(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.normal_(self.property_embed.weight, std=0.02)

    def forward(
        self,
        var_embeddings: torch.Tensor,
        return_augmented: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Predict properties and optionally return augmented embeddings.

        Args:
            var_embeddings: [num_vars, hidden_dim] variable node embeddings
            return_augmented: If True, return property-augmented embeddings

        Returns:
            Dict with:
                - 'logits': [num_vars, num_properties] raw logits
                - 'probs': [num_vars, num_properties] sigmoid probabilities
                - 'augmented': [num_vars, hidden_dim + property_dim] (if return_augmented)
        """
        logits = self.mlp(var_embeddings)  # [num_vars, num_properties]
        probs = torch.sigmoid(logits)

        result = {'logits': logits, 'probs': probs}

        if return_augmented:
            # Weight property embeddings by predicted probabilities
            # [num_vars, num_properties] @ [num_properties, property_dim]
            property_context = probs @ self.property_embed.weight
            # Concatenate with original embeddings
            result['augmented'] = torch.cat([var_embeddings, property_context], dim=-1)

        return result


class InteractionPropertyHead(nn.Module):
    """
    Bilinear attention head for variable interaction properties.

    Given pairs of variable embeddings, predicts interaction properties
    like multiplicative coupling, XOR-like behavior, cancellation patterns.

    Architecture:
        (var_i, var_j) -> bilinear attention -> [num_interactions] logits
    """

    def __init__(
        self,
        hidden_dim: int,
        num_properties: int = NUM_INTERACTION_PROPERTIES,
        num_heads: int = SEMANTIC_HGT_INTERACTION_HEADS,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_properties = num_properties
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Multi-head bilinear attention
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)

        # Property classifier on concatenated attended features
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_properties),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.query_proj.weight)
        nn.init.xavier_uniform_(self.key_proj.weight)
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        var_embeddings: torch.Tensor,
        var_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Predict interaction properties for all variable pairs.

        Args:
            var_embeddings: [num_vars, hidden_dim]
            var_mask: [num_vars] boolean mask for valid variables

        Returns:
            Dict with:
                - 'logits': [num_vars, num_vars, num_properties] pairwise logits
                - 'probs': [num_vars, num_vars, num_properties] probabilities
                - 'attention': [num_vars, num_vars] attention weights
        """
        num_vars = var_embeddings.size(0)

        # Project to queries and keys
        Q = self.query_proj(var_embeddings)  # [num_vars, hidden_dim]
        K = self.key_proj(var_embeddings)    # [num_vars, hidden_dim]

        # Compute attention weights
        attn = torch.matmul(Q, K.transpose(-2, -1)) / (self.hidden_dim ** 0.5)

        if var_mask is not None:
            # Mask invalid variable pairs
            mask_2d = var_mask.unsqueeze(0) & var_mask.unsqueeze(1)
            attn = attn.masked_fill(~mask_2d, float('-inf'))

        attn_weights = F.softmax(attn, dim=-1)

        # For each pair (i, j), concatenate embeddings and classify
        # Efficient: use outer product structure
        var_i = var_embeddings.unsqueeze(1).expand(-1, num_vars, -1)  # [n, n, h]
        var_j = var_embeddings.unsqueeze(0).expand(num_vars, -1, -1)  # [n, n, h]
        pair_features = torch.cat([var_i, var_j], dim=-1)  # [n, n, 2h]

        logits = self.classifier(pair_features)  # [n, n, num_properties]
        probs = torch.sigmoid(logits)

        return {
            'logits': logits,
            'probs': probs,
            'attention': attn_weights,
        }


class InvariantDetector(nn.Module):
    """
    Combines variable and interaction properties with fingerprint features.

    This is the main interface for property detection, integrating:
    1. VariablePropertyHead for per-variable properties
    2. InteractionPropertyHead for pairwise interactions
    3. WalshSpectrumEncoder for explicit spectral features (grokking shortcut)
    4. Fingerprint context (corner cases, derivatives)

    Key insight: By providing Walsh coefficients explicitly, we give the model
    direct access to spectral features that would otherwise require extended
    training ("grokking") to discover.
    """

    def __init__(
        self,
        hidden_dim: int,
        fingerprint_dim: int = 448,
        walsh_output_dim: int = 64,
    ):
        super().__init__()

        # Validate fingerprint mode (RULE 0 CRITICAL)
        if FINGERPRINT_MODE != "full":
            raise ValueError(
                f"InvariantDetector requires FINGERPRINT_MODE='full', "
                f"got '{FINGERPRINT_MODE}'. Walsh-Hadamard features need "
                f"truth table AND non-truth-table components."
            )

        self.hidden_dim = hidden_dim
        self.fingerprint_dim = fingerprint_dim

        if fingerprint_dim != FINGERPRINT_DIM:
            raise ValueError(
                f"Fingerprint dimension mismatch: expected {FINGERPRINT_DIM}, "
                f"got {fingerprint_dim}. Check constants.py configuration."
            )

        # Property detection heads
        self.var_property_head = VariablePropertyHead(hidden_dim)
        self.interaction_head = InteractionPropertyHead(hidden_dim)

        # Walsh-Hadamard spectrum encoder (explicit Fourier features)
        # This is the key "grokking shortcut" - instead of waiting for the
        # network to discover Fourier representations, we provide them directly
        self.walsh_encoder = WalshSpectrumEncoder(
            input_dim=17,  # Base Walsh features
            hidden_dim=64,
            output_dim=walsh_output_dim,
            include_raw_spectrum=True,
            top_k=16,
        )

        # Fingerprint integration (non-truth-table parts)
        # Fingerprint structure: symbolic(32) + corners(256) + random(64) + derivatives(32) + truth_table(64)
        # We handle truth_table separately via Walsh encoder
        non_tt_dim = fingerprint_dim - 64  # = 384
        self.fingerprint_proj = nn.Linear(non_tt_dim, hidden_dim)

        # Final projection for augmented variable embeddings
        # Combines: original + property context + Walsh features
        augmented_dim = hidden_dim + SEMANTIC_HGT_PROPERTY_DIM + walsh_output_dim
        self.output_proj = nn.Linear(augmented_dim, hidden_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fingerprint_proj.weight)
        nn.init.zeros_(self.fingerprint_proj.bias)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        node_embeddings: torch.Tensor,
        node_types: torch.Tensor,
        fingerprint: torch.Tensor,
        batch: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Detect properties and create augmented embeddings.

        Args:
            node_embeddings: [total_nodes, hidden_dim] from HGT
            node_types: [total_nodes] node type IDs (0=VAR, 1=CONST, ...)
            fingerprint: [batch_size, 448] semantic fingerprints
            batch: [total_nodes] batch assignment

        Returns:
            Dict with:
                - 'augmented_embeddings': [total_nodes, hidden_dim]
                - 'var_properties': Dict from VariablePropertyHead
                - 'interactions': Dict from InteractionPropertyHead (per batch)
                - 'walsh_features': [batch_size, walsh_output_dim] Walsh embeddings
                - 'walsh_raw': Dict with linearity indicators and spectrum stats
        """
        device = node_embeddings.device
        batch_size = fingerprint.size(0)

        # Validate fingerprint dimensions at runtime (RULE 0 CRITICAL)
        if fingerprint.size(1) != self.fingerprint_dim:
            raise RuntimeError(
                f"Fingerprint dimension mismatch: expected {self.fingerprint_dim}, "
                f"got {fingerprint.size(1)}. Check FINGERPRINT_MODE setting."
            )

        # Extract truth table (last 64 dims) and compute Walsh features
        truth_table = fingerprint[:, -64:]  # [batch_size, 64]
        walsh_features = self.walsh_encoder(truth_table)  # [batch_size, walsh_output_dim]

        # Also get raw Walsh statistics for interpretability
        from src.data.walsh_hadamard import compute_walsh_features
        walsh_raw = compute_walsh_features(truth_table)  # [batch_size, 17]

        # Process non-truth-table fingerprint components
        non_tt_fingerprint = fingerprint[:, :-64]  # [batch_size, 384]
        fp_context = self.fingerprint_proj(non_tt_fingerprint)  # [batch_size, hidden_dim]

        # Identify variable nodes (type 0)
        var_mask = (node_types == 0)

        # Process each batch item separately for variable properties
        all_var_props = []
        all_interactions = []
        augmented_embeddings = node_embeddings.clone()

        for b in range(batch_size):
            batch_mask = (batch == b)
            batch_var_mask = batch_mask & var_mask

            if batch_var_mask.sum() == 0:
                continue

            # Get variable embeddings for this batch
            var_embeds = node_embeddings[batch_var_mask]  # [num_vars_b, hidden_dim]

            # Predict variable properties
            var_props = self.var_property_head(var_embeds)
            all_var_props.append(var_props)

            # Predict interaction properties
            interactions = self.interaction_head(var_embeds)
            all_interactions.append(interactions)

            # Create augmented embeddings for variables
            # Combine: original + property context + Walsh context
            walsh_ctx = walsh_features[b:b+1].expand(var_embeds.size(0), -1)
            combined = torch.cat([
                var_props['augmented'],  # [num_vars, hidden_dim + property_dim]
                walsh_ctx,               # [num_vars, walsh_output_dim]
            ], dim=-1)

            augmented_vars = self.output_proj(combined)
            augmented_embeddings[batch_var_mask] = augmented_vars

        return {
            'augmented_embeddings': augmented_embeddings,
            'var_properties': all_var_props,
            'interactions': all_interactions,
            'walsh_features': walsh_features,
            'walsh_raw': {
                'features': walsh_raw,
                'is_linear': walsh_raw[:, 13],      # Index 13 is is_linear
                'nonlinearity': walsh_raw[:, 14],   # Index 14 is nonlinearity score
                'degree_estimate': walsh_raw[:, 15], # Index 15 is degree estimate
            },
        }
