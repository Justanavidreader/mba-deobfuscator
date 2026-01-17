"""
GNN encoders for AST graph encoding.

All encoder classes inherit from BaseEncoder for consistent interface
in ablation studies. See encoder_base.py for interface specification.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, MessagePassing
from torch_geometric.utils import softmax
from torch_scatter import scatter_mean, scatter_max
from typing import Optional
from src.constants import (
    NODE_DIM, HIDDEN_DIM, NUM_ENCODER_LAYERS, NUM_ENCODER_HEADS,
    ENCODER_DROPOUT, NUM_EDGE_TYPES, GGNN_TIMESTEPS, FINGERPRINT_DIM,
    NUM_OPTIMIZED_EDGE_TYPES,
    HGT_USE_GLOBAL_ATTENTION, HGT_GLOBAL_ATTN_INTERVAL, HGT_GLOBAL_ATTN_HEADS,
    HGT_GLOBAL_ATTN_FFN_RATIO, HGT_GLOBAL_ATTN_CHECKPOINT,
    # Path encoding integration
    GGNN_USE_PATH_ENCODING, HGT_USE_PATH_ENCODING,
    HGT_PATH_INJECTION_INTERVAL, HGT_PATH_INJECTION_SCALE,
    PATH_MAX_LENGTH, PATH_MAX_PATHS, PATH_AGGREGATION,
    # GCNII over-smoothing mitigation
    GCNII_ALPHA, GCNII_LAMBDA,
    GCNII_USE_INITIAL_RESIDUAL, GCNII_USE_IDENTITY_MAPPING,
)
from src.models.encoder_base import BaseEncoder
from src.models.global_attention import GlobalAttentionBlock
from src.models.operation_aware_aggregator import OperationAwareAggregator
from src.models.path_encoding import PathBasedEdgeEncoder


def _infer_node_types(x: torch.Tensor) -> torch.Tensor:
    """
    Convert node features to type IDs.

    Args:
        x: Either [num_nodes] type IDs or [num_nodes, node_dim] one-hot features

    Returns:
        [num_nodes] tensor of node type IDs
    """
    return x.argmax(dim=-1) if x.dim() > 1 else x


class GATJKNetEncoder(BaseEncoder):
    """
    Graph Attention Network with Jumping Knowledge.
    Default encoder for depth <= 10 expressions.

    Inherits from BaseEncoder for ablation study compatibility.
    Does NOT require edge types (homogeneous GNN).
    """

    def __init__(self, node_dim: int = NODE_DIM, hidden_dim: int = HIDDEN_DIM,
                 num_layers: int = NUM_ENCODER_LAYERS, num_heads: int = NUM_ENCODER_HEADS,
                 dropout: float = ENCODER_DROPOUT, **kwargs):
        """
        Initialize GAT with Jumping Knowledge encoder.

        Args:
            node_dim: Input node feature dimension
            hidden_dim: Hidden dimension
            num_layers: Number of GAT layers
            num_heads: Number of attention heads
            dropout: Dropout probability
            **kwargs: Ignored (for interface compatibility)
        """
        super().__init__(hidden_dim=hidden_dim)
        self.num_layers = num_layers
        self.node_dim = node_dim

        self.node_embedding = nn.Linear(node_dim, hidden_dim)

        self.gat_layers = nn.ModuleList()
        for i in range(num_layers):
            in_channels = hidden_dim
            out_channels = hidden_dim // num_heads
            self.gat_layers.append(
                GATConv(in_channels, out_channels, heads=num_heads,
                        dropout=dropout, concat=True)
            )

        self.dropout = nn.Dropout(dropout)
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])

        self.jk_projection = nn.Linear(hidden_dim * num_layers, hidden_dim)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        nn.init.xavier_uniform_(self.node_embedding.weight)
        nn.init.zeros_(self.node_embedding.bias)
        nn.init.xavier_uniform_(self.jk_projection.weight)
        nn.init.zeros_(self.jk_projection.bias)

    @property
    def requires_edge_types(self) -> bool:
        """GAT+JKNet does not use edge types."""
        return False

    @property
    def requires_node_features(self) -> bool:
        """GAT+JKNet expects [total_nodes, node_dim] features."""
        return True

    def _forward_impl(self, x: torch.Tensor, edge_index: torch.Tensor,
                      batch: torch.Tensor,
                      edge_type: Optional[torch.Tensor] = None,
                      dag_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode AST graph.

        Args:
            x: [total_nodes, node_dim] node features
            edge_index: [2, num_edges] edge indices
            batch: [total_nodes] batch assignment
            edge_type: Ignored (not used by GAT)
            dag_pos: [total_nodes, 4] DAG positional features (ignored by GAT)

        Returns:
            [total_nodes, hidden_dim] node embeddings
        """
        # dag_pos ignored: GAT uses homogeneous message passing
        x = self.node_embedding(x)
        x = F.elu(x)

        layer_outputs = []

        for i, (gat_layer, layer_norm) in enumerate(zip(self.gat_layers, self.layer_norms)):
            x_in = x
            x = gat_layer(x, edge_index)
            x = x_in + self.dropout(x)  # Residual connection before layer norm
            x = layer_norm(x)
            x = F.elu(x)  # ELU activation per spec
            layer_outputs.append(x)

        jk_concat = torch.cat(layer_outputs, dim=-1)
        x = self.jk_projection(jk_concat)

        return x


class GGNNEncoder(BaseEncoder):
    """
    Gated Graph Neural Network with edge type support.
    Better for depth 10+ expressions.

    Inherits from BaseEncoder for ablation study compatibility.
    REQUIRES edge types (heterogeneous GNN).
    """

    def __init__(self, node_dim: int = NODE_DIM, hidden_dim: int = HIDDEN_DIM,
                 num_timesteps: int = GGNN_TIMESTEPS, num_edge_types: int = NUM_EDGE_TYPES,
                 # Edge type mode: "legacy" (6-type) or "optimized" (8-type)
                 edge_type_mode: str = "legacy",
                 # Path encoding parameters
                 use_path_encoding: bool = GGNN_USE_PATH_ENCODING,
                 path_max_length: int = PATH_MAX_LENGTH,
                 path_max_paths: int = PATH_MAX_PATHS,
                 path_aggregation: str = PATH_AGGREGATION,
                 **kwargs):
        """
        Initialize GGNN encoder.

        Args:
            node_dim: Input node feature dimension
            hidden_dim: Hidden dimension
            num_timesteps: Number of message passing iterations
            num_edge_types: Number of edge types
            edge_type_mode: Edge type system - "legacy" (6-type) or "optimized" (8-type)
            use_path_encoding: Enable path-based edge encoding
            path_max_length: Maximum path length to consider
            path_max_paths: Maximum paths per edge
            path_aggregation: Path aggregation method ('mean', 'max', 'attention')
            **kwargs: Ignored (for interface compatibility)
        """
        super().__init__(hidden_dim=hidden_dim)
        self.num_timesteps = num_timesteps
        self.node_dim = node_dim

        # Edge type mode configuration
        if edge_type_mode not in ("legacy", "optimized"):
            raise ValueError(f"edge_type_mode must be 'legacy' or 'optimized', got: {edge_type_mode}")
        self.edge_type_mode = edge_type_mode

        # Set num_edge_types based on mode
        if edge_type_mode == "legacy":
            self.num_edge_types = NUM_EDGE_TYPES  # 6
        else:
            self.num_edge_types = NUM_OPTIMIZED_EDGE_TYPES  # 8

        self.node_embedding = nn.Linear(node_dim, hidden_dim)

        self.message_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ) for _ in range(num_edge_types)
        ])

        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

        # Path-based edge encoding (simple residual for ablation comparison)
        self.use_path_encoding = use_path_encoding
        self.path_encoder: Optional[PathBasedEdgeEncoder] = None
        if use_path_encoding:
            self.path_encoder = PathBasedEdgeEncoder(
                hidden_dim=hidden_dim,
                num_node_types=10,  # MBA node types
                num_edge_types=num_edge_types,
                max_path_length=path_max_length,
                max_paths=path_max_paths,
                aggregation=path_aggregation,
            )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        nn.init.xavier_uniform_(self.node_embedding.weight)
        nn.init.zeros_(self.node_embedding.bias)

        for mlp in self.message_mlps:
            for layer in mlp:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)

    @property
    def requires_edge_types(self) -> bool:
        """GGNN requires edge types for message passing."""
        return True

    @property
    def requires_node_features(self) -> bool:
        """GGNN expects [total_nodes, node_dim] features."""
        return True

    def _forward_impl(self, x: torch.Tensor, edge_index: torch.Tensor,
                      batch: torch.Tensor,
                      edge_type: Optional[torch.Tensor] = None,
                      dag_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode AST graph with edge types.

        Args:
            x: [total_nodes, node_dim] node features
            edge_index: [2, num_edges] edge indices
            batch: [total_nodes] batch assignment
            edge_type: [num_edges] edge type indices (REQUIRED)
            dag_pos: [total_nodes, 4] DAG positional features (optional)

        Returns:
            [total_nodes, hidden_dim] node embeddings
        """
        # edge_type is validated by BaseEncoder.forward()
        # Additional validation: check edge types are in valid range for configured mode
        if edge_type is not None and edge_type.numel() > 0:
            max_edge = edge_type.max().item()
            if max_edge >= self.num_edge_types:
                raise ValueError(
                    f"Edge type {max_edge} exceeds limit {self.num_edge_types - 1} for "
                    f"edge_type_mode='{self.edge_type_mode}'. "
                    f"Check that edge_type_mode in config matches dataset format."
                )

        num_nodes = x.size(0)
        h = self.node_embedding(x)
        h = F.elu(h)

        # Compute path-based edge embeddings once (simple residual for ablation)
        path_edge_emb = None
        if self.use_path_encoding and self.path_encoder is not None:
            node_types = _infer_node_types(x)
            path_edge_emb = self.path_encoder(edge_index, edge_type, node_types)

        for _ in range(self.num_timesteps):
            messages = torch.zeros(num_nodes, self.hidden_dim, device=x.device)

            for edge_t in range(len(self.message_mlps)):
                mask = edge_type == edge_t
                if mask.sum() == 0:
                    continue

                edge_idx = edge_index[:, mask]
                src, dst = edge_idx[0], edge_idx[1]

                h_src = h[src]
                msg = self.message_mlps[edge_t](h_src)

                # Simple residual: add path context to message
                if path_edge_emb is not None:
                    msg = msg + path_edge_emb[mask]

                messages.index_add_(0, dst, msg)

            h = self.gru(messages, h)

        # dag_pos integration deferred to encoder configuration (use_dag_features flag)
        return h


class GraphReadout(nn.Module):
    """Aggregate node embeddings to graph-level representation."""

    def __init__(self, hidden_dim: int = HIDDEN_DIM):
        """
        Initialize graph readout layer.

        Args:
            hidden_dim: Node embedding dimension
        """
        super().__init__()
        self.hidden_dim = hidden_dim

        self.cls_token = nn.Parameter(torch.randn(1, hidden_dim))

        self.projection = nn.Linear(hidden_dim * 3, hidden_dim)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        nn.init.normal_(self.cls_token, mean=0, std=0.02)
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Aggregate node embeddings to graph-level.

        Args:
            x: [total_nodes, hidden_dim] node embeddings
            batch: [total_nodes] batch assignment

        Returns:
            [batch_size, hidden_dim] graph-level embeddings
        """
        batch_size = batch.max().item() + 1

        mean_pool = scatter_mean(x, batch, dim=0, dim_size=batch_size)

        max_pool, _ = scatter_max(x, batch, dim=0, dim_size=batch_size)

        cls_expanded = self.cls_token.expand(batch_size, -1)

        aggregated = torch.cat([mean_pool, max_pool, cls_expanded], dim=-1)

        out = self.projection(aggregated)
        return out


class FingerprintEncoder(nn.Module):
    """Project semantic fingerprint to model dimension."""

    def __init__(self, fp_dim: int = FINGERPRINT_DIM, hidden_dim: int = HIDDEN_DIM):
        """
        Initialize fingerprint encoder.

        Args:
            fp_dim: Fingerprint dimension (448)
            hidden_dim: Output dimension (256)
        """
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(fp_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for layer in self.encoder:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, fp: torch.Tensor) -> torch.Tensor:
        """
        Encode semantic fingerprint.

        Args:
            fp: [batch, FINGERPRINT_DIM] semantic fingerprint

        Returns:
            [batch, hidden_dim] encoded fingerprint
        """
        return self.encoder(fp)


# =============================================================================
# SCALED ENCODERS (for 360M parameter model)
# =============================================================================

try:
    import torch_geometric.nn as pyg_nn
    from torch_geometric.nn import global_mean_pool, global_max_pool
    HAS_PYG_EXTRAS = True
except ImportError:
    HAS_PYG_EXTRAS = False


class HGTEncoder(BaseEncoder):
    """
    Heterogeneous Graph Transformer encoder for scaled MBA model.
    Uses type-specific projections. Recommended for 360M model.
    Architecture: 12 layers, 16 heads, 768 hidden dim (~60M params)

    Inherits from BaseEncoder for ablation study compatibility.
    REQUIRES edge types (heterogeneous GNN).

    Uses complete semantically-valid triplet set (312 triplets) generated
    from MBA expression graph semantics. See scripts/generate_triplets.py.
    """

    # Auto-generated valid triplets from semantic analysis (312 triplets)
    # Format: (src_node_type, edge_type, dst_node_type)
    # Node types (from constants.py): 0=VAR, 1=CONST, 2=ADD, 3=SUB, 4=MUL, 5=AND, 6=OR, 7=XOR, 8=NOT, 9=NEG
    # Edge types: 0=LEFT_OP, 1=RIGHT_OP, 2=UNARY_OP, 3=LEFT_INV, 4=RIGHT_INV, 5=UNARY_INV, 6=BRIDGE_DOWN, 7=BRIDGE_UP
    VALID_TRIPLETS = [
        # LEFT_OPERAND edges (binary op -> any child as left operand)
        (2, 0, 0), (2, 0, 1), (2, 0, 2), (2, 0, 3), (2, 0, 4), (2, 0, 5), (2, 0, 6), (2, 0, 7), (2, 0, 8), (2, 0, 9),
        (3, 0, 0), (3, 0, 1), (3, 0, 2), (3, 0, 3), (3, 0, 4), (3, 0, 5), (3, 0, 6), (3, 0, 7), (3, 0, 8), (3, 0, 9),
        (4, 0, 0), (4, 0, 1), (4, 0, 2), (4, 0, 3), (4, 0, 4), (4, 0, 5), (4, 0, 6), (4, 0, 7), (4, 0, 8), (4, 0, 9),
        (5, 0, 0), (5, 0, 1), (5, 0, 2), (5, 0, 3), (5, 0, 4), (5, 0, 5), (5, 0, 6), (5, 0, 7), (5, 0, 8), (5, 0, 9),
        (6, 0, 0), (6, 0, 1), (6, 0, 2), (6, 0, 3), (6, 0, 4), (6, 0, 5), (6, 0, 6), (6, 0, 7), (6, 0, 8), (6, 0, 9),
        (7, 0, 0), (7, 0, 1), (7, 0, 2), (7, 0, 3), (7, 0, 4), (7, 0, 5), (7, 0, 6), (7, 0, 7), (7, 0, 8), (7, 0, 9),
        # RIGHT_OPERAND edges (binary op -> any child as right operand)
        (2, 1, 0), (2, 1, 1), (2, 1, 2), (2, 1, 3), (2, 1, 4), (2, 1, 5), (2, 1, 6), (2, 1, 7), (2, 1, 8), (2, 1, 9),
        (3, 1, 0), (3, 1, 1), (3, 1, 2), (3, 1, 3), (3, 1, 4), (3, 1, 5), (3, 1, 6), (3, 1, 7), (3, 1, 8), (3, 1, 9),
        (4, 1, 0), (4, 1, 1), (4, 1, 2), (4, 1, 3), (4, 1, 4), (4, 1, 5), (4, 1, 6), (4, 1, 7), (4, 1, 8), (4, 1, 9),
        (5, 1, 0), (5, 1, 1), (5, 1, 2), (5, 1, 3), (5, 1, 4), (5, 1, 5), (5, 1, 6), (5, 1, 7), (5, 1, 8), (5, 1, 9),
        (6, 1, 0), (6, 1, 1), (6, 1, 2), (6, 1, 3), (6, 1, 4), (6, 1, 5), (6, 1, 6), (6, 1, 7), (6, 1, 8), (6, 1, 9),
        (7, 1, 0), (7, 1, 1), (7, 1, 2), (7, 1, 3), (7, 1, 4), (7, 1, 5), (7, 1, 6), (7, 1, 7), (7, 1, 8), (7, 1, 9),
        # UNARY_OPERAND edges (NOT/NEG -> any child)
        (8, 2, 0), (8, 2, 1), (8, 2, 2), (8, 2, 3), (8, 2, 4), (8, 2, 5), (8, 2, 6), (8, 2, 7), (8, 2, 8), (8, 2, 9),
        (9, 2, 0), (9, 2, 1), (9, 2, 2), (9, 2, 3), (9, 2, 4), (9, 2, 5), (9, 2, 6), (9, 2, 7), (9, 2, 8), (9, 2, 9),
        # LEFT_OPERAND_INV edges (any child -> binary op as left operand inverse)
        (0, 3, 2), (0, 3, 3), (0, 3, 4), (0, 3, 5), (0, 3, 6), (0, 3, 7),
        (1, 3, 2), (1, 3, 3), (1, 3, 4), (1, 3, 5), (1, 3, 6), (1, 3, 7),
        (2, 3, 2), (2, 3, 3), (2, 3, 4), (2, 3, 5), (2, 3, 6), (2, 3, 7),
        (3, 3, 2), (3, 3, 3), (3, 3, 4), (3, 3, 5), (3, 3, 6), (3, 3, 7),
        (4, 3, 2), (4, 3, 3), (4, 3, 4), (4, 3, 5), (4, 3, 6), (4, 3, 7),
        (5, 3, 2), (5, 3, 3), (5, 3, 4), (5, 3, 5), (5, 3, 6), (5, 3, 7),
        (6, 3, 2), (6, 3, 3), (6, 3, 4), (6, 3, 5), (6, 3, 6), (6, 3, 7),
        (7, 3, 2), (7, 3, 3), (7, 3, 4), (7, 3, 5), (7, 3, 6), (7, 3, 7),
        (8, 3, 2), (8, 3, 3), (8, 3, 4), (8, 3, 5), (8, 3, 6), (8, 3, 7),
        (9, 3, 2), (9, 3, 3), (9, 3, 4), (9, 3, 5), (9, 3, 6), (9, 3, 7),
        # RIGHT_OPERAND_INV edges (any child -> binary op as right operand inverse)
        (0, 4, 2), (0, 4, 3), (0, 4, 4), (0, 4, 5), (0, 4, 6), (0, 4, 7),
        (1, 4, 2), (1, 4, 3), (1, 4, 4), (1, 4, 5), (1, 4, 6), (1, 4, 7),
        (2, 4, 2), (2, 4, 3), (2, 4, 4), (2, 4, 5), (2, 4, 6), (2, 4, 7),
        (3, 4, 2), (3, 4, 3), (3, 4, 4), (3, 4, 5), (3, 4, 6), (3, 4, 7),
        (4, 4, 2), (4, 4, 3), (4, 4, 4), (4, 4, 5), (4, 4, 6), (4, 4, 7),
        (5, 4, 2), (5, 4, 3), (5, 4, 4), (5, 4, 5), (5, 4, 6), (5, 4, 7),
        (6, 4, 2), (6, 4, 3), (6, 4, 4), (6, 4, 5), (6, 4, 6), (6, 4, 7),
        (7, 4, 2), (7, 4, 3), (7, 4, 4), (7, 4, 5), (7, 4, 6), (7, 4, 7),
        (8, 4, 2), (8, 4, 3), (8, 4, 4), (8, 4, 5), (8, 4, 6), (8, 4, 7),
        (9, 4, 2), (9, 4, 3), (9, 4, 4), (9, 4, 5), (9, 4, 6), (9, 4, 7),
        # UNARY_OPERAND_INV edges (any child -> NOT/NEG as unary operand inverse)
        (0, 5, 8), (0, 5, 9), (1, 5, 8), (1, 5, 9), (2, 5, 8), (2, 5, 9), (3, 5, 8), (3, 5, 9), (4, 5, 8), (4, 5, 9),
        (5, 5, 8), (5, 5, 9), (6, 5, 8), (6, 5, 9), (7, 5, 8), (7, 5, 9), (8, 5, 8), (8, 5, 9), (9, 5, 8), (9, 5, 9),
        # DOMAIN_BRIDGE_DOWN edges (bool parent -> arith child for cross-domain)
        (5, 6, 2), (5, 6, 3), (5, 6, 4), (5, 6, 9),
        (6, 6, 2), (6, 6, 3), (6, 6, 4), (6, 6, 9),
        (7, 6, 2), (7, 6, 3), (7, 6, 4), (7, 6, 9),
        (8, 6, 2), (8, 6, 3), (8, 6, 4), (8, 6, 9),
        # DOMAIN_BRIDGE_UP edges (arith child -> bool parent for cross-domain inverse)
        (2, 7, 5), (2, 7, 6), (2, 7, 7), (2, 7, 8),
        (3, 7, 5), (3, 7, 6), (3, 7, 7), (3, 7, 8),
        (4, 7, 5), (4, 7, 6), (4, 7, 7), (4, 7, 8),
        (9, 7, 5), (9, 7, 6), (9, 7, 7), (9, 7, 8),
    ]

    def __init__(
        self,
        hidden_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 16,
        dropout: float = 0.1,
        num_node_types: int = 10,
        num_edge_types: int = NUM_OPTIMIZED_EDGE_TYPES,
        # Edge type mode: HGT only supports "optimized" (8-type)
        edge_type_mode: str = "optimized",
        # GraphGPS-style global attention parameters
        use_global_attention: bool = HGT_USE_GLOBAL_ATTENTION,
        global_attn_interval: int = HGT_GLOBAL_ATTN_INTERVAL,
        global_attn_heads: int = HGT_GLOBAL_ATTN_HEADS,
        global_attn_ffn_ratio: float = HGT_GLOBAL_ATTN_FFN_RATIO,
        global_attn_checkpoint: bool = HGT_GLOBAL_ATTN_CHECKPOINT,
        # Operation-aware aggregation parameters
        operation_aware: bool = False,
        operation_aware_strict: bool = True,
        # Path encoding parameters
        use_path_encoding: bool = HGT_USE_PATH_ENCODING,
        path_max_length: int = PATH_MAX_LENGTH,
        path_max_paths: int = PATH_MAX_PATHS,
        path_aggregation: str = PATH_AGGREGATION,
        path_injection_interval: int = HGT_PATH_INJECTION_INTERVAL,
        path_injection_scale: float = HGT_PATH_INJECTION_SCALE,
        # GCNII over-smoothing mitigation parameters
        gcnii_alpha: float = GCNII_ALPHA,
        gcnii_lambda: float = GCNII_LAMBDA,
        use_initial_residual: bool = GCNII_USE_INITIAL_RESIDUAL,
        use_identity_mapping: bool = GCNII_USE_IDENTITY_MAPPING,
        **kwargs,
    ):
        super().__init__(hidden_dim=hidden_dim)
        if not HAS_PYG_EXTRAS:
            raise ImportError("HGTEncoder requires torch_geometric with HGTConv")

        # HGT only supports optimized 8-type edges
        if edge_type_mode != "optimized":
            raise ValueError(
                f"HGTEncoder only supports optimized 8-type edges. "
                f"Got edge_type_mode='{edge_type_mode}'. "
                f"Use GGNNEncoder for legacy 6-type datasets or convert dataset to optimized format."
            )
        self.edge_type_mode = edge_type_mode

        self.num_node_types = num_node_types
        self._num_edge_types = num_edge_types
        self.num_layers = num_layers
        self.use_global_attention = use_global_attention
        self.global_attn_interval = global_attn_interval
        self.node_type_embed = nn.Embedding(num_node_types, hidden_dim)

        # GCNII over-smoothing mitigation parameters
        self.gcnii_alpha = gcnii_alpha
        self.gcnii_lambda = gcnii_lambda
        self.use_initial_residual = use_initial_residual
        self.use_identity_mapping = use_identity_mapping

        # Use pruned metadata - only ~312 triplets instead of 700
        # HGTConv requires string type names for ModuleDict keys
        node_types = [str(i) for i in range(num_node_types)]
        # Convert triplets to use string types (src_node, edge_type, dst_node)
        edge_triplets = [
            (str(src), str(edge_t), str(dst))
            for src, edge_t, dst in self.VALID_TRIPLETS
        ]

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            conv = pyg_nn.HGTConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                metadata=(node_types, edge_triplets),
                heads=num_heads,
            )
            self.convs.append(conv)
            self.norms.append(nn.LayerNorm(hidden_dim))

        # Global attention blocks (inserted every global_attn_interval layers)
        self.global_attn_blocks = None
        if use_global_attention:
            # Number of global blocks = floor((num_layers - 1) / interval)
            # E.g., 12 layers, interval=2: blocks after layers 1,3,5,7,9 = 5 blocks
            num_global_blocks = (num_layers - 1) // global_attn_interval
            assert num_global_blocks > 0, (
                f"No global attention blocks with num_layers={num_layers}, "
                f"interval={global_attn_interval}. Need num_layers > interval."
            )
            self.global_attn_blocks = nn.ModuleList([
                GlobalAttentionBlock(
                    hidden_dim=hidden_dim,
                    num_heads=global_attn_heads,
                    ffn_ratio=global_attn_ffn_ratio,
                    dropout=dropout,
                    use_checkpoint=global_attn_checkpoint,
                )
                for _ in range(num_global_blocks)
            ])

        self.dropout = nn.Dropout(dropout)

        # Operation-aware aggregation (treats commutative/non-commutative ops differently)
        self.operation_aware = operation_aware
        self.aggregator: Optional[OperationAwareAggregator] = None
        if operation_aware:
            self.aggregator = OperationAwareAggregator(
                hidden_dim=hidden_dim,
                dropout=dropout,
                strict_validation=operation_aware_strict,
            )

        # Path-based edge encoding for shared subexpression detection
        self.use_path_encoding = use_path_encoding
        self.path_injection_interval = path_injection_interval
        self.path_injection_scale = path_injection_scale
        self.path_encoder: Optional[PathBasedEdgeEncoder] = None
        self.path_projectors: Optional[nn.ModuleList] = None

        if use_path_encoding:
            self.path_encoder = PathBasedEdgeEncoder(
                hidden_dim=hidden_dim,
                num_node_types=num_node_types,
                num_edge_types=num_edge_types,
                max_path_length=path_max_length,
                max_paths=path_max_paths,
                aggregation=path_aggregation,
            )
            # One projector per injection point (after layers interval, 2*interval, etc.)
            # Quality fix: handle num_injections=0 case
            num_injections = (num_layers - 1) // path_injection_interval
            if num_injections > 0:
                self.path_projectors = nn.ModuleList([
                    nn.Linear(hidden_dim, hidden_dim)
                    for _ in range(num_injections)
                ])

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.node_type_embed.weight, mean=0, std=0.02)
        if self.path_projectors is not None:
            for proj in self.path_projectors:
                nn.init.xavier_uniform_(proj.weight)
                nn.init.zeros_(proj.bias)

    @property
    def requires_edge_types(self) -> bool:
        """HGT requires edge types for heterogeneous message passing."""
        return True

    @property
    def requires_node_features(self) -> bool:
        """HGT expects [total_nodes] node type IDs (embeds internally)."""
        return False

    def _to_heterogeneous(self, x: torch.Tensor, edge_index: torch.Tensor,
                          edge_type: torch.Tensor):
        """
        Convert flat tensors to heterogeneous dict format for HGTConv.

        Args:
            x: [num_nodes] node type IDs
            edge_index: [2, num_edges] edge indices
            edge_type: [num_edges] edge type IDs

        Returns:
            x_dict: {node_type_str: [n, hidden_dim]} node features per type
            edge_index_dict: {(src_str, edge_t, dst_str): [2, n]} edges per triplet
        """
        h = self.node_type_embed(x)

        x_dict = {}
        node_offsets = {}  # Map global node idx to local idx per type

        for ntype in range(self.num_node_types):
            mask = (x == ntype)
            if mask.any():
                ntype_str = str(ntype)
                x_dict[ntype_str] = h[mask]
                node_offsets[ntype] = {
                    global_idx.item(): local_idx
                    for local_idx, global_idx in enumerate(mask.nonzero().squeeze(-1))
                }

        # Build edge index dict
        edge_index_dict = {}
        src_types = x[edge_index[0]]
        dst_types = x[edge_index[1]]

        for triplet in self.VALID_TRIPLETS:
            src_t, e_t, dst_t = triplet
            mask = (src_types == src_t) & (edge_type == e_t) & (dst_types == dst_t)
            if mask.any():
                edges = edge_index[:, mask]
                if src_t in node_offsets and dst_t in node_offsets:
                    local_src = torch.tensor(
                        [node_offsets[src_t].get(s.item(), 0) for s in edges[0]],
                        device=x.device
                    )
                    local_dst = torch.tensor(
                        [node_offsets[dst_t].get(d.item(), 0) for d in edges[1]],
                        device=x.device
                    )
                    # Convert triplet to string format for HGTConv
                    triplet_str = (str(src_t), str(e_t), str(dst_t))
                    edge_index_dict[triplet_str] = torch.stack([local_src, local_dst])

        return x_dict, edge_index_dict

    def _aggregate_path_to_nodes(
        self,
        path_edge_emb: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        """
        Aggregate path-based edge embeddings to destination nodes.

        Each node receives mean of incoming edge path embeddings.

        Args:
            path_edge_emb: [num_edges, hidden_dim] path-based edge embeddings
            edge_index: [2, num_edges] edge indices
            num_nodes: Total number of nodes

        Returns:
            [num_nodes, hidden_dim] aggregated path context per node
        """
        device = path_edge_emb.device
        node_path_context = torch.zeros(num_nodes, self.hidden_dim, device=device)
        edge_counts = torch.zeros(num_nodes, device=device)

        dst_nodes = edge_index[1]
        node_path_context.index_add_(0, dst_nodes, path_edge_emb)
        edge_counts.index_add_(0, dst_nodes, torch.ones(dst_nodes.size(0), device=device))

        # Avoid division by zero for nodes with no incoming edges
        edge_counts = edge_counts.clamp(min=1).unsqueeze(-1)
        node_path_context = node_path_context / edge_counts

        return node_path_context

    def _inject_path_context(
        self,
        x_dict: dict,
        path_edge_emb: torch.Tensor,
        edge_index: torch.Tensor,
        original_node_types: torch.Tensor,
        path_inject_idx: int,
    ) -> dict:
        """
        Inject path context into node features after HGT layer.

        Converts heterogeneous dict to flat, adds scaled path context,
        then converts back to heterogeneous dict.

        Args:
            x_dict: Heterogeneous node features {node_type_str: [n, hidden_dim]}
            path_edge_emb: Path-based edge embeddings [num_edges, hidden_dim]
            edge_index: Edge indices [2, num_edges]
            original_node_types: Node type IDs [num_nodes]
            path_inject_idx: Index of path projector to use

        Returns:
            Updated x_dict with path context injected
        """
        num_nodes = original_node_types.size(0)
        device = original_node_types.device

        # Convert heterogeneous dict to flat tensor
        h_flat = torch.zeros(num_nodes, self.hidden_dim, device=device)
        for ntype_str in x_dict:
            ntype_int = int(ntype_str)
            mask = (original_node_types == ntype_int)
            h_flat[mask] = x_dict[ntype_str]

        # Aggregate path embeddings to nodes and project
        path_context = self._aggregate_path_to_nodes(path_edge_emb, edge_index, num_nodes)
        path_context = path_context.to(h_flat.device)  # Quality fix: device consistency
        path_context = self.path_projectors[path_inject_idx](path_context)

        # Residual addition with configurable scale
        h_flat = h_flat + self.path_injection_scale * path_context

        # Convert flat tensor back to heterogeneous dict
        for ntype_str in x_dict:
            ntype_int = int(ntype_str)
            mask = (original_node_types == ntype_int)
            x_dict[ntype_str] = h_flat[mask]

        return x_dict

    def _forward_impl(self, x: torch.Tensor, edge_index: torch.Tensor,
                      batch: torch.Tensor,
                      edge_type: Optional[torch.Tensor] = None,
                      dag_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with automatic flat-to-heterogeneous conversion.

        When use_global_attention=True, global self-attention blocks are interleaved
        with HGT layers every global_attn_interval layers. This enables O(1) detection
        of repeated subexpressions (e.g., detecting that (a & b) appears twice).

        Args:
            x: [total_nodes] node type IDs (0-9)
            edge_index: [2, num_edges] edge indices
            batch: [total_nodes] batch assignment
            edge_type: [num_edges] edge type IDs (0-7) (REQUIRED)
            dag_pos: [total_nodes, 4] DAG positional features (optional)
                     Integration via concat/residual controlled by HGT_USE_DAG_FEATURES

        Returns:
            [total_nodes, hidden_dim] node embeddings (flat format)
        """
        # edge_type is validated by BaseEncoder.forward()
        # dag_pos integration deferred to encoder configuration (use_dag_features flag)
        x_dict, edge_index_dict = self._to_heterogeneous(x, edge_index, edge_type)

        # Track original node types for final reconstruction
        original_node_types = x

        # GCNII: Store initial embeddings for residual connection
        # h^(0) = initial node embeddings (before any message passing)
        h_0_dict = {ntype_str: h.clone() for ntype_str, h in x_dict.items()}

        # Compute path-based edge embeddings once (expensive, so not per-layer)
        path_edge_emb = None
        if self.use_path_encoding and self.path_encoder is not None:
            node_types = _infer_node_types(x)
            path_edge_emb = self.path_encoder(edge_index, edge_type, node_types)

        global_block_idx = 0
        path_inject_idx = 0

        for layer_idx, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            # Compute beta for identity mapping (decreases with depth)
            beta = self._compute_gcnii_beta(layer_idx, self.gcnii_lambda, self.use_identity_mapping)

            # Local HGT message passing
            x_dict_new = conv(x_dict, edge_index_dict)

            for ntype_str in x_dict:
                if ntype_str not in x_dict_new:
                    continue

                # Current features
                h_current = x_dict[ntype_str]

                # Transformed features from HGT
                h_transformed = x_dict_new[ntype_str]

                # ============================================================
                # TECHNIQUE 1: Identity Mapping
                # Mix transformed features with identity (unchanged features)
                # Deep layers have lower beta → act more like identity
                # ============================================================
                if self.use_identity_mapping:
                    h_transformed = beta * h_transformed + (1 - beta) * h_current

                # ============================================================
                # TECHNIQUE 2: Standard Residual + LayerNorm
                # ============================================================
                h_residual = norm(h_current + self.dropout(h_transformed))

                # ============================================================
                # TECHNIQUE 3: GCNII-Style Initial Residual
                # Mix current features with ORIGINAL input features
                # Ensures h^(0) is always present in representation
                # ============================================================
                if self.use_initial_residual and ntype_str in h_0_dict:
                    h_final = (
                        self.gcnii_alpha * h_0_dict[ntype_str] +
                        (1 - self.gcnii_alpha) * h_residual
                    )
                else:
                    h_final = h_residual

                x_dict[ntype_str] = F.elu(h_final)

            # Insert global attention after every N layers (except last layer)
            # Global attention needs flat format, so convert back and forth
            if (self.use_global_attention
                and self.global_attn_blocks is not None
                and (layer_idx + 1) % self.global_attn_interval == 0
                and layer_idx < len(self.convs) - 1):

                # Convert heterogeneous dict to flat tensor for global attention
                num_nodes = original_node_types.size(0)
                h_flat = torch.zeros(num_nodes, self.hidden_dim, device=original_node_types.device)
                for ntype_str in x_dict:
                    ntype_int = int(ntype_str)
                    mask = (original_node_types == ntype_int)
                    h_flat[mask] = x_dict[ntype_str]

                # Apply global self-attention (batch mask prevents cross-graph attention)
                h_flat = self.global_attn_blocks[global_block_idx](h_flat, batch)
                global_block_idx += 1

                # Convert flat tensor back to heterogeneous dict
                for ntype_str in x_dict:
                    ntype_int = int(ntype_str)
                    mask = (original_node_types == ntype_int)
                    x_dict[ntype_str] = h_flat[mask]

            # Insert path context injection after every N layers (except last layer)
            if (self.use_path_encoding
                and path_edge_emb is not None
                and self.path_projectors is not None
                and (layer_idx + 1) % self.path_injection_interval == 0
                and layer_idx < len(self.convs) - 1
                and path_inject_idx < len(self.path_projectors)):

                x_dict = self._inject_path_context(
                    x_dict, path_edge_emb, edge_index, original_node_types, path_inject_idx
                )
                path_inject_idx += 1

        # Convert back to flat format
        num_nodes = original_node_types.size(0)
        h_out = torch.zeros(num_nodes, self.hidden_dim, device=original_node_types.device)

        for ntype_str in x_dict:
            ntype_int = int(ntype_str)
            mask = (original_node_types == ntype_int)
            h_out[mask] = x_dict[ntype_str]

        # Apply operation-aware aggregation if enabled
        # Commutative ops (ADD, MUL, AND, OR, XOR) use order-invariant sum.
        # Non-commutative ops (SUB) use order-preserving concat+project.
        if self.operation_aware and self.aggregator is not None:
            h_out = self.aggregator(
                node_features=h_out,
                edge_index=edge_index,
                edge_types=edge_type,
                node_types=original_node_types,
                messages=h_out,
            )

        return h_out


class RGCNEncoder(BaseEncoder):
    """
    Relational GCN encoder (alternative to HGT).
    Simpler implementation, uses edge-type-specific weight matrices.
    Architecture: 12 layers, 768 hidden dim (~60M params)

    Inherits from BaseEncoder for ablation study compatibility.
    REQUIRES edge types (relational GCN).
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        num_layers: int = 12,
        dropout: float = 0.1,
        num_node_types: int = 10,
        num_edge_types: int = NUM_OPTIMIZED_EDGE_TYPES,
        # Edge type mode: RGCN only supports "optimized" (8-type)
        edge_type_mode: str = "optimized",
        # GCNII over-smoothing mitigation parameters
        gcnii_alpha: float = GCNII_ALPHA,
        gcnii_lambda: float = GCNII_LAMBDA,
        use_initial_residual: bool = GCNII_USE_INITIAL_RESIDUAL,
        use_identity_mapping: bool = GCNII_USE_IDENTITY_MAPPING,
        **kwargs,
    ):
        super().__init__(hidden_dim=hidden_dim)
        if not HAS_PYG_EXTRAS:
            raise ImportError("RGCNEncoder requires torch_geometric with RGCNConv")

        # RGCN only supports optimized 8-type edges
        if edge_type_mode != "optimized":
            raise ValueError(
                f"RGCNEncoder only supports optimized 8-type edges. "
                f"Got edge_type_mode='{edge_type_mode}'. "
                f"Use GGNNEncoder for legacy 6-type datasets or convert dataset to optimized format."
            )
        self.edge_type_mode = edge_type_mode

        self.num_node_types = num_node_types
        self._num_edge_types = num_edge_types
        self.num_layers = num_layers  # Add this if not present
        self.node_embed = nn.Embedding(num_node_types, hidden_dim)

        # GCNII over-smoothing mitigation parameters
        self.gcnii_alpha = gcnii_alpha
        self.gcnii_lambda = gcnii_lambda
        self.use_initial_residual = use_initial_residual
        self.use_identity_mapping = use_identity_mapping

        self.convs = nn.ModuleList([
            pyg_nn.RGCNConv(hidden_dim, hidden_dim, num_relations=num_edge_types)
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.node_embed.weight, mean=0, std=0.02)

    @property
    def requires_edge_types(self) -> bool:
        """RGCN requires edge types for relational convolutions."""
        return True

    @property
    def requires_node_features(self) -> bool:
        """RGCN expects [total_nodes] node type IDs (embeds internally)."""
        return False

    def _forward_impl(self, x: torch.Tensor, edge_index: torch.Tensor,
                      batch: torch.Tensor,
                      edge_type: Optional[torch.Tensor] = None,
                      dag_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through RGCN layers with GCNII-style initial residuals and identity mapping.

        Args:
            x: [total_nodes] node type IDs (0-9)
            edge_index: [2, num_edges] edge indices
            batch: [total_nodes] batch assignment
            edge_type: [num_edges] edge type IDs (0-7) (REQUIRED)
            dag_pos: [total_nodes, 4] DAG positional features (optional)

        Returns:
            [total_nodes, hidden_dim] node embeddings
        """
        # edge_type is validated by BaseEncoder.forward()
        # dag_pos integration deferred to encoder configuration (use_dag_features flag)

        # Initial embedding h^(0)
        h_0 = self.node_embed(x)
        h = h_0

        for layer_idx, (conv, norm) in enumerate(zip(self.convs, self.layer_norms)):
            # Compute beta for identity mapping (decreases with depth)
            beta = self._compute_gcnii_beta(layer_idx, self.gcnii_lambda, self.use_identity_mapping)

            # RGCN transformation
            h_transformed = conv(h, edge_index, edge_type)
            h_transformed = F.elu(h_transformed)

            # ============================================================
            # TECHNIQUE 1: Identity Mapping
            # Mix transformed features with identity (unchanged features)
            # ============================================================
            if self.use_identity_mapping:
                h_transformed = beta * h_transformed + (1 - beta) * h

            # ============================================================
            # TECHNIQUE 2: Standard Residual + LayerNorm
            # ============================================================
            h_residual = norm(h + self.dropout(h_transformed))

            # ============================================================
            # TECHNIQUE 3: GCNII-Style Initial Residual
            # Mix with ORIGINAL input features h^(0)
            # ============================================================
            if self.use_initial_residual:
                h = (
                    self.gcnii_alpha * h_0 +
                    (1 - self.gcnii_alpha) * h_residual
                )
            else:
                h = h_residual

        return h


class ScaledGraphReadout(nn.Module):
    """Graph-level readout for scaled encoder (no CLS token, just pooling)."""

    def __init__(self, hidden_dim: int = 768):
        super().__init__()
        self.hidden_dim = hidden_dim
        # Mean + max pooling concatenated
        self.projection = nn.Linear(hidden_dim * 2, hidden_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Aggregate node embeddings to graph-level.

        Args:
            x: [total_nodes, hidden_dim] node embeddings
            batch: [total_nodes] batch assignment

        Returns:
            [batch_size, hidden_dim] graph-level embeddings
        """
        if HAS_PYG_EXTRAS:
            mean_pool = global_mean_pool(x, batch)
            max_pool = global_max_pool(x, batch)
        else:
            batch_size = batch.max().item() + 1
            mean_pool = scatter_mean(x, batch, dim=0, dim_size=batch_size)
            max_pool, _ = scatter_max(x, batch, dim=0, dim_size=batch_size)

        aggregated = torch.cat([mean_pool, max_pool], dim=-1)
        return self.projection(aggregated)
