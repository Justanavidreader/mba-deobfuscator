"""
GNN encoders for AST graph encoding.

All encoder classes inherit from BaseEncoder for consistent interface
in ablation studies. See encoder_base.py for interface specification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, MessagePassing
from torch_geometric.utils import softmax
from torch_scatter import scatter_mean, scatter_max
from typing import Optional
from src.constants import (
    NODE_DIM, HIDDEN_DIM, NUM_ENCODER_LAYERS, NUM_ENCODER_HEADS,
    ENCODER_DROPOUT, NUM_EDGE_TYPES, GGNN_TIMESTEPS, FINGERPRINT_DIM
)
from src.models.encoder_base import BaseEncoder


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
                      edge_type: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode AST graph.

        Args:
            x: [total_nodes, node_dim] node features
            edge_index: [2, num_edges] edge indices
            batch: [total_nodes] batch assignment
            edge_type: Ignored (not used by GAT)

        Returns:
            [total_nodes, hidden_dim] node embeddings
        """
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
                 **kwargs):
        """
        Initialize GGNN encoder.

        Args:
            node_dim: Input node feature dimension
            hidden_dim: Hidden dimension
            num_timesteps: Number of message passing iterations
            num_edge_types: Number of edge types
            **kwargs: Ignored (for interface compatibility)
        """
        super().__init__(hidden_dim=hidden_dim)
        self.num_timesteps = num_timesteps
        self.num_edge_types = num_edge_types
        self.node_dim = node_dim

        self.node_embedding = nn.Linear(node_dim, hidden_dim)

        self.message_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ) for _ in range(num_edge_types)
        ])

        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

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
                      edge_type: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode AST graph with edge types.

        Args:
            x: [total_nodes, node_dim] node features
            edge_index: [2, num_edges] edge indices
            batch: [total_nodes] batch assignment
            edge_type: [num_edges] edge type indices (REQUIRED)

        Returns:
            [total_nodes, hidden_dim] node embeddings
        """
        # edge_type is validated by BaseEncoder.forward()
        num_nodes = x.size(0)
        h = self.node_embedding(x)
        h = F.elu(h)

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

                messages.index_add_(0, dst, msg)

            h = self.gru(messages, h)

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

    IMPORTANT: This uses a pruned metadata set - only edge triplets that
    actually occur in MBA expression graphs are registered, NOT all 800
    theoretical combinations (10 node types × 8 edge types × 10 node types).
    """

    # Pre-computed valid edge triplets for MBA expressions (~40 combinations)
    # Edge types: 0=LEFT_OPERAND, 1=RIGHT_OPERAND, 2=UNARY_OPERAND,
    #             3=LEFT_OPERAND_INV, 4=RIGHT_OPERAND_INV, 5=UNARY_OPERAND_INV,
    #             6=DOMAIN_BRIDGE_DOWN, 7=DOMAIN_BRIDGE_UP
    # Node types: 0=ADD, 1=SUB, 2=MUL, 3=NEG, 4=AND, 5=OR, 6=XOR, 7=NOT, 8=VAR, 9=CONST
    VALID_TRIPLETS = [
        # Arithmetic operators to operands (LEFT/RIGHT_OPERAND)
        (0, 0, 8), (0, 1, 8), (0, 0, 9), (0, 1, 9),  # ADD -> VAR/CONST
        (1, 0, 8), (1, 1, 8), (1, 0, 9), (1, 1, 9),  # SUB -> VAR/CONST
        (2, 0, 8), (2, 1, 8), (2, 0, 9), (2, 1, 9),  # MUL -> VAR/CONST
        (3, 2, 8), (3, 2, 9),  # NEG -> operand (unary)
        # Boolean operators to operands
        (4, 0, 8), (4, 1, 8), (5, 0, 8), (5, 1, 8),  # AND/OR -> VAR
        (6, 0, 8), (6, 1, 8), (7, 2, 8),  # XOR -> VAR, NOT -> VAR
        # Inverse edges (operand -> operator) using _INV edge types
        (8, 3, 0), (8, 4, 0), (8, 3, 1), (8, 4, 1),  # VAR -> ADD/SUB (LEFT/RIGHT_INV)
        (8, 3, 2), (8, 4, 2), (8, 5, 3), (8, 5, 7),  # VAR -> MUL/NEG/NOT
        (9, 3, 0), (9, 4, 0), (9, 3, 2), (9, 4, 2),  # CONST -> operators
        # Nested operators (operator -> operator)
        (0, 0, 0), (0, 1, 0), (0, 0, 2), (0, 1, 2),  # ADD -> ADD/MUL
        (2, 0, 2), (2, 1, 2), (2, 0, 0), (2, 1, 0),  # MUL -> MUL/ADD
        # Domain bridge DOWN (bool parent -> arith child)
        (4, 6, 0), (4, 6, 1), (4, 6, 2),  # AND -> ADD/SUB/MUL
        (5, 6, 0), (5, 6, 1), (6, 6, 0),  # OR/XOR -> arith ops
        # Domain bridge UP (arith child -> bool parent)
        (0, 7, 4), (1, 7, 4), (2, 7, 4),  # ADD/SUB/MUL -> AND
        (0, 7, 5), (1, 7, 5), (0, 7, 6),  # arith ops -> OR/XOR
    ]

    def __init__(
        self,
        hidden_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 16,
        dropout: float = 0.1,
        num_node_types: int = 10,
        num_edge_types: int = 8,
        **kwargs,
    ):
        super().__init__(hidden_dim=hidden_dim)
        if not HAS_PYG_EXTRAS:
            raise ImportError("HGTEncoder requires torch_geometric with HGTConv")

        self.num_node_types = num_node_types
        self._num_edge_types = num_edge_types
        self.node_type_embed = nn.Embedding(num_node_types, hidden_dim)

        # Use pruned metadata - only ~40 triplets instead of 700
        node_types = list(range(num_node_types))
        edge_triplets = self.VALID_TRIPLETS

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

        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.node_type_embed.weight, mean=0, std=0.02)

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
            x_dict: {node_type: [n, hidden_dim]} node features per type
            edge_index_dict: {(src_t, edge_t, dst_t): [2, n]} edges per triplet
        """
        h = self.node_type_embed(x)

        x_dict = {}
        node_offsets = {}  # Map global node idx to local idx per type

        for ntype in range(self.num_node_types):
            mask = (x == ntype)
            if mask.any():
                x_dict[ntype] = h[mask]
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
                    edge_index_dict[triplet] = torch.stack([local_src, local_dst])

        return x_dict, edge_index_dict

    def _forward_impl(self, x: torch.Tensor, edge_index: torch.Tensor,
                      batch: torch.Tensor,
                      edge_type: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with automatic flat-to-heterogeneous conversion.

        Args:
            x: [total_nodes] node type IDs (0-9)
            edge_index: [2, num_edges] edge indices
            batch: [total_nodes] batch assignment
            edge_type: [num_edges] edge type IDs (0-6) (REQUIRED)

        Returns:
            [total_nodes, hidden_dim] node embeddings (flat format)
        """
        # edge_type is validated by BaseEncoder.forward()
        x_dict, edge_index_dict = self._to_heterogeneous(x, edge_index, edge_type)

        for conv, norm in zip(self.convs, self.norms):
            x_dict_new = conv(x_dict, edge_index_dict)
            for ntype in x_dict:
                if ntype in x_dict_new:
                    x_dict[ntype] = norm(x_dict[ntype] + self.dropout(x_dict_new[ntype]))
                    x_dict[ntype] = F.elu(x_dict[ntype])

        # Convert back to flat format
        num_nodes = x.size(0)
        h_out = torch.zeros(num_nodes, self.hidden_dim, device=x.device)

        for ntype in x_dict:
            mask = (x == ntype)
            h_out[mask] = x_dict[ntype]

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
        num_edge_types: int = 8,
        **kwargs,
    ):
        super().__init__(hidden_dim=hidden_dim)
        if not HAS_PYG_EXTRAS:
            raise ImportError("RGCNEncoder requires torch_geometric with RGCNConv")

        self.num_node_types = num_node_types
        self._num_edge_types = num_edge_types
        self.node_embed = nn.Embedding(num_node_types, hidden_dim)

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
                      edge_type: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through RGCN layers.

        Args:
            x: [total_nodes] node type IDs (0-9)
            edge_index: [2, num_edges] edge indices
            batch: [total_nodes] batch assignment
            edge_type: [num_edges] edge type IDs (0-6) (REQUIRED)

        Returns:
            [total_nodes, hidden_dim] node embeddings
        """
        # edge_type is validated by BaseEncoder.forward()
        h = self.node_embed(x)

        for conv, norm in zip(self.convs, self.layer_norms):
            h_new = conv(h, edge_index, edge_type)
            h = norm(h + self.dropout(F.elu(h_new)))

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
