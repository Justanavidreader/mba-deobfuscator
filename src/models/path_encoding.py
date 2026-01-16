"""
Path-based edge encoding for DAGs with shared subexpressions.

Computes edge embeddings that aggregate information along all paths
between source and destination nodes, capturing structural relationships
that single-hop message passing misses.

Usage:
    from src.models.path_encoding import PathBasedEdgeEncoder

    encoder = PathBasedEdgeEncoder(hidden_dim=256)
    edge_emb = encoder(edge_index, edge_type, node_types)
"""

import torch
import torch.nn as nn
import logging
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)


class PathFinder:
    """
    Efficiently finds all paths between nodes in a DAG.

    Uses depth-limiting to handle large graphs. MBA expression DAGs
    are typically shallow (depth <= 14) with limited path counts.
    """

    def __init__(
        self,
        max_path_length: int = 6,
        max_paths: int = 16,
        warn_truncation: bool = False,
    ):
        """
        Args:
            max_path_length: Maximum path length to consider (>= 2)
            max_paths: Maximum paths to return per (src, dst) pair (>= 1)
            warn_truncation: Log when paths are truncated
        """
        if max_path_length < 2:
            raise ValueError(f"max_path_length must be >= 2, got {max_path_length}")
        if max_paths < 1:
            raise ValueError(f"max_paths must be >= 1, got {max_paths}")

        self.max_path_length = max_path_length
        self.max_paths = max_paths
        self.warn_truncation = warn_truncation

    def build_adjacency(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
    ) -> Dict[int, List[int]]:
        """
        Build adjacency list from edge index.

        Args:
            edge_index: [2, num_edges] edge source/destination indices
            num_nodes: Total number of nodes (for bounds checking)

        Returns:
            Dict mapping source node to list of destination nodes
        """
        adj: Dict[int, List[int]] = defaultdict(list)
        for i in range(edge_index.size(1)):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            # Bounds check
            if 0 <= src < num_nodes and 0 <= dst < num_nodes:
                adj[src].append(dst)
        return adj

    def find_paths(
        self,
        adj: Dict[int, List[int]],
        src: int,
        dst: int,
    ) -> List[List[int]]:
        """
        Find all paths from src to dst using DFS with depth limit.

        Args:
            adj: Adjacency list
            src: Source node index
            dst: Destination node index

        Returns:
            List of paths, each path is a list of node indices.
            Returns empty list if no paths found within limits.
        """
        if src == dst:
            return [[src]]

        paths: List[List[int]] = []
        stack: List[Tuple[int, List[int]]] = [(src, [src])]
        truncated = False

        while stack:
            if len(paths) >= self.max_paths:
                truncated = True
                break

            node, path = stack.pop()

            if len(path) > self.max_path_length:
                truncated = True
                continue

            for neighbor in adj.get(node, []):
                if neighbor in path:  # Avoid cycles (shouldn't exist in DAG)
                    continue

                new_path = path + [neighbor]

                if neighbor == dst:
                    paths.append(new_path)
                elif len(new_path) < self.max_path_length:
                    stack.append((neighbor, new_path))

        if truncated and self.warn_truncation:
            logger.debug(f"Path search truncated: src={src}, dst={dst}, found={len(paths)}")

        return paths


class PathEncoder(nn.Module):
    """
    Encodes paths as fixed-size embeddings using Transformer or LSTM.

    Processes sequences of (node_type, edge_type) pairs along each path.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_node_types: int = 10,
        num_edge_types: int = 8,
        max_path_length: int = 6,
        use_transformer: bool = True,
    ):
        """
        Args:
            hidden_dim: Output embedding dimension (must be even)
            num_node_types: Number of node types in the graph
            num_edge_types: Number of edge types in the graph
            max_path_length: Maximum path length for positional encoding
            use_transformer: Use Transformer (True) or LSTM (False)
        """
        super().__init__()
        if hidden_dim % 2 != 0:
            raise ValueError(f"hidden_dim must be even, got {hidden_dim}")

        self.hidden_dim = hidden_dim
        self.max_path_length = max_path_length
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types

        # Embeddings with +1 for padding index 0 (actual types are 1-indexed)
        self.node_type_embed = nn.Embedding(num_node_types + 1, hidden_dim // 2, padding_idx=0)
        self.edge_type_embed = nn.Embedding(num_edge_types + 1, hidden_dim // 2, padding_idx=0)
        self.position_embed = nn.Embedding(max_path_length, hidden_dim)

        if use_transformer:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=4,
                dim_feedforward=hidden_dim * 2,
                dropout=0.1,
                batch_first=True,
            )
            self.sequence_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        else:
            self.sequence_encoder = nn.LSTM(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=2,
                batch_first=True,
            )

        self.use_transformer = use_transformer
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.node_type_embed.weight, std=0.02)
        nn.init.normal_(self.edge_type_embed.weight, std=0.02)
        nn.init.normal_(self.position_embed.weight, std=0.02)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        node_types: torch.Tensor,
        edge_types: torch.Tensor,
        path_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode paths to fixed-size embeddings.

        Args:
            node_types: [batch, max_path_len] 1-indexed node types (0=pad)
            edge_types: [batch, max_path_len - 1] 1-indexed edge types (0=pad)
            path_lengths: [batch] actual length of each path

        Returns:
            [batch, hidden_dim] path embeddings
        """
        batch_size, max_len = node_types.shape
        device = node_types.device

        # Embed nodes
        node_emb = self.node_type_embed(node_types)  # [batch, path_len, hidden//2]

        # Pad edge_types to match node length (last position has no outgoing edge)
        edge_types_padded = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
        edge_types_padded[:, :-1] = edge_types
        edge_emb = self.edge_type_embed(edge_types_padded)  # [batch, path_len, hidden//2]

        # Concatenate node and edge embeddings
        path_emb = torch.cat([node_emb, edge_emb], dim=-1)  # [batch, path_len, hidden]

        # Add positional embeddings
        positions = torch.arange(max_len, device=device).unsqueeze(0).expand(batch_size, -1)
        positions = positions.clamp(max=self.max_path_length - 1)
        path_emb = path_emb + self.position_embed(positions)

        # Create attention mask for padding (True = ignore)
        mask = torch.arange(max_len, device=device).unsqueeze(0) >= path_lengths.unsqueeze(1)

        if self.use_transformer:
            encoded = self.sequence_encoder(path_emb, src_key_padding_mask=mask)
            # Mean pooling over non-padded positions
            mask_expanded = (~mask).unsqueeze(-1).float()
            pooled = (encoded * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            # LSTM: use final hidden state
            lengths_clamped = path_lengths.clamp(min=1).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(
                path_emb, lengths_clamped, batch_first=True, enforce_sorted=False
            )
            _, (hidden, _) = self.sequence_encoder(packed)
            pooled = hidden[-1]  # Last layer hidden state

        return self.output_proj(pooled)


class PathBasedEdgeEncoder(nn.Module):
    """
    Computes path-aware edge embeddings by batching all paths
    across all edges for efficient encoding.

    For each edge (u, v), finds all paths from u to v, encodes them,
    and aggregates into a single embedding. This is combined with
    the direct edge embedding to produce the final edge representation.

    Integration points:
    - Can replace or augment standard edge embeddings in GNN layers
    - Works with HGT, GGNN, and GAT encoders
    - Adds ~2M parameters for path encoding
    """

    def __init__(
        self,
        hidden_dim: int,
        num_node_types: int = 10,
        num_edge_types: int = 8,
        max_path_length: int = 6,
        max_paths: int = 16,
        aggregation: str = 'mean',
        warn_truncation: bool = False,
        use_transformer: bool = True,
    ):
        """
        Args:
            hidden_dim: Embedding dimension
            num_node_types: Number of node types
            num_edge_types: Number of edge types
            max_path_length: Maximum path length to consider
            max_paths: Maximum paths per edge
            aggregation: 'mean', 'max', or 'attention'
            warn_truncation: Log when paths are truncated
            use_transformer: Use Transformer (True) or LSTM (False) for path encoding
        """
        super().__init__()
        if aggregation not in ('mean', 'max', 'attention'):
            raise ValueError(f"aggregation must be 'mean', 'max', or 'attention', got {aggregation}")

        self.hidden_dim = hidden_dim
        self.max_path_length = max_path_length
        self.max_paths = max_paths
        self.aggregation = aggregation
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types

        self.path_finder = PathFinder(max_path_length, max_paths, warn_truncation)
        self.path_encoder = PathEncoder(
            hidden_dim=hidden_dim,
            num_node_types=num_node_types,
            num_edge_types=num_edge_types,
            max_path_length=max_path_length,
            use_transformer=use_transformer,
        )

        # Direct edge embedding (always used, combined with path embedding)
        self.direct_edge_embed = nn.Embedding(num_edge_types, hidden_dim)

        # Attention aggregation (if enabled)
        if aggregation == 'attention':
            self.attn_query = nn.Linear(hidden_dim, hidden_dim)
            self.attn_key = nn.Linear(hidden_dim, hidden_dim)

        # Combine direct and path-based embeddings
        self.combine = nn.Linear(hidden_dim * 2, hidden_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.direct_edge_embed.weight, std=0.02)
        nn.init.xavier_uniform_(self.combine.weight)
        nn.init.zeros_(self.combine.bias)

    def forward(
        self,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        node_types: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute path-aware embeddings for all edges.

        Args:
            edge_index: [2, num_edges] edge source/destination indices
            edge_type: [num_edges] edge type indices (0 to num_edge_types-1)
            node_types: [num_nodes] node type indices (0 to num_node_types-1)

        Returns:
            [num_edges, hidden_dim] edge embeddings
        """
        num_edges = edge_index.size(1)
        num_nodes = node_types.size(0)
        device = edge_index.device

        # Clamp edge types to valid range for safety
        edge_type_clamped = edge_type.clamp(0, self.num_edge_types - 1)
        direct_emb = self.direct_edge_embed(edge_type_clamped)

        if num_edges == 0:
            return direct_emb

        adj = self.path_finder.build_adjacency(edge_index, num_nodes)

        # Find all paths for all edges
        edge_paths = self._find_all_edge_paths(edge_index, adj)

        # Collect path info into batched tensors
        path_info = self._collect_path_info(
            edge_paths, edge_index, edge_type, node_types, num_nodes, device
        )

        if path_info is None:
            # No paths found for any edge - use zero path embeddings
            zeros = torch.zeros(num_edges, self.hidden_dim, device=device)
            combined = torch.cat([direct_emb, zeros], dim=-1)
            return self.combine(combined)

        all_node_types, all_edge_types, all_lengths, edge_indices, paths_per_edge = path_info

        # Batch encode all paths at once
        encoded_paths = self.path_encoder(all_node_types, all_edge_types, all_lengths)

        # Aggregate per edge
        path_emb = self._aggregate_per_edge(
            encoded_paths, edge_indices, paths_per_edge, num_edges, direct_emb, device
        )

        combined = torch.cat([direct_emb, path_emb], dim=-1)
        return self.combine(combined)

    def _find_all_edge_paths(
        self,
        edge_index: torch.Tensor,
        adj: Dict[int, List[int]],
    ) -> List[List[List[int]]]:
        """
        Find paths for each edge.

        Returns:
            List of path lists, one per edge. Each path is a list of node indices.
        """
        edge_paths: List[List[List[int]]] = []
        for i in range(edge_index.size(1)):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            paths = self.path_finder.find_paths(adj, src, dst)
            edge_paths.append(paths)
        return edge_paths

    def _collect_path_info(
        self,
        edge_paths: List[List[List[int]]],
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        node_types: torch.Tensor,
        num_nodes: int,
        device: torch.device,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int], List[int]]]:
        """
        Collect path node/edge types into batched tensors.

        Returns None if no paths found for any edge.
        """
        all_path_nodes: List[List[int]] = []
        all_path_edges: List[List[int]] = []
        all_path_lengths: List[int] = []
        edge_indices: List[int] = []  # Which edge each path belongs to
        paths_per_edge: List[int] = []

        for edge_idx, paths in enumerate(edge_paths):
            count = 0
            for path in paths:
                if not path:
                    continue

                # Node types (1-indexed for embedding, 0=pad)
                p_nodes: List[int] = []
                for n in path:
                    if 0 <= n < num_nodes:
                        nt = node_types[n].item()
                        # Clamp and 1-index
                        nt = max(0, min(nt, self.num_node_types - 1))
                        p_nodes.append(nt + 1)
                    else:
                        p_nodes.append(1)  # Default type (1-indexed)

                # Edge types (1-indexed)
                p_edges: List[int] = []
                for j in range(len(path) - 1):
                    edge_mask = (edge_index[0] == path[j]) & (edge_index[1] == path[j + 1])
                    if edge_mask.any():
                        et = edge_type[edge_mask][0].item()
                        # Clamp and 1-index
                        et = max(0, min(et, self.num_edge_types - 1))
                        p_edges.append(et + 1)
                    else:
                        p_edges.append(1)  # Default type

                # Record original length before padding
                orig_len = len(p_nodes)

                # Pad to max_path_length
                pad_len = self.max_path_length - len(p_nodes)
                p_nodes = p_nodes + [0] * pad_len
                p_edges = p_edges + [0] * (self.max_path_length - 1 - len(p_edges))

                # Truncate if necessary
                all_path_nodes.append(p_nodes[:self.max_path_length])
                all_path_edges.append(p_edges[:self.max_path_length - 1])
                all_path_lengths.append(min(orig_len, self.max_path_length))
                edge_indices.append(edge_idx)
                count += 1

            paths_per_edge.append(count)

        if not all_path_nodes:
            return None

        return (
            torch.tensor(all_path_nodes, device=device, dtype=torch.long),
            torch.tensor(all_path_edges, device=device, dtype=torch.long),
            torch.tensor(all_path_lengths, device=device, dtype=torch.long),
            edge_indices,
            paths_per_edge,
        )

    def _aggregate_per_edge(
        self,
        encoded_paths: torch.Tensor,
        edge_indices: List[int],
        paths_per_edge: List[int],
        num_edges: int,
        direct_emb: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Aggregate encoded paths back to per-edge embeddings.

        Args:
            encoded_paths: [total_paths, hidden_dim] all encoded paths
            edge_indices: Which edge each path belongs to
            paths_per_edge: Count of paths per edge
            num_edges: Total number of edges
            direct_emb: [num_edges, hidden_dim] direct edge embeddings (for attention)
            device: Device for output tensor

        Returns:
            [num_edges, hidden_dim] aggregated path embeddings
        """
        path_emb = torch.zeros(num_edges, self.hidden_dim, device=device)

        offset = 0
        for edge_idx, count in enumerate(paths_per_edge):
            if count == 0:
                continue

            edge_encodings = encoded_paths[offset:offset + count]

            if self.aggregation == 'mean':
                path_emb[edge_idx] = edge_encodings.mean(dim=0)
            elif self.aggregation == 'max':
                path_emb[edge_idx] = edge_encodings.max(dim=0)[0]
            elif self.aggregation == 'attention':
                # Attention-weighted aggregation using direct embedding as query
                query = self.attn_query(direct_emb[edge_idx:edge_idx + 1])  # [1, hidden]
                keys = self.attn_key(edge_encodings)  # [count, hidden]
                attn_scores = torch.matmul(query, keys.T) / (self.hidden_dim ** 0.5)
                attn_weights = torch.softmax(attn_scores, dim=-1)  # [1, count]
                path_emb[edge_idx] = torch.matmul(attn_weights, edge_encodings).squeeze(0)

            offset += count

        return path_emb
