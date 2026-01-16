# Implementation Plan: Path-Based Edge Encoding

**Status: IMPLEMENTING** (2024-01-17)

## Overview

Add path-based edge encoding to capture relationships in MBA expression DAGs where subexpressions are shared. When the same subexpression `(a & b)` appears multiple times, standard message passing treats each occurrence independently. Path-based encoding aggregates information along all paths between nodes, enabling the model to recognize structural patterns across shared subexpressions.

## Quality Review Fixes Applied

Based on quality-reviewer feedback (CRITICAL_ISSUES):

1. **Bounds checking for node types**: Added validation in `_collect_path_info` to prevent index out of bounds
2. **Empty path handling**: `forward()` now returns direct embeddings when no paths found (no tensor stack crash)
3. **Path truncation transparency**: Added `warn_truncation` parameter; truncation logged if enabled
4. **Edge type validation**: Added bounds checking before embedding lookup
5. **Batched encoding**: Rewrote forward pass to batch all paths across all edges, not per-edge sequential
6. **Refactored God Function**: Split 93-line `forward()` into smaller focused methods
7. **Consistent error handling**: ValueError for config errors, explicit fallback for runtime edge cases

## Motivation

MBA obfuscation frequently introduces shared subexpressions:

```
Original: x | y
Obfuscated: (x & y) + (x ^ y) + 2*((x & y) & ~(x & y))
                ^                      ^
                └──── shared (x & y) ────┘
```

Current GNN encoders process edges independently. Path-based encoding enables:
1. Recognition that distant nodes share structural relationships via common ancestors
2. Better detection of identity patterns like `(x & ~x)` across expression tree
3. Improved semantic similarity detection between isomorphic subgraphs

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  PATH-BASED EDGE ENCODING PIPELINE                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input: Edge (u, v) with edge_type                              │
│                        │                                        │
│                        ▼                                        │
│  ┌─────────────────────────────────────┐                        │
│  │  1. Find All Paths (u → v)          │                        │
│  │     - BFS/DFS with depth limit      │                        │
│  │     - Cache paths for efficiency    │                        │
│  └─────────────────────────────────────┘                        │
│                        │                                        │
│                        ▼                                        │
│  ┌─────────────────────────────────────┐                        │
│  │  2. Encode Each Path                │                        │
│  │     - Sequence of edge types        │                        │
│  │     - Sequence of node types        │                        │
│  │     - Path length encoding          │                        │
│  └─────────────────────────────────────┘                        │
│                        │                                        │
│                        ▼                                        │
│  ┌─────────────────────────────────────┐                        │
│  │  3. Aggregate Path Encodings        │                        │
│  │     - Mean pooling (default)        │
│  │     - Attention-weighted pooling    │                        │
│  │     - Max pooling                   │                        │
│  └─────────────────────────────────────┘                        │
│                        │                                        │
│                        ▼                                        │
│  Output: Path-aware edge embedding [hidden_dim]                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Implementation Details

### 1. New Module: `src/models/path_encoding.py`

Key design changes from quality review:

1. **Batched path encoding**: All paths across all edges are collected first, then encoded in a single batch
2. **Explicit bounds checking**: Node indices validated before access
3. **Empty path fallback**: Returns direct embeddings instead of crashing
4. **Refactored into focused methods**: `_find_all_edge_paths()`, `_collect_path_info()`, `_batch_encode_paths()`, `_aggregate_per_edge()`
5. **Edge type bounds checking**: Validates edge types before embedding lookup

```python
"""
Path-based edge encoding for DAGs with shared subexpressions.

Computes edge embeddings that aggregate information along all paths
between source and destination nodes, capturing structural relationships
that single-hop message passing misses.
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
    are typically shallow (depth ≤ 14) with limited path counts.
    """

    def __init__(
        self,
        max_path_length: int = 6,
        max_paths: int = 16,
        warn_truncation: bool = False,
    ):
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
        """Build adjacency list from edge index."""
        adj = defaultdict(list)
        for i in range(edge_index.size(1)):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
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

        Returns:
            List of paths, each path is a list of node indices.
            Returns empty list if no paths found within limits.
        """
        if src == dst:
            return [[src]]

        paths = []
        stack = [(src, [src])]
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
                if neighbor in path:  # Avoid cycles
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
    """

    def __init__(
        self,
        hidden_dim: int,
        num_node_types: int = 10,
        num_edge_types: int = 8,
        max_path_length: int = 6,
        use_transformer: bool = True,
    ):
        super().__init__()
        if hidden_dim % 2 != 0:
            raise ValueError(f"hidden_dim must be even, got {hidden_dim}")

        self.hidden_dim = hidden_dim
        self.max_path_length = max_path_length
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types

        # Embeddings (with +1 for padding index 0)
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
        node_types: torch.Tensor,   # [batch, max_path_len], 1-indexed (0=pad)
        edge_types: torch.Tensor,   # [batch, max_path_len - 1], 1-indexed (0=pad)
        path_lengths: torch.Tensor, # [batch]
    ) -> torch.Tensor:
        """Encode paths. Returns [batch, hidden_dim]."""
        batch_size, max_len = node_types.shape
        device = node_types.device

        node_emb = self.node_type_embed(node_types)

        edge_types_padded = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
        edge_types_padded[:, :-1] = edge_types
        edge_emb = self.edge_type_embed(edge_types_padded)

        path_emb = torch.cat([node_emb, edge_emb], dim=-1)

        positions = torch.arange(max_len, device=device).unsqueeze(0).expand(batch_size, -1)
        positions = positions.clamp(max=self.max_path_length - 1)
        path_emb = path_emb + self.position_embed(positions)

        mask = torch.arange(max_len, device=device).unsqueeze(0) >= path_lengths.unsqueeze(1)

        if self.use_transformer:
            encoded = self.sequence_encoder(path_emb, src_key_padding_mask=mask)
            mask_expanded = (~mask).unsqueeze(-1).float()
            pooled = (encoded * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            lengths_clamped = path_lengths.clamp(min=1).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(
                path_emb, lengths_clamped, batch_first=True, enforce_sorted=False
            )
            _, (hidden, _) = self.sequence_encoder(packed)
            pooled = hidden[-1]

        return self.output_proj(pooled)


class PathBasedEdgeEncoder(nn.Module):
    """
    Computes path-aware edge embeddings by batching all paths
    across all edges for efficient encoding.
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
    ):
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
        )

        self.direct_edge_embed = nn.Embedding(num_edge_types, hidden_dim)

        if aggregation == 'attention':
            self.attn_query = nn.Linear(hidden_dim, hidden_dim)
            self.attn_key = nn.Linear(hidden_dim, hidden_dim)

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
        num_edges = edge_index.size(1)
        num_nodes = node_types.size(0)
        device = edge_index.device

        # Clamp edge types to valid range
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
            # No paths found for any edge
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
        """Find paths for each edge. Returns list of path lists per edge."""
        edge_paths = []
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
        Returns None if no paths found.
        """
        all_path_nodes = []
        all_path_edges = []
        all_path_lengths = []
        edge_indices = []  # Which edge each path belongs to
        paths_per_edge = []

        for edge_idx, paths in enumerate(edge_paths):
            count = 0
            for path in paths:
                if not path:
                    continue

                # Node types (1-indexed for embedding, 0=pad)
                p_nodes = []
                for n in path:
                    if 0 <= n < num_nodes:
                        p_nodes.append(node_types[n].item() + 1)  # +1 for 1-indexing
                    else:
                        p_nodes.append(1)  # Default to type 0 (1-indexed)

                # Edge types (1-indexed)
                p_edges = []
                for j in range(len(path) - 1):
                    edge_mask = (edge_index[0] == path[j]) & (edge_index[1] == path[j + 1])
                    if edge_mask.any():
                        et = edge_type[edge_mask][0].item()
                        et = max(0, min(et, self.num_edge_types - 1))
                        p_edges.append(et + 1)
                    else:
                        p_edges.append(1)

                # Pad
                orig_len = len(p_nodes)
                pad_len = self.max_path_length - orig_len
                p_nodes = p_nodes + [0] * pad_len
                p_edges = p_edges + [0] * (self.max_path_length - 1 - len(p_edges))

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
        """Aggregate encoded paths back to per-edge embeddings."""
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
                query = self.attn_query(direct_emb[edge_idx:edge_idx + 1])
                keys = self.attn_key(edge_encodings)
                attn_scores = torch.matmul(query, keys.T) / (self.hidden_dim ** 0.5)
                attn_weights = torch.softmax(attn_scores, dim=-1)
                path_emb[edge_idx] = torch.matmul(attn_weights, edge_encodings).squeeze(0)

            offset += count

        return path_emb
```

### 2. Integration with Encoders

**Option A: Augment edge features in message passing**

```python
# In GGNNEncoder._forward_impl
def _forward_impl(self, x, edge_index, batch, edge_type=None):
    # Compute path-aware edge embeddings
    if self.use_path_encoding:
        edge_emb = self.path_encoder(edge_index, edge_type, x)
    else:
        edge_emb = self.edge_embed(edge_type)

    # Use edge_emb in message passing...
```

**Option B: Add as separate feature to node representations**

```python
# Compute path-based features and add to node embeddings
path_features = aggregate_incoming_path_features(node_idx, edge_index, edge_emb)
h = h + self.path_projection(path_features)
```

### 3. Update `src/constants.py`

```python
# Path-based edge encoding
PATH_ENCODING_ENABLED: bool = False  # Default off
PATH_MAX_LENGTH: int = 6             # Max path length to consider
PATH_MAX_PATHS: int = 16             # Max paths per edge
PATH_AGGREGATION: str = 'mean'       # 'mean', 'max', 'attention'
PATH_USE_TRANSFORMER: bool = True    # Use Transformer vs LSTM for path encoding
```

### 4. Testing

```python
# tests/test_path_encoding.py

def test_path_finder_simple_dag():
    """Test path finding on simple DAG."""
    # DAG: 0 -> 1 -> 3
    #      0 -> 2 -> 3
    edge_index = torch.tensor([[0, 0, 1, 2], [1, 2, 3, 3]])
    finder = PathFinder(max_path_length=4, max_paths=10)
    adj = finder.build_adjacency(edge_index, num_nodes=4)

    paths = finder.find_paths(adj, 0, 3)
    assert len(paths) == 2
    assert [0, 1, 3] in paths
    assert [0, 2, 3] in paths


def test_path_encoder_shape():
    """Test path encoder output shape."""
    encoder = PathEncoder(hidden_dim=64, num_node_types=10, num_edge_types=8)

    node_types = torch.randint(0, 10, (4, 6))  # 4 paths, max len 6
    edge_types = torch.randint(0, 8, (4, 5))   # 5 edges per path
    path_lengths = torch.tensor([3, 4, 5, 6])

    out = encoder(node_types, edge_types, path_lengths)
    assert out.shape == (4, 64)


def test_path_edge_encoder_integration():
    """Test full path-based edge encoding."""
    encoder = PathBasedEdgeEncoder(hidden_dim=64)

    # Simple graph
    edge_index = torch.tensor([[0, 0, 1, 2], [1, 2, 3, 3]])
    edge_type = torch.tensor([0, 1, 2, 3])
    node_types = torch.tensor([0, 1, 2, 3])

    edge_emb = encoder(edge_index, edge_type, node_types)
    assert edge_emb.shape == (4, 64)
```

## Task Breakdown

| Task | File | Status | Priority |
|------|------|--------|----------|
| Create PathFinder class | `src/models/path_encoding.py` | DONE | P0 |
| Create PathEncoder module | `src/models/path_encoding.py` | DONE | P0 |
| Create PathBasedEdgeEncoder | `src/models/path_encoding.py` | DONE | P0 |
| Add path encoding constants | `src/constants.py` | DONE | P0 |
| Write unit tests | `tests/test_path_encoding.py` | DONE | P0 |
| Integrate with GGNNEncoder | `src/models/encoder.py` | TODO | P1 |
| Integrate with HGTEncoder | `src/models/encoder.py` | TODO | P1 |
| Benchmark performance impact | - | TODO | P2 |
| Add caching for path computation | `src/models/path_encoding.py` | TODO | P2 |

## Complexity Analysis

**Time Complexity:**
- Path finding: O(V + E) per edge with depth limit, O(E * (V + E)) total
- Path encoding: O(num_paths * path_length) per edge
- With caching: Amortized O(1) for repeated queries

**Space Complexity:**
- Adjacency list: O(E)
- Path cache: O(V² * max_paths * max_path_length) worst case
- Practical: Much smaller due to DAG structure and depth limits

**MBA Expression Graphs:**
- Typical: 50-200 nodes, 100-400 edges
- Max depth: 14 (curriculum stage 4)
- Expected paths per edge: 1-4 (due to tree-like structure)
- Path computation overhead: ~10-50ms per batch

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Exponential paths in dense graphs | OOM/timeout | max_paths limit, depth limit |
| Slow path computation | Training bottleneck | Caching, batch parallelization |
| Overfitting on path patterns | Poor generalization | Dropout on path embeddings, regularization |
| Memory overhead | OOM on large batches | Lazy computation, gradient checkpointing |

## Success Criteria

1. Unit tests pass for PathFinder, PathEncoder, PathBasedEdgeEncoder
2. Path computation adds <20% overhead to forward pass
3. Model with path encoding achieves ≥2% accuracy improvement on depth 10+ expressions
4. Memory usage increase <500MB for batch_size=32

## Rollback Plan

Path encoding is opt-in via `PATH_ENCODING_ENABLED=False` default. No changes to existing encoder behavior unless explicitly enabled.

## Future Enhancements (P3+)

1. **Learnable path importance**: Weight paths by learned relevance scores
2. **Hierarchical path encoding**: Encode subpaths and compose
3. **Cross-attention between paths**: Allow paths to attend to each other
4. **Path-based contrastive learning**: Use path similarity in Phase 1 pretraining
