# Implementation Plan: DAG Positional Features for GNN Encoders

**Status: PLANNING** (2026-01-17)
**Revision: 4** (Addresses quality review findings)

## Overview

Add DAG-aware positional features to GNN encoders to capture structural properties unique to DAG (Directed Acyclic Graph) representations of MBA expressions. Unlike tree positional encoding which only considers depth, DAG features capture shared subexpression patterns through metrics like `in_degree` and `is_shared`.

## Motivation

Current positional encoding assumes tree structure:
- `depth` is computed as simple tree depth
- No awareness of shared subexpressions (DAG structure)
- Missing signal about node reachability and importance

With DAG positional features:
- **Shared subexpressions are explicit**: `is_shared=1.0` when node has multiple parents
- **Structural importance captured**: `in_degree` indicates how many operations depend on this node
- **Subtree size for pruning heuristics**: Large subtrees are harder to simplify
- **Normalized features**: All values in [0, 1] for stable training

**Example:**
```
Expression: (x & y) + (x & y)

Tree view:          DAG view (with sharing):
     +                   +
   /   \               /   \
  &     &    -->      &     |
 / \   / \           / \    |
x   y x   y         x   y   |
                     \______/  (& node has in_degree=2, is_shared=1.0)
```

## Planning Context

### Decision Log

| Decision | Rationale |
|----------|-----------|
| Use Kahn's topological sort for depth | Topological sort ensures depth[node] is computed only after all children are processed, making depth deterministic for DAGs with shared nodes. |
| Return cached subtree_size for visited nodes | Semantically correct: each parent's computation depends on full shared subtree. Alternative (return 0) undercounts dependencies. |
| Helper function decomposition (5 functions) | Enables isolated unit testing and profiling. Improves maintainability vs monolithic 150-line function. |
| Opt-in via use_dag_features=False | Backward compatibility requirement. Existing models continue working unchanged. |
| Concat fusion for HGT, residual for GGNN | HGT has large hidden_dim (768), can afford concat. GGNN has smaller hidden_dim (256), residual preserves capacity. |
| Explicit cycle detection with ValueError | Fail-fast on malformed input. Error message lists unprocessed nodes and suggests checking graph construction. |
| Clamp normalization denominators to minimum 1 | Prevents division by zero for single-node graphs (depth=0, in_degree=0). Ensures empty features map to 0.0 consistently. |
| Identity projection in concat fusion | Linear layer (4×4=16 weights) learns adaptive scaling of DAG features vs node embeddings. Adds flexibility with negligible parameter cost. |
| Tanh activation for residual DAG projector | Bounds residual contribution to [-1, 1], preventing DAG features from overwhelming node embeddings. More stable than unbounded activations. |
| Leaf detection via node_types ∈ {0, 1} | Encoding from constants.py: VAR=0, CONST=1 are terminal nodes (out_degree=0). Operators have node_types ≥ 2. |
| Device consistency validation | Explicit check that edge_index and node_types are on same device. Prevents cryptic tensor operation errors deep in computation. |

### Rejected Alternatives

| Alternative | Why Rejected |
|-------------|--------------|
| BFS for depth computation | Non-deterministic when node has multiple parents. Depth depends on which parent visits first. |
| Return 0 for visited nodes in subtree DFS | Semantically incorrect. Shared node contributes to ALL parents, not just first visitor. |
| Monolithic feature computation function | Hard to test, profile, and maintain. |
| Always-on DAG features | Breaking change for existing models. Requires retraining all checkpoints. |
| Silent handling of cyclic graphs | Produces incorrect depth=0 without alerting caller. Hard to debug. |
| Wrapper function compute_dag_features_from_ast() | Redundant with expr_to_optimized_graph(use_dag_features=True). |

### Constraints & Assumptions

- **Constraint**: Must maintain BaseEncoder interface compatibility (all encoders accept dag_pos)
- **Constraint**: Backward compatibility required (default use_dag_features=False)
- **Constraint**: All features must normalize to [0, 1] for training stability
- **Assumption**: MBA expression graphs are always DAGs (cycles are malformed input)
- **Assumption**: Shared subexpressions are common enough to justify 4 extra features per node
- **Assumption**: Feature computation overhead (<5ms) is negligible vs GNN forward pass (50ms)

### Known Risks (Acknowledged)

| Risk | Mitigation | Status |
|------|------------|--------|
| Concat fusion reduces embedding capacity | Ablate concat vs residual; residual is fallback | Mitigated |
| DAG features don't improve accuracy | Opt-in design allows easy rollback; ablation study planned | Mitigated |
| Feature computation slow on large graphs | Profile; consider preprocessing cache if needed | Accepted |

## Architecture

### Feature Computation Flow

```
+---------------------------------------------------------------------+
|  DAG POSITIONAL FEATURE PIPELINE                                    |
+---------------------------------------------------------------------+
|                                                                     |
|  Input: ASTNode tree + edge_index from graph construction           |
|                        |                                            |
|                        v                                            |
|  +------------------------------------------------------------------+
|  |  compute_dag_positional_features()                               |
|  |  Orchestrates 4 helper functions:                                |
|  |                                                                  |
|  |  1. _build_adjacency()      -> children, parents dicts           |
|  |  2. _compute_in_degrees()   -> in_degrees tensor, max_in_degree  |
|  |  3. _compute_depths_topo()  -> depths tensor, max_depth          |
|  |  4. _compute_subtree_sizes()-> subtree_sizes tensor              |
|  |                                                                  |
|  |  5. _normalize_to_unit_range() for all features                  |
|  |  6. Stack: [depth, subtree_size, in_degree, is_shared]           |
|  +------------------------------------------------------------------+
|                        |                                            |
|                        v                                            |
|  [num_nodes, 4] DAG positional features                             |
|                        |                                            |
|                        v                                            |
|  +------------------------------------------------------------------+
|  |  Encoder Integration                                             |
|  |                                                                  |
|  |  Option A: Concatenate with node embedding (HGT default)         |
|  |     node_emb [hidden_dim-4] || dag_pos [4] -> [hidden_dim]       |
|  |     Requires: hidden_dim > dag_feature_dim                       |
|  |                                                                  |
|  |  Option B: Residual addition after projection (GGNN default)     |
|  |     node_emb [hidden_dim] + MLP(dag_pos) [hidden_dim]            |
|  |     No dimension constraint                                      |
|  +------------------------------------------------------------------+
|                        |                                            |
|                        v                                            |
|  Enhanced node embeddings passed to GNN layers                      |
|                                                                     |
+---------------------------------------------------------------------+
```

### Integration Strategy

**Option A: Early Fusion (RECOMMENDED for HGT)**

Concatenate DAG features with node type embedding, then use full hidden_dim.

```python
# In HGTEncoder.__init__
if use_dag_features and dag_fusion_method == 'concat':
    node_embed_dim = hidden_dim - dag_feature_dim
    # CRITICAL: Validate dimension constraint
    assert node_embed_dim > 0, (
        f"Concat fusion requires hidden_dim ({hidden_dim}) > dag_feature_dim ({dag_feature_dim}). "
        f"Current node_embed_dim = {node_embed_dim}. "
        f"Either increase hidden_dim or use dag_fusion_method='residual'."
    )
    self.node_type_embed = nn.Embedding(num_node_types, node_embed_dim)
    self.dag_projector = nn.Linear(dag_feature_dim, dag_feature_dim)

# In forward
node_type_emb = self.node_type_embed(x)  # [num_nodes, hidden_dim - dag_dim]
dag_features = self.dag_projector(dag_pos)  # [num_nodes, dag_dim]
h = torch.cat([node_type_emb, dag_features], dim=-1)  # [num_nodes, hidden_dim]
```

**Pros:**
- DAG structure visible from first layer
- Simple implementation
- No extra parameters for fusion

**Cons:**
- Reduces node type embedding dimension
- Requires hidden_dim > dag_feature_dim (validated at init)

**Option B: Residual Addition (RECOMMENDED for GGNN)**

Project DAG features to hidden_dim and add as residual after node embedding.

```python
# In GGNNEncoder.__init__
self.node_embedding = nn.Linear(node_dim, hidden_dim)
self.dag_projector = nn.Sequential(
    nn.Linear(dag_dim, hidden_dim),
    nn.LayerNorm(hidden_dim),
    nn.Tanh(),
)

# In forward
h = self.node_embedding(x)  # [num_nodes, hidden_dim]
h = F.elu(h)
h = h + self.dag_projector(dag_pos)  # Residual addition
```

**Pros:**
- Preserves full node embedding dimension
- No dimension constraint
- Residual connection prevents disrupting existing learned features
- Easy to ablate (can scale residual down)

**Cons:**
- Additional projection parameters

## Implementation Details

### 1. Core Feature Computation

**File:** `src/data/dag_features.py` (NEW FILE)

```python
"""
DAG positional feature computation for MBA expression graphs.

Captures structural properties unique to DAG representations:
- depth: Distance from leaves (normalized)
- subtree_size: Number of descendant nodes (normalized)
- in_degree: Number of incoming edges (normalized)
- is_shared: Binary indicator for shared subexpressions (in_degree > 1)

Design decisions:
- Uses topological sort for deterministic depth computation on DAGs
- Subtree size correctly handles shared nodes (each parent gets full subtree count)
- All features normalized to [0, 1] for training stability
- Helper functions are separated for testability and profiling
"""

import torch
from typing import Dict, List, Tuple, Union
from collections import defaultdict


# =============================================================================
# HELPER FUNCTIONS (separated for testability and profiling)
# =============================================================================

def _build_adjacency(
    edge_index: torch.Tensor,
    num_nodes: int,
) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    """
    Build forward (children) and reverse (parents) adjacency lists.

    Args:
        edge_index: [2, num_edges] edge indices (source -> dest)
        num_nodes: Total number of nodes

    Returns:
        children: node -> [child1, child2, ...] (forward edges)
        parents: node -> [parent1, parent2, ...] (reverse edges)
    """
    children: Dict[int, List[int]] = defaultdict(list)
    parents: Dict[int, List[int]] = defaultdict(list)

    for i in range(edge_index.size(1)):
        src, dst = edge_index[0, i].item(), edge_index[1, i].item()
        children[src].append(dst)
        parents[dst].append(src)

    return children, parents


def _compute_in_degrees(
    num_nodes: int,
    parents: Dict[int, List[int]],
    device: torch.device,
) -> Tuple[torch.Tensor, int]:
    """
    Compute in-degree for each node.

    Args:
        num_nodes: Total number of nodes
        parents: node -> [parent1, parent2, ...] adjacency
        device: Torch device for output tensor

    Returns:
        in_degrees: [num_nodes] tensor of in-degree counts
        max_in_degree: Maximum in-degree (for normalization)
    """
    in_degrees = torch.zeros(num_nodes, dtype=torch.long, device=device)

    for node in range(num_nodes):
        in_degrees[node] = len(parents[node])

    max_in_degree = in_degrees.max().item() if num_nodes > 0 else 1
    return in_degrees, max(max_in_degree, 1)


def _compute_depths_topological(
    num_nodes: int,
    children: Dict[int, List[int]],
    parents: Dict[int, List[int]],
    node_types: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, int]:
    """
    Compute depth using reverse topological order (bottom-up from leaves).

    Topological sort ensures depth[node] is computed only after all children
    are processed. This makes depth deterministic for DAGs with shared nodes:
    depth[node] = 1 + max(depth[child]), independent of parent visit order.

    Algorithm:
    1. Identify leaf nodes (terminals: VAR=0, CONST=1)
    2. Topological sort using Kahn's algorithm (queue from leaves)
    3. Process nodes in topological order: depth[node] = 1 + max(depth[child])
    4. Leaves have depth 0, root has maximum depth
    5. Detect cycles: unprocessed nodes indicate malformed graph

    Args:
        num_nodes: Total number of nodes
        children: node -> [child1, child2, ...] adjacency
        parents: node -> [parent1, parent2, ...] adjacency
        node_types: [num_nodes] node type IDs (for identifying leaves)
        device: Torch device for output tensor

    Returns:
        depths: [num_nodes] tensor of depth values
        max_depth: Maximum depth (for normalization)

    Raises:
        ValueError: If graph contains cycles (unprocessed nodes after sort)
    """
    depths = torch.zeros(num_nodes, dtype=torch.long, device=device)

    if num_nodes == 0:
        return depths, 1

    # Identify leaves (terminals: VAR=0, CONST=1)
    is_leaf = (node_types == 0) | (node_types == 1)

    # Compute out-degree (number of children) for topological sort
    out_degree = torch.zeros(num_nodes, dtype=torch.long, device=device)
    for node in range(num_nodes):
        out_degree[node] = len(children[node])

    # Initialize: leaves have depth 0 and out_degree 0
    # Use Kahn's algorithm from leaves (nodes with out_degree=0)
    from collections import deque

    # Queue nodes with no children (leaves in expression tree)
    queue = deque()
    for node in range(num_nodes):
        if out_degree[node] == 0:
            queue.append(node)
            depths[node] = 0

    # Track remaining children to process per node
    remaining_children = out_degree.clone()

    # Process in topological order (leaves first, root last)
    processed = set()
    while queue:
        node = queue.popleft()

        if node in processed:
            continue
        processed.add(node)

        # Compute depth as 1 + max(child depths)
        if children[node]:
            child_depths = [depths[c].item() for c in children[node]]
            depths[node] = max(child_depths) + 1
        else:
            depths[node] = 0  # Leaf

        # Decrement remaining children count for all parents
        for parent in parents[node]:
            remaining_children[parent] -= 1
            if remaining_children[parent] == 0:
                queue.append(parent)

    # CRITICAL: Detect cycles (unprocessed nodes indicate cycle or malformed graph)
    unprocessed = [n for n in range(num_nodes) if n not in processed]
    if unprocessed:
        raise ValueError(
            f"Graph contains cycles or disconnected components. "
            f"Unprocessed nodes: {unprocessed[:10]}{'...' if len(unprocessed) > 10 else ''}. "
            f"DAG positional features require acyclic graphs. "
            f"Check input graph construction for errors."
        )

    max_depth = depths.max().item() if num_nodes > 0 else 1
    return depths, max(max_depth, 1)


def _compute_subtree_sizes(
    num_nodes: int,
    children: Dict[int, List[int]],
    parents: Dict[int, List[int]],
    device: torch.device,
) -> torch.Tensor:
    """
    Compute subtree size for each node using DFS from roots.

    CRITICAL: For DAGs with shared nodes, each parent's subtree size includes
    the FULL subtree of shared children. This is semantically correct because
    the shared subtree contributes to the computation of ALL its parents.

    The memoization returns cached subtree_size for already-visited nodes,
    which is added to each parent's count. This ensures:
    - No infinite loops on DAGs
    - Each parent gets credit for its entire computational subtree
    - Shared nodes are counted multiple times in ancestor subtree sizes

    Example: (a + b) + (a + b) where ADD is shared
    - Shared ADD has subtree_size=3 (itself + a + b)
    - Root ADD has subtree_size=1 + 3 + 3 = 7 (not 4)
    - This correctly reflects that root depends on all 7 node computations

    Args:
        num_nodes: Total number of nodes
        children: node -> [child1, child2, ...] adjacency
        parents: node -> [parent1, parent2, ...] adjacency
        device: Torch device for output tensor

    Returns:
        subtree_sizes: [num_nodes] tensor of subtree sizes
    """
    subtree_sizes = torch.zeros(num_nodes, dtype=torch.long, device=device)
    visited = set()

    def count_descendants(node: int) -> int:
        """Recursively count descendants including shared nodes."""
        if node in visited:
            # CRITICAL: Return cached size for shared nodes
            # This ensures each parent gets the full subtree contribution
            return subtree_sizes[node].item()

        visited.add(node)
        count = 1  # Count self

        for child in children[node]:
            count += count_descendants(child)

        subtree_sizes[node] = count
        return count

    # Find root nodes (nodes with no parents)
    root_nodes = [n for n in range(num_nodes) if len(parents[n]) == 0]

    # DFS from each root
    for root in root_nodes:
        count_descendants(root)

    # Handle orphaned nodes (shouldn't happen in well-formed graphs)
    for node in range(num_nodes):
        if node not in visited:
            count_descendants(node)

    return subtree_sizes


def _normalize_to_unit_range(
    feature: torch.Tensor,
    max_val: Union[int, float],
) -> torch.Tensor:
    """
    Normalize feature to [0, 1] range using max value.

    Args:
        feature: Input tensor to normalize
        max_val: Maximum value for normalization (clamped to >= 1)

    Returns:
        Normalized tensor with values in [0, 1]
    """
    return feature.float() / max(float(max_val), 1.0)


# =============================================================================
# MAIN API
# =============================================================================

def compute_dag_positional_features(
    num_nodes: int,
    edge_index: torch.Tensor,
    node_types: torch.Tensor,
) -> torch.Tensor:
    """
    Compute DAG positional features for all nodes.

    Orchestrates helper functions to compute 4 structural features:
    - depth: Distance from leaves, computed via topological sort (deterministic for DAGs)
    - subtree_size: Number of descendant nodes (shared nodes contribute to all parents)
    - in_degree: Number of incoming edges (indicates sharing)
    - is_shared: Binary flag (1.0 if in_degree > 1)

    Args:
        num_nodes: Total number of nodes in graph
        edge_index: [2, num_edges] edge indices (source -> dest)
        node_types: [num_nodes] node type IDs (0=VAR, 1=CONST for leaf detection)

    Returns:
        [num_nodes, 4] tensor with columns [depth, subtree_size, in_degree, is_shared]
        All features normalized to [0, 1] except is_shared which is binary.

    Raises:
        ValueError: If edge_index and node_types are on different devices
        ValueError: If graph contains cycles (detected via topological sort)
    """
    # Validate device consistency to prevent cryptic tensor operation errors
    if edge_index.numel() > 0 and edge_index.device != node_types.device:
        raise ValueError(
            f"edge_index and node_types must be on same device. "
            f"Got edge_index on {edge_index.device}, node_types on {node_types.device}. "
            f"Move tensors to same device before calling this function."
        )

    device = edge_index.device if edge_index.numel() > 0 else node_types.device

    # Handle empty graph
    if num_nodes == 0:
        return torch.zeros((0, 4), dtype=torch.float, device=device)

    # Step 1: Build adjacency lists
    children, parents = _build_adjacency(edge_index, num_nodes)

    # Step 2: Compute in-degrees
    in_degrees, max_in_degree = _compute_in_degrees(num_nodes, parents, device)

    # Step 3: Compute depths via topological sort
    depths, max_depth = _compute_depths_topological(
        num_nodes, children, parents, node_types, device
    )

    # Step 4: Compute subtree sizes via DFS
    subtree_sizes = _compute_subtree_sizes(num_nodes, children, parents, device)

    # Step 5: Normalize all features to [0, 1]
    depth_norm = _normalize_to_unit_range(depths, max_depth)
    subtree_size_norm = _normalize_to_unit_range(subtree_sizes, num_nodes)
    in_degree_norm = _normalize_to_unit_range(in_degrees, max_in_degree)
    is_shared = (in_degrees > 1).float()

    # Step 6: Stack features [num_nodes, 4]
    dag_features = torch.stack([
        depth_norm,
        subtree_size_norm,
        in_degree_norm,
        is_shared,
    ], dim=1)

    return dag_features
```

### 2. Integration with Graph Construction

**File:** `src/data/ast_parser.py` (MODIFICATIONS)

```python
# Add import at top
from src.data.dag_features import compute_dag_positional_features

def ast_to_optimized_graph(ast: ASTNode, use_dag_features: bool = False) -> Data:
    """
    Convert AST to PyTorch Geometric Data with optimized 8-type edge system.

    Args:
        ast: Root ASTNode
        use_dag_features: If True, compute and attach DAG positional features

    Returns:
        PyTorch Geometric Data object with:
        - x: [num_nodes] node type IDs
        - edge_index: [2, num_edges] edge connectivity
        - edge_type: [num_edges] edge type IDs (0-7)
        - dag_pos: [num_nodes, 4] DAG features (only if use_dag_features=True)
    """
    # ... existing node collection and edge construction ...

    # Convert to tensors
    if edge_list:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_type = torch.tensor(edge_types, dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_type = torch.zeros((0,), dtype=torch.long)

    data = Data(x=node_type_tensor, edge_index=edge_index, edge_type=edge_type)

    # Compute DAG positional features if requested
    if use_dag_features:
        dag_pos = compute_dag_positional_features(
            num_nodes=num_nodes,
            edge_index=edge_index,
            node_types=node_type_tensor,
        )
        data.dag_pos = dag_pos

    return data


def expr_to_optimized_graph(expr: str, use_dag_features: bool = False) -> Data:
    """
    Convenience function: expression string to optimized graph.

    Primary API for converting expressions to graphs with optional DAG features.

    Args:
        expr: Expression string
        use_dag_features: If True, compute and attach DAG positional features

    Returns:
        PyTorch Geometric Data object with optimized edge types and optional dag_pos
    """
    ast = parse_to_ast(expr)
    return ast_to_optimized_graph(ast, use_dag_features=use_dag_features)
```

### 3. BaseEncoder Interface Update

**File:** `src/models/encoder_base.py` (MODIFICATIONS)

CRITICAL: Both `forward()` AND `_forward_impl()` must accept `dag_pos` parameter.
All encoder implementations must update their signatures even if they ignore the parameter.

```python
from abc import ABC, abstractmethod
from typing import Optional
import torch
import torch.nn as nn


class BaseEncoder(ABC, nn.Module):
    """
    Base class for all GNN encoders.

    All encoders must implement _forward_impl with the full signature including
    optional dag_pos parameter. Encoders that don't use DAG features should
    accept and ignore the parameter for interface compatibility.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

    @property
    def requires_edge_types(self) -> bool:
        """Override to True if encoder needs edge_type parameter."""
        return False

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        edge_type: Optional[torch.Tensor] = None,
        dag_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode graph to node embeddings.

        Args:
            x: Node features (format depends on encoder)
            edge_index: [2, num_edges] edge connectivity
            batch: [num_nodes] batch assignment
            edge_type: [num_edges] edge type IDs (required for heterogeneous GNNs)
            dag_pos: [num_nodes, 4] DAG positional features (optional)
                     Columns: [depth, subtree_size, in_degree, is_shared]

        Returns:
            [num_nodes, hidden_dim] node embeddings
        """
        if self.requires_edge_types and edge_type is None:
            raise ValueError(
                f"{self.__class__.__name__} requires edge_type parameter"
            )

        return self._forward_impl(x, edge_index, batch, edge_type, dag_pos)

    @abstractmethod
    def _forward_impl(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        edge_type: Optional[torch.Tensor],
        dag_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Implementation-specific forward pass.

        Args:
            x: Node features
            edge_index: Edge connectivity
            batch: Batch assignment
            edge_type: Edge types (may be None for homogeneous encoders)
            dag_pos: DAG positional features (may be None if not using DAG features)
                     Encoders not using DAG features should accept and ignore this.

        Returns:
            Node embeddings [num_nodes, hidden_dim]
        """
        pass
```

### 4. Encoder Integration: HGTEncoder

**File:** `src/models/encoder.py` (MODIFICATIONS)

```python
class HGTEncoder(BaseEncoder):
    """
    Heterogeneous Graph Transformer encoder with optional DAG positional features.
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 16,
        num_node_types: int = 10,
        num_edge_types: int = 8,
        dropout: float = 0.1,
        # DAG feature parameters
        use_dag_features: bool = False,
        dag_feature_dim: int = 4,
        dag_fusion_method: str = 'concat',  # 'concat' or 'residual'
        **kwargs,
    ):
        super().__init__(hidden_dim=hidden_dim)

        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types
        self.use_dag_features = use_dag_features
        self.dag_feature_dim = dag_feature_dim
        self.dag_fusion_method = dag_fusion_method

        # Setup node embedding with DAG feature fusion
        if use_dag_features and dag_fusion_method == 'concat':
            node_embed_dim = hidden_dim - dag_feature_dim
            # Validate dimension constraint: error includes current values,
            # threshold requirement, and alternative solution (residual fusion)
            assert node_embed_dim > 0, (
                f"Concat fusion requires hidden_dim ({hidden_dim}) > dag_feature_dim ({dag_feature_dim}). "
                f"Current node_embed_dim = {node_embed_dim}. "
                f"Either increase hidden_dim or use dag_fusion_method='residual'."
            )
            self.node_type_embed = nn.Embedding(num_node_types, node_embed_dim)
            self.dag_projector = nn.Linear(dag_feature_dim, dag_feature_dim)
        else:
            self.node_type_embed = nn.Embedding(num_node_types, hidden_dim)
            self.dag_projector = None

        # Residual fusion: project DAG features to hidden_dim
        if use_dag_features and dag_fusion_method == 'residual':
            self.dag_projector = nn.Sequential(
                nn.Linear(dag_feature_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.Tanh(),
            )

        # ... rest of HGT init (layers, attention, etc.) ...

    @property
    def requires_edge_types(self) -> bool:
        return True

    def _fuse_dag_features(
        self,
        node_type_emb: torch.Tensor,
        dag_pos: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Fuse DAG positional features with node embeddings.

        Args:
            node_type_emb: [num_nodes, embed_dim] node type embeddings
            dag_pos: [num_nodes, 4] DAG positional features (optional)

        Returns:
            [num_nodes, hidden_dim] fused embeddings
        """
        if not self.use_dag_features or dag_pos is None:
            return node_type_emb

        if self.dag_fusion_method == 'concat':
            dag_features = self.dag_projector(dag_pos)
            return torch.cat([node_type_emb, dag_features], dim=-1)
        elif self.dag_fusion_method == 'residual':
            return node_type_emb + self.dag_projector(dag_pos)
        else:
            return node_type_emb

    def _forward_impl(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        edge_type: Optional[torch.Tensor],
        dag_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with optional DAG positional features.

        Args:
            x: [total_nodes] node type IDs (0-9)
            edge_index: [2, num_edges] edge indices
            batch: [total_nodes] batch assignment
            edge_type: [num_edges] edge type IDs (0-7)
            dag_pos: [total_nodes, 4] DAG positional features (optional)

        Returns:
            [total_nodes, hidden_dim] node embeddings
        """
        # Get node type embeddings
        node_type_emb = self.node_type_embed(x)

        # Fuse with DAG features
        h = self._fuse_dag_features(node_type_emb, dag_pos)

        # ... rest of HGT forward pass (convert to heterogeneous, run layers) ...

        return h
```

### 5. Encoder Integration: GGNNEncoder

**File:** `src/models/encoder.py` (MODIFICATIONS)

```python
class GGNNEncoder(BaseEncoder):
    """
    Gated Graph Neural Network encoder with optional DAG positional features.
    Always uses residual fusion for DAG features.
    """

    def __init__(
        self,
        node_dim: int = 32,
        hidden_dim: int = 256,
        num_timesteps: int = 8,
        num_edge_types: int = 8,
        # DAG feature parameters
        use_dag_features: bool = False,
        dag_feature_dim: int = 4,
        **kwargs,
    ):
        super().__init__(hidden_dim=hidden_dim)

        self.use_dag_features = use_dag_features
        self.node_embedding = nn.Linear(node_dim, hidden_dim)

        # DAG projector (always residual for GGNN)
        self.dag_projector = None
        if use_dag_features:
            self.dag_projector = nn.Sequential(
                nn.Linear(dag_feature_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.Tanh(),
            )

        # ... rest of GGNN init ...

    @property
    def requires_edge_types(self) -> bool:
        return True

    def _forward_impl(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        edge_type: Optional[torch.Tensor],
        dag_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with optional DAG positional features.
        """
        h = self.node_embedding(x)
        h = F.elu(h)

        # Add DAG features as residual
        if self.use_dag_features and self.dag_projector is not None and dag_pos is not None:
            h = h + self.dag_projector(dag_pos)

        # ... rest of GGNN message passing ...

        return h
```

### 6. Encoder Integration: GAT+JKNet and RGCN (No-Op)

These encoders don't use DAG features but must accept the parameter for interface compatibility.

**File:** `src/models/encoder.py` (MODIFICATIONS)

```python
class GATJKNetEncoder(BaseEncoder):
    """GAT with Jumping Knowledge. Does not use DAG features."""

    def _forward_impl(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        edge_type: Optional[torch.Tensor],
        dag_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # dag_pos unused: GAT operates on homogeneous graphs where all nodes
        # share the same embedding space. DAG structural features (depth,
        # in_degree) are captured implicitly through attention over neighbors.
        # ... existing implementation unchanged ...
        pass


class RGCNEncoder(BaseEncoder):
    """Relational GCN. Does not use DAG features."""

    def _forward_impl(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        edge_type: Optional[torch.Tensor],
        dag_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # dag_pos unused: RGCN uses edge types for relation-specific
        # transformations. Current implementation operates on edge types only.
        # ... existing implementation unchanged ...
        pass
```

### 7. Constants Configuration

**File:** `src/constants.py` (ADDITIONS)

```python
# =============================================================================
# DAG POSITIONAL FEATURES
# =============================================================================
# Structural features for DAG-aware positional encoding.
# Captures shared subexpressions and reachability information.

DAG_FEATURE_DIM: int = 4  # [depth, subtree_size, in_degree, is_shared]

# DAG feature integration with encoders (opt-in, backward compatible)
USE_DAG_FEATURES: bool = False              # Global default
GGNN_USE_DAG_FEATURES: bool = False         # GGNN-specific
HGT_USE_DAG_FEATURES: bool = False          # HGT-specific

# Fusion method per encoder
HGT_DAG_FUSION_METHOD: str = 'concat'       # 'concat' or 'residual'
GGNN_DAG_FUSION_METHOD: str = 'residual'    # GGNN always uses residual

# Validation: concat fusion requires hidden_dim > dag_feature_dim
# This is enforced at encoder init time with a clear error message.
# Example: SCALED_HIDDEN_DIM=768 > DAG_FEATURE_DIM=4 -> OK
# Example: hidden_dim=4 with concat would fail -> use residual
```

### 8. Dataset Integration

**File:** `src/data/dataset.py` (MODIFICATIONS)

```python
class MBADataset(Dataset):
    def __init__(
        self,
        data_path: str,
        tokenizer: MBATokenizer,
        max_seq_len: int = 64,
        max_depth: Optional[int] = None,
        use_dag_features: bool = False,  # NEW
    ):
        self.use_dag_features = use_dag_features
        # ... rest of init ...

    def __getitem__(self, idx: int) -> Dict:
        item = self.data[idx]
        obfuscated_expr = item['obfuscated']

        # Build graph with optional DAG features
        graph = expr_to_optimized_graph(
            obfuscated_expr,
            use_dag_features=self.use_dag_features,
        )

        # ... rest of item construction ...
        return {
            'graph': graph,  # Contains dag_pos if use_dag_features=True
            'target_tokens': target_tokens,
            # ...
        }
```

### 9. Testing

**File:** `tests/test_dag_features.py` (NEW FILE)

```python
"""
Tests for DAG positional feature computation.

Test categories:
1. Helper function unit tests (isolated, easy to debug)
2. Main API tests (feature computation correctness)
3. Edge case tests (single node, empty graph, same depth)
4. Encoder integration tests (HGT, GGNN)
5. End-to-end tests (dataset, batching)
"""

import pytest
import torch
from collections import defaultdict

from src.data.dag_features import (
    compute_dag_positional_features,
    _build_adjacency,
    _compute_in_degrees,
    _compute_depths_topological,
    _compute_subtree_sizes,
    _normalize_to_unit_range,
)
from src.data.ast_parser import expr_to_optimized_graph


# =============================================================================
# HELPER FUNCTION UNIT TESTS
# =============================================================================

class TestBuildAdjacency:
    def test_simple_tree(self):
        """Test adjacency for simple tree: + -> x, y"""
        edge_index = torch.tensor([[0, 0], [1, 2]])  # 0->1, 0->2
        children, parents = _build_adjacency(edge_index, num_nodes=3)

        assert children[0] == [1, 2]
        assert parents[1] == [0]
        assert parents[2] == [0]
        assert children[1] == []
        assert children[2] == []

    def test_dag_with_sharing(self):
        """Test adjacency for DAG with shared node."""
        # 0 -> 2, 1 -> 2 (node 2 is shared)
        edge_index = torch.tensor([[0, 1], [2, 2]])
        children, parents = _build_adjacency(edge_index, num_nodes=3)

        assert children[0] == [2]
        assert children[1] == [2]
        assert parents[2] == [0, 1]  # Two parents

    def test_empty_graph(self):
        """Test adjacency for graph with no edges."""
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        children, parents = _build_adjacency(edge_index, num_nodes=1)

        assert children[0] == []
        assert parents[0] == []


class TestComputeInDegrees:
    def test_tree_in_degrees(self):
        """All nodes except root have in_degree=1 in tree."""
        parents = {0: [], 1: [0], 2: [0]}
        in_degrees, max_in = _compute_in_degrees(3, parents, torch.device('cpu'))

        assert in_degrees[0].item() == 0  # Root
        assert in_degrees[1].item() == 1
        assert in_degrees[2].item() == 1
        assert max_in == 1

    def test_shared_node_in_degree(self):
        """Shared node has in_degree > 1."""
        parents = {0: [], 1: [], 2: [0, 1]}  # Node 2 has two parents
        in_degrees, max_in = _compute_in_degrees(3, parents, torch.device('cpu'))

        assert in_degrees[2].item() == 2
        assert max_in == 2


class TestComputeDepthsTopological:
    def test_simple_tree_depths(self):
        """Test depths: leaves=0, root=max."""
        # Tree: root(0) -> left(1), right(2)
        children = {0: [1, 2], 1: [], 2: []}
        parents = {0: [], 1: [0], 2: [0]}
        node_types = torch.tensor([2, 0, 0])  # ADD, VAR, VAR

        depths, max_depth = _compute_depths_topological(
            3, children, parents, node_types, torch.device('cpu')
        )

        assert depths[1].item() == 0  # Leaf
        assert depths[2].item() == 0  # Leaf
        assert depths[0].item() == 1  # Root = max(0, 0) + 1
        assert max_depth == 1

    def test_dag_depths_deterministic(self):
        """
        CRITICAL TEST: Depth must be deterministic for DAGs.

        Structure:
            root(0)
           /      \
        add1(1)  add2(2)
           \      /
           shared(3)  <- shared node with two parents
              |
            leaf(4)

        Expected depths:
        - leaf(4) = 0
        - shared(3) = 1
        - add1(1) = add2(2) = 2 (both see shared at depth 1)
        - root(0) = 3
        """
        children = {0: [1, 2], 1: [3], 2: [3], 3: [4], 4: []}
        parents = {0: [], 1: [0], 2: [0], 3: [1, 2], 4: [3]}
        node_types = torch.tensor([2, 2, 2, 2, 0])  # ADD, ADD, ADD, ADD, VAR

        depths, max_depth = _compute_depths_topological(
            5, children, parents, node_types, torch.device('cpu')
        )

        assert depths[4].item() == 0  # Leaf
        assert depths[3].item() == 1  # shared = leaf + 1
        assert depths[1].item() == 2  # add1 = shared + 1
        assert depths[2].item() == 2  # add2 = shared + 1 (SAME as add1)
        assert depths[0].item() == 3  # root = max(2, 2) + 1
        assert max_depth == 3

    def test_single_node_depth(self):
        """Single node has depth 0."""
        children = {0: []}
        parents = {0: []}
        node_types = torch.tensor([0])  # VAR

        depths, max_depth = _compute_depths_topological(
            1, children, parents, node_types, torch.device('cpu')
        )

        assert depths[0].item() == 0
        assert max_depth == 1  # Clamped to 1 for normalization


class TestComputeSubtreeSizes:
    def test_tree_subtree_sizes(self):
        """Test subtree sizes for simple tree."""
        # root(0) -> left(1), right(2)
        children = {0: [1, 2], 1: [], 2: []}
        parents = {0: [], 1: [0], 2: [0]}

        sizes = _compute_subtree_sizes(3, children, parents, torch.device('cpu'))

        assert sizes[1].item() == 1  # Leaf
        assert sizes[2].item() == 1  # Leaf
        assert sizes[0].item() == 3  # Root = 1 + 1 + 1

    def test_shared_node_subtree_size(self):
        """
        CRITICAL TEST: Shared node contributes to ALL parent subtrees.

        Structure: root -> add1, add2; add1 -> shared; add2 -> shared

        Expected:
        - shared = 1
        - add1 = 1 + 1 = 2 (itself + shared)
        - add2 = 1 + 1 = 2 (itself + shared, even though shared already visited)
        - root = 1 + 2 + 2 = 5 (not 3!)
        """
        children = {0: [1, 2], 1: [3], 2: [3], 3: []}
        parents = {0: [], 1: [0], 2: [0], 3: [1, 2]}

        sizes = _compute_subtree_sizes(4, children, parents, torch.device('cpu'))

        assert sizes[3].item() == 1  # Shared leaf
        assert sizes[1].item() == 2  # add1 = 1 + shared(1)
        assert sizes[2].item() == 2  # add2 = 1 + shared(1)
        assert sizes[0].item() == 5  # root = 1 + add1(2) + add2(2)


class TestNormalize:
    def test_normalize_range(self):
        """Normalized values should be in [0, 1]."""
        feature = torch.tensor([0, 5, 10])
        norm = _normalize_to_unit_range(feature, 10)

        assert norm[0].item() == 0.0
        assert norm[1].item() == 0.5
        assert norm[2].item() == 1.0

    def test_normalize_zero_max(self):
        """max_val=0 should clamp to 1 to avoid division by zero."""
        feature = torch.tensor([0])
        norm = _normalize_to_unit_range(feature, 0)

        assert norm[0].item() == 0.0  # 0 / 1 = 0


# =============================================================================
# MAIN API TESTS
# =============================================================================

class TestComputeDAGPositionalFeatures:
    def test_simple_expression(self):
        """Test features on x + y."""
        graph = expr_to_optimized_graph("x + y", use_dag_features=True)

        assert hasattr(graph, 'dag_pos')
        assert graph.dag_pos.shape == (graph.x.size(0), 4)

        # All features in [0, 1]
        assert (graph.dag_pos >= 0.0).all()
        assert (graph.dag_pos <= 1.0).all()

        # No shared nodes in simple tree
        is_shared = graph.dag_pos[:, 3]
        assert (is_shared == 0.0).all()

    def test_shared_subexpression(self):
        """Test is_shared flag on (x & y) + (x & y)."""
        graph = expr_to_optimized_graph("(x & y) + (x & y)", use_dag_features=True)

        is_shared = graph.dag_pos[:, 3]
        shared_count = (is_shared == 1.0).sum().item()

        # Should have shared nodes (the AND and possibly x, y)
        assert shared_count > 0

    def test_depth_max_at_root(self):
        """Root should have depth = 1.0 (normalized)."""
        graph = expr_to_optimized_graph("((x + y) * z)", use_dag_features=True)

        depth_norm = graph.dag_pos[:, 0]
        assert depth_norm.max().item() == 1.0

    def test_subtree_size_max_at_root(self):
        """Root should have subtree_size = 1.0 (normalized)."""
        graph = expr_to_optimized_graph("x + (y * z)", use_dag_features=True)

        subtree_norm = graph.dag_pos[:, 1]
        assert subtree_norm.max().item() == 1.0


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestEdgeCases:
    def test_single_node_graph(self):
        """Single variable: depth=0, subtree=1, in_degree=0, is_shared=0."""
        graph = expr_to_optimized_graph("x", use_dag_features=True)

        assert graph.dag_pos.shape == (1, 4)
        assert graph.dag_pos[0, 0].item() == 0.0  # depth: 0/1=0
        assert graph.dag_pos[0, 1].item() == 1.0  # subtree: 1/1=1
        assert graph.dag_pos[0, 2].item() == 0.0  # in_degree: 0/1=0
        assert graph.dag_pos[0, 3].item() == 0.0  # is_shared: False

    def test_constant_node(self):
        """Single constant."""
        graph = expr_to_optimized_graph("42", use_dag_features=True)

        assert graph.dag_pos.shape == (1, 4)
        assert not torch.isnan(graph.dag_pos).any()

    def test_flat_expression_same_depth_leaves(self):
        """All leaves at same level: x + y + z."""
        graph = expr_to_optimized_graph("(x + y) + z", use_dag_features=True)

        # Should not produce NaN
        assert not torch.isnan(graph.dag_pos).any()
        assert not torch.isinf(graph.dag_pos).any()

    def test_deep_expression(self):
        """Deep expression doesn't overflow."""
        expr = "((((x + y) * z) & a) | b)"
        graph = expr_to_optimized_graph(expr, use_dag_features=True)

        assert not torch.isnan(graph.dag_pos).any()
        assert (graph.dag_pos >= 0.0).all()
        assert (graph.dag_pos <= 1.0).all()

    def test_cyclic_graph_raises_error(self):
        """
        CRITICAL TEST: Cyclic graphs should raise ValueError.

        MBA expressions are semantically always DAGs, but malformed
        input (e.g., from buggy parser) could contain cycles. The
        implementation should detect and reject cycles rather than
        produce incorrect output silently.
        """
        # Create artificial cycle: 0 -> 1 -> 0
        edge_index = torch.tensor([[0, 1], [1, 0]])  # Bidirectional = cycle
        node_types = torch.tensor([2, 2])  # Two ADD nodes

        with pytest.raises(ValueError, match="Graph contains cycles"):
            compute_dag_positional_features(
                num_nodes=2,
                edge_index=edge_index,
                node_types=node_types,
            )

    def test_self_loop_raises_error(self):
        """Self-loop (node pointing to itself) should raise ValueError."""
        edge_index = torch.tensor([[0], [0]])  # Node 0 -> Node 0
        node_types = torch.tensor([2])  # ADD node

        with pytest.raises(ValueError, match="Graph contains cycles"):
            compute_dag_positional_features(
                num_nodes=1,
                edge_index=edge_index,
                node_types=node_types,
            )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_device_mismatch_raises_error(self):
        """Tensors on different devices should raise ValueError."""
        edge_index = torch.tensor([[0], [1]]).cuda()  # GPU
        node_types = torch.tensor([2, 0])  # CPU

        with pytest.raises(ValueError, match="must be on same device"):
            compute_dag_positional_features(
                num_nodes=2,
                edge_index=edge_index,
                node_types=node_types,
            )

    def test_device_consistency_cpu(self):
        """Both tensors on CPU should work."""
        edge_index = torch.tensor([[0], [1]])
        node_types = torch.tensor([2, 0])

        # Should not raise
        result = compute_dag_positional_features(
            num_nodes=2,
            edge_index=edge_index,
            node_types=node_types,
        )
        assert result.device == torch.device('cpu')


# =============================================================================
# ENCODER INTEGRATION TESTS
# =============================================================================

class TestHGTIntegration:
    @pytest.fixture
    def sample_input(self):
        return {
            'x': torch.randint(0, 10, (50,)),
            'edge_index': torch.randint(0, 50, (2, 100)),
            'edge_type': torch.randint(0, 8, (100,)),
            'batch': torch.zeros(50, dtype=torch.long),
            'dag_pos': torch.rand(50, 4),
        }

    def test_hgt_concat_fusion_output_shape(self, sample_input):
        """HGT with concat fusion produces correct output shape."""
        from src.models.encoder import HGTEncoder

        encoder = HGTEncoder(
            hidden_dim=256,
            num_layers=2,
            use_dag_features=True,
            dag_fusion_method='concat',
        )

        output = encoder(
            sample_input['x'],
            sample_input['edge_index'],
            sample_input['batch'],
            edge_type=sample_input['edge_type'],
            dag_pos=sample_input['dag_pos'],
        )

        assert output.shape == (50, 256)

    def test_hgt_residual_fusion_output_shape(self, sample_input):
        """HGT with residual fusion produces correct output shape."""
        from src.models.encoder import HGTEncoder

        encoder = HGTEncoder(
            hidden_dim=256,
            num_layers=2,
            use_dag_features=True,
            dag_fusion_method='residual',
        )

        output = encoder(
            sample_input['x'],
            sample_input['edge_index'],
            sample_input['batch'],
            edge_type=sample_input['edge_type'],
            dag_pos=sample_input['dag_pos'],
        )

        assert output.shape == (50, 256)

    def test_hgt_backward_compatible_without_dag(self, sample_input):
        """HGT works without DAG features (backward compatible)."""
        from src.models.encoder import HGTEncoder

        encoder = HGTEncoder(
            hidden_dim=256,
            num_layers=2,
            use_dag_features=False,
        )

        # Should work without dag_pos
        output = encoder(
            sample_input['x'],
            sample_input['edge_index'],
            sample_input['batch'],
            edge_type=sample_input['edge_type'],
        )

        assert output.shape == (50, 256)

    def test_hgt_concat_dimension_validation(self):
        """HGT concat fusion validates hidden_dim > dag_feature_dim."""
        from src.models.encoder import HGTEncoder

        with pytest.raises(AssertionError) as exc_info:
            HGTEncoder(
                hidden_dim=4,  # Same as dag_feature_dim
                use_dag_features=True,
                dag_fusion_method='concat',
                dag_feature_dim=4,
            )

        # Error message should suggest residual fusion
        assert 'residual' in str(exc_info.value).lower()

    def test_hgt_gradient_flow(self, sample_input):
        """Gradients flow through DAG projector."""
        from src.models.encoder import HGTEncoder

        encoder = HGTEncoder(
            hidden_dim=256,
            num_layers=2,
            use_dag_features=True,
            dag_fusion_method='concat',
        )

        output = encoder(
            sample_input['x'],
            sample_input['edge_index'],
            sample_input['batch'],
            edge_type=sample_input['edge_type'],
            dag_pos=sample_input['dag_pos'],
        )

        loss = output.sum()
        loss.backward()

        # DAG projector should have gradients
        if encoder.dag_projector is not None:
            has_grad = any(
                p.grad is not None and p.grad.abs().sum() > 0
                for p in encoder.dag_projector.parameters()
            )
            assert has_grad


class TestGGNNIntegration:
    @pytest.fixture
    def sample_input(self):
        return {
            'x': torch.rand(50, 32),  # GGNN uses continuous features
            'edge_index': torch.randint(0, 50, (2, 100)),
            'edge_type': torch.randint(0, 8, (100,)),
            'batch': torch.zeros(50, dtype=torch.long),
            'dag_pos': torch.rand(50, 4),
        }

    def test_ggnn_residual_fusion_output_shape(self, sample_input):
        """GGNN with DAG features produces correct output shape."""
        from src.models.encoder import GGNNEncoder

        encoder = GGNNEncoder(
            node_dim=32,
            hidden_dim=256,
            num_timesteps=4,
            use_dag_features=True,
        )

        output = encoder(
            sample_input['x'],
            sample_input['edge_index'],
            sample_input['batch'],
            edge_type=sample_input['edge_type'],
            dag_pos=sample_input['dag_pos'],
        )

        assert output.shape == (50, 256)

    def test_ggnn_backward_compatible(self, sample_input):
        """GGNN works without DAG features."""
        from src.models.encoder import GGNNEncoder

        encoder = GGNNEncoder(
            node_dim=32,
            hidden_dim=256,
            use_dag_features=False,
        )

        output = encoder(
            sample_input['x'],
            sample_input['edge_index'],
            sample_input['batch'],
            edge_type=sample_input['edge_type'],
        )

        assert output.shape == (50, 256)


# =============================================================================
# END-TO-END TESTS
# =============================================================================

class TestEndToEnd:
    def test_batch_dag_features(self):
        """PyG batching preserves DAG features."""
        from torch_geometric.data import Batch

        graph1 = expr_to_optimized_graph("x + y", use_dag_features=True)
        graph2 = expr_to_optimized_graph("(a & b) * (a & b)", use_dag_features=True)

        batched = Batch.from_data_list([graph1, graph2])

        # DAG features should be concatenated
        assert hasattr(batched, 'dag_pos')
        expected_nodes = graph1.x.size(0) + graph2.x.size(0)
        assert batched.dag_pos.shape == (expected_nodes, 4)

    def test_multiple_shared_subexpressions(self):
        """Complex expression with multiple levels of sharing."""
        # (x & y) appears twice, (a | b) appears twice
        expr = "((x & y) + (x & y)) * ((a | b) - (a | b))"
        graph = expr_to_optimized_graph(expr, use_dag_features=True)

        is_shared = graph.dag_pos[:, 3]
        shared_count = (is_shared == 1.0).sum().item()

        # Should detect multiple shared nodes
        assert shared_count >= 2
```

## Feature Dimensions and Normalization

| Feature | Raw Range | Normalization | Interpretation |
|---------|-----------|---------------|----------------|
| `depth` | [0, max_depth] | / max(max_depth, 1) | Distance from leaves; 1.0 = root |
| `subtree_size` | [1, num_nodes] | / max(num_nodes, 1) | Fraction of graph; 1.0 = root |
| `in_degree` | [0, max_in_degree] | / max(max_in_degree, 1) | Relative popularity; shared nodes > 0 |
| `is_shared` | {0, 1} | Binary | 1.0 if node has multiple parents |

## Configuration Flags

**File:** `src/constants.py`

```python
# Enable/disable DAG features per encoder (opt-in, backward compatible)
USE_DAG_FEATURES: bool = False              # Global default
GGNN_USE_DAG_FEATURES: bool = False         # GGNN-specific
HGT_USE_DAG_FEATURES: bool = False          # HGT-specific

# Fusion method
HGT_DAG_FUSION_METHOD: str = 'concat'       # 'concat' or 'residual'
GGNN_DAG_FUSION_METHOD: str = 'residual'    # Always residual for GGNN

# Feature dimension
DAG_FEATURE_DIM: int = 4                    # Fixed: [depth, subtree_size, in_degree, is_shared]
```

**File:** `configs/scaled_model.yaml`

```yaml
encoder:
  type: hgt
  hidden_dim: 768
  num_layers: 12
  use_dag_features: true
  dag_fusion_method: concat

data:
  use_dag_features: true
```

## Backward Compatibility

DAG features are **opt-in** with `use_dag_features=False` as default.

**Compatibility guarantees:**
1. Existing models without DAG features continue to work unchanged
2. All encoders accept `dag_pos` parameter (interface compatible)
3. Encoders gracefully handle `dag_pos=None` (treats as no DAG features)
4. Dataset loader backward compatible (graphs without dag_pos work normally)
5. Config files without DAG settings use default (disabled)

**Migration path:**
1. Enable DAG features in dataset: `use_dag_features=True`
2. Enable in encoder: `HGT_USE_DAG_FEATURES=True`
3. Train new model or fine-tune existing model with new features
4. Compare accuracy via ablation study

## Task Breakdown

### Critical: Atomic Signature Update

The following changes **MUST be applied in a single commit** to avoid breaking the build:

1. Update `BaseEncoder._forward_impl` abstract method signature (add `dag_pos` parameter)
2. Update ALL existing encoder implementations simultaneously:
   - `GATJKNetEncoder._forward_impl` - accept and ignore `dag_pos`
   - `GGNNEncoder._forward_impl` - accept and use `dag_pos` if enabled
   - `HGTEncoder._forward_impl` - accept and use `dag_pos` if enabled
   - `RGCNEncoder._forward_impl` - accept and ignore `dag_pos`

**Why atomic?** Python abstract methods require exact signature match. Adding a parameter to the abstract method without updating all implementations causes `TypeError` at runtime when any encoder is instantiated.

**Verification:** After applying changes, run `pytest tests/test_models.py -v` to verify all encoders instantiate correctly.

### Task List

| Task | File | Complexity | Priority |
|------|------|------------|----------|
| Implement `_build_adjacency()` helper | `src/data/dag_features.py` | Low | P0 |
| Implement `_compute_in_degrees()` helper | `src/data/dag_features.py` | Low | P0 |
| Implement `_compute_depths_topological()` with Kahn's algorithm | `src/data/dag_features.py` | Medium | P0 |
| Implement `_compute_subtree_sizes()` with correct shared node handling | `src/data/dag_features.py` | Medium | P0 |
| Implement `_normalize_to_unit_range()` helper | `src/data/dag_features.py` | Low | P0 |
| Implement `compute_dag_positional_features()` orchestrator | `src/data/dag_features.py` | Low | P0 |
| Update `BaseEncoder._forward_impl` signature with dag_pos | `src/models/encoder_base.py` | Low | P0 |
| Update `BaseEncoder.forward` signature with dag_pos | `src/models/encoder_base.py` | Low | P0 |
| Add `use_dag_features` param to `ast_to_optimized_graph()` | `src/data/ast_parser.py` | Low | P0 |
| Update HGTEncoder with DAG params and fusion | `src/models/encoder.py` | Medium | P0 |
| Add dimension validation with helpful error for concat fusion | `src/models/encoder.py` | Low | P0 |
| Update GGNNEncoder with DAG params and residual fusion | `src/models/encoder.py` | Low | P0 |
| Update GAT+JKNet signature to accept dag_pos (no-op) | `src/models/encoder.py` | Low | P0 |
| Update RGCNEncoder signature to accept dag_pos (no-op) | `src/models/encoder.py` | Low | P0 |
| Add constants to `constants.py` | `src/constants.py` | Low | P0 |
| Update MBADataset with `use_dag_features` | `src/data/dataset.py` | Low | P0 |
| Write helper function unit tests | `tests/test_dag_features.py` | Medium | P0 |
| Write main API tests | `tests/test_dag_features.py` | Medium | P0 |
| Write edge case tests (single node, constants) | `tests/test_dag_features.py` | Medium | P0 |
| Write cycle detection tests (cyclic graph, self-loop) | `tests/test_dag_features.py` | Low | P0 |
| Write device consistency tests (mismatch, CPU) | `tests/test_dag_features.py` | Low | P0 |
| Write DAG-specific tests (shared nodes, determinism) | `tests/test_dag_features.py` | Medium | P0 |
| Write encoder integration tests | `tests/test_dag_features.py` | Medium | P0 |
| Write end-to-end tests (batching) | `tests/test_dag_features.py` | Low | P1 |
| Update training scripts to pass dag_pos | `scripts/train.py` | Low | P1 |
| Update config files with DAG settings | `configs/*.yaml` | Low | P1 |
| Benchmark feature computation overhead | - | Low | P2 |
| Ablation study: DAG features vs baseline | - | High | P2 |

## Complexity Analysis

**Time Complexity:**
- Build adjacency: O(E)
- Compute in-degrees: O(V)
- Topological sort + depth: O(V + E)
- Subtree sizes (DFS): O(V + E)
- Total: **O(V + E)** per graph

**Space Complexity:**
- Adjacency lists: O(V + E)
- Feature storage: O(V * 4) = O(V)
- Total: **O(V + E)**

**Overhead Analysis:**
For typical MBA graphs (V=50-200, E=100-400):
- Time: ~1-5ms per graph (negligible vs 50ms GNN forward pass)
- Memory: ~5KB per graph (negligible vs 1-10MB model state)

## Risks and Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Cyclic graphs in input | ValueError raised | Low | Explicit cycle detection with clear error message; MBA expressions are always DAGs |
| Concat fusion reduces embedding capacity | Accuracy drop | Medium | Ablate concat vs residual; residual is fallback |
| DAG features don't improve accuracy | Wasted effort | Medium | Ablation study; opt-in design allows easy rollback |
| Feature computation slow on large graphs | Training bottleneck | Low | Profile; consider preprocessing cache |
| Breaking change to BaseEncoder signature | Build failure | High | Atomic commit of all encoder updates (documented above) |

## Success Criteria

1. **Correctness**: All unit tests pass including edge cases
2. **Determinism**: DAG depth computation is deterministic (critical test passes)
3. **Shared node handling**: Subtree sizes correctly count shared contributions
4. **Cycle detection**: Cyclic graphs raise `ValueError` with clear message
5. **Device validation**: Tensors on different devices raise `ValueError` with clear message
6. **Interface compatibility**: All encoders accept dag_pos parameter
7. **Backward compatibility**: Models without DAG features work identically
8. **Atomic update**: All encoder signature changes applied in single commit
9. **Performance**: Feature computation adds <5% overhead to data loading
10. **Ablation**: DAG features improve accuracy on shared-subexpression test set

## Rollback Plan

DAG features are opt-in via `use_dag_features=False` default. If accuracy regresses:
1. Disable in config: `use_dag_features: false`
2. Revert to baseline encoder (no code changes needed)
3. Investigate fusion method (try residual instead of concat)
4. Ablate individual features to identify harmful ones

## Appendix: Algorithm Details

### Depth Computation (Topological Sort)

```
Uses Kahn's algorithm for deterministic depth on DAGs:

1. Initialize: out_degree[node] = |children[node]| for all nodes
2. Queue: all nodes with out_degree=0 (leaves)
3. While queue not empty:
     node = queue.pop()
     if node has children:
         depth[node] = 1 + max(depth[child] for child in children[node])
     else:
         depth[node] = 0  # Leaf

     For each parent of node:
         remaining_children[parent] -= 1
         if remaining_children[parent] == 0:
             queue.push(parent)

4. CYCLE DETECTION: If any nodes unprocessed, raise ValueError
   (unprocessed nodes indicate cycle or disconnected component)

5. Normalize: depth[i] / max(depth)

Key insight: Processing in topological order (leaves first) ensures all
children are processed before their parents, making depth computation
deterministic regardless of which parent visits a shared node first.

Defensive programming: Cycles cause nodes to never have remaining_children=0,
so they never enter the queue. We detect this and fail fast with a clear error.
```

### Subtree Size (DFS with Memoization)

```
1. Initialize: subtree_size[i] = 0, visited = {}
2. For each root node (no parents):
     count_descendants(root)
3. Normalize: subtree_size[i] / num_nodes

Function count_descendants(node):
    if node in visited:
        return subtree_size[node]  # Return cached size

    visited.add(node)
    count = 1  # Count self

    For each child of node:
        count += count_descendants(child)

    subtree_size[node] = count
    return count

Key insight: Returning cached size (not 0) for visited nodes means shared
nodes contribute their full subtree to ALL parents, which is semantically
correct - each parent's computation depends on the shared subtree.
```

## References

- Kahn's Algorithm: Topological sorting for DAGs
- PyTorch Geometric: Data batching and graph utilities
- MBA deobfuscation domain: Shared subexpressions are key optimization targets
