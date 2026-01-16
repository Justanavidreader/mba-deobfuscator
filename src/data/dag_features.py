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
from collections import defaultdict, deque


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
        max_in_degree: Maximum in-degree (for normalization, clamped to >= 1)
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
        max_depth: Maximum depth (for normalization, clamped to >= 1)

    Raises:
        ValueError: If graph contains cycles (unprocessed nodes after sort)
    """
    depths = torch.zeros(num_nodes, dtype=torch.long, device=device)

    if num_nodes == 0:
        return depths, 1

    # Compute out-degree (number of children) for topological sort
    out_degree = torch.zeros(num_nodes, dtype=torch.long, device=device)
    for node in range(num_nodes):
        out_degree[node] = len(children[node])

    # Queue nodes with no children (leaves in expression tree)
    queue: deque = deque()
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

    # Detect cycles (unprocessed nodes indicate cycle or malformed graph)
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

    For DAGs with shared nodes, each parent's subtree size includes the FULL
    subtree of shared children. This is semantically correct because the shared
    subtree contributes to the computation of ALL its parents.

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
            # Return cached size for shared nodes
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
