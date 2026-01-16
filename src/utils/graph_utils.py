"""
Graph traversal utilities for ablation study encoders.

Addresses issues from quality review:
- Cyclic graph detection and handling
- Disconnected component handling
- Root validation for ASTs
"""

from typing import List, Optional, Set, Tuple

import torch


def compute_dfs_order(
    edge_index: torch.Tensor,
    node_mask: torch.Tensor,
    validate_acyclic: bool = True,
) -> torch.Tensor:
    """
    Compute DFS traversal order for subgraph.

    Args:
        edge_index: [2, num_edges] full edge index
        node_mask: [total_nodes] boolean mask for subgraph
        validate_acyclic: If True, raise ValueError on cycles

    Returns:
        [num_nodes_in_subgraph] indices in DFS order (local indices 0 to N-1)

    Raises:
        ValueError: If graph has cycles (when validate_acyclic=True)
        ValueError: If no root node found (all nodes have incoming edges)
    """
    device = edge_index.device

    # Get subgraph nodes (global indices)
    nodes = torch.where(node_mask)[0]
    num_subgraph_nodes = nodes.size(0)

    if num_subgraph_nodes == 0:
        return torch.tensor([], dtype=torch.long, device=device)

    if num_subgraph_nodes == 1:
        return torch.tensor([0], dtype=torch.long, device=device)

    # Build global -> local index mapping
    global_to_local = {g.item(): i for i, g in enumerate(nodes)}
    local_to_global = {i: g.item() for i, g in enumerate(nodes)}

    # Filter edges to subgraph
    src_in = node_mask[edge_index[0]]
    dst_in = node_mask[edge_index[1]]
    subgraph_mask = src_in & dst_in
    subgraph_edges = edge_index[:, subgraph_mask]

    # Build adjacency list (local indices)
    adjacency: List[List[int]] = [[] for _ in range(num_subgraph_nodes)]
    in_degree = torch.zeros(num_subgraph_nodes, dtype=torch.long, device=device)

    for i in range(subgraph_edges.size(1)):
        src_global = subgraph_edges[0, i].item()
        dst_global = subgraph_edges[1, i].item()

        src_local = global_to_local[src_global]
        dst_local = global_to_local[dst_global]

        adjacency[src_local].append(dst_local)
        in_degree[dst_local] += 1

    # Find root(s) - nodes with no incoming edges
    root_mask = in_degree == 0
    roots = torch.where(root_mask)[0].tolist()

    if not roots:
        raise ValueError(
            "No root node found in subgraph. All nodes have incoming edges, "
            "suggesting a cycle or malformed AST. ASTs must have a single root."
        )

    # DFS traversal starting from all roots (handles disconnected components)
    visited: Set[int] = set()
    order: List[int] = []
    in_stack: Set[int] = set()  # For cycle detection

    def dfs(node: int, path: List[int]) -> bool:
        """
        DFS with cycle detection.

        Returns False if cycle detected (when validate_acyclic=True).
        """
        if node in in_stack:
            if validate_acyclic:
                cycle_path = path[path.index(node):] + [node]
                raise ValueError(
                    f"Cycle detected in graph: {cycle_path}. "
                    "ASTs should be acyclic trees."
                )
            return False

        if node in visited:
            return True

        in_stack.add(node)
        visited.add(node)
        order.append(node)

        # Sort children for deterministic traversal
        children = sorted(adjacency[node])
        for child in children:
            if not dfs(child, path + [node]):
                return False

        in_stack.remove(node)
        return True

    # Start DFS from each root
    for root in sorted(roots):
        if root not in visited:
            dfs(root, [])

    # Handle any remaining disconnected nodes (shouldn't happen in valid AST)
    # This ensures all nodes are included even if graph is malformed
    for node in range(num_subgraph_nodes):
        if node not in visited:
            visited.add(node)
            order.append(node)

    return torch.tensor(order, dtype=torch.long, device=device)


def compute_bfs_order(
    edge_index: torch.Tensor,
    node_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute BFS traversal order for subgraph.

    Fallback for when DFS fails (cycles, etc.)

    Args:
        edge_index: [2, num_edges] full edge index
        node_mask: [total_nodes] boolean mask for subgraph

    Returns:
        [num_nodes_in_subgraph] indices in BFS order (local indices)
    """
    from collections import deque

    device = edge_index.device
    nodes = torch.where(node_mask)[0]
    num_subgraph_nodes = nodes.size(0)

    if num_subgraph_nodes == 0:
        return torch.tensor([], dtype=torch.long, device=device)

    if num_subgraph_nodes == 1:
        return torch.tensor([0], dtype=torch.long, device=device)

    # Build mappings
    global_to_local = {g.item(): i for i, g in enumerate(nodes)}

    # Filter edges and build adjacency
    src_in = node_mask[edge_index[0]]
    dst_in = node_mask[edge_index[1]]
    subgraph_mask = src_in & dst_in
    subgraph_edges = edge_index[:, subgraph_mask]

    adjacency: List[List[int]] = [[] for _ in range(num_subgraph_nodes)]
    in_degree = torch.zeros(num_subgraph_nodes, dtype=torch.long, device=device)

    for i in range(subgraph_edges.size(1)):
        src_local = global_to_local[subgraph_edges[0, i].item()]
        dst_local = global_to_local[subgraph_edges[1, i].item()]
        adjacency[src_local].append(dst_local)
        in_degree[dst_local] += 1

    # Find roots
    roots = torch.where(in_degree == 0)[0].tolist()
    if not roots:
        # Fallback: use node 0 as root
        roots = [0]

    visited: Set[int] = set()
    order: List[int] = []
    queue: deque = deque()

    for root in sorted(roots):
        if root not in visited:
            queue.append(root)
            visited.add(root)

            while queue:
                node = queue.popleft()
                order.append(node)

                for child in sorted(adjacency[node]):
                    if child not in visited:
                        visited.add(child)
                        queue.append(child)

    # Handle disconnected nodes
    for node in range(num_subgraph_nodes):
        if node not in visited:
            order.append(node)

    return torch.tensor(order, dtype=torch.long, device=device)


def safe_dfs_order(
    edge_index: torch.Tensor,
    node_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute traversal order with fallback to BFS on failure.

    Use this in production code for robustness.

    Args:
        edge_index: [2, num_edges] full edge index
        node_mask: [total_nodes] boolean mask for subgraph

    Returns:
        [num_nodes_in_subgraph] indices in traversal order
    """
    try:
        return compute_dfs_order(edge_index, node_mask, validate_acyclic=True)
    except ValueError:
        # Fall back to BFS on cyclic or malformed graphs
        return compute_bfs_order(edge_index, node_mask)


def validate_ast_structure(
    edge_index: torch.Tensor,
    node_mask: torch.Tensor,
) -> Tuple[bool, Optional[str]]:
    """
    Validate that subgraph is a valid AST (tree).

    Args:
        edge_index: [2, num_edges] full edge index
        node_mask: [total_nodes] boolean mask for subgraph

    Returns:
        (is_valid, error_message) - error_message is None if valid
    """
    nodes = torch.where(node_mask)[0]
    num_nodes = nodes.size(0)

    if num_nodes == 0:
        return True, None

    # Filter edges
    src_in = node_mask[edge_index[0]]
    dst_in = node_mask[edge_index[1]]
    subgraph_mask = src_in & dst_in
    num_edges = subgraph_mask.sum().item()

    # Tree property: num_edges == num_nodes - 1
    if num_edges != num_nodes - 1:
        return False, f"Expected {num_nodes - 1} edges for tree, got {num_edges}"

    # Check for single root
    global_to_local = {g.item(): i for i, g in enumerate(nodes)}
    subgraph_edges = edge_index[:, subgraph_mask]

    in_degree = torch.zeros(num_nodes, dtype=torch.long, device=edge_index.device)
    for i in range(subgraph_edges.size(1)):
        dst_local = global_to_local[subgraph_edges[1, i].item()]
        in_degree[dst_local] += 1

    roots = (in_degree == 0).sum().item()
    if roots != 1:
        return False, f"Expected 1 root, found {roots}"

    # Check connectivity via DFS
    try:
        order = compute_dfs_order(edge_index, node_mask, validate_acyclic=True)
        if len(order) != num_nodes:
            return False, f"DFS only reached {len(order)} of {num_nodes} nodes"
    except ValueError as e:
        return False, str(e)

    return True, None
