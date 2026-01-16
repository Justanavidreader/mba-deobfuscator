# Rich Feature System Implementation Plan

> **Revision History:**
> - v1.1 (Current): Applied quality review fixes
>   - Added `compute_node_features()` dispatcher function
>   - Fixed fragile child lookup with `_graph_id` assignment
>   - Added `use_rich_features` and `use_edge_context` flags to encoders
>   - Moved constants to centralized `src/constants.py` section
>   - Fixed duplicate domain computation (pass precomputed features)
>   - Replaced magic number 14 with `MAX_AST_DEPTH_NORM` constant
>   - Added `_to_heterogeneous_from_embeddings()` for type-safe HGT processing

## Overview

This document specifies the implementation of a rich feature system for the GNN encoder (HGT/RGCN) in the MBA deobfuscation pipeline. The system enhances node and edge features beyond simple type IDs to improve the model's ability to learn expression simplifications.

**Goals:**
1. Add domain awareness (ARITHMETIC/BOOLEAN/BITWISE/LEAF) to operator nodes
2. Provide structural context through depth and root flags
3. Enable terminals (variables/constants) to signal usage context across multiple domains
4. Enrich edges with parent context domain for better message passing
5. Modify HGT attention to incorporate edge context

**Impact:** Expected 5-10% accuracy improvement on depth 10+ expressions based on domain-aware attention.

---

## Domain Taxonomy

### Domain Enum

```python
# Add to src/models/edge_types.py

from enum import IntEnum

class Domain(IntEnum):
    """
    Semantic domain classification for MBA expression nodes.
    Used for domain-aware message passing in HGT/RGCN encoders.
    """
    ARITHMETIC = 0  # ADD, SUB, MUL, NEG
    BOOLEAN = 1     # AND, OR, XOR, NOT
    BITWISE = 2     # (Reserved for future shift operators)
    LEAF = 3        # VAR, CONST (domain-agnostic terminals)
    NONE = 4        # Used for relational edges without domain context

    @staticmethod
    def from_node_type(node_type: int) -> int:
        """
        Map node type to domain.

        Args:
            node_type: NodeType enum value (0-9)

        Returns:
            Domain enum value
        """
        if NodeType.is_arithmetic(node_type):
            return int(Domain.ARITHMETIC)
        elif NodeType.is_boolean(node_type):
            return int(Domain.BOOLEAN)
        elif NodeType.is_terminal(node_type):
            return int(Domain.LEAF)
        else:
            return int(Domain.NONE)
```

---

## Data Structures

### Extended Node Representation

Nodes now carry both intrinsic features (type, value) and contextual features (depth, usage domains).

```python
# Node feature components (conceptual - implemented as tensors)

class NodeFeatures:
    """
    Logical grouping of node features. Not a runtime class -
    features are flattened into tensors during graph construction.
    """
    # === Intrinsic Features ===
    node_type: int              # NodeType enum (0-9)
    domain: int                 # Domain enum (0-4)

    # For CONST nodes:
    const_value: float          # Normalized to [-1, 1]
    is_zero: bool
    is_one: bool
    is_neg_one: bool

    # For VAR nodes:
    var_index: int              # 0-7 for x0-x7

    # === Structural Features ===
    depth: float                # Normalized depth in AST (root=1.0, decreases)
    is_root: bool

    # === Contextual Features ===
    context_domains: List[int]  # Multi-hot: [ARITH, BOOL, BITWISE] usage
```

**Feature Dimension Calculation:**

| Feature Group | Components | Dimension |
|---------------|------------|-----------|
| node_type embedding | 10 types | `node_type_embed_dim` (e.g., 64) |
| domain embedding | 5 domains | `domain_embed_dim` (e.g., 32) |
| var_index embedding | 8 variables | `var_embed_dim` (e.g., 16) |
| const features | value(1) + flags(3) | 4 scalars |
| structural | depth(1) + is_root(1) | 2 scalars |
| context_domains | 3-element multi-hot | 3 scalars |
| **Total input** | - | **embeddings + 9 scalars** |

After embedding and concatenation, projected to `hidden_dim` (256 or 768).

### Extended Edge Representation

```python
class EdgeFeatures:
    """
    Logical grouping of edge features.
    """
    edge_type: int           # EdgeType enum (0-7)
    context_domain: int      # Domain enum (0-4) - domain of source/parent node
```

**Context Domain Assignment Rules:**

| Edge Type | Direction | Context Domain Source | Rationale |
|-----------|-----------|----------------------|-----------|
| LEFT_OPERAND (0) | parent→child | parent.domain | Child receives parent's domain context |
| RIGHT_OPERAND (1) | parent→child | parent.domain | Child receives parent's domain context |
| UNARY_OPERAND (2) | parent→child | parent.domain | Child receives parent's domain context |
| LEFT_OPERAND_INV (3) | child→parent | child.domain | Parent receives child's result domain |
| RIGHT_OPERAND_INV (4) | child→parent | child.domain | Parent receives child's result domain |
| UNARY_OPERAND_INV (5) | child→parent | child.domain | Parent receives child's result domain |
| DOMAIN_BRIDGE_DOWN (6) | parent→child | parent.domain | Explicit cross-domain signal |
| DOMAIN_BRIDGE_UP (7) | child→parent | child.domain | Explicit cross-domain signal |

**Implementation Note:** Context domain is always the **source node's domain** for the edge direction. Forward edges use parent domain, inverse edges use child domain, domain bridges use their respective source.

---

## Feature Computation Pipeline

### Phase 1: Build DAG Structure

Parse expression to AST and collect nodes with IDs.

```python
# In src/data/ast_parser.py

def collect_nodes_with_metadata(ast: ASTNode) -> List[Dict]:
    """
    Collect all nodes from AST with metadata.

    Returns:
        List of dicts with keys: id, node, parent_id, depth

    NOTE: Assigns unique _graph_id to each node for reliable lookup.
    This avoids fragile identity-based comparisons that can fail
    if AST nodes are copied during traversal.
    """
    nodes = []

    def traverse(node: ASTNode, parent_id: Optional[int], depth: int):
        node_id = len(nodes)
        node._graph_id = node_id  # Store ID on node for O(1) lookup
        nodes.append({
            'id': node_id,
            'node': node,
            'parent_id': parent_id,
            'depth': depth
        })
        for child in node.children:
            traverse(child, node_id, depth + 1)

    traverse(ast, None, 0)
    return nodes
```

### Phase 2: Compute Node Intrinsic Features

#### For OPERATORS (ADD, SUB, MUL, NEG, AND, OR, XOR, NOT)

```python
def compute_operator_features(node: ASTNode, depth: int, is_root: bool,
                              max_depth: int = MAX_AST_DEPTH_NORM) -> Dict:
    """
    Compute features for operator nodes.

    Args:
        node: AST node (operator type)
        depth: Node depth in AST
        is_root: Whether this is the root node
        max_depth: Maximum depth for normalization (from constants)

    Returns:
        {
            'node_type': int,
            'domain': int,
            'depth': float,       # Normalized: 1.0 - (depth / max_depth)
            'is_root': float      # 1.0 or 0.0
        }
    """
    node_type_id = NODE_TYPE_MAP[node.type]
    domain = Domain.from_node_type(node_type_id)

    # Normalize depth (root=1.0, decreases with depth)
    normalized_depth = 1.0 - (depth / max_depth)

    return {
        'node_type': node_type_id,
        'domain': domain,
        'depth': normalized_depth,
        'is_root': 1.0 if is_root else 0.0
    }
```

#### For VARIABLES

```python
def compute_variable_features(node: ASTNode, depth: int,
                              max_depth: int = MAX_AST_DEPTH_NORM) -> Dict:
    """
    Compute features for variable nodes.

    Args:
        node: AST node (VAR type)
        depth: Node depth in AST
        max_depth: Maximum depth for normalization (from constants)

    Returns:
        {
            'node_type': int,
            'domain': int (LEAF),
            'var_index': int,
            'depth': float,
            'is_root': float,
            'context_domains': None  # Computed in Phase 4
        }
    """
    node_type_id = NODE_TYPE_MAP['VAR']

    # Extract variable index (x0-x7)
    var_index = 0
    if node.value and node.value.startswith('x'):
        try:
            var_index = int(node.value[1:])
            var_index = min(max(var_index, 0), 7)  # Clamp to [0, 7]
        except ValueError:
            var_index = 0

    normalized_depth = 1.0 - (depth / max_depth)

    return {
        'node_type': node_type_id,
        'domain': int(Domain.LEAF),
        'var_index': var_index,
        'depth': normalized_depth,
        'is_root': 0.0,  # Variables never at root
        'context_domains': None  # Filled in Phase 4
    }
```

#### For CONSTANTS

```python
def compute_constant_features(node: ASTNode, depth: int,
                              max_depth: int = MAX_AST_DEPTH_NORM,
                              const_norm: float = CONST_VALUE_NORM) -> Dict:
    """
    Compute features for constant nodes.

    Args:
        node: AST node (CONST type)
        depth: Node depth in AST
        max_depth: Maximum depth for normalization (from constants)
        const_norm: Constant value normalization factor (from constants)

    Returns:
        {
            'node_type': int,
            'domain': int (LEAF),
            'const_value': float,  # Normalized to [-1, 1]
            'is_zero': float,
            'is_one': float,
            'is_neg_one': float,
            'depth': float,
            'is_root': float,
            'context_domains': None  # Computed in Phase 4
        }
    """
    node_type_id = NODE_TYPE_MAP['CONST']

    # Parse constant value
    try:
        const_val = int(node.value) if node.value else 0
    except ValueError:
        const_val = 0

    # Normalize to [-1, 1] using constant from config
    normalized_value = max(-1.0, min(1.0, const_val / const_norm))

    # Special value flags
    is_zero = 1.0 if const_val == 0 else 0.0
    is_one = 1.0 if const_val == 1 else 0.0
    is_neg_one = 1.0 if const_val == -1 else 0.0

    normalized_depth = 1.0 - (depth / max_depth)

    return {
        'node_type': node_type_id,
        'domain': int(Domain.LEAF),
        'const_value': normalized_value,
        'is_zero': is_zero,
        'is_one': is_one,
        'is_neg_one': is_neg_one,
        'depth': normalized_depth,
        'is_root': 0.0,  # Constants never at root
        'context_domains': None  # Filled in Phase 4
    }
```

#### Dispatcher Function

```python
def compute_node_features(metadata: Dict) -> Dict:
    """
    Dispatch to appropriate feature computation based on node type.

    Args:
        metadata: Dict from collect_nodes_with_metadata() with keys:
            - node: ASTNode
            - depth: int
            - parent_id: Optional[int]

    Returns:
        Feature dict appropriate for the node type
    """
    node = metadata['node']
    depth = metadata['depth']
    is_root = metadata['parent_id'] is None
    node_type_id = NODE_TYPE_MAP.get(node.type, NODE_TYPE_MAP.get(node.type.upper(), 0))

    if NodeType.is_terminal(node_type_id):
        if node.type in ('VAR', 'var') or node_type_id == NodeType.VAR:
            return compute_variable_features(node, depth)
        else:
            return compute_constant_features(node, depth)
    else:
        return compute_operator_features(node, depth, is_root)
```

### Phase 3: Build Edge List with Context Domains

```python
def build_edges_with_context(nodes_metadata: List[Dict],
                             node_features: List[Dict]) -> List[Tuple[int, int, int, int]]:
    """
    Build edge list with context domains.

    Args:
        nodes_metadata: List of dicts from collect_nodes_with_metadata()
        node_features: Precomputed features from Phase 2 (avoids recomputation)

    Returns:
        List of (src, dst, edge_type, context_domain) tuples
    """
    edges = []

    for i, metadata in enumerate(nodes_metadata):
        node = metadata['node']
        parent_domain = node_features[i]['domain']

        if node.is_unary() and len(node.children) >= 1:
            # Unary operator edges
            child_idx = find_child_index(node.children[0], nodes_metadata)
            child_domain = node_features[child_idx]['domain']

            # Forward: parent -> child (parent's domain context)
            edges.append((i, child_idx, int(EdgeType.UNARY_OPERAND), parent_domain))

            # Inverse: child -> parent (child's domain context)
            edges.append((child_idx, i, int(EdgeType.UNARY_OPERAND_INV), child_domain))

        elif node.is_binary():
            # Binary operator edges
            if len(node.children) >= 1:
                child_idx = find_child_index(node.children[0], nodes_metadata)
                child_domain = node_features[child_idx]['domain']

                # Forward: parent -> left child
                edges.append((i, child_idx, int(EdgeType.LEFT_OPERAND), parent_domain))

                # Inverse: left child -> parent
                edges.append((child_idx, i, int(EdgeType.LEFT_OPERAND_INV), child_domain))

            if len(node.children) >= 2:
                child_idx = find_child_index(node.children[1], nodes_metadata)
                child_domain = node_features[child_idx]['domain']

                # Forward: parent -> right child
                edges.append((i, child_idx, int(EdgeType.RIGHT_OPERAND), parent_domain))

                # Inverse: right child -> parent
                edges.append((child_idx, i, int(EdgeType.RIGHT_OPERAND_INV), child_domain))

            # Domain bridge edges (boolean <-> arithmetic transitions)
            if NodeType.is_boolean(node_features[i]['node_type']):
                for child in node.children:
                    child_idx = find_child_index(child, nodes_metadata)
                    child_type = node_features[child_idx]['node_type']

                    if NodeType.is_arithmetic(child_type):
                        # Bridge DOWN: boolean parent -> arithmetic child
                        edges.append((i, child_idx, int(EdgeType.DOMAIN_BRIDGE_DOWN), parent_domain))

                        # Bridge UP: arithmetic child -> boolean parent
                        edges.append((child_idx, i, int(EdgeType.DOMAIN_BRIDGE_UP), child_domain))

    return edges

def find_child_index(child_node: ASTNode, nodes_metadata: List[Dict]) -> int:
    """
    Find index of child node in nodes_metadata list.

    Uses _graph_id assigned during collect_nodes_with_metadata() for O(1) lookup.
    Falls back to identity comparison if _graph_id not available.
    """
    # O(1) lookup via pre-assigned ID (preferred)
    if hasattr(child_node, '_graph_id'):
        return child_node._graph_id

    # Fallback to identity comparison (for backward compatibility)
    for i, metadata in enumerate(nodes_metadata):
        if metadata['node'] is child_node:
            return i

    raise ValueError(f"Child node not found in metadata: {child_node}")
```

### Phase 4: Compute Context Domains for Terminals

Context domains track which semantic domains (ARITHMETIC, BOOLEAN, BITWISE) use this terminal node.

```python
def compute_context_domains(nodes_metadata: List[Dict],
                           node_features: List[Dict],
                           edges: List[Tuple[int, int, int, int]]) -> Dict[int, List[int]]:
    """
    Compute context_domains multi-hot vector for each terminal node.

    Walks up the tree from each terminal to collect domains of ancestor operators.

    Args:
        nodes_metadata: Node metadata from Phase 1
        node_features: Precomputed features from Phase 2 (avoids recomputation)
        edges: Edge list from Phase 3

    Returns:
        Dict mapping node_id -> [ARITH, BOOL, BITWISE] (1 if used, 0 otherwise)
    """
    context_domains = {}

    # Build parent lookup (child_idx -> [parent_idx, ...])
    parent_map = defaultdict(list)
    for src, dst, edge_type, _ in edges:
        if edge_type in [int(EdgeType.LEFT_OPERAND_INV),
                        int(EdgeType.RIGHT_OPERAND_INV),
                        int(EdgeType.UNARY_OPERAND_INV)]:
            # Inverse edges point from child to parent
            parent_map[src].append(dst)

    # For each terminal node, walk up to root collecting domains
    for i, metadata in enumerate(nodes_metadata):
        node = metadata['node']

        if NodeType.is_terminal(node_features[i]['node_type']):
            # Walk up tree collecting operator domains
            visited_domains = set()
            queue = [i]
            visited = set()

            while queue:
                node_idx = queue.pop(0)
                if node_idx in visited:
                    continue
                visited.add(node_idx)

                node_domain = node_features[node_idx]['domain']

                # Record operator domain (skip LEAF)
                if node_domain in [int(Domain.ARITHMETIC),
                                  int(Domain.BOOLEAN),
                                  int(Domain.BITWISE)]:
                    visited_domains.add(node_domain)

                # Add parents to queue
                for parent_idx in parent_map.get(node_idx, []):
                    queue.append(parent_idx)

            # Convert to multi-hot [ARITH, BOOL, BITWISE]
            context_domains[i] = [
                1.0 if int(Domain.ARITHMETIC) in visited_domains else 0.0,
                1.0 if int(Domain.BOOLEAN) in visited_domains else 0.0,
                1.0 if int(Domain.BITWISE) in visited_domains else 0.0
            ]
        else:
            # Non-terminals don't have context_domains feature
            context_domains[i] = [0.0, 0.0, 0.0]

    return context_domains
```

### Phase 5: Featurize Nodes and Edges

Convert logical feature dicts to tensors for PyTorch Geometric.

```python
def featurize_graph(nodes_metadata: List[Dict],
                   node_features: List[Dict],
                   edges: List[Tuple[int, int, int, int]],
                   context_domains: Dict[int, List[int]]) -> Data:
    """
    Convert nodes and edges to PyTorch Geometric Data object.

    Args:
        nodes_metadata: Node metadata from Phase 1
        node_features: Precomputed features from Phase 2
        edges: Edge list from Phase 3
        context_domains: Context domains from Phase 4

    Returns:
        Data object with:
            x: Dict of tensors (rich features)
            edge_index: [2, num_edges]
            edge_type: [num_edges]
            edge_attr: [num_edges, 2] - (edge_type, context_domain) for embedding
    """
    num_nodes = len(nodes_metadata)

    # Merge context_domains into node features
    node_features_list = []
    for i, features in enumerate(node_features):
        features_copy = features.copy()
        features_copy['context_domains'] = context_domains[i]
        node_features_list.append(features_copy)

    # Convert to tensor format
    # Each node gets a variable-length feature vector depending on type
    # Model will embed and project to hidden_dim

    # Strategy: Create structured dict that encoder can process
    node_data = {
        'node_type': torch.tensor([f['node_type'] for f in node_features_list], dtype=torch.long),
        'domain': torch.tensor([f['domain'] for f in node_features_list], dtype=torch.long),
        'depth': torch.tensor([f['depth'] for f in node_features_list], dtype=torch.float),
        'is_root': torch.tensor([f['is_root'] for f in node_features_list], dtype=torch.float),
        'context_domains': torch.tensor([f['context_domains'] for f in node_features_list], dtype=torch.float),
    }

    # Add type-specific features (sparse)
    var_indices = []
    const_values = []
    const_flags = []

    for f in node_features_list:
        # VAR features
        var_indices.append(f.get('var_index', 0))

        # CONST features
        const_values.append(f.get('const_value', 0.0))
        const_flags.append([
            f.get('is_zero', 0.0),
            f.get('is_one', 0.0),
            f.get('is_neg_one', 0.0)
        ])

    node_data['var_index'] = torch.tensor(var_indices, dtype=torch.long)
    node_data['const_value'] = torch.tensor(const_values, dtype=torch.float)
    node_data['const_flags'] = torch.tensor(const_flags, dtype=torch.float)

    # Build edge tensors
    edge_src = [e[0] for e in edges]
    edge_dst = [e[1] for e in edges]
    edge_types = [e[2] for e in edges]
    edge_contexts = [e[3] for e in edges]

    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    edge_type = torch.tensor(edge_types, dtype=torch.long)
    edge_attr = torch.stack([
        torch.tensor(edge_types, dtype=torch.long),
        torch.tensor(edge_contexts, dtype=torch.long)
    ], dim=1)

    return Data(
        x=node_data,  # Dict of tensors, not a single tensor
        edge_index=edge_index,
        edge_type=edge_type,
        edge_attr=edge_attr
    )
```

---

## Model Integration

### Critical Design: Handling Type Mismatch

**Problem:** The existing `HGTEncoder._to_heterogeneous()` calls `self.node_type_embed(x)` expecting `x` to be a `LongTensor[num_nodes]`. With rich features, `x` is a `Dict[str, Tensor]`. This causes a runtime TypeError.

**Solution:** Add `use_rich_features: bool` flag to encoder initialization. When True, use `RichNodeFeatureEmbedding` instead of simple `node_type_embed`.

```python
# In HGTEncoder.__init__()
def __init__(self, ..., use_rich_features: bool = False, use_edge_context: bool = False):
    ...
    self.use_rich_features = use_rich_features
    self.use_edge_context = use_edge_context

    if use_rich_features:
        self.node_feature_embed = RichNodeFeatureEmbedding(hidden_dim)
    else:
        self.node_type_embed = nn.Embedding(num_node_types, hidden_dim)

    # Only create edge context embedding if needed (Phase 5)
    # Avoids 10-15% compute overhead when not using edge context
    if use_edge_context:
        self.edge_context_embed = EdgeContextEmbedding(hidden_dim)
```

### Node Feature Embedding Layer

Add to encoder classes (HGTEncoder, RGCNEncoder).

```python
# In src/models/encoder.py

class RichNodeFeatureEmbedding(nn.Module):
    """
    Embed rich node features to hidden dimension.

    Handles variable-length feature vectors for different node types.
    """

    def __init__(self, hidden_dim: int = 768,
                 node_type_embed_dim: int = 64,
                 domain_embed_dim: int = 32,
                 var_embed_dim: int = 16):
        super().__init__()

        self.node_type_embed = nn.Embedding(10, node_type_embed_dim)  # 10 node types
        self.domain_embed = nn.Embedding(5, domain_embed_dim)         # 5 domains
        self.var_index_embed = nn.Embedding(8, var_embed_dim)         # 8 variables

        # Total input dimension after embedding + scalars
        # = node_type_embed + domain_embed + var_embed + depth(1) + is_root(1)
        #   + const_value(1) + const_flags(3) + context_domains(3)
        total_dim = node_type_embed_dim + domain_embed_dim + var_embed_dim + 9

        self.projection = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU()
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.node_type_embed.weight, std=0.02)
        nn.init.normal_(self.domain_embed.weight, std=0.02)
        nn.init.normal_(self.var_index_embed.weight, std=0.02)

        for layer in self.projection:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, node_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Embed node features.

        Args:
            node_data: Dict with keys:
                - node_type: [num_nodes] long
                - domain: [num_nodes] long
                - var_index: [num_nodes] long
                - depth: [num_nodes] float
                - is_root: [num_nodes] float
                - const_value: [num_nodes] float
                - const_flags: [num_nodes, 3] float
                - context_domains: [num_nodes, 3] float

        Returns:
            [num_nodes, hidden_dim] embedded features
        """
        # Embed categorical features
        node_type_emb = self.node_type_embed(node_data['node_type'])     # [N, 64]
        domain_emb = self.domain_embed(node_data['domain'])               # [N, 32]
        var_emb = self.var_index_embed(node_data['var_index'])           # [N, 16]

        # Concatenate all features
        features = torch.cat([
            node_type_emb,                              # 64
            domain_emb,                                 # 32
            var_emb,                                    # 16
            node_data['depth'].unsqueeze(-1),           # 1
            node_data['is_root'].unsqueeze(-1),         # 1
            node_data['const_value'].unsqueeze(-1),     # 1
            node_data['const_flags'],                   # 3
            node_data['context_domains']                # 3
        ], dim=-1)  # Total: 121

        # Project to hidden_dim
        return self.projection(features)
```

### Edge Context Embedding (for HGT)

```python
class EdgeContextEmbedding(nn.Module):
    """
    Embed edge features: edge_type and context_domain.
    """

    def __init__(self, hidden_dim: int = 768,
                 edge_type_embed_dim: int = 32,
                 context_domain_embed_dim: int = 16):
        super().__init__()

        self.edge_type_embed = nn.Embedding(8, edge_type_embed_dim)           # 8 edge types
        self.context_domain_embed = nn.Embedding(5, context_domain_embed_dim) # 5 domains

        total_dim = edge_type_embed_dim + context_domain_embed_dim
        self.projection = nn.Linear(total_dim, hidden_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.edge_type_embed.weight, std=0.02)
        nn.init.normal_(self.context_domain_embed.weight, std=0.02)
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Embed edge attributes.

        Args:
            edge_attr: [num_edges, 2] with (edge_type, context_domain)

        Returns:
            [num_edges, hidden_dim] embedded edge features
        """
        edge_type = edge_attr[:, 0]
        context_domain = edge_attr[:, 1]

        edge_type_emb = self.edge_type_embed(edge_type)                # [E, 32]
        context_emb = self.context_domain_embed(context_domain)        # [E, 16]

        features = torch.cat([edge_type_emb, context_emb], dim=-1)    # [E, 48]
        return self.projection(features)                               # [E, hidden_dim]
```

### Modified HGT Attention

HGT natively uses edge-type-specific transformations. We extend this with additive edge context.

**Conceptual modification** (actual HGTConv is in torch_geometric):

```python
# Pseudocode - actual implementation requires modifying HGTConv or wrapping it

class HGTWithEdgeContext(nn.Module):
    """
    HGT layer with edge context injection into attention.

    Standard HGT attention:
        attention = softmax(Q · W_edge[edge_type] · K^T / sqrt(d))

    Modified attention:
        context_emb = EdgeContextEmbedding(context_domain)
        K_contextual = K + context_emb
        attention = softmax(Q · W_edge[edge_type] · K_contextual^T / sqrt(d))
    """

    def __init__(self, hidden_dim, num_heads, metadata, use_edge_context=True):
        super().__init__()
        self.use_edge_context = use_edge_context

        # Standard HGTConv
        self.hgt_conv = HGTConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            metadata=metadata,
            heads=num_heads
        )

        if use_edge_context:
            self.edge_context_embed = EdgeContextEmbedding(hidden_dim)

    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None):
        """
        Args:
            x_dict: {node_type: [n, hidden_dim]}
            edge_index_dict: {(src_t, edge_t, dst_t): [2, n]}
            edge_attr_dict: {(src_t, edge_t, dst_t): [n, 2]} - optional edge attributes

        Returns:
            x_dict_out: {node_type: [n, hidden_dim]}
        """
        if self.use_edge_context and edge_attr_dict is not None:
            # Inject edge context into messages
            # This requires access to HGTConv internals or custom implementation

            # For each edge triplet, embed context and add to key vectors
            # (Implementation details depend on torch_geometric version)

            # Simplified approach: Use edge attributes as additive bias in attention
            # Actual implementation would modify HGTConv source or use message passing framework
            pass

        return self.hgt_conv(x_dict, edge_index_dict)
```

**Practical Implementation Strategy:**

Since `torch_geometric.nn.HGTConv` doesn't expose internal attention computation, we have two options:

1. **Copy and modify HGTConv source** - Add edge context injection to attention computation
2. **Post-HGT context fusion** - Apply edge context as a separate message passing step after HGT

**Recommended: Post-HGT Context Fusion** (simpler, less invasive):

```python
class HGTEncoderWithRichFeatures(HGTEncoder):
    """
    HGT encoder with rich node features and optional edge context fusion.

    Handles the type mismatch between dict node features and tensor-based processing.
    """

    def __init__(self, *args, use_rich_features: bool = True,
                 use_edge_context: bool = False, **kwargs):
        super().__init__(*args, **kwargs)

        self.use_rich_features = use_rich_features
        self.use_edge_context = use_edge_context

        if use_rich_features:
            # Replace simple embedding with rich feature embedding
            self.node_feature_embed = RichNodeFeatureEmbedding(self._hidden_dim)
            # Remove the simple embedding to save memory
            if hasattr(self, 'node_type_embed'):
                del self.node_type_embed

        if use_edge_context:
            self.edge_context_embed = EdgeContextEmbedding(self._hidden_dim)
            # Context fusion layer: applies edge-contextualized messages
            self.context_fusion = nn.ModuleList([
                nn.Linear(self._hidden_dim * 2, self._hidden_dim)
                for _ in range(len(self.convs))
            ])

    def _forward_impl(self, x, edge_index, batch, edge_type=None, edge_attr=None):
        """
        Extended forward pass with rich features and optional edge context.

        Args:
            x: Either LongTensor[num_nodes] (simple) or Dict[str, Tensor] (rich)
            edge_attr: [num_edges, 2] with (edge_type, context_domain) - optional
        """
        # Step 1: Embed node features based on mode
        if self.use_rich_features and isinstance(x, dict):
            # Rich features: x is dict, embed via RichNodeFeatureEmbedding
            h = self.node_feature_embed(x)  # [N, hidden_dim]
            node_types = x['node_type']  # For heterogeneous grouping
        else:
            # Simple features: x is LongTensor of node type IDs
            h = self.node_type_embed(x)  # [N, hidden_dim]
            node_types = x

        # Step 2: Convert to heterogeneous format for HGT
        x_dict, edge_index_dict = self._to_heterogeneous_from_embeddings(
            h, node_types, edge_index, edge_type
        )

        # Step 3: Embed edge context if enabled (Phase 5 only)
        edge_context = None
        if self.use_edge_context and edge_attr is not None:
            edge_context = self.edge_context_embed(edge_attr)  # [E, hidden_dim]

        # Step 4: Apply HGT layers
        for layer_idx, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            # Standard HGT message passing
            x_dict_new = conv(x_dict, edge_index_dict)

            # Apply edge context fusion (Phase 5 only)
            if edge_context is not None and self.use_edge_context:
                x_dict_new = self._apply_edge_context(
                    x_dict, x_dict_new, edge_index, edge_context, layer_idx
                )

            # Residual + norm
            for ntype in x_dict:
                if ntype in x_dict_new:
                    x_dict[ntype] = norm(x_dict[ntype] + self.dropout(x_dict_new[ntype]))
                    x_dict[ntype] = F.elu(x_dict[ntype])

        # Step 5: Convert back to flat format
        return self._flatten_heterogeneous(x_dict, node_types)

    def _to_heterogeneous_from_embeddings(self, h, node_types, edge_index, edge_type):
        """
        Convert pre-embedded node features to heterogeneous format.

        Unlike _to_heterogeneous() which embeds node types, this takes
        already-embedded features and groups them by node type.
        """
        # Group nodes by type
        x_dict = {}
        node_offsets = {}

        for ntype in range(self.num_node_types):
            mask = node_types == ntype
            if mask.any():
                x_dict[ntype] = h[mask]
                node_offsets[ntype] = mask.nonzero(as_tuple=True)[0]

        # Build edge_index_dict (same as original _to_heterogeneous)
        edge_index_dict = {}
        src_types = node_types[edge_index[0]]
        dst_types = node_types[edge_index[1]]

        for triplet in self.VALID_TRIPLETS:
            src_t, e_t, dst_t = triplet
            mask = (src_types == src_t) & (edge_type == e_t) & (dst_types == dst_t)
            if mask.any():
                edges = edge_index[:, mask]
                if src_t in node_offsets and dst_t in node_offsets:
                    # Remap to local indices
                    src_local = self._remap_indices(edges[0], node_offsets[src_t])
                    dst_local = self._remap_indices(edges[1], node_offsets[dst_t])
                    edge_index_dict[(src_t, e_t, dst_t)] = torch.stack([src_local, dst_local])

        return x_dict, edge_index_dict

    def _apply_edge_context(self, x_dict_in, x_dict_out, edge_index, edge_context, layer_idx):
        """
        Apply edge context as additional message pass.

        For each node, aggregate context-weighted messages from neighbors.
        """
        # Simplified: Apply context as gate on messages
        # Full implementation would recompute messages with context-conditioned attention

        # Convert to flat format for aggregation
        num_nodes = sum(len(x_dict_in[t]) for t in x_dict_in)
        h_flat = torch.zeros(num_nodes, self.hidden_dim, device=edge_context.device)

        # ... (flatten x_dict_out, apply scatter with edge_context weights, unflatten)

        return x_dict_out  # Placeholder
```

**Note:** Full edge context integration into HGT attention requires either:
- Modifying `torch_geometric/nn/conv/hgt_conv.py` source
- Implementing custom heterogeneous message passing

For initial implementation, **proceed without HGT attention modification** and add as Phase 2 enhancement.

---

## Implementation Phases

### Phase 1: Core Infrastructure (Priority: P0)

**CRITICAL FIRST STEP:** Add constants to `src/constants.py` BEFORE writing any other code.

**Files to create/modify:**

1. **`src/constants.py`** (modify FIRST)
   - Add `USE_RICH_FEATURES`, `USE_EDGE_CONTEXT` flags
   - Add embedding dimension constants
   - Add normalization constants
   - See "Configuration Constants" section above

2. **`src/models/edge_types.py`** (modify)
   - Add `Domain` enum
   - Add `Domain.from_node_type()` method

3. **`src/data/rich_features.py`** (new)
   - `collect_nodes_with_metadata()` (with `_graph_id` assignment)
   - `compute_operator_features()`
   - `compute_variable_features()`
   - `compute_constant_features()`
   - `compute_node_features()` (dispatcher)
   - `build_edges_with_context()` (accepts precomputed node_features)
   - `compute_context_domains()` (accepts precomputed node_features)
   - `featurize_graph()` (accepts all precomputed data)
   - `find_child_index()` (ID-based lookup)

4. **`src/models/node_embedding.py`** (new)
   - `RichNodeFeatureEmbedding` class

5. **`src/models/edge_embedding.py`** (new)
   - `EdgeContextEmbedding` class (created but NOT used until Phase 5)

**Testing:**
- Unit test for each feature computation function
- Integration test: parse expression -> featurize -> check tensor shapes
- Verify `_graph_id` assignment and lookup works correctly

### Phase 2: Encoder Integration (Priority: P0)

**Files to modify:**

1. **`src/models/encoder.py`**
   - Add `HGTEncoderWithRichFeatures` class (or modify `HGTEncoder`)
   - Add `use_rich_features: bool = False` and `use_edge_context: bool = False` to `__init__()`
   - Conditionally create `RichNodeFeatureEmbedding` vs `node_type_embed`
   - Add `_to_heterogeneous_from_embeddings()` method for pre-embedded features
   - Update `_forward_impl()` to handle dict `x` input
   - Accept `edge_attr` argument but IGNORE it (until Phase 5)
   - Update `requires_node_features` property to return True when rich features enabled

2. **`src/models/encoder.py`**
   - Modify `RGCNEncoder` similarly with `use_rich_features` flag

3. **`src/models/encoder_registry.py`**
   - Register new encoder variants or add flags to existing registrations

**Key Design Decision:**
- `use_edge_context=False` by default to avoid 10-15% compute overhead
- Edge context only activated in Phase 5

**Testing:**
- Forward pass test with `use_rich_features=True` and dict `x`
- Forward pass test with `use_rich_features=False` and tensor `x` (backward compat)
- Gradient flow test
- Output dimension validation
- Memory usage comparison

### Phase 3: Dataset Integration (Priority: P0)

**Files to modify:**

1. **`src/data/dataset.py`**
   - Modify `ScaledMBADataset._build_optimized_graph()` to call `featurize_graph()`
   - Update `__getitem__()` to return `edge_attr` in graph Data object

2. **`src/data/ast_parser.py`**
   - Add `ast_to_rich_graph()` function using new pipeline

**Testing:**
- Dataset loading test
- Collate function compatibility
- Batch construction with rich features

### Phase 4: Training Pipeline (Priority: P1)

**Files to modify:**

1. **`src/constants.py`**
   - Add `USE_RICH_FEATURES: bool = True`
   - Add embedding dimension constants

2. **Training scripts**
   - Pass `use_rich_features` flag to model constructor
   - No other changes needed (backward compatible)

**Testing:**
- End-to-end training smoke test (1 batch)
- Checkpoint save/load with rich features

### Phase 5: HGT Attention Modification (Priority: P2)

**Implementation decision pending:**
- Option A: Custom HGTConv implementation
- Option B: Post-attention context fusion (implemented in Phase 2)
- Option C: Use edge_attr in different way

**Deferred to after Phase 1-4 validation.**

---

## Backward Compatibility

**Ensuring existing code continues to work:**

1. **Feature flag:** All encoders accept `use_rich_features=False` (default initially)
   - When False, falls back to simple `node_type` embedding

2. **Dataset compatibility:**
   - `expr_to_graph()` returns simple graph (old format)
   - `ast_to_rich_graph()` returns rich graph (new format)
   - Dataset classes detect graph format automatically

3. **Checkpoint compatibility:**
   - Model saved with `use_rich_features=True` cannot load in old code
   - Model saved with `use_rich_features=False` loads in both old and new code
   - Add version field to checkpoint metadata

---

## Testing Strategy

### Unit Tests

Create `tests/test_rich_features.py`:

```python
def test_domain_enum():
    """Test domain classification from node types."""
    assert Domain.from_node_type(NodeType.ADD) == Domain.ARITHMETIC
    assert Domain.from_node_type(NodeType.AND) == Domain.BOOLEAN
    assert Domain.from_node_type(NodeType.VAR) == Domain.LEAF

def test_operator_features():
    """Test operator feature computation."""
    node = ASTNode(type='ADD')
    features = compute_operator_features(node, depth=2, is_root=True)
    assert features['node_type'] == NodeType.ADD
    assert features['domain'] == Domain.ARITHMETIC
    assert 0.0 <= features['depth'] <= 1.0
    assert features['is_root'] == 1.0

def test_variable_features():
    """Test variable feature computation."""
    node = ASTNode(type='VAR', value='x3')
    features = compute_variable_features(node, depth=5)
    assert features['node_type'] == NodeType.VAR
    assert features['var_index'] == 3
    assert features['domain'] == Domain.LEAF

def test_constant_features():
    """Test constant feature computation."""
    node = ASTNode(type='CONST', value='0')
    features = compute_constant_features(node, depth=6)
    assert features['is_zero'] == 1.0
    assert features['is_one'] == 0.0
    assert features['const_value'] == 0.0

def test_context_domains():
    """Test context domain propagation for terminals."""
    # Build small AST: ADD(VAR(x), CONST(1))
    # VAR should have context_domains = [1, 0, 0] (ARITHMETIC only)
    expr = "x + 1"
    ast = parse_to_ast(expr)
    nodes_metadata = collect_nodes_with_metadata(ast)
    edges = build_edges_with_context(nodes_metadata)
    context_domains = compute_context_domains(nodes_metadata, edges)

    # Find VAR node
    var_idx = next(i for i, m in enumerate(nodes_metadata) if m['node'].type == 'VAR')
    assert context_domains[var_idx][0] == 1.0  # ARITHMETIC
    assert context_domains[var_idx][1] == 0.0  # BOOLEAN

def test_edge_context_assignment():
    """Test context domain assignment for edges."""
    expr = "(x & y) + 1"  # Mixed boolean and arithmetic
    ast = parse_to_ast(expr)
    nodes_metadata = collect_nodes_with_metadata(ast)
    edges = build_edges_with_context(nodes_metadata)

    # Check that edges have correct context domains
    for src, dst, edge_type, context_domain in edges:
        if edge_type == EdgeType.LEFT_OPERAND:
            # Forward structural edge: uses parent (src) domain
            parent_domain = compute_node_features(nodes_metadata[src])['domain']
            assert context_domain == parent_domain
        elif edge_type == EdgeType.LEFT_OPERAND_INV:
            # Inverse edge: uses child (src) domain
            child_domain = compute_node_features(nodes_metadata[src])['domain']
            assert context_domain == child_domain

def test_rich_graph_construction():
    """Test end-to-end graph construction with rich features."""
    expr = "(x & y) + (x ^ 2)"
    graph = ast_to_rich_graph(expr)

    assert 'x' in graph and isinstance(graph.x, dict)
    assert 'node_type' in graph.x
    assert 'domain' in graph.x
    assert 'edge_attr' in graph
    assert graph.edge_attr.shape[1] == 2  # (edge_type, context_domain)
```

### Integration Tests

Create `tests/test_rich_encoder.py`:

```python
def test_node_embedding():
    """Test rich node feature embedding."""
    embedding = RichNodeFeatureEmbedding(hidden_dim=256)

    # Create mock node data
    node_data = {
        'node_type': torch.tensor([0, 4, 8, 9]),  # ADD, AND, VAR, CONST
        'domain': torch.tensor([0, 1, 3, 3]),     # ARITH, BOOL, LEAF, LEAF
        'var_index': torch.tensor([0, 0, 2, 0]),
        'depth': torch.tensor([1.0, 0.8, 0.6, 0.6]),
        'is_root': torch.tensor([1.0, 0.0, 0.0, 0.0]),
        'const_value': torch.tensor([0.0, 0.0, 0.0, 0.5]),
        'const_flags': torch.tensor([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]),
        'context_domains': torch.tensor([[0, 0, 0], [0, 0, 0], [1, 1, 0], [1, 0, 0]])
    }

    output = embedding(node_data)
    assert output.shape == (4, 256)
    assert not torch.isnan(output).any()

def test_hgt_with_rich_features():
    """Test HGT encoder with rich features."""
    encoder = HGTEncoder(hidden_dim=256, use_rich_features=True)

    # Create mock graph
    graph = create_mock_rich_graph()
    batch = torch.zeros(graph.x['node_type'].size(0), dtype=torch.long)

    output = encoder.forward(
        x=graph.x,
        edge_index=graph.edge_index,
        batch=batch,
        edge_type=graph.edge_type,
        edge_attr=graph.edge_attr
    )

    assert output.shape[0] == graph.x['node_type'].size(0)
    assert output.shape[1] == 256
    assert not torch.isnan(output).any()

def test_backward_compatibility():
    """Test that old graph format still works."""
    encoder = HGTEncoder(hidden_dim=256, use_rich_features=False)

    # Old format: x is simple node type tensor
    x = torch.tensor([0, 1, 8, 9], dtype=torch.long)
    edge_index = torch.tensor([[0, 0, 1, 1], [2, 3, 2, 3]], dtype=torch.long)
    edge_type = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    batch = torch.zeros(4, dtype=torch.long)

    output = encoder.forward(x, edge_index, batch, edge_type)
    assert output.shape == (4, 256)
```

### Ablation Study

After implementation, run ablation to measure impact:

```bash
# Baseline (no rich features)
python scripts/run_ablation.py --encoder hgt --rich-features False --run-id baseline

# With rich features
python scripts/run_ablation.py --encoder hgt --rich-features True --run-id rich_features

# Compare
python scripts/compare_ablations.py --baseline baseline --treatment rich_features
```

**Expected improvements:**
- Depth 8-10: +3-5% accuracy
- Depth 11-14: +5-10% accuracy
- Cross-domain expressions: +8-12% accuracy

---

## Feature Dimension Summary

### Node Feature Dimensions

| Component | Raw Size | Embedded Size | Source |
|-----------|----------|---------------|--------|
| node_type | 1 (10 classes) | 64 | Embedding |
| domain | 1 (5 classes) | 32 | Embedding |
| var_index | 1 (8 classes) | 16 | Embedding (sparse) |
| depth | 1 | 1 | Scalar |
| is_root | 1 | 1 | Scalar |
| const_value | 1 | 1 | Scalar |
| const_flags | 3 | 3 | Scalar |
| context_domains | 3 | 3 | Scalar (multi-hot) |
| **Total** | - | **121** | After concat |
| **Projected** | - | **hidden_dim** (256 or 768) | Linear + LN |

### Edge Feature Dimensions

| Component | Raw Size | Embedded Size | Source |
|-----------|----------|---------------|--------|
| edge_type | 1 (8 classes) | 32 | Embedding |
| context_domain | 1 (5 classes) | 16 | Embedding |
| **Total** | - | **48** | After concat |
| **Projected** | - | **hidden_dim** (optional) | Linear |

---

## Configuration Constants

**CRITICAL:** These constants MUST be added to `src/constants.py` in Phase 1, BEFORE any feature computation code is written. All modules import from here to ensure consistency.

Add to `src/constants.py` after line 259 (after `SCALED_CURRICULUM_STAGES`):

```python
# =============================================================================
# RICH FEATURE SYSTEM
# =============================================================================

# Feature flags (for backward compatibility)
USE_RICH_FEATURES: bool = True      # Enable rich node/edge features
USE_EDGE_CONTEXT: bool = False      # Enable edge context in HGT attention (Phase 5)

# Node feature embedding dimensions
NODE_TYPE_EMBED_DIM: int = 64       # Embedding dim for 10 node types
DOMAIN_EMBED_DIM: int = 32          # Embedding dim for 5 domains
VAR_INDEX_EMBED_DIM: int = 16       # Embedding dim for 8 variable indices

# Edge feature embedding dimensions
EDGE_TYPE_EMBED_DIM: int = 32       # Embedding dim for 8 edge types
CONTEXT_DOMAIN_EMBED_DIM: int = 16  # Embedding dim for context domain

# Feature normalization constants
MAX_AST_DEPTH_NORM: int = 14        # Max depth for normalization (matches curriculum)
CONST_VALUE_NORM: float = 256.0     # Constant value range [-256, 256] -> [-1, 1]

# Computed dimensions (for reference)
RICH_NODE_FEATURE_DIM: int = (
    NODE_TYPE_EMBED_DIM +  # 64
    DOMAIN_EMBED_DIM +     # 32
    VAR_INDEX_EMBED_DIM +  # 16
    9                       # depth(1) + is_root(1) + const_value(1) + const_flags(3) + context_domains(3)
)  # Total: 121

RICH_EDGE_FEATURE_DIM: int = EDGE_TYPE_EMBED_DIM + CONTEXT_DOMAIN_EMBED_DIM  # 48
```

**Import in all rich feature modules:**
```python
from src.constants import (
    USE_RICH_FEATURES, USE_EDGE_CONTEXT,
    NODE_TYPE_EMBED_DIM, DOMAIN_EMBED_DIM, VAR_INDEX_EMBED_DIM,
    EDGE_TYPE_EMBED_DIM, CONTEXT_DOMAIN_EMBED_DIM,
    MAX_AST_DEPTH_NORM, CONST_VALUE_NORM,
    RICH_NODE_FEATURE_DIM, RICH_EDGE_FEATURE_DIM
)
```

---

## Example Usage

```python
from src.data.ast_parser import parse_to_ast
from src.data.rich_features import ast_to_rich_graph
from src.models.encoder import HGTEncoder

# Parse expression
expr = "(x & y) + (x ^ y)"
ast = parse_to_ast(expr)

# Build rich graph
graph = ast_to_rich_graph(ast)

# graph.x is now a dict:
# {
#   'node_type': tensor([...]),
#   'domain': tensor([...]),
#   'depth': tensor([...]),
#   ...
# }

# graph.edge_attr: [num_edges, 2] = (edge_type, context_domain)

# Create encoder with rich features
encoder = HGTEncoder(
    hidden_dim=768,
    num_layers=12,
    num_heads=16,
    use_rich_features=True
)

# Forward pass
batch = torch.zeros(graph.x['node_type'].size(0), dtype=torch.long)
node_embeddings = encoder(
    x=graph.x,
    edge_index=graph.edge_index,
    batch=batch,
    edge_type=graph.edge_type,
    edge_attr=graph.edge_attr
)

# node_embeddings: [num_nodes, 768]
```

---

## Performance Considerations

### Memory Overhead

**Per graph:**
- Old format: `num_nodes * 1` (type ID only) = ~10-50 bytes
- Rich format: `num_nodes * (121 floats)` = ~500-2500 bytes
- Increase: **50x memory per graph**

**Mitigation:**
- Batch size reduction: 128 -> 64 (2x less graphs in memory)
- Pre-compute and cache features during dataset preprocessing
- Use mixed precision (float16) for non-critical features

### Computation Overhead

**Feature computation:**
- Old: O(num_nodes) for type ID lookup
- Rich: O(num_nodes + num_edges) for context domain propagation

**Mitigation:**
- Parallelize across batch (already done in DataLoader)
- Pre-compute during dataset generation (recommended for training)

### Training Time Impact

**Expected slowdown:** 10-15% due to:
- Larger embedding layers (121 input vs 1 input)
- Edge attribute processing
- Context domain computation

**Mitigation:**
- Use DataLoader with `num_workers > 0` for parallel feature computation
- Cache featurized graphs to disk if dataset fits in storage

---

## Open Questions

1. **HGT attention modification:**
   - Should we modify torch_geometric HGTConv source, or use post-attention fusion?
   - Answer: Start with post-attention fusion (Phase 2), defer native integration to Phase 5

2. **Context domain computation:**
   - Should we limit ancestor walk depth for context_domains?
   - Answer: No limit initially; evaluate if performance becomes issue

3. **Embedding dimension tuning:**
   - Are 64/32/16 the right embedding dims?
   - Answer: Use these as defaults; hyperparameter sweep after Phase 2

4. **Edge context vs. node context:**
   - Should we also add node-level context features (e.g., "used in boolean expression")?
   - Answer: context_domains already provides this for terminals; defer for operators

---

## Success Criteria

**Phase 1-2 complete when:**
- All unit tests pass
- Integration tests pass
- Forward pass runs without errors
- Backward pass computes gradients correctly
- Checkpoint save/load works

**Phase 3-4 complete when:**
- Can train for 1 epoch on small dataset (1000 samples)
- Loss decreases
- No memory leaks
- Inference pipeline works

**Full success when:**
- Accuracy improvement ≥3% on depth 10+ test set
- Training overhead ≤15%
- Memory usage ≤2x baseline
- Backward compatible with old checkpoints (use_rich_features=False)

---

## References

- **Edge Types:** `src/models/edge_types.py` - EdgeType and NodeType enums
- **AST Parser:** `src/data/ast_parser.py` - Expression parsing and graph construction
- **Encoder Base:** `src/models/encoder.py` - GNN encoder implementations
- **Dataset:** `src/data/dataset.py` - PyTorch dataset classes
- **Constants:** `src/constants.py` - Hyperparameters and dimensions
- **HOList Research:** Referenced in edge_types.py for bidirectional edge design

---

## Appendix: Example Feature Vectors

### Example 1: Arithmetic Operator (ADD at root)

```python
{
    'node_type': 0,          # ADD
    'domain': 0,             # ARITHMETIC
    'var_index': 0,          # N/A (not a variable)
    'depth': 1.0,            # Root node
    'is_root': 1.0,
    'const_value': 0.0,      # N/A (not a constant)
    'const_flags': [0, 0, 0],
    'context_domains': [0, 0, 0]  # Not a terminal
}
```

### Example 2: Boolean Operator (AND at depth 1)

```python
{
    'node_type': 4,          # AND
    'domain': 1,             # BOOLEAN
    'var_index': 0,          # N/A
    'depth': 0.93,           # (1.0 - 1/14)
    'is_root': 0.0,
    'const_value': 0.0,      # N/A
    'const_flags': [0, 0, 0],
    'context_domains': [0, 0, 0]  # Not a terminal
}
```

### Example 3: Variable (x2 at depth 3, used in ARITH and BOOL contexts)

```python
{
    'node_type': 8,          # VAR
    'domain': 3,             # LEAF
    'var_index': 2,          # x2
    'depth': 0.79,           # (1.0 - 3/14)
    'is_root': 0.0,
    'const_value': 0.0,      # N/A
    'const_flags': [0, 0, 0],
    'context_domains': [1, 1, 0]  # Used in ARITHMETIC and BOOLEAN
}
```

### Example 4: Constant (value=0 at depth 5)

```python
{
    'node_type': 9,          # CONST
    'domain': 3,             # LEAF
    'var_index': 0,          # N/A
    'depth': 0.64,           # (1.0 - 5/14)
    'is_root': 0.0,
    'const_value': 0.0,      # Normalized value
    'const_flags': [1, 0, 0], # is_zero=1
    'context_domains': [1, 0, 0]  # Used only in ARITHMETIC
}
```

### Example 5: Edge with Context

```python
# Edge: ADD (parent) -> VAR (child) via LEFT_OPERAND
(src=0, dst=2, edge_type=0, context_domain=0)
# edge_type=0: LEFT_OPERAND
# context_domain=0: ARITHMETIC (from parent ADD node)

# Inverse edge: VAR (child) -> ADD (parent) via LEFT_OPERAND_INV
(src=2, dst=0, edge_type=3, context_domain=3)
# edge_type=3: LEFT_OPERAND_INV
# context_domain=3: LEAF (from child VAR node)
```

---

## End of Document

**Next Steps:**
1. Review this plan with team
2. Create GitHub issues for each phase
3. Begin Phase 1 implementation
4. Set up CI tests for rich features
5. Schedule ablation study after Phase 4
