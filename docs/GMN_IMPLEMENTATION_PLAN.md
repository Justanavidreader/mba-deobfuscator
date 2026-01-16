# Graph Matching Networks (GMN) Implementation Plan

## 1. Overview

### 1.1 What is GMN?

Graph Matching Networks (GMN) enhance Siamese graph encoders by introducing **cross-graph attention** to detect node correspondences between two graphs. Unlike standard Siamese approaches that encode graphs independently, GMN enables nodes in one graph to "attend to" nodes in the other graph, capturing structural similarities and differences explicitly.

### 1.2 Why GMN Benefits MBA Equivalence Detection

**Current Limitation (Siamese Approach):**
```
Obfuscated Graph  → HGT Encoder → Embedding₁ ─┐
                                               ├─► Cosine Similarity
Simplified Graph  → HGT Encoder → Embedding₂ ─┘
```

Each graph is encoded in isolation. The model must learn to map semantically equivalent expressions to nearby points in embedding space, but cannot explicitly identify which nodes correspond across graphs.

**GMN Enhancement:**
```
Obfuscated Graph  → HGT Encoder ──┬──► Cross-Attention ──► Embedding₁
                                  ✕                          │
Simplified Graph  → HGT Encoder ──┴──► Cross-Attention ──► Embedding₂
                                                            │
                                                    Matching Score
```

Cross-attention enables the model to learn:
- **Variable correspondence**: "This `x` in the obfuscated expression corresponds to that `x` in the simplified expression"
- **Cancellation detection**: "These AND nodes in the obfuscated expression cancel out and have no correspondence"
- **Structural alignment**: "This subexpression structure matches that subexpression, even if operators differ"

**Concrete Example:**
```
Obfuscated:  (x & m) | (x & ~m)    Simplified: x

Without GMN:
  - Model learns: "expressions that evaluate to x map to similar embeddings"
  - Problem: Must memorize all algebraic identities

With GMN:
  - Cross-attention learns: "x nodes strongly attend to x node"
  - Cross-attention learns: "m and ~m nodes have no correspondence (they cancel)"
  - Matching vector captures: "these extra operations cancel to produce x"
  - Generalization: Model understands cancellation pattern, applies to new variables
```

### 1.3 Key Advantages

| Aspect | Siamese (Current) | GMN (Proposed) |
|--------|------------------|----------------|
| Node correspondence | Implicit (in embedding space) | Explicit (via attention) |
| Subgraph matching | O(embedding_dim) comparison | O(num_nodes) fine-grained |
| Cancellation detection | Must memorize patterns | Learns generic cancellation |
| Variable renaming | Requires data augmentation | Natural via correspondence |
| Interpretability | Opaque embedding similarity | Attention weights show matching |

---

## 2. Architecture Design

### 2.1 Core Components

#### 2.1.1 CrossGraphAttention Module

The fundamental building block that computes matching vectors between two graphs.

```python
class CrossGraphAttention(nn.Module):
    """
    Cross-graph attention for node correspondence detection.

    Given embeddings from two graphs, computes attention scores and
    matching vectors that capture how nodes in graph 1 correspond to
    nodes in graph 2.

    For MBA: Detects variable correspondence, identifies cancellations,
    and recognizes structural equivalences.
    """

    def __init__(self, hidden_dim: int = 768, dropout: float = 0.1):
        """
        Args:
            hidden_dim: Node embedding dimension (must match encoder output)
            dropout: Dropout probability for attention weights
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.scale = hidden_dim ** -0.5  # Scaling factor for dot-product attention

        # Linear projections for query, key, value
        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)

        # Dropout for attention weights
        self.attn_dropout = nn.Dropout(dropout)

        # Projection for matching vector
        self.match_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # Concat [h, attended]
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, h1: torch.Tensor, h2: torch.Tensor,
                mask1: Optional[torch.Tensor] = None,
                mask2: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute cross-graph attention from h1 to h2.

        Args:
            h1: [N1, hidden_dim] node embeddings from graph 1
            h2: [N2, hidden_dim] node embeddings from graph 2
            mask1: [N1] boolean mask for valid nodes in graph 1
            mask2: [N2] boolean mask for valid nodes in graph 2

        Returns:
            matching_vector: [N1, hidden_dim] matching representation for graph 1
            attention_weights: [N1, N2] attention weights (for visualization)
        """
        # Project to query, key, value
        Q = self.W_q(h1)  # [N1, hidden_dim]
        K = self.W_k(h2)  # [N2, hidden_dim]
        V = self.W_v(h2)  # [N2, hidden_dim]

        # Compute attention scores: how much each node in g1 attends to each node in g2
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [N1, N2]

        # Apply mask for padding (if graphs have different sizes)
        # Use -1e9 instead of -inf to avoid NaN when all elements in a row are masked
        if mask2 is not None:
            # mask2: [N2] or [N1, N2], expand if needed for broadcasting
            if mask2.dim() == 1:
                attn_scores = attn_scores.masked_fill(~mask2.unsqueeze(0), -1e9)
            else:
                attn_scores = attn_scores.masked_fill(~mask2, -1e9)

        # Softmax over graph 2 nodes
        attn_weights = F.softmax(attn_scores, dim=-1)  # [N1, N2]

        # Handle all-masked rows (no valid nodes to attend to) - replace NaN with uniform
        # This can happen when batch sizes are mismatched or graphs are empty
        nan_mask = torch.isnan(attn_weights)
        if nan_mask.any():
            uniform_weight = 1.0 / attn_weights.size(-1)
            attn_weights = torch.where(nan_mask, torch.full_like(attn_weights, uniform_weight), attn_weights)

        attn_weights = self.attn_dropout(attn_weights)

        # Compute attended representation: weighted sum of values
        h2_attended = torch.matmul(attn_weights, V)  # [N1, hidden_dim]

        # Matching vector: captures difference between h1 and what it attends to in h2
        # Intuition: If h1[i] strongly attends to h2[j] and they're similar,
        # the matching vector will be small (nodes correspond).
        # If h1[i] has no good match in h2, matching vector is large (node unique to g1).
        matching_input = torch.cat([h1, h2_attended], dim=-1)  # [N1, hidden_dim*2]
        matching_vector = self.match_projection(matching_input)  # [N1, hidden_dim]

        return matching_vector, attn_weights
```

**Key Design Decisions:**

1. **Why difference-based matching?**
   - `matching_vector = f(h1, h2_attended)` captures what's unique about h1 relative to h2
   - For equivalent nodes (e.g., same variable): matching vector ≈ 0
   - For cancelled nodes (e.g., `m & ~m`): matching vector is large (no good correspondence)

2. **Why separate Q, K, V projections?**
   - Standard attention pattern, allows model to learn task-specific transformations
   - Q/K projections learn "similarity" function, V projection learns "information to aggregate"

3. **Masking strategy:**
   - Graphs have variable sizes (depth-2 expression vs depth-14 expression)
   - Padding masks ensure attention doesn't leak to padding tokens

#### 2.1.2 Multi-Head CrossGraphAttention

Extends single-head attention to capture diverse matching patterns.

```python
class MultiHeadCrossGraphAttention(nn.Module):
    """
    Multi-head cross-graph attention for diverse matching patterns.

    Different heads can specialize:
      - Head 1: Variable correspondence (x ↔ x)
      - Head 2: Constant matching (5 ↔ 5)
      - Head 3: Structural patterns (AND-subtree ↔ simplified node)
      - Head 4: Cancellation detection (no correspondence)
    """

    def __init__(self, hidden_dim: int = 768, num_heads: int = 8, dropout: float = 0.1):
        """
        Args:
            hidden_dim: Must be divisible by num_heads
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Projections (shared across heads, split in forward)
        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)

        self.attn_dropout = nn.Dropout(dropout)

        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # Matching projection
        self.match_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, h1: torch.Tensor, h2: torch.Tensor,
                mask1: Optional[torch.Tensor] = None,
                mask2: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Multi-head cross-attention.

        Args:
            h1: [N1, hidden_dim]
            h2: [N2, hidden_dim]
            mask1: [N1] boolean mask
            mask2: [N2] boolean mask

        Returns:
            matching_vector: [N1, hidden_dim]
            attention_weights: [num_heads, N1, N2] (for visualization)
        """
        N1, N2 = h1.size(0), h2.size(0)

        # Project and reshape for multi-head: [N, hidden_dim] -> [N, num_heads, head_dim]
        Q = self.W_q(h1).view(N1, self.num_heads, self.head_dim).transpose(0, 1)  # [num_heads, N1, head_dim]
        K = self.W_k(h2).view(N2, self.num_heads, self.head_dim).transpose(0, 1)  # [num_heads, N2, head_dim]
        V = self.W_v(h2).view(N2, self.num_heads, self.head_dim).transpose(0, 1)  # [num_heads, N2, head_dim]

        # Compute attention: [num_heads, N1, head_dim] @ [num_heads, head_dim, N2]
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [num_heads, N1, N2]

        # Apply mask (use -1e9 instead of -inf to avoid NaN)
        if mask2 is not None:
            # mask2: [N2] or [N1, N2], expand for multi-head broadcasting
            if mask2.dim() == 1:
                # [N2] -> [1, 1, N2] -> broadcast to [num_heads, N1, N2]
                attn_scores = attn_scores.masked_fill(~mask2.unsqueeze(0).unsqueeze(1), -1e9)
            else:
                # [N1, N2] -> [1, N1, N2] -> broadcast to [num_heads, N1, N2]
                attn_scores = attn_scores.masked_fill(~mask2.unsqueeze(0), -1e9)

        attn_weights = F.softmax(attn_scores, dim=-1)  # [num_heads, N1, N2]

        # Handle all-masked rows (replace NaN with uniform distribution)
        nan_mask = torch.isnan(attn_weights)
        if nan_mask.any():
            uniform_weight = 1.0 / attn_weights.size(-1)
            attn_weights = torch.where(nan_mask, torch.full_like(attn_weights, uniform_weight), attn_weights)

        attn_weights = self.attn_dropout(attn_weights)

        # Aggregate: [num_heads, N1, N2] @ [num_heads, N2, head_dim]
        h2_attended = torch.matmul(attn_weights, V)  # [num_heads, N1, head_dim]

        # Concatenate heads: [num_heads, N1, head_dim] -> [N1, num_heads * head_dim]
        h2_attended = h2_attended.transpose(0, 1).contiguous().view(N1, self.hidden_dim)
        h2_attended = self.out_proj(h2_attended)  # [N1, hidden_dim]

        # Matching vector
        matching_input = torch.cat([h1, h2_attended], dim=-1)
        matching_vector = self.match_projection(matching_input)

        return matching_vector, attn_weights

```

#### 2.1.3 GraphMatchingNetwork Class

Top-level module that combines encoder and cross-graph attention.

```python
class GraphMatchingNetwork(nn.Module):
    """
    Graph Matching Network with cross-graph attention.

    Architecture:
      1. Encode both graphs independently with HGT
      2. Apply cross-graph attention (bidirectional)
      3. Aggregate to graph-level embeddings
      4. Compute matching score
    """

    def __init__(
        self,
        encoder: nn.Module,  # Pre-trained HGT encoder
        hidden_dim: int = 768,
        num_attention_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        aggregation: str = 'mean_max',  # 'mean', 'max', 'mean_max', 'attention'
    ):
        """
        Args:
            encoder: Pre-trained graph encoder (HGT/GGNN/etc)
            hidden_dim: Must match encoder output dimension
            num_attention_layers: Number of cross-attention layers (stacked)
            num_heads: Attention heads per layer
            dropout: Dropout probability
            aggregation: Graph-level aggregation method
        """
        super().__init__()
        self.encoder = encoder
        self.hidden_dim = hidden_dim
        self.aggregation = aggregation

        # Stack of cross-attention layers (alternating directions)
        self.cross_attn_layers = nn.ModuleList([
            MultiHeadCrossGraphAttention(hidden_dim, num_heads, dropout)
            for _ in range(num_attention_layers)
        ])

        # Layer norms for residual connections
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_attention_layers * 2)
        ])

        # Graph-level aggregation
        if aggregation == 'attention':
            # Learnable attention-based pooling
            self.graph_attention = nn.Sequential(
                nn.Linear(hidden_dim, 1),
                nn.Tanh()
            )

        # Matching score predictor
        self.match_score = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def encode_graph(self, graph_batch) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode graph using HGT encoder.

        Args:
            graph_batch: PyG Batch with x, edge_index, edge_type, batch

        Returns:
            node_embeddings: [total_nodes, hidden_dim]
            batch_indices: [total_nodes] batch assignment
        """
        x = graph_batch.x
        edge_index = graph_batch.edge_index
        edge_type = graph_batch.edge_type
        batch = graph_batch.batch

        # Use encoder's forward method
        node_embeddings = self.encoder(x, edge_index, batch, edge_type)

        return node_embeddings, batch

    def cross_attention_pass(
        self,
        h1: torch.Tensor,
        h2: torch.Tensor,
        batch1: torch.Tensor,
        batch2: torch.Tensor,
        track_attention: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Apply stacked cross-attention layers (bidirectional).

        IMPORTANT: Creates attention mask to prevent cross-graph attention leak.
        Nodes in graph pair (i) can only attend to nodes in graph pair (i),
        not to nodes from other pairs in the batch.

        Args:
            h1: [N1, hidden_dim] graph 1 node embeddings
            h2: [N2, hidden_dim] graph 2 node embeddings
            batch1: [N1] batch assignment for graph 1
            batch2: [N2] batch assignment for graph 2
            track_attention: If True, store attention weights for visualization

        Returns:
            h1_matched: [N1, hidden_dim] updated embeddings for graph 1
            h2_matched: [N2, hidden_dim] updated embeddings for graph 2
            attention_dict: (optional) attention weights per layer if track_attention=True
        """
        # Create cross-attention mask: nodes in batch1[i] can only attend to nodes in batch2[i]
        # Shape: [N1, 1] == [1, N2] -> [N1, N2] boolean mask (True = allow attention)
        cross_mask_1to2 = batch1.unsqueeze(-1) == batch2.unsqueeze(0)  # [N1, N2]
        cross_mask_2to1 = batch2.unsqueeze(-1) == batch1.unsqueeze(0)  # [N2, N1]

        attention_dict = {} if track_attention else None
        h1_out, h2_out = h1, h2

        for i, cross_attn in enumerate(self.cross_attn_layers):
            # Bidirectional attention: h1 attends to h2, h2 attends to h1
            # Pass cross-attention mask to prevent attending across different graph pairs

            # h1 -> h2 attention (with mask)
            match_1to2, attn_1to2 = cross_attn(h1_out, h2_out, mask1=None, mask2=cross_mask_1to2)
            h1_out = self.layer_norms[i * 2](h1_out + match_1to2)  # Residual

            # h2 -> h1 attention (symmetric, with transposed mask)
            match_2to1, attn_2to1 = cross_attn(h2_out, h1_out, mask1=None, mask2=cross_mask_2to1)
            h2_out = self.layer_norms[i * 2 + 1](h2_out + match_2to1)  # Residual

            if track_attention:
                attention_dict[f'layer_{i}'] = {
                    'h1_to_h2': attn_1to2.detach(),
                    'h2_to_h1': attn_2to1.detach()
                }

        return h1_out, h2_out, attention_dict

    def aggregate_graph(self, node_embeddings: torch.Tensor,
                        batch: torch.Tensor) -> torch.Tensor:
        """
        Aggregate node embeddings to graph-level.

        Args:
            node_embeddings: [total_nodes, hidden_dim]
            batch: [total_nodes] batch assignment

        Returns:
            graph_embedding: [batch_size, hidden_dim]
        """
        batch_size = batch.max().item() + 1

        if self.aggregation == 'mean':
            return scatter_mean(node_embeddings, batch, dim=0, dim_size=batch_size)

        elif self.aggregation == 'max':
            graph_emb, _ = scatter_max(node_embeddings, batch, dim=0, dim_size=batch_size)
            return graph_emb

        elif self.aggregation == 'mean_max':
            mean_pool = scatter_mean(node_embeddings, batch, dim=0, dim_size=batch_size)
            max_pool, _ = scatter_max(node_embeddings, batch, dim=0, dim_size=batch_size)
            return mean_pool + max_pool  # Element-wise sum

        elif self.aggregation == 'attention':
            # Learnable attention weights per node
            attn_scores = self.graph_attention(node_embeddings)  # [total_nodes, 1]
            attn_weights = scatter_softmax(attn_scores.squeeze(-1), batch, dim=0)  # [total_nodes]
            weighted = node_embeddings * attn_weights.unsqueeze(-1)
            return scatter_mean(weighted, batch, dim=0, dim_size=batch_size)

        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

    def _compute_match_score(
        self,
        h1_matched: torch.Tensor,
        h2_matched: torch.Tensor,
        batch1: torch.Tensor,
        batch2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute matching score from matched node embeddings.

        Extracts graph-level aggregation and score computation for clarity.

        Args:
            h1_matched: [N1, hidden_dim] matched embeddings for graph 1
            h2_matched: [N2, hidden_dim] matched embeddings for graph 2
            batch1: [N1] batch assignment for graph 1
            batch2: [N2] batch assignment for graph 2

        Returns:
            match_score: [batch_size, 1] similarity score (0-1)
        """
        # Aggregate to graph-level embeddings
        g1_embedding = self.aggregate_graph(h1_matched, batch1)
        g2_embedding = self.aggregate_graph(h2_matched, batch2)

        # Compute matching score
        combined = torch.cat([g1_embedding, g2_embedding], dim=-1)
        match_score = torch.sigmoid(self.match_score(combined))  # [batch_size, 1]

        return match_score

    def forward(
        self,
        graph1_batch,
        graph2_batch,
        return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        Forward pass for graph pair.

        Args:
            graph1_batch: PyG Batch for obfuscated expression
            graph2_batch: PyG Batch for simplified expression
            return_attention: If True, return attention weights for visualization

        Returns:
            match_score: [batch_size, 1] similarity score (0-1)
            attention_dict: (optional) attention weights per layer
        """
        # Step 1: Encode both graphs
        h1, batch1 = self.encode_graph(graph1_batch)
        h2, batch2 = self.encode_graph(graph2_batch)

        # Step 2: Cross-graph attention (bidirectional) with optional attention tracking
        h1_matched, h2_matched, attention_dict = self.cross_attention_pass(
            h1, h2, batch1, batch2, track_attention=return_attention
        )

        # Step 3: Compute matching score (includes aggregation)
        match_score = self._compute_match_score(h1_matched, h2_matched, batch1, batch2)

        if return_attention:
            return match_score, attention_dict

        return match_score
```

### 2.2 Integration with HGT Encoder

#### 2.2.1 Wrapper for Frozen HGT

```python
class HGTWithGMN(BaseEncoder):
    """
    Combines pre-trained HGT encoder with GMN cross-attention.

    Subclasses BaseEncoder for compatibility with encoder_registry.

    Training strategy:
      - Phase 1a: Train HGT alone (existing Phase 1 contrastive)
      - Phase 1b: Freeze HGT, train GMN cross-attention layers
      - Phase 1c: (Optional) Fine-tune entire network end-to-end
    """

    def __init__(self, hgt_checkpoint_path: str, gmn_config: Dict):
        """
        Args:
            hgt_checkpoint_path: Path to pre-trained HGT weights
            gmn_config: Config dict for GMN (num_layers, num_heads, etc)

        Raises:
            ValueError: If HGT encoder hidden_dim doesn't match gmn_config['hidden_dim']
        """
        super().__init__()

        # Load pre-trained HGT encoder
        checkpoint = torch.load(hgt_checkpoint_path)
        self.hgt_encoder = HGTEncoder(**checkpoint['encoder_config'])
        self.hgt_encoder.load_state_dict(checkpoint['encoder_state_dict'])

        # CRITICAL: Validate dimension compatibility
        encoder_hidden_dim = checkpoint['encoder_config']['hidden_dim']
        gmn_hidden_dim = gmn_config['hidden_dim']
        if encoder_hidden_dim != gmn_hidden_dim:
            raise ValueError(
                f"Dimension mismatch: HGT encoder has hidden_dim={encoder_hidden_dim}, "
                f"but gmn_config specifies hidden_dim={gmn_hidden_dim}. "
                f"Set gmn_config['hidden_dim']={encoder_hidden_dim} or retrain HGT."
            )

        # Freeze HGT (optional, controlled by config)
        if gmn_config.get('freeze_encoder', True):
            for param in self.hgt_encoder.parameters():
                param.requires_grad = False

        # Initialize GMN
        self.gmn = GraphMatchingNetwork(
            encoder=self.hgt_encoder,
            hidden_dim=gmn_config['hidden_dim'],
            num_attention_layers=gmn_config['num_attention_layers'],
            num_heads=gmn_config['num_heads'],
            dropout=gmn_config['dropout'],
            aggregation=gmn_config['aggregation']
        )

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor,
               batch: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        """
        BaseEncoder interface: encode single graph.

        Args:
            x: Node features
            edge_index: Edge indices
            batch: Batch assignment
            edge_type: Edge types

        Returns:
            Node embeddings [num_nodes, hidden_dim]
        """
        return self.hgt_encoder(x, edge_index, batch, edge_type)

    def forward(self, graph1_batch, graph2_batch):
        """Forward pass through GMN for graph pair matching."""
        return self.gmn(graph1_batch, graph2_batch)

    def unfreeze_encoder(self):
        """Unfreeze HGT encoder for end-to-end fine-tuning."""
        for param in self.hgt_encoder.parameters():
            param.requires_grad = True
            # Clear stale gradients from Phase 1b to prevent corruption
            if param.grad is not None:
                param.grad.zero_()
```

### 2.3 Batch Handling for Variable-Size Graphs

**Challenge:** Graphs have variable sizes (e.g., depth-2 vs depth-14 expressions). Cross-attention requires handling pairs of different sizes.

**Solution:** Packed batching with attention masking.

```python
class GMNBatchCollator:
    """
    Collate function for batching graph pairs with GMN.

    Handles variable-size graphs by:
      1. Packing graphs into PyG Batch (standard)
      2. Creating attention masks for cross-attention
      3. Tracking pair indices for loss computation
    """

    def __call__(self, batch_list: List[Tuple[Data, Data, int]]):
        """
        Collate batch of graph pairs.

        Args:
            batch_list: List of (graph1, graph2, label) tuples
              - graph1: PyG Data (obfuscated)
              - graph2: PyG Data (simplified)
              - label: 1 if equivalent, 0 if not

        Returns:
            graph1_batch: PyG Batch
            graph2_batch: PyG Batch
            labels: [batch_size] tensor
            pair_indices: [batch_size, 2] tensor mapping batch indices to pair IDs
        """
        graphs1, graphs2, labels = zip(*batch_list)

        # Standard PyG batching (handles variable sizes automatically)
        from torch_geometric.data import Batch
        graph1_batch = Batch.from_data_list(graphs1)
        graph2_batch = Batch.from_data_list(graphs2)

        labels = torch.tensor(labels, dtype=torch.float32)

        # Pair indices: [i, j] means graph1[i] pairs with graph2[j]
        batch_size = len(batch_list)
        pair_indices = torch.arange(batch_size).unsqueeze(-1).repeat(1, 2)

        return graph1_batch, graph2_batch, labels, pair_indices
```

**Memory Efficiency:** For large batches, cross-attention has O(N1 × N2) memory per pair. Mitigation strategies:
- **Gradient checkpointing:** Trade compute for memory (recompute activations in backward pass)
- **Sparse attention:** Limit attention to top-k most similar nodes
- **Smaller batch sizes:** Reduce from 32 to 16 graphs per batch

---

## 3. Implementation Details

### 3.1 File Structure

```
src/models/
├── encoder.py                    # Existing (HGT, GGNN, etc)
├── encoder_base.py               # Existing (BaseEncoder)
├── encoder_registry.py           # UPDATE: Add GMN variants
├── gmn/
│   ├── __init__.py
│   ├── cross_attention.py        # NEW: CrossGraphAttention, MultiHeadCrossGraphAttention
│   ├── graph_matching.py         # NEW: GraphMatchingNetwork
│   ├── gmn_encoder_wrapper.py   # NEW: HGTWithGMN
│   └── batch_collator.py         # NEW: GMNBatchCollator
├── full_model.py                 # UPDATE: Add GMN mode
└── heads.py                      # Existing (TokenHead, etc)
```

### 3.2 Class Hierarchies

```
BaseEncoder (existing)
    ├── HGTEncoder (existing)
    ├── GATJKNetEncoder (existing)
    ├── ... (other encoders)
    ├── HGTWithGMN (new) ─────────────────── Subclasses BaseEncoder for registry compatibility
    │       ├── encode(): Delegates to hgt_encoder (BaseEncoder interface)
    │       ├── forward(): GMN graph pair matching
    │       └── contains: gmn (GraphMatchingNetwork)
    └── GATWithGMN (new)
            └── (same pattern as HGTWithGMN)

nn.Module
    ├── CrossGraphAttention (new)
    ├── MultiHeadCrossGraphAttention (new)
    └── GraphMatchingNetwork (new)
            └── uses: encoder (BaseEncoder instance)
```

### 3.3 Configuration Schema

Add to `configs/phase1_gmn.yaml`:

```yaml
# Phase 1b: GMN Cross-Attention Training
model:
  encoder_type: hgt_gmn  # New encoder type
  encoder_checkpoint: checkpoints/phase1_hgt.pt  # Pre-trained HGT

  gmn:
    freeze_encoder: true  # Freeze HGT, train only cross-attention
    num_attention_layers: 2  # Stack 2 cross-attention layers
    num_heads: 8
    dropout: 0.1
    aggregation: mean_max  # or 'attention'

training:
  phase: 1b
  loss: contrastive_gmn  # Use GMN-specific contrastive loss
  batch_size: 16  # Smaller due to O(N1*N2) memory
  num_epochs: 10
  learning_rate: 1e-4  # Lower LR for fine-tuning

  # Contrastive sampling
  positive_pairs: 10000  # (obfuscated, simplified) equivalent pairs
  negative_pairs: 10000  # (expr1, expr2) non-equivalent pairs
  hard_negative_mining: true  # Sample hard negatives (same variables, different semantics)
```

### 3.4 Key Implementation Considerations

#### 3.4.1 Attention Masking

```python
def create_cross_attention_mask(batch1: torch.Tensor, batch2: torch.Tensor) -> torch.Tensor:
    """
    Create attention mask for batched cross-attention.

    Prevents attention across different graph pairs in the batch.

    Args:
        batch1: [N1] batch indices for graph 1 nodes
        batch2: [N2] batch indices for graph 2 nodes

    Returns:
        mask: [N1, N2] boolean mask (True = allow attention)
    """
    # Each node in batch1[i] can only attend to nodes in batch2[i]
    # Shape: [N1, 1] == [1, N2] -> [N1, N2]
    mask = batch1.unsqueeze(-1) == batch2.unsqueeze(0)
    return mask
```

#### 3.4.2 Efficient Batching with Sparse Attention

For very large graphs (depth-14, ~1000 nodes), full O(N²) attention is expensive.

```python
class SparseTopKCrossAttention(nn.Module):
    """
    Cross-attention with top-k sparsity.

    Only compute attention for top-k most similar nodes (by cosine similarity).
    Reduces memory from O(N1*N2) to O(N1*k).
    """

    def __init__(self, hidden_dim: int, top_k: int = 32):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.top_k = top_k
        # ... (similar to CrossGraphAttention)

    def forward(self, h1: torch.Tensor, h2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h1: [N1, hidden_dim]
            h2: [N2, hidden_dim]
        """
        # Compute cosine similarity: h1 @ h2.T / (||h1|| * ||h2||)
        h1_norm = F.normalize(h1, dim=-1)
        h2_norm = F.normalize(h2, dim=-1)
        similarity = torch.matmul(h1_norm, h2_norm.T)  # [N1, N2]

        # For each node in h1, keep only top-k matches in h2
        topk_vals, topk_indices = torch.topk(similarity, k=min(self.top_k, h2.size(0)), dim=-1)
        # topk_indices: [N1, k]

        # Sparse attention: only compute for top-k
        # (Implementation: use torch.scatter or index_select)
        # ... (details omitted for brevity)

        return matching_vector
```

---

## 4. Integration Points

### 4.1 encoder_registry.py Updates

```python
# src/models/encoder_registry.py

def _get_encoder_classes() -> Dict[str, Type[BaseEncoder]]:
    """Get encoder classes with lazy imports."""
    from src.models.encoder import (
        GATJKNetEncoder,
        GGNNEncoder,
        HGTEncoder,
        RGCNEncoder,
    )
    from src.models.encoder_ablation import (
        HybridGREATEncoder,
        TransformerOnlyEncoder,
    )
    # NEW: Import GMN variants
    from src.models.gmn import (
        HGTWithGMN,
        GATWithGMN,  # Could also add GMN for GAT encoder
    )

    return {
        "gat_jknet": GATJKNetEncoder,
        "ggnn": GGNNEncoder,
        "hgt": HGTEncoder,
        "rgcn": RGCNEncoder,
        "transformer_only": TransformerOnlyEncoder,
        "hybrid_great": HybridGREATEncoder,
        # NEW: GMN-enhanced encoders
        "hgt_gmn": HGTWithGMN,
        "gat_gmn": GATWithGMN,
    }
```

### 4.2 full_model.py Updates

Add GMN support to `MBADeobfuscator` for Phase 1 contrastive training.

```python
# src/models/full_model.py

class MBADeobfuscator(nn.Module):
    def __init__(self, encoder_type: str = 'gat', use_gmn: bool = False, **kwargs):
        """
        Args:
            encoder_type: 'gat', 'ggnn', 'hgt', 'rgcn'
            use_gmn: If True, wrap encoder with GMN for contrastive training
            **kwargs: Additional arguments
        """
        super().__init__()
        self.encoder_type = encoder_type
        self.use_gmn = use_gmn

        # ... (existing encoder initialization)

        # NEW: GMN wrapper for Phase 1
        if use_gmn:
            gmn_config = kwargs.get('gmn_config', {})
            self.gmn = GraphMatchingNetwork(
                encoder=self.graph_encoder,
                hidden_dim=gmn_config.get('hidden_dim', 768),
                num_attention_layers=gmn_config.get('num_attention_layers', 2),
                num_heads=gmn_config.get('num_heads', 8),
                dropout=gmn_config.get('dropout', 0.1),
                aggregation=gmn_config.get('aggregation', 'mean_max')
            )
        else:
            self.gmn = None

        # ... (rest of initialization)

    def encode_pair_gmn(self, graph1_batch, graph2_batch,
                        fingerprint1: torch.Tensor,
                        fingerprint2: torch.Tensor,
                        return_attention: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        Encode pair of graphs with GMN cross-attention.

        Delegates entirely to GMN to avoid duplicate encoding logic.
        All graph encoding is handled within GraphMatchingNetwork.encode_graph().

        Used in Phase 1 contrastive training with GMN.

        Args:
            graph1_batch: PyG batch for graph 1
            graph2_batch: PyG batch for graph 2
            fingerprint1: [batch, FINGERPRINT_DIM] for graph 1 (reserved for future use)
            fingerprint2: [batch, FINGERPRINT_DIM] for graph 2 (reserved for future use)
            return_attention: If True, return attention weights for visualization

        Returns:
            match_score: [batch, 1] similarity score
            attention_dict: (optional) attention weights per layer if return_attention=True
        """
        if self.gmn is None:
            raise ValueError("GMN not enabled. Set use_gmn=True.")

        # Delegate to GMN (consolidates all graph encoding logic)
        return self.gmn(graph1_batch, graph2_batch, return_attention=return_attention)
```

### 4.3 Phase 1 Training Modifications

Update `src/training/phase1_trainer.py` to support GMN training mode.

```python
# src/training/phase1_trainer.py

class Phase1Trainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.use_gmn = config.get('use_gmn', False)

        # Loss functions
        if self.use_gmn:
            self.criterion = nn.BCELoss()  # Binary classification (equivalent or not)
        else:
            self.criterion = InfoNCELoss(temperature=INFONCE_TEMPERATURE)

    def train_step_gmn(self, batch):
        """
        Training step for GMN mode.

        Batch contains pairs: (graph1, graph2, label)
        - label=1: Equivalent expressions (positive pair)
        - label=0: Non-equivalent expressions (negative pair)
        """
        graph1_batch, graph2_batch, labels, _ = batch

        # Forward pass through GMN
        match_scores = self.model.encode_pair_gmn(
            graph1_batch,
            graph2_batch,
            graph1_batch.fingerprint,
            graph2_batch.fingerprint
        )

        # Binary cross-entropy loss
        loss = self.criterion(match_scores.squeeze(), labels)

        # Compute accuracy
        preds = (match_scores.squeeze() > 0.5).float()
        acc = (preds == labels).float().mean()

        return loss, {'gmn_loss': loss.item(), 'gmn_acc': acc.item()}

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0

        for batch in dataloader:
            self.optimizer.zero_grad()

            if self.use_gmn:
                loss, metrics = self.train_step_gmn(batch)
            else:
                loss, metrics = self.train_step_standard(batch)  # Existing InfoNCE

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)
```

---

## 5. Training Considerations

### 5.1 Three-Stage Training Strategy

```
┌─────────────────────────────────────────────────────────────┐
│ Phase 1a: HGT Encoder Pre-training (Existing)               │
│   - Loss: InfoNCE + MaskLM                                   │
│   - Output: Pre-trained HGT encoder (checkpoints/phase1.pt) │
│   - Duration: 20 epochs                                      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Phase 1b: GMN Cross-Attention Training (New)                │
│   - Freeze HGT encoder                                       │
│   - Train only cross-attention layers                        │
│   - Loss: Binary classification (equivalent or not)          │
│   - Data: (obfuscated, simplified, label) pairs             │
│   - Duration: 10 epochs                                      │
│   - Output: HGT + GMN (checkpoints/phase1b_gmn.pt)          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Phase 1c: End-to-End Fine-Tuning (Optional)                 │
│   - Unfreeze entire network                                  │
│   - Fine-tune HGT + GMN together                             │
│   - Loss: Binary classification                              │
│   - Duration: 5 epochs                                       │
│   - Learning rate: 1e-5 (very low)                           │
│   - Output: Fine-tuned HGT+GMN (checkpoints/phase1c_gmn.pt) │
└─────────────────────────────────────────────────────────────┘
                            ↓
                 Use in Phase 2 & 3 (seq2seq training)
```

### 5.2 Loss Functions

#### 5.2.1 Binary Classification Loss (Phase 1b)

```python
class GMNContrastiveLoss(nn.Module):
    """
    Binary classification loss for GMN training.

    Positive pairs: (obfuscated, simplified) where both semantically equivalent
    Negative pairs: (expr1, expr2) where semantically different
    """

    def __init__(self, pos_weight: float = 1.0):
        """
        Args:
            pos_weight: Weight for positive class (to handle class imbalance)
        """
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))

    def forward(self, match_scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            match_scores: [batch_size, 1] predicted similarity (pre-sigmoid)
            labels: [batch_size] ground truth (1=equivalent, 0=not)

        Returns:
            loss: Scalar
        """
        return self.criterion(match_scores.squeeze(), labels)
```

#### 5.2.2 Triplet Loss (Alternative)

```python
class GMNTripletLoss(nn.Module):
    """
    Triplet loss for GMN: (anchor, positive, negative).

    Encourages:
      - dist(anchor, positive) < dist(anchor, negative) + margin

    May perform better than binary classification when hard negative mining is effective.
    """

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        anchor_emb: torch.Tensor,
        positive_emb: torch.Tensor,
        negative_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            anchor_emb: [batch, hidden_dim] obfuscated expression
            positive_emb: [batch, hidden_dim] simplified (equivalent)
            negative_emb: [batch, hidden_dim] non-equivalent expression

        Returns:
            loss: Scalar
        """
        pos_dist = F.pairwise_distance(anchor_emb, positive_emb, p=2)
        neg_dist = F.pairwise_distance(anchor_emb, negative_emb, p=2)

        loss = F.relu(pos_dist - neg_dist + self.margin).mean()
        return loss
```

### 5.3 Data Sampling Strategy

**Positive Pairs:** Straightforward from dataset.
```
(obfuscated, simplified)
Example: ("(x&y)+(x^y)", "x|y")
```

**Negative Pairs:** Require careful sampling to avoid trivial negatives.

```python
class NegativeSampler:
    """Sample hard negative pairs for GMN training."""

    def __init__(self, mode: str = 'hard'):
        """
        Args:
            mode: 'random', 'hard', or 'semi-hard'
              - random: Any two different expressions
              - hard: Same variables, different semantics (most challenging)
              - semi-hard: Partially overlapping variables
        """
        self.mode = mode

    def sample_negative(self, anchor_expr: str, dataset: List[str]) -> str:
        """
        Sample negative expression for anchor.

        Args:
            anchor_expr: Anchor expression (e.g., "x | y")
            dataset: Pool of expressions

        Returns:
            negative_expr: Non-equivalent expression
        """
        if self.mode == 'random':
            # Random sampling (easiest)
            negative = random.choice(dataset)
            while self.are_equivalent(anchor_expr, negative):
                negative = random.choice(dataset)
            return negative

        elif self.mode == 'hard':
            # Same variables, different semantics
            anchor_vars = self.extract_variables(anchor_expr)
            candidates = [
                expr for expr in dataset
                if self.extract_variables(expr) == anchor_vars
                and not self.are_equivalent(anchor_expr, expr)
            ]
            if not candidates:
                return self.sample_negative(anchor_expr, dataset)  # Fallback to random
            return random.choice(candidates)

        elif self.mode == 'semi-hard':
            # Overlapping variables (medium difficulty)
            anchor_vars = self.extract_variables(anchor_expr)
            candidates = [
                expr for expr in dataset
                if len(self.extract_variables(expr) & anchor_vars) > 0
                and not self.are_equivalent(anchor_expr, expr)
            ]
            return random.choice(candidates) if candidates else self.sample_negative(anchor_expr, dataset)

    def are_equivalent(self, expr1: str, expr2: str, timeout_ms: int = 5000) -> bool:
        """
        Check semantic equivalence using Z3 with timeout.

        Args:
            expr1: First expression
            expr2: Second expression
            timeout_ms: Z3 solver timeout in milliseconds (default: 5000)

        Returns:
            True if expressions are semantically equivalent, False otherwise
            (returns False on timeout or error for conservative negative sampling)
        """
        from src.utils.z3_interface import verify_equivalence
        try:
            result = verify_equivalence(expr1, expr2, timeout_ms=timeout_ms)
            return result.is_equivalent if result else False
        except TimeoutError:
            # Conservative: assume non-equivalent on timeout
            logging.warning(f"Z3 timeout on equivalence check: {expr1} vs {expr2}")
            return False
        except Exception as e:
            logging.error(f"Z3 error during equivalence check: {e}")
            return False

    def extract_variables(self, expr: str) -> Set[str]:
        """Extract variable names from expression."""
        import re
        return set(re.findall(r'\b[xyz]\d?\b', expr))
```

### 5.4 Curriculum Learning Integration

Extend curriculum to GMN training:

```yaml
# configs/phase1b_gmn_curriculum.yaml

curriculum:
  stages:
    - name: easy
      max_depth: 5  # Shallow expressions first
      epochs: 5
      negative_mode: random  # Easy negatives

    - name: medium
      max_depth: 10
      epochs: 3
      negative_mode: semi-hard  # Medium negatives

    - name: hard
      max_depth: 14
      epochs: 2
      negative_mode: hard  # Hard negatives (same variables)
```

---

## 6. API Design

### 6.1 Public Interfaces

```python
# src/models/gmn/__init__.py

from .cross_attention import CrossGraphAttention, MultiHeadCrossGraphAttention
from .graph_matching import GraphMatchingNetwork
from .gmn_encoder_wrapper import HGTWithGMN, GATWithGMN

__all__ = [
    'CrossGraphAttention',
    'MultiHeadCrossGraphAttention',
    'GraphMatchingNetwork',
    'HGTWithGMN',
    'GATWithGMN',
]
```

### 6.2 Configuration Options

```python
# Example usage in training script

from src.models.gmn import HGTWithGMN
from src.training.phase1_gmn_trainer import Phase1GMNTrainer

# Initialize model
model = HGTWithGMN(
    hgt_checkpoint_path='checkpoints/phase1_hgt.pt',
    gmn_config={
        'freeze_encoder': True,
        'num_attention_layers': 2,
        'num_heads': 8,
        'dropout': 0.1,
        'aggregation': 'mean_max'
    }
)

# Initialize trainer
trainer = Phase1GMNTrainer(
    model=model,
    config={
        'batch_size': 16,
        'learning_rate': 1e-4,
        'num_epochs': 10,
        'negative_mode': 'hard',
        'loss_type': 'binary'  # or 'triplet'
    }
)

# Train
trainer.train(train_loader, val_loader)
```

### 6.3 Usage Examples

#### Example 1: Train GMN from Scratch

```python
import torch
from src.models.encoder import HGTEncoder
from src.models.gmn import GraphMatchingNetwork
from src.data.dataset import MBAPairDataset

# Load pre-trained HGT
hgt = HGTEncoder(hidden_dim=768, num_layers=12, num_heads=16)
hgt.load_state_dict(torch.load('checkpoints/phase1_hgt.pt'))

# Initialize GMN
gmn = GraphMatchingNetwork(
    encoder=hgt,
    hidden_dim=768,
    num_attention_layers=2,
    num_heads=8
)

# Freeze HGT
for param in hgt.parameters():
    param.requires_grad = False

# Load dataset
dataset = MBAPairDataset(
    data_path='data/train_pairs.json',
    negative_mode='hard'
)

# Training loop
optimizer = torch.optim.Adam(gmn.parameters(), lr=1e-4)
criterion = nn.BCELoss()

for epoch in range(10):
    for graph1, graph2, label in dataset:
        optimizer.zero_grad()

        score = gmn(graph1, graph2)
        loss = criterion(score.squeeze(), label)

        loss.backward()
        optimizer.step()
```

#### Example 2: Use Trained GMN for Equivalence Checking

```python
from src.models.gmn import HGTWithGMN
from src.data.ast_parser import parse_expression

# Load trained model
model = HGTWithGMN.from_pretrained('checkpoints/phase1c_gmn.pt')
model.eval()

# Parse two expressions
expr1 = "(x & y) + (x ^ y)"
expr2 = "x | y"

graph1 = parse_expression(expr1)
graph2 = parse_expression(expr2)

# Compute similarity
with torch.no_grad():
    score = model(graph1, graph2)

print(f"Equivalence score: {score.item():.4f}")  # Expected: ~0.95 (high similarity)
```

#### Example 3: Visualize Cross-Attention Weights

```python
import matplotlib.pyplot as plt

# Forward pass with attention tracking
score, attn_dict = model(graph1, graph2, return_attention=True)

# Visualize first attention layer
attn_weights = attn_dict['layer_0']['h1_to_h2']  # [num_nodes_1, num_nodes_2]

plt.figure(figsize=(10, 8))
plt.imshow(attn_weights.cpu().numpy(), cmap='viridis')
plt.xlabel('Nodes in Simplified Expression')
plt.ylabel('Nodes in Obfuscated Expression')
plt.title('Cross-Graph Attention Weights')
plt.colorbar()
plt.savefig('attention_visualization.png')
```

---

## 7. Testing Strategy

### 7.1 Unit Tests

#### Test 1: CrossGraphAttention Forward Pass

```python
# tests/test_gmn_cross_attention.py

import torch
from src.models.gmn import CrossGraphAttention

def test_cross_attention_forward():
    """Test that cross-attention produces correct output shapes."""
    hidden_dim = 256
    N1, N2 = 10, 15  # Variable graph sizes

    cross_attn = CrossGraphAttention(hidden_dim=hidden_dim)

    h1 = torch.randn(N1, hidden_dim)
    h2 = torch.randn(N2, hidden_dim)

    match_vector, attn_weights = cross_attn(h1, h2)

    # Check shapes
    assert match_vector.shape == (N1, hidden_dim)
    assert attn_weights.shape == (N1, N2)

    # Check attention weights sum to 1
    assert torch.allclose(attn_weights.sum(dim=-1), torch.ones(N1), atol=1e-5)

def test_cross_attention_masking():
    """Test that masking correctly prevents attention to invalid nodes."""
    hidden_dim = 256
    N1, N2 = 10, 15

    cross_attn = CrossGraphAttention(hidden_dim=hidden_dim)

    h1 = torch.randn(N1, hidden_dim)
    h2 = torch.randn(N2, hidden_dim)

    # Mask out last 5 nodes in h2
    mask2 = torch.ones(N2, dtype=torch.bool)
    mask2[-5:] = False

    match_vector, attn_weights = cross_attn(h1, h2, mask2=mask2)

    # Attention weights should be 0 for masked nodes
    assert torch.allclose(attn_weights[:, -5:], torch.zeros(N1, 5), atol=1e-5)
```

#### Test 2: GraphMatchingNetwork Equivalence Detection

```python
# tests/test_gmn_matching.py

import torch
from src.models.gmn import GraphMatchingNetwork
from src.models.encoder import HGTEncoder
from src.data.ast_parser import parse_expression

def test_gmn_same_expression():
    """Test that GMN gives high score for identical expressions."""
    hgt = HGTEncoder(hidden_dim=256, num_layers=4)
    gmn = GraphMatchingNetwork(encoder=hgt, hidden_dim=256)

    # Same expression
    expr = "x | y"
    graph1 = parse_expression(expr)
    graph2 = parse_expression(expr)

    score = gmn(graph1, graph2)

    # Should be very high (near 1.0)
    assert score.item() > 0.9

def test_gmn_equivalent_expressions():
    """Test that GMN gives high score for semantically equivalent expressions."""
    hgt = HGTEncoder(hidden_dim=256, num_layers=4)
    gmn = GraphMatchingNetwork(encoder=hgt, hidden_dim=256)

    # Equivalent expressions
    graph1 = parse_expression("(x & y) + (x ^ y)")
    graph2 = parse_expression("x | y")

    # Note: Without training, score may be random
    # This test requires a trained model
    score = gmn(graph1, graph2)

    assert 0.0 <= score.item() <= 1.0  # Valid range

def test_gmn_different_expressions():
    """Test that GMN gives low score for non-equivalent expressions."""
    hgt = HGTEncoder(hidden_dim=256, num_layers=4)
    gmn = GraphMatchingNetwork(encoder=hgt, hidden_dim=256)

    # Different expressions
    graph1 = parse_expression("x | y")
    graph2 = parse_expression("x & y")

    score = gmn(graph1, graph2)

    # Should be low (near 0.0) after training
    assert 0.0 <= score.item() <= 1.0
```

#### Test 3: Batch Collation

```python
# tests/test_gmn_batch.py

from src.models.gmn import GMNBatchCollator
from src.data.ast_parser import parse_expression

def test_batch_collator():
    """Test that batch collator handles variable-size graphs correctly."""
    collator = GMNBatchCollator()

    # Create batch with variable sizes
    batch_list = [
        (parse_expression("x | y"), parse_expression("x"), 1),  # Positive
        (parse_expression("x & y"), parse_expression("x | y"), 0),  # Negative
        (parse_expression("(x & y) + z"), parse_expression("(x | y) + z"), 0),  # Negative
    ]

    graph1_batch, graph2_batch, labels, pair_indices = collator(batch_list)

    # Check batch structure
    assert graph1_batch.num_graphs == 3
    assert graph2_batch.num_graphs == 3
    assert labels.shape == (3,)
    assert pair_indices.shape == (3, 2)
```

### 7.2 Integration Tests

#### Test 4: End-to-End Training

```python
# tests/test_gmn_training.py

import torch
from src.models.gmn import HGTWithGMN
from src.training.phase1_gmn_trainer import Phase1GMNTrainer
from src.data.dataset import MBAPairDataset

def test_gmn_training_step():
    """Test that GMN training step runs without errors."""
    # Initialize model
    hgt_config = {'hidden_dim': 256, 'num_layers': 4}
    model = HGTWithGMN(
        hgt_checkpoint_path=None,  # Random init for test
        gmn_config={'num_attention_layers': 1, 'num_heads': 4}
    )

    # Create dummy batch
    batch_list = [
        (parse_expression("x | y"), parse_expression("x"), 1),
        (parse_expression("x & y"), parse_expression("x | y"), 0),
    ]
    collator = GMNBatchCollator()
    batch = collator(batch_list)

    # Forward pass
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCELoss()

    graph1_batch, graph2_batch, labels, _ = batch

    optimizer.zero_grad()
    scores = model(graph1_batch, graph2_batch)
    loss = criterion(scores.squeeze(), labels)
    loss.backward()
    optimizer.step()

    # Check that loss is finite
    assert torch.isfinite(loss).all()
```

### 7.3 Performance Benchmarks

#### Benchmark 1: Memory Usage

```python
# tests/benchmark_gmn_memory.py

import torch
from src.models.gmn import GraphMatchingNetwork
from src.models.encoder import HGTEncoder

def benchmark_memory_scaling():
    """Measure memory usage vs graph size."""
    hgt = HGTEncoder(hidden_dim=768, num_layers=12)
    gmn = GraphMatchingNetwork(encoder=hgt, hidden_dim=768)

    graph_sizes = [10, 50, 100, 500, 1000]  # Number of nodes

    for N in graph_sizes:
        # Create dummy graphs
        graph1 = create_random_graph(num_nodes=N)
        graph2 = create_random_graph(num_nodes=N)

        torch.cuda.reset_peak_memory_stats()

        with torch.no_grad():
            score = gmn(graph1, graph2)

        peak_memory = torch.cuda.max_memory_allocated() / 1e9  # GB

        print(f"N={N}: Peak memory = {peak_memory:.2f} GB")

        # Expected: O(N^2) for cross-attention
        # Should be manageable up to N=1000
```

#### Benchmark 2: Inference Speed

```python
# tests/benchmark_gmn_speed.py

import time
import torch
from src.models.gmn import GraphMatchingNetwork

def benchmark_inference_speed():
    """Measure inference time vs graph size."""
    gmn = GraphMatchingNetwork(...)
    gmn.eval()

    graph_sizes = [10, 50, 100, 500, 1000]
    num_trials = 100

    for N in graph_sizes:
        graph1 = create_random_graph(num_nodes=N)
        graph2 = create_random_graph(num_nodes=N)

        # Warmup
        for _ in range(10):
            gmn(graph1, graph2)

        # Measure
        start = time.time()
        for _ in range(num_trials):
            with torch.no_grad():
                score = gmn(graph1, graph2)
        elapsed = time.time() - start

        avg_time = elapsed / num_trials * 1000  # ms
        print(f"N={N}: {avg_time:.2f} ms per pair")
```

---

## 8. Implementation Phases

### Phase Breakdown

| Phase | Component | Dependencies | Estimated Effort | Priority |
|-------|-----------|--------------|------------------|----------|
| **0** | **Prerequisites** | | | |
| 0.1 | Verify HGT encoder trained | None | 0 days (existing) | P0 |
| 0.2 | Create pair dataset generator | Dataset generation | 2 days | P0 |

#### Phase 0.2 Detail: Pair Dataset Generation

The GMN requires (graph1, graph2, label) tuples, but existing data format is JSONL with `{"obfuscated": str, "simplified": str, "depth": int}`. This phase creates the conversion utility.

**New Script: `scripts/generate_gmn_pairs.py`**

```python
#!/usr/bin/env python3
"""
Generate GMN pair dataset from existing JSONL format.

Input:  {"obfuscated": str, "simplified": str, "depth": int}
Output: {"graph1": str, "graph2": str, "label": int}

Usage:
    python scripts/generate_gmn_pairs.py --input data/train.jsonl --output data/train_pairs.json
    python scripts/generate_gmn_pairs.py --input data/train.jsonl --output data/train_pairs.json --neg-mode hard
"""

import argparse
import json
import random
import logging
from pathlib import Path
from typing import List, Set, Tuple
from tqdm import tqdm

from src.utils.z3_interface import verify_equivalence

logger = logging.getLogger(__name__)


def extract_variables(expr: str) -> Set[str]:
    """Extract variable names from expression."""
    import re
    return set(re.findall(r'\b[a-z]\d?\b', expr))


def are_equivalent(expr1: str, expr2: str, timeout_ms: int = 5000) -> bool:
    """
    Check semantic equivalence using Z3 with timeout.

    Args:
        expr1: First expression
        expr2: Second expression
        timeout_ms: Z3 solver timeout in milliseconds

    Returns:
        True if expressions are semantically equivalent
    """
    try:
        result = verify_equivalence(expr1, expr2, timeout_ms=timeout_ms)
        return result.is_equivalent if result else False
    except TimeoutError:
        logger.warning(f"Z3 timeout on equivalence check: {expr1} vs {expr2}")
        return False  # Conservative: assume non-equivalent on timeout
    except Exception as e:
        logger.error(f"Z3 error: {e}")
        return False


def sample_negative(
    anchor_expr: str,
    candidates: List[str],
    mode: str = 'hard',
    max_attempts: int = 100
) -> Tuple[str, bool]:
    """
    Sample negative expression for anchor.

    Args:
        anchor_expr: Anchor expression
        candidates: Pool of candidate expressions
        mode: 'random', 'hard', or 'semi-hard'
        max_attempts: Maximum sampling attempts before fallback

    Returns:
        (negative_expr, found) tuple
    """
    anchor_vars = extract_variables(anchor_expr)

    if mode == 'random':
        # Random sampling (easiest)
        for _ in range(max_attempts):
            negative = random.choice(candidates)
            if negative != anchor_expr and not are_equivalent(anchor_expr, negative):
                return negative, True
        return random.choice(candidates), False

    elif mode == 'hard':
        # Same variables, different semantics (most challenging)
        hard_candidates = [
            expr for expr in candidates
            if extract_variables(expr) == anchor_vars and expr != anchor_expr
        ]
        for candidate in random.sample(hard_candidates, min(max_attempts, len(hard_candidates))):
            if not are_equivalent(anchor_expr, candidate):
                return candidate, True
        # Fallback to semi-hard
        return sample_negative(anchor_expr, candidates, mode='semi-hard')

    elif mode == 'semi-hard':
        # Overlapping variables (medium difficulty)
        semi_candidates = [
            expr for expr in candidates
            if len(extract_variables(expr) & anchor_vars) > 0 and expr != anchor_expr
        ]
        for candidate in random.sample(semi_candidates, min(max_attempts, len(semi_candidates))):
            if not are_equivalent(anchor_expr, candidate):
                return candidate, True
        # Fallback to random
        return sample_negative(anchor_expr, candidates, mode='random')

    raise ValueError(f"Unknown mode: {mode}")


def generate_pairs(
    input_path: str,
    output_path: str,
    neg_mode: str = 'hard',
    neg_ratio: float = 1.0,
    seed: int = 42
):
    """
    Convert JSONL dataset to GMN pair format.

    Args:
        input_path: Path to input JSONL file
        output_path: Path to output JSON file
        neg_mode: Negative sampling mode ('random', 'hard', 'semi-hard')
        neg_ratio: Ratio of negatives to positives
        seed: Random seed
    """
    random.seed(seed)

    # Load existing dataset
    logger.info(f"Loading data from {input_path}")
    samples = []
    with open(input_path, 'r') as f:
        for line in f:
            samples.append(json.loads(line))

    # Extract all expressions for negative sampling
    all_simplified = list(set(s['simplified'] for s in samples))

    pairs = []
    neg_failed = 0

    logger.info(f"Generating pairs with {neg_mode} negative sampling...")
    for sample in tqdm(samples):
        obfuscated = sample['obfuscated']
        simplified = sample['simplified']

        # Positive pair: (obfuscated, simplified, label=1)
        pairs.append({
            'graph1': obfuscated,
            'graph2': simplified,
            'label': 1,
            'depth': sample.get('depth', 0)
        })

        # Negative pair(s)
        for _ in range(int(neg_ratio)):
            negative, found = sample_negative(simplified, all_simplified, mode=neg_mode)
            if not found:
                neg_failed += 1
            pairs.append({
                'graph1': obfuscated,
                'graph2': negative,
                'label': 0,
                'depth': sample.get('depth', 0)
            })

    # Shuffle
    random.shuffle(pairs)

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(pairs, f, indent=2)

    logger.info(f"Generated {len(pairs)} pairs to {output_path}")
    if neg_failed > 0:
        logger.warning(f"Failed to find {neg_failed} hard negatives (used fallback)")


def main():
    parser = argparse.ArgumentParser(description='Generate GMN pair dataset')
    parser.add_argument('--input', '-i', required=True, help='Input JSONL file')
    parser.add_argument('--output', '-o', required=True, help='Output JSON file')
    parser.add_argument('--neg-mode', default='hard', choices=['random', 'hard', 'semi-hard'])
    parser.add_argument('--neg-ratio', type=float, default=1.0, help='Negatives per positive')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    generate_pairs(args.input, args.output, args.neg_mode, args.neg_ratio, args.seed)


if __name__ == '__main__':
    main()
```

**Output Format:**
```json
[
  {"graph1": "(x & y) + (x ^ y)", "graph2": "x | y", "label": 1, "depth": 3},
  {"graph1": "(x & y) + (x ^ y)", "graph2": "x & y", "label": 0, "depth": 3},
  ...
]
```
| **1** | **Core GMN Components** | | | |
| 1.1 | Implement `CrossGraphAttention` | None | 2 days | P0 |
| 1.2 | Implement `MultiHeadCrossGraphAttention` | Phase 1.1 | 1 day | P0 |
| 1.3 | Unit tests for attention modules | Phase 1.1-1.2 | 1 day | P0 |
| **2** | **Graph Matching Network** | | | |
| 2.1 | Implement `GraphMatchingNetwork` | Phase 1.2 | 3 days | P0 |
| 2.2 | Implement `HGTWithGMN` wrapper | Phase 2.1 | 1 day | P0 |
| 2.3 | Unit tests for GMN | Phase 2.1-2.2 | 2 days | P0 |
| **3** | **Batch Processing** | | | |
| 3.1 | Implement `GMNBatchCollator` | None | 1 day | P0 |
| 3.2 | Add attention masking for batches | Phase 3.1 | 1 day | P0 |
| 3.3 | Test variable-size batch handling | Phase 3.2 | 1 day | P0 |
| **4** | **Training Infrastructure** | | | |
| 4.1 | Implement `Phase1GMNTrainer` | Phase 2.2, 3.1 | 3 days | P0 |
| 4.2 | Implement `NegativeSampler` | None | 2 days | P0 |
| 4.3 | Add GMN loss functions | None | 1 day | P0 |
| 4.4 | Integration tests for training | Phase 4.1-4.3 | 2 days | P0 |
| **5** | **Integration with Existing Code** | | | |
| 5.1 | Update `encoder_registry.py` | Phase 2.2 | 0.5 days | P0 |
| 5.2 | Update `full_model.py` | Phase 2.2 | 1 day | P1 |
| 5.3 | Create config files | None | 0.5 days | P0 |
| 5.4 | Update documentation | All | 1 day | P1 |
| **6** | **Optimization & Extensions** | | | |
| 6.1 | Implement sparse attention (optional) | Phase 2.1 | 3 days | P2 |
| 6.2 | Add gradient checkpointing | Phase 2.1 | 1 day | P2 |
| 6.3 | Attention visualization tools | Phase 2.1 | 2 days | P3 |
| 6.4 | Curriculum learning for GMN | Phase 4.1 | 2 days | P2 |
| **7** | **Evaluation & Ablation** | | | |
| 7.1 | Train GMN on full dataset | All | 3 days | P0 |
| 7.2 | Ablation study: GMN vs Siamese | Phase 7.1 | 2 days | P1 |
| 7.3 | Analyze attention patterns | Phase 7.1 | 2 days | P2 |
| 7.4 | Performance benchmarks | Phase 7.1 | 1 day | P1 |

### Estimated Timeline

- **Core Implementation (Phases 1-3):** 12 days
- **Training Infrastructure (Phase 4):** 8 days
- **Integration (Phase 5):** 3 days
- **Optimization (Phase 6, optional):** 8 days
- **Evaluation (Phase 7):** 8 days

**Total (Critical Path):** 23 days
**Total (With Optimizations):** 39 days

### Dependencies Graph

```
Phase 0 (Prerequisites)
    ├── Phase 1.1 (CrossGraphAttention)
    │       ├── Phase 1.2 (MultiHead)
    │       │       └── Phase 2.1 (GraphMatchingNetwork)
    │       │               ├── Phase 2.2 (HGTWithGMN)
    │       │               │       ├── Phase 4.1 (Trainer)
    │       │               │       └── Phase 5.1 (Registry)
    │       │               └── Phase 6.1 (Sparse attention)
    │       └── Phase 1.3 (Unit tests)
    ├── Phase 3.1 (BatchCollator)
    │       └── Phase 3.2 (Masking)
    │               └── Phase 4.1 (Trainer)
    └── Phase 4.2 (NegativeSampler)
            └── Phase 4.1 (Trainer)
```

---

## 9. Evaluation Metrics

### 9.1 GMN-Specific Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Positive Pair Accuracy** | % of equivalent pairs scored > 0.5 | >95% |
| **Negative Pair Accuracy** | % of non-equivalent pairs scored < 0.5 | >90% |
| **AUC-ROC** | Area under ROC curve | >0.95 |
| **Hard Negative Accuracy** | Accuracy on hard negatives (same vars) | >85% |
| **Variable Renaming Robustness** | Score unchanged when variables renamed | >0.98 |

### 9.2 Comparison to Baseline (Siamese HGT)

```python
# scripts/evaluate_gmn_vs_siamese.py

def compare_gmn_vs_siamese():
    """
    Compare GMN to baseline Siamese approach.

    Metrics:
      - Equivalence detection accuracy
      - Generalization to unseen patterns
      - Robustness to variable renaming
      - Inference time
      - Memory usage
    """
    gmn_model = load_model('checkpoints/phase1c_gmn.pt')
    siamese_model = load_model('checkpoints/phase1_hgt.pt')

    test_pairs = load_test_data('data/test_pairs.json')

    results = {
        'gmn': evaluate_model(gmn_model, test_pairs),
        'siamese': evaluate_model(siamese_model, test_pairs)
    }

    # Compare
    print("=== GMN vs Siamese Comparison ===")
    for metric in ['accuracy', 'auc', 'hard_neg_acc']:
        gmn_score = results['gmn'][metric]
        siamese_score = results['siamese'][metric]
        improvement = (gmn_score - siamese_score) / siamese_score * 100
        print(f"{metric}: GMN={gmn_score:.3f}, Siamese={siamese_score:.3f} (+{improvement:.1f}%)")
```

### 9.3 Ablation Studies

| Ablation | Description | Expected Impact |
|----------|-------------|-----------------|
| **No cross-attention** | Remove GMN, use Siamese | Baseline performance |
| **Single-head attention** | num_heads=1 | -5% accuracy (less diverse patterns) |
| **No residual connections** | Remove residual in cross-attention | -10% accuracy (training instability) |
| **Frozen encoder** | Never fine-tune HGT (Phase 1c) | -3% accuracy (suboptimal joint optimization) |
| **Random negatives only** | No hard negative mining | -15% accuracy (easier training, worse generalization) |

---

## 10. Open Questions & Future Work

### 10.1 Open Questions

1. **Optimal number of cross-attention layers?**
   - Hypothesis: 2 layers sufficient for MBA (expressions are not deeply nested)
   - Experiment: Ablate num_attention_layers ∈ {1, 2, 3, 4}

2. **Best aggregation strategy?**
   - Options: mean, max, mean_max, attention-based
   - Hypothesis: Attention-based best for variable-size graphs
   - Experiment: Compare on depth-14 expressions

3. **Should we fine-tune HGT end-to-end (Phase 1c)?**
   - Trade-off: Better joint optimization vs risk of catastrophic forgetting
   - Recommendation: Try both, monitor Phase 2 performance

4. **How to integrate GMN with Phase 2 (seq2seq)?**
   - Option A: Use GMN embeddings as encoder context
   - Option B: Only use GMN for Phase 1, discard afterward
   - Option C: Add GMN equivalence check as auxiliary loss in Phase 2

5. **Sparse attention necessary for depth-14 expressions?**
   - Full attention: O(N²) = O(1000²) = 1M memory per pair
   - Sparse top-k: O(N × k) = O(1000 × 32) = 32K memory per pair
   - Benchmark: Measure actual memory usage before optimizing

### 10.2 Future Enhancements

#### Enhancement 1: Hierarchical Cross-Attention

```
Instead of node-level attention, attend at subexpression level.

Example:
  Obfuscated:  ((x & y) + (x ^ y)) | z
  Simplified:  (x | y) | z

Standard GMN: Attends at leaf nodes (x, y, z, operators)
Hierarchical: Attends at subexpressions ((x&y)+(x^y) ↔ (x|y))

Benefit: Better captures structural equivalences.
```

#### Enhancement 2: Contrastive Learning with GMN

```python
class ContrastiveGMNLoss(nn.Module):
    """
    Combine InfoNCE with GMN matching scores.

    Loss = InfoNCE(embeddings) + λ * BCE(gmn_scores, labels)

    Encourages both:
      - Good embedding space (InfoNCE)
      - Accurate node correspondence (GMN)
    """
    pass
```

#### Enhancement 3: Graph Pooling with GMN

```
Use GMN attention weights to guide graph pooling (coarsening).

Nodes with strong mutual attention → merge into supernode.

Benefit: Hierarchical graph representation for very deep expressions.
```

#### Enhancement 4: Multi-Task Learning

```
Train GMN jointly with:
  - Task 1: Equivalence detection (current)
  - Task 2: Simplification quality estimation (predict complexity reduction)
  - Task 3: Subexpression matching (identify which subexprs correspond)

Shared encoder, task-specific heads.
```

### 10.3 Research Directions

1. **Attention Mechanism Variants**
   - Explore relative positional encodings in cross-attention
   - Try Performer/LinFormer for linear-complexity attention
   - Investigate graph-specific attention (e.g., edge-aware attention)

2. **Beyond Pairwise Matching**
   - Extend GMN to match 3+ graphs (obfuscated, simplified, canonical form)
   - Learn embeddings where all equivalent expressions cluster

3. **Interpretability**
   - Visualize attention weights to understand what patterns GMN learns
   - Extract "matching rules" from trained GMN (e.g., "x & ~x always cancels")
   - Use attention for explainability ("simplification possible because nodes X and Y cancel")

4. **Transfer Learning**
   - Pre-train GMN on synthetic MBA expressions
   - Fine-tune on real-world obfuscated binaries
   - Investigate domain adaptation techniques

---

## 11. Conclusion

### Summary

Graph Matching Networks with Cross-Graph Attention provide a principled approach to enhance MBA equivalence detection by:
- **Explicitly modeling node correspondences** between obfuscated and simplified expressions
- **Learning cancellation patterns** (nodes with no correspondence)
- **Improving generalization** to unseen algebraic identities

The proposed implementation integrates seamlessly with the existing MBA Deobfuscator architecture, requiring minimal changes to existing code. The three-stage training strategy (pre-train HGT, train GMN, optional fine-tuning) balances training efficiency with performance.

### Expected Benefits

| Aspect | Expected Improvement |
|--------|---------------------|
| Equivalence detection accuracy | +10-15% on hard negatives |
| Robustness to variable renaming | Near-perfect (learned correspondence) |
| Generalization to unseen patterns | +5-10% (learns generic cancellation) |
| Interpretability | Attention weights show matching |
| Inference time | ~2× slower (cross-attention overhead) |

### Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| O(N²) memory for large graphs | Implement sparse top-k attention (Phase 6.1) |
| GMN benefits unclear | Run ablation study early (Phase 7.2) |
| Training instability | Use gradient checkpointing, lower LR |
| Overfitting to training patterns | Hard negative mining, diverse data augmentation |

### Next Steps

1. **Implement core GMN components** (Phases 1-2, ~2 weeks)
2. **Integrate with existing training** (Phases 3-5, ~1.5 weeks)
3. **Train and evaluate** (Phase 7.1, ~3 days training)
4. **Ablation study** (Phase 7.2, ~2 days)
5. **Decide on deployment** based on ablation results

If ablation shows >10% improvement on equivalence detection, proceed with full integration into Phase 2 training. Otherwise, treat GMN as optional auxiliary loss or Phase 1-only feature.

---

## Appendix A: Code Templates

### A.1 Training Script Template

```python
# scripts/train_gmn.py

import torch
from src.models.gmn import HGTWithGMN
from src.training.phase1_gmn_trainer import Phase1GMNTrainer
from src.data.dataset import MBAPairDataset
from src.models.gmn import GMNBatchCollator

def main():
    # Config
    config = {
        'hgt_checkpoint': 'checkpoints/phase1_hgt.pt',
        'gmn_config': {
            'freeze_encoder': True,
            'num_attention_layers': 2,
            'num_heads': 8,
            'dropout': 0.1,
            'aggregation': 'mean_max'
        },
        'batch_size': 16,
        'learning_rate': 1e-4,
        'num_epochs': 10,
        'negative_mode': 'hard',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    # Model
    model = HGTWithGMN(
        hgt_checkpoint_path=config['hgt_checkpoint'],
        gmn_config=config['gmn_config']
    ).to(config['device'])

    # Data
    train_dataset = MBAPairDataset('data/train_pairs.json', negative_mode=config['negative_mode'])
    val_dataset = MBAPairDataset('data/val_pairs.json', negative_mode='random')

    collator = GMNBatchCollator()
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config['batch_size'],
        shuffle=True, collate_fn=collator
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config['batch_size'],
        collate_fn=collator
    )

    # Trainer
    trainer = Phase1GMNTrainer(model, config)

    # Train
    trainer.train(train_loader, val_loader)

    # Save
    torch.save(model.state_dict(), 'checkpoints/phase1b_gmn.pt')

if __name__ == '__main__':
    main()
```

### A.2 Evaluation Script Template

```python
# scripts/evaluate_gmn.py

import torch
from sklearn.metrics import roc_auc_score, accuracy_score
from src.models.gmn import HGTWithGMN
from src.data.dataset import MBAPairDataset

def evaluate(model, dataloader, device):
    """Evaluate GMN on test set."""
    model.eval()
    all_scores = []
    all_labels = []

    with torch.no_grad():
        for graph1_batch, graph2_batch, labels, _ in dataloader:
            graph1_batch = graph1_batch.to(device)
            graph2_batch = graph2_batch.to(device)

            scores = model(graph1_batch, graph2_batch).squeeze().cpu()
            all_scores.extend(scores.tolist())
            all_labels.extend(labels.tolist())

    # Metrics
    all_scores = torch.tensor(all_scores)
    all_labels = torch.tensor(all_labels)

    preds = (all_scores > 0.5).float()

    acc = accuracy_score(all_labels, preds)
    auc = roc_auc_score(all_labels, all_scores)

    print(f"Accuracy: {acc:.4f}")
    print(f"AUC-ROC: {auc:.4f}")

    return {'accuracy': acc, 'auc': auc}

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = HGTWithGMN.from_pretrained('checkpoints/phase1c_gmn.pt').to(device)

    test_dataset = MBAPairDataset('data/test_pairs.json')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16)

    results = evaluate(model, test_loader, device)

if __name__ == '__main__':
    main()
```

---

## Appendix B: Configuration Examples

### B.1 Phase 1b Config (GMN Training)

```yaml
# configs/phase1b_gmn.yaml

model:
  encoder_type: hgt_gmn
  encoder_checkpoint: checkpoints/phase1_hgt.pt

  gmn:
    freeze_encoder: true
    num_attention_layers: 2
    num_heads: 8
    dropout: 0.1
    aggregation: mean_max

training:
  phase: 1b
  loss: binary_classification
  batch_size: 16
  num_epochs: 10
  learning_rate: 1e-4
  weight_decay: 1e-5

  optimizer: adam
  scheduler: cosine
  warmup_steps: 500

  negative_sampling:
    mode: hard  # 'random', 'hard', 'semi-hard'
    ratio: 1.0  # 1 negative per positive

  logging:
    log_interval: 100
    eval_interval: 500
    save_interval: 1000

data:
  train_path: data/train_pairs.json
  val_path: data/val_pairs.json
  test_path: data/test_pairs.json

  num_workers: 4
  pin_memory: true

device: cuda
seed: 42
```

### B.2 Phase 1c Config (End-to-End Fine-Tuning)

```yaml
# configs/phase1c_gmn_finetune.yaml

model:
  encoder_type: hgt_gmn
  encoder_checkpoint: checkpoints/phase1b_gmn.pt

  gmn:
    freeze_encoder: false  # UNFREEZE for end-to-end training
    num_attention_layers: 2
    num_heads: 8
    dropout: 0.1
    aggregation: mean_max

training:
  phase: 1c
  loss: binary_classification
  batch_size: 16
  num_epochs: 5  # Fewer epochs for fine-tuning
  learning_rate: 1e-5  # LOWER learning rate
  weight_decay: 1e-5

  optimizer: adam
  scheduler: cosine

  negative_sampling:
    mode: hard
    ratio: 1.0

data:
  train_path: data/train_pairs.json
  val_path: data/val_pairs.json

device: cuda
seed: 42
```

---

**Document Version:** 1.0
**Last Updated:** 2026-01-17
**Authors:** Technical Writing Assistant (Claude Code)
**Status:** Planning Document (No Implementation Yet)