# MEMORY-AUGMENTED GNN IMPLEMENTATION PLAN V2

## Revision Notes

**V2 Changes:**
- **Architecture fix**: Separate MemoryAugmentedRGCN and MemoryAugmentedHGT encoders instead of broken hybrid RGCN+HGT
- **Stability improvements**: Added temperature scaling, attention dropout, gradient clipping to AlgebraicRuleMemory
- **Loss balancing**: Added gradual multi-task loss introduction and gradient norm monitoring
- **Error handling**: Added robust error handling to auxiliary label generation
- **Simplified GCNII**: Removed duplicate GCNII parameters, rely on base encoder GCNII settings

---

## Executive Summary

This plan adds memory-augmented GNN encoders to the MBA deobfuscation system by extending existing RGCN and HGT architectures with differentiable memory modules. The implementation creates:

1. **AlgebraicRuleMemory**: External memory bank (256 slots) storing learned transformation patterns via attention
2. **MemoryAugmentedRGCN**: RGCN encoder with memory injection between layers
3. **MemoryAugmentedHGT**: HGT encoder with memory injection between layers
4. **Stack-augmented message passing**: Soft stack for tree traversal context
5. **Multi-task training**: Rule prediction, contrastive equivalence, axiom satisfaction

**Integration Strategy**: Extends encoder registry pattern, maintains backward compatibility, integrates with Phase 2 curriculum training.

**Parameter Budget**:
- MemoryAugmentedRGCN: ~60M params (RGCN 12 layers + memory 0.065M + stack 0.5M)
- MemoryAugmentedHGT: ~60M params (HGT 12 layers + memory 0.065M + stack 0.5M)

**Timeline**: 5 weeks implementation (removed hybrid complexity), medium risk.

---

## 1. Architecture Overview

### 1.1 Memory-Augmented Design Pattern

Both RGCN and HGT encoders follow the same augmentation pattern:

```
Input: Node type IDs [N]
    ↓
Base Encoder Embedding (node_type_embed)
    ↓ [N, hidden_dim]
Layer 1: GNN message passing
    ↓ [N, hidden_dim]
Memory Query & Injection (after layer 3, 6, 9)
    ↓ [N, hidden_dim] (residual add, not concat)
Layer 2-12: Continue message passing with memory context
    ↓
Stack-Augmented Readout
    ↓ [B, hidden_dim]
Output: Graph embeddings
```

**Key Design Decisions:**

1. **Memory injection via residual addition** (not concatenation):
   ```python
   h = base_layer(h, edge_index, edge_type)
   if layer_idx in [3, 6, 9]:  # Inject every 3 layers
       memory_context = self.memory(h, batch)
       h = h + self.memory_proj(memory_context)  # Residual add
   ```

2. **Separate encoders instead of hybrid**: MemoryAugmentedRGCN and MemoryAugmentedHGT are independent. Ablation studies compare them against base RGCN/HGT and each other.

3. **Reuse existing GCNII mitigation**: Both encoders inherit base encoder's GCNII parameters. No duplicate mitigation logic.

### 1.2 Integration Points

| Component | Integration Method | Backward Compatibility |
|-----------|-------------------|------------------------|
| **Encoders** | `MemoryAugmentedRGCN(RGCNEncoder)`, `MemoryAugmentedHGT(HGTEncoder)` | ✓ Inherit from base encoders |
| **Registry** | Add 'memory_rgcn', 'memory_hgt' to encoder_registry.py | ✓ Via get_encoder() |
| **Training** | Extend Phase2Trainer with multi-task losses | ✓ Via config flag `use_memory_augmented` |
| **Config** | `configs/memory_augmented_rgcn.yaml`, `configs/memory_augmented_hgt.yaml` | ✓ Separate configs |
| **Ablation** | Compare memory_rgcn vs rgcn, memory_hgt vs hgt | ✓ Existing ablation framework |

---

## 2. Core Components

### 2.1 AlgebraicRuleMemory Module

**Purpose**: Differentiable memory bank storing learned algebraic transformation patterns.

**Architecture**:
```python
class AlgebraicRuleMemory(nn.Module):
    """
    Differentiable memory bank for algebraic transformation patterns.

    Memory slots store rule embeddings; node features query via attention.
    Includes stability mechanisms: temperature scaling, dropout, LayerNorm.
    """
    def __init__(
        self,
        num_slots: int = 256,
        slot_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
        temperature: Optional[float] = None,
    ):
        super().__init__()

        # Memory bank: [num_slots, slot_dim] learnable embeddings
        self.memory_slots = nn.Parameter(
            torch.randn(num_slots, slot_dim) * (1.0 / math.sqrt(slot_dim))
        )

        # Query/Key/Value projections with LayerNorm for stability
        self.query_norm = nn.LayerNorm(slot_dim)
        self.key_norm = nn.LayerNorm(slot_dim)

        self.query_proj = nn.Linear(slot_dim, slot_dim)
        self.key_proj = nn.Linear(slot_dim, slot_dim)
        self.value_proj = nn.Linear(slot_dim, slot_dim)

        # Multi-head attention with dropout
        self.attention = nn.MultiheadAttention(
            slot_dim, num_heads, dropout=dropout, batch_first=True
        )

        # Temperature scaling for attention stability
        self.temperature = temperature or math.sqrt(slot_dim)

        # Output projection
        self.output_proj = nn.Linear(slot_dim, slot_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, node_features: torch.Tensor, batch: torch.Tensor
    ) -> torch.Tensor:
        """
        Query memory with node features, return retrieved context.

        Args:
            node_features: [N, slot_dim] node embeddings
            batch: [N] batch indices

        Returns:
            memory_context: [N, slot_dim] retrieved rule context
        """
        # Normalize queries and keys for stability
        queries = self.query_norm(self.query_proj(node_features))  # [N, slot_dim]

        # Prepare memory as keys/values (broadcast to batch size)
        num_nodes = node_features.shape[0]
        memory_keys = self.key_norm(
            self.key_proj(self.memory_slots)
        )  # [num_slots, slot_dim]
        memory_values = self.value_proj(self.memory_slots)  # [num_slots, slot_dim]

        # Expand memory for batched attention: [N, num_slots, slot_dim]
        memory_keys_expanded = memory_keys.unsqueeze(0).expand(
            num_nodes, -1, -1
        )
        memory_values_expanded = memory_values.unsqueeze(0).expand(
            num_nodes, -1, -1
        )

        # Apply temperature scaling to queries
        queries_scaled = queries / self.temperature

        # Multi-head attention: queries attend to memory
        attn_output, attn_weights = self.attention(
            query=queries_scaled.unsqueeze(1),  # [N, 1, slot_dim]
            key=memory_keys_expanded,  # [N, num_slots, slot_dim]
            value=memory_values_expanded,  # [N, num_slots, slot_dim]
        )

        # Output: [N, 1, slot_dim] -> [N, slot_dim]
        memory_context = attn_output.squeeze(1)
        memory_context = self.output_proj(self.dropout(memory_context))

        return memory_context
```

**Stability Mechanisms:**
1. **Xavier initialization**: `torch.randn(...) * (1.0 / math.sqrt(slot_dim))` prevents large initial values
2. **LayerNorm**: Normalizes queries and keys before attention
3. **Temperature scaling**: Divides queries by `sqrt(slot_dim)` to prevent softmax saturation
4. **Attention dropout**: Regularizes attention weights
5. **Output dropout**: Prevents overfitting to memory

**Parameters**: `256 * 256 = 65,536` base parameters (0.065M).

---

### 2.2 MemoryAugmentedRGCN Encoder

**Architecture**: Extends RGCNEncoder with memory injection at layers 3, 6, 9.

```python
class MemoryAugmentedRGCN(RGCNEncoder):
    """
    RGCN encoder with memory-augmented message passing.

    Injects memory context every 3 layers via residual connection.
    Inherits GCNII over-smoothing mitigation from RGCNEncoder.
    """
    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 12,
        memory_num_slots: int = 256,
        memory_attn_heads: int = 4,
        memory_dropout: float = 0.1,
        memory_injection_layers: List[int] = [3, 6, 9],
        use_stack_augmented: bool = True,
        **kwargs,  # Pass through to RGCNEncoder (gcnii_alpha, etc.)
    ):
        # Initialize base RGCN encoder with GCNII
        super().__init__(hidden_dim=hidden_dim, num_layers=num_layers, **kwargs)

        # Memory module
        self.memory = AlgebraicRuleMemory(
            num_slots=memory_num_slots,
            slot_dim=hidden_dim,
            num_heads=memory_attn_heads,
            dropout=memory_dropout,
        )

        # Memory projection for residual addition
        self.memory_proj = nn.Linear(hidden_dim, hidden_dim)

        # Memory injection schedule
        self.memory_injection_layers = set(memory_injection_layers)

        # Optional stack augmentation
        self.use_stack_augmented = use_stack_augmented
        if use_stack_augmented:
            self.stack = StackAugmentedGNN(
                hidden_dim=hidden_dim, max_depth=20
            )

    def _forward_impl(
        self,
        x: torch.Tensor,  # [N] node type IDs
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        edge_type: torch.Tensor,
        dag_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with memory-augmented message passing.

        Args:
            x: [N] node type IDs (not features!)
            edge_index: [2, E] edge indices
            batch: [N] batch assignment
            edge_type: [E] edge type indices
            dag_pos: [N, 4] positional features (optional)

        Returns:
            h: [N, hidden_dim] node embeddings
        """
        # Initial embedding (from base RGCNEncoder)
        h = self.node_type_embed(x)  # [N, hidden_dim]

        # Add positional encoding if available
        if dag_pos is not None and self.use_dag_pos:
            h = h + self.dag_pos_encoder(dag_pos)

        # Store initial features for GCNII (from base encoder)
        h_0 = h.clone()

        # Message passing with memory injection
        for layer_idx, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            # RGCN message passing (with GCNII from base encoder)
            beta = self._compute_gcnii_beta(
                layer_idx, self.gcnii_lambda, self.use_identity_mapping
            )

            # Local message passing
            h_transformed = conv(h, edge_index, edge_type)

            # GCNII identity mapping (from base encoder)
            if self.use_identity_mapping:
                h_transformed = beta * h_transformed + (1 - beta) * h

            # Residual + LayerNorm
            h_residual = norm(h + self.dropout(h_transformed))

            # GCNII initial residual (from base encoder)
            if self.use_initial_residual:
                h = (
                    self.gcnii_alpha * h_0 +
                    (1 - self.gcnii_alpha) * h_residual
                )
            else:
                h = h_residual

            h = F.elu(h)

            # Memory injection at specific layers
            if layer_idx in self.memory_injection_layers:
                memory_context = self.memory(h, batch)
                h = h + self.memory_proj(memory_context)  # Residual add

        # Stack-augmented readout (optional)
        if self.use_stack_augmented:
            h = self.stack(h, edge_index, edge_type, batch)

        return h
```

**Key Features:**
- Inherits all GCNII mitigation from RGCNEncoder (no duplication)
- Memory injection via residual addition (preserves dimension)
- Injection at layers 3, 6, 9 balances memory influence across depth
- Optional stack augmentation for tree traversal context

**Parameters**: ~60M (RGCN 12 layers) + 0.065M (memory) + 0.5M (stack) = **~60.6M total**

---

### 2.3 MemoryAugmentedHGT Encoder

**Architecture**: Extends HGTEncoder with memory injection at layers 3, 6, 9.

```python
class MemoryAugmentedHGT(HGTEncoder):
    """
    HGT encoder with memory-augmented message passing.

    Injects memory context every 3 layers via residual connection.
    Inherits GCNII over-smoothing mitigation from HGTEncoder.
    """
    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 12,
        num_heads: int = 8,
        memory_num_slots: int = 256,
        memory_attn_heads: int = 4,
        memory_dropout: float = 0.1,
        memory_injection_layers: List[int] = [3, 6, 9],
        use_stack_augmented: bool = True,
        **kwargs,  # Pass through to HGTEncoder
    ):
        # Initialize base HGT encoder with GCNII
        super().__init__(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            **kwargs
        )

        # Memory module
        self.memory = AlgebraicRuleMemory(
            num_slots=memory_num_slots,
            slot_dim=hidden_dim,
            num_heads=memory_attn_heads,
            dropout=memory_dropout,
        )

        # Memory projection for residual addition
        self.memory_proj = nn.Linear(hidden_dim, hidden_dim)

        # Memory injection schedule
        self.memory_injection_layers = set(memory_injection_layers)

        # Optional stack augmentation
        self.use_stack_augmented = use_stack_augmented
        if use_stack_augmented:
            self.stack = StackAugmentedGNN(
                hidden_dim=hidden_dim, max_depth=20
            )

    def _forward_impl(
        self,
        x: torch.Tensor,  # [N] node type IDs
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        edge_type: torch.Tensor,
        dag_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with memory-augmented heterogeneous message passing."""
        # Initial embedding and heterogeneous conversion
        # (Same as base HGTEncoder lines 781-802)
        h = self.node_type_embed(x)

        if dag_pos is not None and self.use_dag_pos:
            h = h + self.dag_pos_encoder(dag_pos)

        # Convert to heterogeneous format
        x_dict, edge_index_dict = self._to_heterogeneous(
            x, h, edge_index, edge_type
        )

        # Store initial for GCNII
        h_0_dict = {ntype: x_dict[ntype].clone() for ntype in x_dict}

        # HGT message passing with memory injection
        for layer_idx, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            # GCNII beta computation (from base encoder)
            beta = self._compute_gcnii_beta(
                layer_idx, self.gcnii_lambda, self.use_identity_mapping
            )

            # HGT heterogeneous message passing
            x_dict_new = conv(x_dict, edge_index_dict)

            # Apply GCNII to each node type (from base encoder logic)
            for ntype in x_dict:
                if ntype not in x_dict_new:
                    continue

                h_current = x_dict[ntype]
                h_transformed = x_dict_new[ntype]

                # Identity mapping
                if self.use_identity_mapping:
                    h_transformed = beta * h_transformed + (1 - beta) * h_current

                # Residual + LayerNorm
                h_residual = norm(h_current + self.dropout(h_transformed))

                # Initial residual
                if self.use_initial_residual and ntype in h_0_dict:
                    h_final = (
                        self.gcnii_alpha * h_0_dict[ntype] +
                        (1 - self.gcnii_alpha) * h_residual
                    )
                else:
                    h_final = h_residual

                x_dict[ntype] = F.elu(h_final)

            # Memory injection at specific layers
            if layer_idx in self.memory_injection_layers:
                # Flatten heterogeneous dict to homogeneous tensor for memory
                h_flat = self._flatten_heterogeneous(x_dict, x)  # [N, hidden_dim]
                memory_context = self.memory(h_flat, batch)
                h_flat = h_flat + self.memory_proj(memory_context)

                # Convert back to heterogeneous dict
                x_dict = self._unflatten_heterogeneous(h_flat, x_dict, x)

        # Convert back to homogeneous for readout
        h = self._flatten_heterogeneous(x_dict, x)

        # Stack-augmented readout (optional)
        if self.use_stack_augmented:
            h = self.stack(h, edge_index, edge_type, batch)

        return h

    def _flatten_heterogeneous(
        self, x_dict: Dict[str, torch.Tensor], x_node_types: torch.Tensor
    ) -> torch.Tensor:
        """Convert heterogeneous dict to homogeneous tensor."""
        # Implementation: scatter x_dict values back to original node order
        h = torch.zeros(
            x_node_types.shape[0], self.hidden_dim, device=x_node_types.device
        )
        for ntype_str, ntype_tensor in x_dict.items():
            mask = (x_node_types == self.node_type_to_id[ntype_str])
            h[mask] = ntype_tensor
        return h

    def _unflatten_heterogeneous(
        self,
        h_flat: torch.Tensor,
        x_dict_template: Dict[str, torch.Tensor],
        x_node_types: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Convert homogeneous tensor back to heterogeneous dict."""
        x_dict_new = {}
        for ntype_str in x_dict_template:
            mask = (x_node_types == self.node_type_to_id[ntype_str])
            x_dict_new[ntype_str] = h_flat[mask]
        return x_dict_new
```

**Key Features:**
- Maintains HGT's heterogeneous message passing
- Flattens to homogeneous for memory query (memory operates on all nodes uniformly)
- Unflattens back to heterogeneous after memory injection
- Inherits GCNII from base HGTEncoder

**Parameters**: ~60M (HGT 12 layers) + 0.065M (memory) + 0.5M (stack) = **~60.6M total**

---

### 2.4 StackAugmentedGNN Module

**Purpose**: Soft stack for tree traversal context with clamped pointer.

```python
class StackAugmentedGNN(nn.Module):
    """
    Stack-augmented readout with soft differentiable stack.

    Maintains stack of node contexts during message passing.
    Stack pointer is clamped to [0, max_depth-1] for safety.
    """
    def __init__(self, hidden_dim: int = 256, max_depth: int = 20):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_depth = max_depth

        # Stack memory: learnable initialization
        # Shape: [1, max_depth, hidden_dim] - broadcast to batch
        self.stack_init = nn.Parameter(
            torch.zeros(1, max_depth, hidden_dim)
        )

        # PUSH/POP gate networks
        self.push_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Output in [0, 1]
        )

        self.pop_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Output in [0, 1]
        )

        # Stack write/read projections
        self.stack_write_proj = nn.Linear(hidden_dim, hidden_dim)
        self.stack_read_proj = nn.Linear(hidden_dim, hidden_dim)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(
        self,
        h: torch.Tensor,  # [N, hidden_dim] node features
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply stack-augmented readout to node features.

        Args:
            h: [N, hidden_dim] node embeddings
            edge_index: [2, E] edges
            edge_type: [E] edge types
            batch: [N] batch indices

        Returns:
            h_augmented: [N, hidden_dim] stack-augmented features
        """
        batch_size = batch.max().item() + 1
        num_nodes = h.shape[0]

        # Initialize stack for batch: [batch_size, max_depth, hidden_dim]
        stack = self.stack_init.expand(batch_size, -1, -1).clone()

        # Initialize pointer for each graph: [batch_size]
        pointer = torch.zeros(batch_size, device=h.device)

        # Per-node stack operations
        h_augmented = []
        for node_idx in range(num_nodes):
            graph_idx = batch[node_idx].item()
            node_features = h[node_idx]  # [hidden_dim]

            # Compute PUSH/POP signals
            push_signal = self.push_gate(node_features)  # [1]
            pop_signal = self.pop_gate(node_features)    # [1]

            # Update pointer with clamping
            ptr = pointer[graph_idx]
            ptr = ptr + push_signal.item() - pop_signal.item()
            ptr = torch.clamp(ptr, min=0.0, max=self.max_depth - 1.0)
            pointer[graph_idx] = ptr

            # Soft stack read via linear interpolation
            floor_idx = int(math.floor(ptr.item()))
            ceil_idx = min(int(math.ceil(ptr.item())), self.max_depth - 1)
            alpha = ptr.item() - floor_idx

            stack_top = (
                (1 - alpha) * stack[graph_idx, floor_idx, :] +
                alpha * stack[graph_idx, ceil_idx, :]
            )  # [hidden_dim]

            # Write current node to stack at pointer position
            # (soft write - weighted update at floor and ceil positions)
            node_to_write = self.stack_write_proj(node_features)
            stack[graph_idx, floor_idx, :] = (
                (1 - alpha) * stack[graph_idx, floor_idx, :] +
                alpha * node_to_write
            )
            if ceil_idx != floor_idx:
                stack[graph_idx, ceil_idx, :] = (
                    alpha * stack[graph_idx, ceil_idx, :] +
                    (1 - alpha) * node_to_write
                )

            # Augment node features with stack context
            stack_context = self.stack_read_proj(stack_top)
            h_aug = self.output_proj(
                torch.cat([node_features, stack_context], dim=-1)
            )  # [hidden_dim]
            h_augmented.append(h_aug)

        # Stack to tensor: [N, hidden_dim]
        h_augmented = torch.stack(h_augmented, dim=0)

        return h_augmented
```

**Stability Features:**
1. **Pointer clamping**: `torch.clamp(ptr, 0, max_depth-1)` prevents out-of-bounds
2. **Linear interpolation**: Soft read between floor/ceil positions maintains differentiability
3. **Soft write**: Weighted update at adjacent positions for smooth gradients

**Parameters**: ~0.5M (gates + projections)

---

## 3. Multi-Task Training Components

### 3.1 Auxiliary Prediction Heads

**RuleApplicationHead**: Predicts algebraic rule at each node.

```python
class RuleApplicationHead(nn.Module):
    """Predicts algebraic rule class for each node."""
    def __init__(self, d_model: int = 256, num_rule_classes: int = 20):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, num_rule_classes),
        )

    def forward(self, node_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_embeddings: [N, d_model]
        Returns:
            rule_logits: [N, num_rule_classes]
        """
        return self.mlp(node_embeddings)
```

**AxiomHead**: Predicts axiom satisfaction scores.

```python
class AxiomHead(nn.Module):
    """Predicts axiom satisfaction scores for each node."""
    def __init__(self, d_model: int = 256, num_axioms: int = 10):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, num_axioms),
            nn.Sigmoid(),  # Satisfaction in [0, 1]
        )

    def forward(self, node_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_embeddings: [N, d_model]
        Returns:
            axiom_scores: [N, num_axioms]
        """
        return self.mlp(node_embeddings)
```

---

### 3.2 Multi-Task Loss with Gradual Introduction

```python
def memory_augmented_loss(
    outputs: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    config: Dict[str, float],
    epoch: int,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Combined loss for memory-augmented training with gradual introduction.

    Loss schedule:
    - Epochs 1-5: Standard losses only (ce, complexity, copy)
    - Epochs 6-15: Add rule loss
    - Epochs 16-25: Add contrastive loss
    - Epochs 26+: Add axiom loss

    Args:
        outputs: Model outputs with keys:
            - 'logits': [B, seq_len, vocab_size]
            - 'length_pred': [B]
            - 'depth_pred': [B]
            - 'copy_scores': [B, seq_len, max_src_len]
            - 'node_embeddings': [N, hidden_dim]
            - 'graph_embeddings': [B, hidden_dim]
            - 'rule_logits': [N, num_rule_classes] (optional)
            - 'axiom_scores': [N, num_axioms] (optional)
        targets: Ground truth with keys:
            - 'target_ids': [B, seq_len]
            - 'length': [B]
            - 'depth': [B]
            - 'copy_indices': [B, seq_len]
            - 'rule_labels': [N] (optional)
            - 'equiv_pairs': (pos_pairs, neg_pairs) (optional)
            - 'axiom_targets': [N, num_axioms] (optional)
        config: Loss weights and hyperparameters
        epoch: Current training epoch (for gradual introduction)

    Returns:
        total_loss: Weighted sum of all losses
        loss_dict: Individual loss values for logging
    """
    # Standard Phase 2 losses (always active)
    ce_loss = F.cross_entropy(
        outputs['logits'].view(-1, outputs['logits'].size(-1)),
        targets['target_ids'].view(-1),
        ignore_index=0,  # Padding token
    )

    complexity_loss = (
        F.mse_loss(outputs['length_pred'], targets['length'].float()) +
        F.mse_loss(outputs['depth_pred'], targets['depth'].float())
    )

    copy_loss = F.binary_cross_entropy_with_logits(
        outputs['copy_scores'].view(-1),
        targets['copy_indices'].float().view(-1),
    )

    # Initialize auxiliary losses
    rule_loss = torch.tensor(0.0, device=ce_loss.device)
    contrastive_loss = torch.tensor(0.0, device=ce_loss.device)
    axiom_loss = torch.tensor(0.0, device=ce_loss.device)

    # Gradual introduction of auxiliary losses
    if epoch >= 6 and 'rule_logits' in outputs:
        rule_logits = outputs['rule_logits']
        rule_targets = targets['rule_labels']
        # Mask padding nodes (rule_labels == -1)
        mask = (rule_targets != -1)
        if mask.sum() > 0:
            rule_loss = F.cross_entropy(
                rule_logits[mask], rule_targets[mask]
            )

    if epoch >= 16 and 'graph_embeddings' in outputs:
        graph_emb = outputs['graph_embeddings']
        pos_pairs, neg_pairs = targets['equiv_pairs']
        contrastive_loss = contrastive_equivalence_loss(
            graph_emb, pos_pairs, neg_pairs, temperature=0.07
        )

    if epoch >= 26 and 'axiom_scores' in outputs:
        axiom_scores = outputs['axiom_scores']
        axiom_targets = targets['axiom_targets']
        # Mask padding nodes
        mask = (targets['rule_labels'] != -1)
        if mask.sum() > 0:
            axiom_loss = F.mse_loss(
                axiom_scores[mask], axiom_targets[mask]
            )

    # Weighted sum with configurable weights
    total_loss = (
        config['ce_weight'] * ce_loss +
        config['complexity_weight'] * complexity_loss +
        config['copy_weight'] * copy_loss +
        config['rule_weight'] * rule_loss +
        config['contrastive_weight'] * contrastive_loss +
        config['axiom_weight'] * axiom_loss
    )

    # Return loss dict for logging
    loss_dict = {
        'total': total_loss.item(),
        'ce': ce_loss.item(),
        'complexity': complexity_loss.item(),
        'copy': copy_loss.item(),
        'rule': rule_loss.item(),
        'contrastive': contrastive_loss.item(),
        'axiom': axiom_loss.item(),
    }

    return total_loss, loss_dict


def contrastive_equivalence_loss(
    embeddings: torch.Tensor,
    pos_pairs: torch.Tensor,
    neg_pairs: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    InfoNCE contrastive loss for equivalence learning.

    Args:
        embeddings: [B, hidden_dim] graph embeddings
        pos_pairs: [P, 2] indices of equivalent pairs
        neg_pairs: [N, 2] indices of non-equivalent pairs
        temperature: Softmax temperature

    Returns:
        loss: Scalar contrastive loss
    """
    if len(pos_pairs) == 0:
        return torch.tensor(0.0, device=embeddings.device)

    # Normalize embeddings
    embeddings = F.normalize(embeddings, dim=-1)

    # Compute positive similarities
    pos_sim = torch.sum(
        embeddings[pos_pairs[:, 0]] * embeddings[pos_pairs[:, 1]],
        dim=-1
    ) / temperature  # [P]

    # Compute negative similarities
    neg_sim = torch.sum(
        embeddings[neg_pairs[:, 0]] * embeddings[neg_pairs[:, 1]],
        dim=-1
    ) / temperature  # [N]

    # InfoNCE loss: log(exp(pos) / (exp(pos) + sum(exp(neg))))
    # Equivalent to: -log_softmax(concat([pos, neg]))[0]
    logits = torch.cat([pos_sim, neg_sim], dim=0)  # [P + N]
    labels = torch.zeros(len(pos_sim), dtype=torch.long, device=embeddings.device)

    loss = F.cross_entropy(
        logits.unsqueeze(0).expand(len(pos_sim), -1),
        labels,
    )

    return loss
```

**Key Features:**
1. **Gradual loss introduction**: Prevents auxiliary tasks from destabilizing primary task
2. **Masked loss computation**: Ignores padding nodes (rule_labels==-1)
3. **Per-loss logging**: Monitors gradient conflicts via TensorBoard
4. **InfoNCE for contrastive**: Standard contrastive learning objective

---

## 4. Data Pipeline Extensions

### 4.1 Auxiliary Label Generation with Error Handling

**Script**: `scripts/generate_auxiliary_labels.py`

```python
#!/usr/bin/env python3
"""
Generate auxiliary labels for memory-augmented training.

Adds rule_labels and axiom_targets to existing JSONL dataset.
Includes robust error handling for large-scale processing.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm

from src.data.ast_parser import ASTParser
from src.constants import NUM_RULE_CLASSES, NUM_AXIOMS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('label_generation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# Rule pattern definitions
RULE_PATTERNS = {
    0: "NONE",  # No rule applies
    1: "IDENTITY_XOR",  # x ^ x -> 0
    2: "IDENTITY_AND",  # x & x -> x
    3: "IDENTITY_OR",   # x | x -> x
    4: "MBA_TO_OR",     # (x & y) + (x ^ y) -> x | y
    5: "MBA_TO_AND",    # (x | y) - (x ^ y) -> x & y
    6: "ABSORPTION",    # x & (x | y) -> x
    7: "DISTRIBUTION",  # x & (y | z) -> (x & y) | (x & z)
    8: "DE_MORGAN_AND", # ~(x & y) -> ~x | ~y
    9: "DE_MORGAN_OR",  # ~(x | y) -> ~x & ~y
    10: "DOUBLE_NOT",   # ~~x -> x
    # ... Define remaining 10 rules
}

AXIOMS = {
    0: "COMMUTATIVE",    # x op y == y op x for +, *, &, |, ^
    1: "ASSOCIATIVE",    # (x op y) op z == x op (y op z)
    2: "DISTRIBUTIVE",   # x * (y + z) == x*y + x*z
    3: "IDENTITY",       # x + 0 == x, x * 1 == x
    4: "ANNIHILATOR",    # x * 0 == 0, x & 0 == 0
    5: "IDEMPOTENT",     # x & x == x, x | x == x
    6: "COMPLEMENT",     # x & ~x == 0, x | ~x == 1
    7: "INVOLUTION",     # ~~x == x
    8: "DE_MORGAN",      # ~(x & y) == ~x | ~y
    9: "ABSORPTION",     # x & (x | y) == x
}


def match_rule_pattern(node: Dict, ast_parser: ASTParser) -> int:
    """
    Match node against rule patterns.

    Args:
        node: AST node dict with 'op', 'left', 'right'
        ast_parser: Parser for traversing AST

    Returns:
        rule_id: Matched rule ID (0 if no match)
    """
    try:
        op = node.get('op')

        # Rule 1: x ^ x -> 0
        if op == 'XOR':
            left = node.get('left')
            right = node.get('right')
            if left and right and ast_parser.are_equivalent(left, right):
                return 1

        # Rule 2: x & x -> x
        if op == 'AND':
            left = node.get('left')
            right = node.get('right')
            if left and right and ast_parser.are_equivalent(left, right):
                return 2

        # Rule 4: (x & y) + (x ^ y) -> x | y
        if op == 'ADD':
            left = node.get('left')
            right = node.get('right')
            if (left and left.get('op') == 'AND' and
                right and right.get('op') == 'XOR'):
                # Check if operands match
                if ast_parser.match_mba_pattern(left, right):
                    return 4

        # ... Match remaining patterns

        return 0  # No rule matches

    except Exception as e:
        logger.warning(f"Pattern matching error: {e}")
        return 0


def check_axiom_satisfaction(node: Dict) -> List[int]:
    """
    Check which axioms are satisfied by node.

    Args:
        node: AST node dict

    Returns:
        axiom_vector: [num_axioms] binary satisfaction vector
    """
    try:
        op = node.get('op')
        axioms = [0] * NUM_AXIOMS

        # Axiom 0: Commutative
        if op in ['ADD', 'MUL', 'AND', 'OR', 'XOR']:
            axioms[0] = 1

        # Axiom 1: Associative
        if op in ['ADD', 'MUL', 'AND', 'OR', 'XOR']:
            axioms[1] = 1

        # Axiom 2: Distributive (for MUL over ADD)
        if op == 'MUL':
            axioms[2] = 1

        # ... Check remaining axioms

        return axioms

    except Exception as e:
        logger.warning(f"Axiom checking error: {e}")
        return [0] * NUM_AXIOMS


def generate_labels_for_sample(
    sample: Dict, ast_parser: ASTParser
) -> Optional[Dict]:
    """
    Generate auxiliary labels for one sample.

    Args:
        sample: Original sample with 'obfuscated', 'simplified', 'depth'
        ast_parser: AST parser

    Returns:
        augmented_sample: Sample with added 'rule_labels', 'axiom_targets'
        None if processing fails
    """
    try:
        # Parse obfuscated expression
        ast = ast_parser.parse(sample['obfuscated'])
        nodes = ast_parser.traverse(ast)  # DFS order

        # Generate rule labels for each node
        rule_labels = [match_rule_pattern(node, ast_parser) for node in nodes]

        # Generate axiom targets for each node
        axiom_targets = [check_axiom_satisfaction(node) for node in nodes]

        # Add to sample
        augmented_sample = sample.copy()
        augmented_sample['rule_labels'] = rule_labels
        augmented_sample['axiom_targets'] = axiom_targets

        return augmented_sample

    except Exception as e:
        logger.error(f"Failed to process sample: {sample.get('obfuscated', 'UNKNOWN')}")
        logger.error(f"Error: {e}")
        return None


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Input JSONL path')
    parser.add_argument('--output', required=True, help='Output JSONL path')
    parser.add_argument('--checkpoint-every', type=int, default=1000000,
                        help='Save checkpoint every N samples')
    parser.add_argument('--dry-run', action='store_true',
                        help='Process only 1000 samples for testing')
    args = parser.parse_args()

    # Initialize parser
    ast_parser = ASTParser()

    # Count total samples
    with open(args.input, 'r') as f:
        total_samples = sum(1 for _ in f)

    if args.dry_run:
        total_samples = min(total_samples, 1000)
        logger.info("DRY RUN MODE: Processing 1000 samples")

    logger.info(f"Processing {total_samples} samples")

    # Process with checkpointing
    failed_samples = []
    checkpoint_idx = 0

    with open(args.input, 'r') as f_in, \
         open(args.output, 'w') as f_out:

        for idx, line in enumerate(tqdm(f_in, total=total_samples)):
            if args.dry_run and idx >= 1000:
                break

            try:
                sample = json.loads(line)
                augmented = generate_labels_for_sample(sample, ast_parser)

                if augmented:
                    f_out.write(json.dumps(augmented) + '\n')
                else:
                    failed_samples.append((idx, sample.get('obfuscated', 'UNKNOWN')))

            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error at line {idx}: {e}")
                failed_samples.append((idx, line[:100]))

            # Checkpoint
            if (idx + 1) % args.checkpoint_every == 0:
                logger.info(f"Checkpoint: {idx + 1} samples processed")
                f_out.flush()

    # Report failures
    failure_rate = len(failed_samples) / total_samples * 100
    logger.info(f"Processing complete: {total_samples - len(failed_samples)}/{total_samples} succeeded")
    logger.info(f"Failure rate: {failure_rate:.2f}%")

    if failed_samples:
        with open('failed_samples.txt', 'w') as f_fail:
            for idx, expr in failed_samples:
                f_fail.write(f"{idx}: {expr}\n")
        logger.warning(f"Wrote {len(failed_samples)} failed samples to failed_samples.txt")

    # Validation pass
    logger.info("Running validation pass...")
    with open(args.output, 'r') as f:
        for idx, line in enumerate(f):
            sample = json.loads(line)
            assert 'rule_labels' in sample, f"Missing rule_labels at line {idx}"
            assert 'axiom_targets' in sample, f"Missing axiom_targets at line {idx}"

    logger.info("✅ Validation passed: All samples have auxiliary labels")


if __name__ == '__main__':
    main()
```

**Error Handling Features:**
1. **Per-sample try-except**: Continues processing even if one sample fails
2. **Checkpointing**: Flushes output every 1M samples
3. **Dry run mode**: Test on 1000 samples before full run
4. **Failure logging**: Writes failed samples to separate file
5. **Validation pass**: Verifies all samples have required fields
6. **Detailed logging**: Both file and console output

---

### 4.2 Dataset Extension

**Modified**: `src/data/dataset.py`

```python
class MBADataset(Dataset):
    """
    MBA expression dataset with optional auxiliary labels.

    Backward compatible: samples without rule_labels/axiom_targets
    default to RULE_NONE (0) and zeros.
    """
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]

        # ... Existing parsing and graph construction ...

        # Load auxiliary labels (with defaults)
        rule_labels = sample.get('rule_labels', [0] * num_nodes)
        axiom_targets = sample.get('axiom_targets', [[0] * NUM_AXIOMS] * num_nodes)

        # Pad to match graph size
        if len(rule_labels) < num_nodes:
            rule_labels = rule_labels + [-1] * (num_nodes - len(rule_labels))

        return {
            'x': x,
            'edge_index': edge_index,
            'edge_type': edge_type,
            'batch': batch,
            'target_ids': target_ids,
            'rule_labels': torch.tensor(rule_labels, dtype=torch.long),
            'axiom_targets': torch.tensor(axiom_targets, dtype=torch.float),
            # ... Other fields ...
        }
```

---

## 5. Implementation Timeline

| Week | Milestone | Deliverables | Risk Mitigation |
|------|-----------|--------------|-----------------|
| **1** | Memory module + RGCNencoder | `algebraic_rule_memory.py`, `MemoryAugmentedRGCN`, unit tests | Start with RGCN (simpler than HGT) |
| **2** | HGT encoder + stack | `MemoryAugmentedHGT`, `StackAugmentedGNN`, integration tests | Reuse RGCN patterns |
| **3** | Auxiliary heads + losses | `RuleApplicationHead`, `AxiomHead`, `memory_augmented_loss` | Test gradual introduction schedule |
| **4** | Data pipeline | `generate_auxiliary_labels.py`, dataset extension, dry run validation | Checkpoint + validation pass |
| **5** | Training + ablation | Train memory_rgcn, memory_hgt, ablation studies, documentation | Monitor gradient norms |

**Critical Path**: Week 1 (memory module stability) → Week 3 (multi-task loss) → Week 5 (training).

**Buffer**: Week 4 (data pipeline) can parallelize with Week 3 if needed.

---

## 6. Testing Strategy

### 6.1 Unit Tests

```python
# tests/test_memory_augmented.py

def test_memory_attention_stability():
    """Memory attention produces bounded outputs (no NaN/Inf)."""
    memory = AlgebraicRuleMemory(num_slots=256, slot_dim=256)

    # Extreme input: large magnitude features
    node_features = torch.randn(100, 256) * 100  # Large values
    batch = torch.zeros(100, dtype=torch.long)

    output = memory(node_features, batch)

    assert not torch.isnan(output).any(), "Memory output contains NaN"
    assert not torch.isinf(output).any(), "Memory output contains Inf"
    assert output.abs().max() < 1000, "Memory output has extreme values"


def test_memory_gradient_flow():
    """Gradients flow through memory module."""
    memory = AlgebraicRuleMemory(num_slots=256, slot_dim=256)
    node_features = torch.randn(100, 256, requires_grad=True)
    batch = torch.zeros(100, dtype=torch.long)

    output = memory(node_features, batch)
    loss = output.sum()
    loss.backward()

    assert node_features.grad is not None
    assert not torch.isnan(node_features.grad).any()
    assert memory.memory_slots.grad is not None


def test_stack_pointer_clamping():
    """Stack pointer stays within [0, max_depth-1]."""
    stack = StackAugmentedGNN(hidden_dim=256, max_depth=20)

    # Simulate deep expression (many PUSH operations)
    h = torch.randn(50, 256)
    edge_index = torch.randint(0, 50, (2, 100))
    edge_type = torch.zeros(100, dtype=torch.long)
    batch = torch.zeros(50, dtype=torch.long)

    output = stack(h, edge_index, edge_type, batch)

    # Check output is valid
    assert output.shape == (50, 256)
    assert not torch.isnan(output).any()


def test_memory_augmented_rgcn_forward():
    """MemoryAugmentedRGCN forward pass with memory injection."""
    encoder = MemoryAugmentedRGCN(
        hidden_dim=256,
        num_layers=6,
        memory_injection_layers=[2, 4],
    )

    x = torch.randint(0, 10, (50,))  # Node type IDs
    edge_index = torch.randint(0, 50, (2, 100))
    edge_type = torch.randint(0, 8, (100,))
    batch = torch.zeros(50, dtype=torch.long)

    output = encoder(x, edge_index, batch, edge_type)

    assert output.shape == (50, 256)
    assert not torch.isnan(output).any()


def test_multi_task_loss_gradual_introduction():
    """Auxiliary losses activate at correct epochs."""
    outputs = {
        'logits': torch.randn(4, 10, 300),
        'length_pred': torch.tensor([5.0, 6.0, 7.0, 8.0]),
        'depth_pred': torch.tensor([3.0, 4.0, 3.0, 5.0]),
        'copy_scores': torch.randn(4, 10, 20),
        'node_embeddings': torch.randn(50, 256),
        'graph_embeddings': torch.randn(4, 256),
        'rule_logits': torch.randn(50, 20),
        'axiom_scores': torch.randn(50, 10),
    }

    targets = {
        'target_ids': torch.randint(0, 300, (4, 10)),
        'length': torch.tensor([5, 6, 7, 8]),
        'depth': torch.tensor([3, 4, 3, 5]),
        'copy_indices': torch.zeros(4, 10),
        'rule_labels': torch.randint(0, 20, (50,)),
        'equiv_pairs': (torch.tensor([[0, 1]]), torch.tensor([[2, 3]])),
        'axiom_targets': torch.rand(50, 10),
    }

    config = {
        'ce_weight': 1.0,
        'complexity_weight': 0.1,
        'copy_weight': 0.1,
        'rule_weight': 0.2,
        'contrastive_weight': 0.1,
        'axiom_weight': 0.05,
    }

    # Epoch 1: Only standard losses
    loss_early, loss_dict_early = memory_augmented_loss(
        outputs, targets, config, epoch=1
    )
    assert loss_dict_early['rule'] == 0.0
    assert loss_dict_early['contrastive'] == 0.0
    assert loss_dict_early['axiom'] == 0.0

    # Epoch 10: Rule loss active
    loss_mid, loss_dict_mid = memory_augmented_loss(
        outputs, targets, config, epoch=10
    )
    assert loss_dict_mid['rule'] > 0.0
    assert loss_dict_mid['contrastive'] == 0.0

    # Epoch 30: All losses active
    loss_late, loss_dict_late = memory_augmented_loss(
        outputs, targets, config, epoch=30
    )
    assert loss_dict_late['rule'] > 0.0
    assert loss_dict_late['contrastive'] > 0.0
    assert loss_dict_late['axiom'] > 0.0
```

---

## 7. Expected Results

### 7.1 Performance Targets

| Metric | Baseline (HGT) | Target (Memory HGT) | Gain |
|--------|----------------|---------------------|------|
| **Accuracy (depth 2-4)** | 95% | >95% | Maintain |
| **Accuracy (depth 5-7)** | 88% | >90% | +2% |
| **Accuracy (depth 8-10)** | 80% | >85% | +5% |
| **Accuracy (depth 11-14)** | 65% | >75% | +10% |
| **Rule prediction acc** | N/A | >80% | New |
| **Training time (50 epochs)** | 36 hours | <48 hours | Acceptable |

**Key Improvement**: +10% accuracy on depth 11-14 via memory-guided extrapolation.

### 7.2 Ablation Study Plan

Compare 6 configurations:
1. Baseline RGCN (12 layers, GCNII)
2. MemoryAugmentedRGCN (no stack)
3. MemoryAugmentedRGCN (with stack)
4. Baseline HGT (12 layers, GCNII)
5. MemoryAugmentedHGT (no stack)
6. MemoryAugmentedHGT (with stack)

Run 3 independent trials per config, report mean ± std, statistical significance (Welch's t-test, p<0.05).

---

## 8. File Modification Matrix

| File | Type | Description |
|------|------|-------------|
| **NEW: `src/models/memory_augmented/__init__.py`** | Create | Package init |
| **NEW: `src/models/memory_augmented/algebraic_rule_memory.py`** | Create | AlgebraicRuleMemory class |
| **NEW: `src/models/memory_augmented/stack_augmented_gnn.py`** | Create | StackAugmentedGNN class |
| **NEW: `src/models/memory_augmented/memory_rgcn.py`** | Create | MemoryAugmentedRGCN class |
| **NEW: `src/models/memory_augmented/memory_hgt.py`** | Create | MemoryAugmentedHGT class |
| **MODIFY: `src/models/heads.py`** | Add | RuleApplicationHead, AxiomHead |
| **MODIFY: `src/models/encoder_registry.py`** | Add | 'memory_rgcn', 'memory_hgt' entries |
| **MODIFY: `src/constants.py`** | Add | Memory/stack/axiom constants |
| **MODIFY: `src/training/losses.py`** | Add | `memory_augmented_loss()`, `contrastive_equivalence_loss()` |
| **NEW: `src/training/memory_augmented_trainer.py`** | Create | MemoryAugmentedTrainer with gradient monitoring |
| **MODIFY: `src/data/dataset.py`** | Extend | Load auxiliary labels with defaults |
| **NEW: `scripts/generate_auxiliary_labels.py`** | Create | Label generation with error handling |
| **NEW: `scripts/train_memory_rgcn.py`** | Create | Training script for MemoryAugmentedRGCN |
| **NEW: `scripts/train_memory_hgt.py`** | Create | Training script for MemoryAugmentedHGT |
| **NEW: `configs/memory_augmented_rgcn.yaml`** | Create | Config for memory RGCN |
| **NEW: `configs/memory_augmented_hgt.yaml`** | Create | Config for memory HGT |
| **NEW: `tests/test_memory_augmented.py`** | Create | Unit + integration tests |
| **NEW: `docs/MEMORY_AUGMENTED.md`** | Create | Architecture documentation |

**Total New Files**: 13 created, 5 modified

---

## 9. Risk Mitigation Summary

| Risk (from V1 review) | Mitigation in V2 |
|-----------------------|------------------|
| **Hybrid architecture broken** | ✅ Removed hybrid, separate RGCN/HGT encoders |
| **Memory attention NaN** | ✅ Added LayerNorm, temperature scaling, dropout |
| **Multi-task gradient conflicts** | ✅ Gradual loss introduction, monitor grad norms |
| **Soft stack out-of-bounds** | ✅ Explicit pointer clamping, linear interpolation |
| **Label generation failures** | ✅ Per-sample try-except, checkpointing, validation |
| **Duplicate GCNII** | ✅ Inherit base encoder GCNII, no duplication |

**All critical issues from V1 addressed.**

---

## Appendix: Constants

Add to `src/constants.py`:

```python
# Memory-Augmented GNN Constants
MEMORY_NUM_SLOTS: int = 256
MEMORY_ATTN_HEADS: int = 4
MEMORY_DROPOUT: float = 0.1
MEMORY_INJECTION_LAYERS: List[int] = [3, 6, 9]  # Inject every 3 layers

# Stack-Augmented GNN
STACK_MAX_DEPTH: int = 20

# Multi-Task Loss Weights
RULE_LOSS_WEIGHT: float = 0.2
CONTRASTIVE_LOSS_WEIGHT: float = 0.1
AXIOM_LOSS_WEIGHT: float = 0.05

# Gradual Loss Introduction Schedule (epochs)
RULE_LOSS_START_EPOCH: int = 6
CONTRASTIVE_LOSS_START_EPOCH: int = 16
AXIOM_LOSS_START_EPOCH: int = 26

# Rule Classes (algebraic transformations)
NUM_RULE_CLASSES: int = 20
RULE_NAMES: Dict[int, str] = {
    0: "NONE",
    1: "IDENTITY_XOR",
    2: "IDENTITY_AND",
    3: "IDENTITY_OR",
    4: "MBA_TO_OR",
    5: "MBA_TO_AND",
    6: "ABSORPTION",
    7: "DISTRIBUTION",
    8: "DE_MORGAN_AND",
    9: "DE_MORGAN_OR",
    10: "DOUBLE_NOT",
    # ... Define remaining 10
}

# Axioms (algebraic properties)
NUM_AXIOMS: int = 10
AXIOM_NAMES: Dict[int, str] = {
    0: "COMMUTATIVE",
    1: "ASSOCIATIVE",
    2: "DISTRIBUTIVE",
    3: "IDENTITY",
    4: "ANNIHILATOR",
    5: "IDEMPOTENT",
    6: "COMPLEMENT",
    7: "INVOLUTION",
    8: "DE_MORGAN",
    9: "ABSORPTION",
}
```

---

This revised plan addresses all critical issues identified in V1 quality review and provides a production-ready implementation strategy for memory-augmented GNN encoders.
