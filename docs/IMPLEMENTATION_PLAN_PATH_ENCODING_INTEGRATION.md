# Implementation Plan: Path Encoding Integration with GNN Encoders

**Status: PLANNING** (2024-01-17)

## Quality Review Fixes Applied

Based on quality-reviewer feedback (NEEDS_CHANGES):

1. **CRITICAL - Device mismatch**: Added `.to(h_flat.device)` after path context aggregation
2. **HIGH - node_types inconsistency**: Added `_infer_node_types()` helper method used by both encoders
3. **HIGH - Projector allocation off-by-one**: Handle `num_injections=0` by setting `path_projectors=None`
4. **SHOULD_FIX - God function**: Extracted `_inject_path_context()` method from HGT._forward_impl
5. **SHOULD_FIX - Testing gaps**: Added parameterized tests and determinism tests
6. **Magic number**: Added `HGT_PATH_INJECTION_SCALE` constant (default 0.1)

## Overview

Integrate `PathBasedEdgeEncoder` with `GGNNEncoder` and `HGTEncoder` to enable path-aware message passing. Path encoding captures relationships along all paths between edge endpoints, improving detection of shared subexpressions in MBA DAGs.

## Motivation

Current encoders process edges independently. With path encoding:
- GGNN can use path context when computing edge-specific messages
- HGT can incorporate path information into heterogeneous attention
- Both benefit from recognizing that `(x & y)` appearing twice shares structural context

## Architecture

### Integration Strategy

**Option A: Edge Embedding Augmentation (RECOMMENDED)**

Compute path-based edge embeddings once at the start, then use them to augment message passing at each layer.

```
┌─────────────────────────────────────────────────────────────────┐
│  FORWARD PASS WITH PATH ENCODING                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input: node_types, edge_index, edge_type                       │
│                        │                                        │
│                        ▼                                        │
│  ┌─────────────────────────────────────┐                        │
│  │  PathBasedEdgeEncoder               │  ← Compute once        │
│  │  edge_emb = [num_edges, hidden_dim] │                        │
│  └─────────────────────────────────────┘                        │
│                        │                                        │
│                        ▼                                        │
│  ┌─────────────────────────────────────┐                        │
│  │  GNN Layer 1                        │                        │
│  │  messages += edge_emb modulation    │  ← Use at each layer   │
│  └─────────────────────────────────────┘                        │
│                        │                                        │
│                        ▼                                        │
│  ┌─────────────────────────────────────┐                        │
│  │  GNN Layer 2...N                    │                        │
│  │  messages += edge_emb modulation    │                        │
│  └─────────────────────────────────────┘                        │
│                        │                                        │
│                        ▼                                        │
│  Output: [num_nodes, hidden_dim]                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Why Option A:**
- Path computation is expensive (O(E * (V + E))), so compute once
- Edge embeddings are static (graph structure doesn't change during forward)
- Memory efficient: single [num_edges, hidden_dim] tensor

**Option B: Per-Layer Path Context (NOT RECOMMENDED)**
- Recompute paths at each layer with updated node features
- Prohibitively expensive for 8+ layer networks
- Marginal benefit over Option A

## Implementation Details

### 0. Shared Helper Method

Both encoders need to infer node types from input. Add to BaseEncoder or as standalone function:

```python
def _infer_node_types(x: torch.Tensor) -> torch.Tensor:
    """
    Convert node features to type IDs.

    Args:
        x: Either [num_nodes] type IDs or [num_nodes, node_dim] features

    Returns:
        [num_nodes] tensor of node type IDs
    """
    return x.argmax(dim=-1) if x.dim() > 1 else x
```

### 1. Modify `GGNNEncoder`

GGNNEncoder currently uses edge-type-specific MLPs. Integration adds path-based modulation.

```python
class GGNNEncoder(BaseEncoder):
    def __init__(
        self,
        # ... existing params ...
        use_path_encoding: bool = False,
        path_max_length: int = 6,
        path_max_paths: int = 16,
        path_aggregation: str = 'mean',
    ):
        # ... existing init ...

        self.use_path_encoding = use_path_encoding
        self.path_encoder = None
        if use_path_encoding:
            self.path_encoder = PathBasedEdgeEncoder(
                hidden_dim=hidden_dim,
                num_node_types=10,  # MBA node types
                num_edge_types=num_edge_types,
                max_path_length=path_max_length,
                max_paths=path_max_paths,
                aggregation=path_aggregation,
            )
            # Gating mechanism to combine message with path context
            self.path_gate = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Sigmoid(),
            )

    def _forward_impl(self, x, edge_index, batch, edge_type=None):
        num_nodes = x.size(0)
        h = self.node_embedding(x)
        h = F.elu(h)

        # Compute path-based edge embeddings once
        path_edge_emb = None
        if self.use_path_encoding and self.path_encoder is not None:
            # Use standardized node type inference (quality fix)
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

                # Modulate message with path context
                if path_edge_emb is not None:
                    edge_path_emb = path_edge_emb[mask]
                    # Gated combination: gate * msg + (1-gate) * path_context
                    gate_input = torch.cat([msg, edge_path_emb], dim=-1)
                    gate = self.path_gate(gate_input)
                    msg = gate * msg + (1 - gate) * edge_path_emb

                messages.index_add_(0, dst, msg)

            h = self.gru(messages, h)

        return h
```

### 2. Modify `HGTEncoder`

HGTEncoder uses PyG's HGTConv which doesn't expose edge features directly. Integration adds path context as a post-processing step after each HGT layer.

```python
class HGTEncoder(BaseEncoder):
    def __init__(
        self,
        # ... existing params ...
        use_path_encoding: bool = False,
        path_max_length: int = 6,
        path_max_paths: int = 16,
        path_aggregation: str = 'mean',
        path_injection_interval: int = 2,  # Inject every N layers
        path_injection_scale: float = HGT_PATH_INJECTION_SCALE,  # Configurable scale (quality fix)
    ):
        # ... existing init ...

        self.use_path_encoding = use_path_encoding
        self.path_injection_interval = path_injection_interval
        self.path_injection_scale = path_injection_scale
        self.path_encoder = None
        self.path_projectors = None

        if use_path_encoding:
            self.path_encoder = PathBasedEdgeEncoder(
                hidden_dim=hidden_dim,
                num_node_types=num_node_types,
                num_edge_types=num_edge_types,
                max_path_length=path_max_length,
                max_paths=path_max_paths,
                aggregation=path_aggregation,
            )
            # One projector per injection point (quality fix: handle zero injections)
            num_injections = (num_layers - 1) // path_injection_interval
            if num_injections > 0:
                self.path_projectors = nn.ModuleList([
                    nn.Linear(hidden_dim, hidden_dim)
                    for _ in range(num_injections)
                ])
            # else: self.path_projectors remains None

    def _aggregate_path_to_nodes(
        self,
        path_edge_emb: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        """
        Aggregate path-based edge embeddings to destination nodes.

        Each node receives mean of incoming edge path embeddings.
        """
        node_path_context = torch.zeros(num_nodes, self.hidden_dim, device=edge_index.device)
        edge_counts = torch.zeros(num_nodes, device=edge_index.device)

        dst_nodes = edge_index[1]
        node_path_context.index_add_(0, dst_nodes, path_edge_emb)
        edge_counts.index_add_(0, dst_nodes, torch.ones_like(dst_nodes, dtype=torch.float))

        # Avoid division by zero
        edge_counts = edge_counts.clamp(min=1).unsqueeze(-1)
        node_path_context = node_path_context / edge_counts

        return node_path_context

    def _inject_path_context(
        self,
        x_dict: Dict[str, torch.Tensor],
        path_edge_emb: torch.Tensor,
        edge_index: torch.Tensor,
        original_node_types: torch.Tensor,
        path_inject_idx: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Inject path context into node features after HGT layer.
        Extracted from _forward_impl for clarity (quality fix: God function).

        Args:
            x_dict: Heterogeneous node features
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

        # Aggregate path embeddings to nodes
        path_context = self._aggregate_path_to_nodes(path_edge_emb, edge_index, num_nodes)
        path_context = path_context.to(h_flat.device)  # Quality fix: device mismatch
        path_context = self.path_projectors[path_inject_idx](path_context)

        # Residual addition with configurable scale
        h_flat = h_flat + self.path_injection_scale * path_context

        # Convert flat tensor back to heterogeneous dict
        for ntype_str in x_dict:
            ntype_int = int(ntype_str)
            mask = (original_node_types == ntype_int)
            x_dict[ntype_str] = h_flat[mask]

        return x_dict

    def _forward_impl(self, x, edge_index, batch, edge_type=None):
        x_dict, edge_index_dict = self._to_heterogeneous(x, edge_index, edge_type)
        original_node_types = x

        # Compute path embeddings once (quality fix: use standardized inference)
        path_edge_emb = None
        if self.use_path_encoding and self.path_encoder is not None:
            node_types = _infer_node_types(x)
            path_edge_emb = self.path_encoder(edge_index, edge_type, node_types)

        global_block_idx = 0
        path_inject_idx = 0

        for layer_idx, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            # HGT layer
            x_dict_new = conv(x_dict, edge_index_dict)
            for ntype_str in x_dict:
                if ntype_str in x_dict_new:
                    x_dict[ntype_str] = norm(x_dict[ntype_str] + self.dropout(x_dict_new[ntype_str]))
                    x_dict[ntype_str] = F.elu(x_dict[ntype_str])

            # Global attention (existing)
            if (self.use_global_attention
                and self.global_attn_blocks is not None
                and (layer_idx + 1) % self.global_attn_interval == 0
                and layer_idx < len(self.convs) - 1):
                # ... existing global attention code ...
                pass

            # Path context injection (quality fix: check path_projectors is not None)
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

        # ... existing output conversion ...
```

### 3. Update `src/constants.py`

Add integration-specific constants (already have base path encoding constants).

```python
# Path encoding integration with encoders
GGNN_USE_PATH_ENCODING: bool = False     # Enable path encoding in GGNN
HGT_USE_PATH_ENCODING: bool = False      # Enable path encoding in HGT
HGT_PATH_INJECTION_INTERVAL: int = 2     # Inject path context every N layers
HGT_PATH_INJECTION_SCALE: float = 0.1    # Scale factor for path context residual (quality fix)
```

### 4. Testing

```python
# tests/test_path_encoding_integration.py

class TestGGNNPathEncoding:
    def test_ggnn_with_path_encoding_shape(self):
        """GGNN with path encoding produces correct output shape."""
        encoder = GGNNEncoder(
            node_dim=32,
            hidden_dim=64,
            num_timesteps=4,
            use_path_encoding=True,
        )
        # ... test shape preservation ...

    def test_ggnn_path_encoding_gradient_flow(self):
        """Gradients flow through path encoder."""
        # ... verify path_encoder params have gradients ...

    def test_ggnn_backward_compatible(self):
        """GGNN without path encoding unchanged."""
        encoder = GGNNEncoder(use_path_encoding=False)
        # ... verify same behavior as before ...

    # Quality fix: Parameterized tests
    @pytest.mark.parametrize("max_length,max_paths,aggregation", [
        (2, 4, 'mean'),
        (6, 16, 'max'),
        (10, 32, 'attention'),
    ])
    def test_ggnn_path_encoding_parameterized(self, max_length, max_paths, aggregation):
        """GGNN works with various path encoding configurations."""
        encoder = GGNNEncoder(
            use_path_encoding=True,
            path_max_length=max_length,
            path_max_paths=max_paths,
            path_aggregation=aggregation,
        )
        # ... test shape and gradient flow ...

    # Quality fix: Determinism test
    def test_ggnn_path_embedding_determinism(self):
        """Path embeddings are deterministic for same input."""
        encoder = GGNNEncoder(use_path_encoding=True)
        encoder.eval()
        # Run forward twice with same input
        output1 = encoder(x, edge_index, batch, edge_type)
        output2 = encoder(x, edge_index, batch, edge_type)
        assert torch.allclose(output1, output2, atol=1e-6)


class TestHGTPathEncoding:
    def test_hgt_with_path_encoding_shape(self):
        """HGT with path encoding produces correct output shape."""
        encoder = HGTEncoder(
            hidden_dim=64,
            num_layers=4,
            use_path_encoding=True,
        )
        # ... test shape preservation ...

    def test_hgt_path_injection_count(self):
        """Verify correct number of path injections."""
        encoder = HGTEncoder(
            num_layers=8,
            use_path_encoding=True,
            path_injection_interval=2,
        )
        # Should have 3 injection points (after layers 1,3,5)
        assert len(encoder.path_projectors) == 3

    # Quality fix: Test zero injections case
    def test_hgt_path_injection_zero(self):
        """Verify no projectors created when num_injections=0."""
        encoder = HGTEncoder(
            num_layers=2,
            use_path_encoding=True,
            path_injection_interval=2,
        )
        # (2-1)//2 = 0 injections, path_projectors should be None
        assert encoder.path_projectors is None

    def test_hgt_combined_global_and_path(self):
        """HGT with both global attention and path encoding."""
        encoder = HGTEncoder(
            use_global_attention=True,
            use_path_encoding=True,
        )
        # ... verify both features work together ...

    # Quality fix: Parameterized tests for HGT
    @pytest.mark.parametrize("injection_interval,scale", [
        (2, 0.1),
        (3, 0.05),
        (4, 0.2),
    ])
    def test_hgt_path_injection_parameterized(self, injection_interval, scale):
        """HGT works with various path injection configurations."""
        encoder = HGTEncoder(
            num_layers=8,
            use_path_encoding=True,
            path_injection_interval=injection_interval,
            path_injection_scale=scale,
        )
        # ... test shape and gradient flow ...

    # Quality fix: Device consistency test
    def test_hgt_path_context_device_consistency(self):
        """Path context is on same device as node features."""
        encoder = HGTEncoder(use_path_encoding=True)
        # Test with CPU tensors (GPU test requires CUDA)
        x = torch.randint(0, 10, (50,))
        edge_index = torch.randint(0, 50, (2, 100))
        edge_type = torch.randint(0, 8, (100,))
        batch = torch.zeros(50, dtype=torch.long)

        output = encoder(x, edge_index, batch, edge_type=edge_type)
        assert output.device == x.device
```

## Task Breakdown

| Task | File | Status | Priority |
|------|------|--------|----------|
| Add _infer_node_types helper function | `src/models/encoder.py` | TODO | P1 |
| Add path encoding params to GGNNEncoder.__init__ | `src/models/encoder.py` | TODO | P1 |
| Implement path modulation in GGNN._forward_impl | `src/models/encoder.py` | TODO | P1 |
| Add path encoding params to HGTEncoder.__init__ | `src/models/encoder.py` | TODO | P1 |
| Implement _aggregate_path_to_nodes helper | `src/models/encoder.py` | TODO | P1 |
| Implement _inject_path_context method (quality fix) | `src/models/encoder.py` | TODO | P1 |
| Implement path injection in HGT._forward_impl | `src/models/encoder.py` | TODO | P1 |
| Add integration constants (including scale) | `src/constants.py` | TODO | P1 |
| Write GGNN integration tests | `tests/test_path_encoding_integration.py` | TODO | P1 |
| Write HGT integration tests | `tests/test_path_encoding_integration.py` | TODO | P1 |
| Write parameterized tests (quality fix) | `tests/test_path_encoding_integration.py` | TODO | P1 |
| Write determinism tests (quality fix) | `tests/test_path_encoding_integration.py` | TODO | P1 |
| Benchmark performance overhead | - | TODO | P2 |

## Complexity Analysis

**Time Overhead:**
- Path encoding computed once per forward pass: O(E * (V + E))
- For typical MBA graphs (50-200 nodes, 100-400 edges): ~10-50ms
- Amortized over 8 timesteps (GGNN) or 12 layers (HGT): negligible per-layer cost

**Memory Overhead:**
- Path edge embeddings: O(E * hidden_dim) = ~400 edges * 256d * 4 bytes = ~400KB
- Path projectors (HGT): O(num_injections * hidden_dim²) = ~5 * 768² * 4 = ~12MB
- Acceptable within 24GB memory budget

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Path computation slow | Training bottleneck | Limit max_paths, cache adjacency |
| HGTConv API incompatibility | Integration breaks | Use post-layer injection instead of modifying HGTConv |
| Path context disrupts convergence | Worse accuracy | Small residual scale (0.1), ablation study |
| Memory overhead on large graphs | OOM | Compute path encoding in eval-only mode for very large graphs |

## Success Criteria

1. Unit tests pass for both GGNN and HGT integration
2. Backward compatibility: `use_path_encoding=False` produces identical results to before
3. Path encoding adds <20% overhead to forward pass time
4. Model with path encoding matches or exceeds baseline accuracy (pending training)

## Rollback Plan

Path encoding is opt-in via `use_path_encoding=False` default. No changes to existing encoder behavior unless explicitly enabled.

## Open Questions

1. **Should path injection happen before or after global attention?**
   - Current plan: independent intervals, may overlap
   - Alternative: single combined interval for both

2. **Should GGNN use gated combination or simple addition?**
   - Current plan: gated combination for more flexibility
   - Alternative: simpler residual addition like HGT

3. ~~**Should node types be inferred from x or passed separately?**~~
   - **RESOLVED**: Added `_infer_node_types()` helper that works with both node features and type IDs
   - Both encoders use the same standardized inference logic
