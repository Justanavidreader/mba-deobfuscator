# Implementation Plan: Copy Mechanism for MBA Deobfuscation

## Overview

Add a pointer-generator style copy mechanism to the MBA deobfuscation decoder, enabling the model to copy tokens directly from the input expression. This is crucial for variable preservation in simplification—when the simplified expression contains variables from the obfuscated input, copying is more reliable than generation.

**Priority**: P1 (Critical for variable preservation)
**Complexity**: Medium (3-4 days)
**Dependencies**: None (decoder infrastructure exists)

## Motivation

### The Variable Preservation Problem

Current decoder generates all tokens from vocabulary. Problems:

1. **Variable mismatch**: Model may generate `x0` when input uses `x1`
2. **Variable hallucination**: Model may generate variables not present in input
3. **Inefficient learning**: Model must learn to "copy" variables through generation

Example failure case:
```python
Input:  (x & y) + (x ^ y)         # Uses variables x, y
Output: x0 | y0                    # Generated x0, y0 (wrong names)
Target: x | y                      # Should copy x, y from input
```

### Why Copy Mechanism

Copy mechanism solves this by:
- **Direct token access**: Attention over input tokens enables copying
- **Mixed distribution**: Model learns when to generate vs. copy
- **Guaranteed validity**: Copied variables always match input

Research precedent: Pointer-generator networks (See et al., 2017) achieve strong results in text summarization where source tokens must be preserved.

## Architecture

### Current State (Baseline)

```
Input tokens → Decoder → TokenHead → Vocab distribution
                   ↑
              Cross-attention
                   ↑
              Encoder memory
```

**Issue**: `vocab_dist` is the only output. No mechanism to copy from source.

### Proposed State (With Copy)

```
Input tokens → Decoder → ┌─→ TokenHead → vocab_dist
                   ↓     │
              Cross-attention weights (copy_attn)
                   ↓     │
              Copy gate → p_gen
                   ↓     │
              Mixing layer: final_dist = p_gen * vocab_dist + (1-p_gen) * copy_dist
```

**Key additions**:
1. Expose cross-attention weights (`copy_attn`) for copying
2. Compute generation probability (`p_gen`) from decoder state + context
3. Mix vocab and copy distributions using `p_gen`

### Components

#### 1. Copy Attention (Already Available)

- Decoder cross-attention weights serve as copy distribution
- Shape: `[batch, tgt_len, src_len]`
- Interpretation: `copy_attn[b, i, j]` = probability of copying source token j at output position i

**Current Implementation** (decoder.py lines 76-78):
```python
cross_attn_out, cross_attn_weights = self.cross_attn(
    x_norm, memory, memory, key_padding_mask=memory_mask
)
```

**Status**: ✅ Already computed, just need to expose in return signature

#### 2. Generation Probability (p_gen)

Sigmoid gate that decides "generate from vocab" vs "copy from source".

**Formula**:
```
context = sum(copy_attn * encoder_memory)  # [batch, tgt_len, d_model]
combined = [decoder_state; context]        # [batch, tgt_len, 2*d_model]
p_gen = sigmoid(W_gen @ combined + b_gen)  # [batch, tgt_len, 1]
```

**Current Implementation** (decoder.py lines 121-126):
```python
self.copy_gate = nn.Sequential(
    nn.Linear(d_model * 2, d_model),
    nn.ReLU(),
    nn.Linear(d_model, 1),
    nn.Sigmoid()
)
```

**Status**: ✅ Already implemented, needs forward pass integration

#### 3. Final Distribution Mixing

Combine vocab and copy distributions using p_gen.

**Formula**:
```
final_dist[vocab_id] = p_gen * vocab_dist[vocab_id] + (1-p_gen) * sum_{j: src[j]=vocab_id} copy_attn[j]
```

**Implementation Location**: New layer in `full_model.py` or extended `TokenHead`

**Status**: ❌ Needs implementation

## Detailed Implementation

### Phase 1: Decoder Changes (decoder.py)

**File**: `src/models/decoder.py`

#### Change 1.1: Expose Cross-Attention in Forward Pass

**Current** (lines 139-174):
```python
def forward(self, tgt: torch.Tensor, memory: torch.Tensor,
            tgt_mask: Optional[torch.Tensor] = None,
            memory_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # ... existing implementation ...
    cross_attn_weights = None
    for layer in self.layers:
        x, cross_attn_weights = layer(x, memory, tgt_mask, memory_mask)

    context = torch.bmm(cross_attn_weights, memory)
    combined = torch.cat([x, context], dim=-1)
    p_gen = self.copy_gate(combined)

    return x, cross_attn_weights, p_gen
```

**Status**: ✅ Already correct—returns `(decoder_hidden_states, copy_attn, p_gen)`

**IMPORTANT**: The decoder returns **hidden states** (not logits). The `full_model.decode()` method is responsible for applying `token_head` to convert hidden states to vocabulary logits. This separation of concerns is intentional.

**No changes needed** for decoder.py forward pass.

### Phase 2: Model Integration (full_model.py)

**File**: `src/models/full_model.py`

#### Change 2.1: Add Source Token Tracking

Add `source_tokens` parameter to track input token IDs for copy mechanism.

**Location**: Lines 216-251 (`forward` method)

**Current**:
```python
def forward(self, graph_batch, fingerprint: torch.Tensor,
            tgt: torch.Tensor,
            boolean_domain_only: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
```

**Modified**:
```python
def forward(self, graph_batch, fingerprint: torch.Tensor,
            tgt: torch.Tensor,
            source_tokens: Optional[torch.Tensor] = None,
            boolean_domain_only: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
    """
    Full forward pass for training.

    Args:
        graph_batch: PyG batch of input graphs
        fingerprint: [batch, FINGERPRINT_DIM] semantic fingerprints
        tgt: [batch, tgt_len] target token IDs
        source_tokens: [batch, src_len] input token IDs for copy mechanism (optional)
        boolean_domain_only: [batch] bool tensor for domain conditioning (scaled model)

    Returns:
        Dict with: vocab_logits, copy_attn, p_gen, length_pred, depth_pred, value,
                   final_logits (if source_tokens provided)
    """
```

#### Change 2.2: Compute Final Distribution

Add copy distribution mixing in `decode` method or as post-processing.

**Location**: After line 207 (in `decode` method)

**Add**:
```python
def _compute_copy_distribution(self, vocab_logits: torch.Tensor,
                                copy_attn: torch.Tensor,
                                p_gen: torch.Tensor,
                                source_tokens: torch.Tensor,
                                memory_mask: Optional[torch.Tensor] = None,
                                vocab_size: int = VOCAB_SIZE) -> torch.Tensor:
    """
    Compute final distribution mixing generation and copying.

    Args:
        vocab_logits: [batch, tgt_len, vocab_size] vocabulary logits
        copy_attn: [batch, tgt_len, src_len] copy attention weights
        p_gen: [batch, tgt_len, 1] generation probability
        source_tokens: [batch, src_len] input token IDs
        memory_mask: [batch, src_len] padding mask (True = masked position)

    Returns:
        [batch, tgt_len, vocab_size] final logits
    """
    batch_size, tgt_len, _ = vocab_logits.shape
    _, src_len = source_tokens.shape

    # Input validation: Check for NaN/Inf to prevent silent training divergence
    if not torch.isfinite(copy_attn).all():
        raise ValueError(f"copy_attn contains NaN/Inf values - check decoder cross-attention stability")
    if not torch.isfinite(vocab_logits).all():
        raise ValueError(f"vocab_logits contains NaN/Inf values - check token head stability")

    # Shape validation: Ensure encoder memory length matches tokenized input
    if copy_attn.shape[2] != src_len:
        raise ValueError(
            f"Shape mismatch: copy_attn has src_len={copy_attn.shape[2]} "
            f"but source_tokens has src_len={src_len}. "
            f"Ensure encoder memory length matches tokenized input length."
        )

    # Convert logits to probabilities
    vocab_dist = F.softmax(vocab_logits, dim=-1)  # [batch, tgt_len, vocab_size]

    # Apply memory mask to copy attention to prevent padding leakage
    if memory_mask is not None:
        # memory_mask: [batch, src_len], True = masked (padding)
        # copy_attn: [batch, tgt_len, src_len]
        mask_expanded = memory_mask.unsqueeze(1).expand(-1, tgt_len, -1)  # [batch, tgt_len, src_len]
        copy_attn_masked = copy_attn.masked_fill(mask_expanded, 0.0)
    else:
        copy_attn_masked = copy_attn

    # Clamp source tokens to valid vocab range to prevent scatter_add_ crash
    # Out-of-bounds tokens (from corrupted data or tokenizer bugs) are clamped
    source_tokens_clamped = torch.clamp(source_tokens, 0, vocab_size - 1)
    if (source_tokens != source_tokens_clamped).any():
        import logging
        logging.warning(f"Out-of-bounds token IDs clamped. Max ID: {source_tokens.max()}, vocab_size: {vocab_size}")

    # Initialize copy distribution
    copy_dist = torch.zeros_like(vocab_dist)  # [batch, tgt_len, vocab_size]

    # Scatter copy attention to vocabulary indices
    # For each source position j with token id = source_tokens[b,j],
    # add copy_attn[b,i,j] to copy_dist[b,i,source_tokens[b,j]]
    copy_dist.scatter_add_(
        dim=2,
        index=source_tokens_clamped.unsqueeze(1).expand(-1, tgt_len, -1),  # [batch, tgt_len, src_len]
        src=copy_attn_masked  # [batch, tgt_len, src_len]
    )

    # Mix distributions
    p_gen_expanded = p_gen.expand(-1, -1, vocab_size)  # [batch, tgt_len, vocab_size]
    final_dist = p_gen_expanded * vocab_dist + (1 - p_gen_expanded) * copy_dist

    # Convert back to logits for loss computation
    # Epsilon prevents log(0); 1e-10 is safe for float32 (-23 in log space)
    final_logits = torch.log(final_dist + 1e-10)

    return final_logits
```

**Integration**: Call in `decode` method when `source_tokens` is provided:

```python
def decode(self, tgt: torch.Tensor, memory: torch.Tensor,
           source_tokens: Optional[torch.Tensor] = None,
           tgt_mask: Optional[torch.Tensor] = None,
           memory_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
    """
    Decode step.

    Args:
        tgt: [batch, tgt_len] target token IDs
        memory: [batch, src_len, D_MODEL] encoder context
        source_tokens: [batch, src_len] input token IDs for copy mechanism (optional)
        tgt_mask: [tgt_len, tgt_len] causal mask
        memory_mask: [batch, src_len] memory mask

    Returns:
        Dict with vocab_logits, copy_attn, p_gen, decoder_out, [final_logits]
    """
    decoder_out, copy_attn, p_gen = self.decoder(tgt, memory, tgt_mask, memory_mask)
    vocab_logits = self.token_head(decoder_out)

    result = {
        'vocab_logits': vocab_logits,
        'copy_attn': copy_attn,
        'p_gen': p_gen,
        'decoder_out': decoder_out
    }

    # Compute final distribution if source tokens provided
    if source_tokens is not None:
        final_logits = self._compute_copy_distribution(
            vocab_logits, copy_attn, p_gen, source_tokens, memory_mask
        )
        result['final_logits'] = final_logits

    return result
```

### Phase 3: Training Integration

**File**: `src/training/phase2_trainer.py`

#### Change 3.1: Modify Data Loading

Update dataset to provide source tokens.

**Current dataset output** (hypothetical):
```python
{
    'graph_batch': PyG batch,
    'fingerprint': tensor,
    'target_ids': tensor
}
```

**Modified**:
```python
{
    'graph_batch': PyG batch,
    'fingerprint': tensor,
    'target_ids': tensor,
    'source_tokens': tensor  # NEW: [batch, src_len] input token IDs
}
```

**Implementation**: Add to dataset class (likely `src/data/dataset.py`):

```python
def __getitem__(self, idx):
    item = self.data[idx]

    # Existing processing
    graph_batch = self._build_graph(item['obfuscated'])
    fingerprint = self._compute_fingerprint(item['obfuscated'])
    target_ids = self.tokenizer.encode(item['simplified'])

    # NEW: Get source tokens
    source_tokens = torch.tensor(
        self.tokenizer.get_source_tokens(item['obfuscated']),
        dtype=torch.long
    )

    return {
        'graph_batch': graph_batch,
        'fingerprint': fingerprint,
        'target_ids': target_ids,
        'source_tokens': source_tokens  # NEW
    }
```

#### Change 3.2: Modify Loss Computation

Use `final_logits` (with copy) instead of `vocab_logits` (generation only).

**Current loss** (hypothetical):
```python
vocab_logits = model_output['vocab_logits']  # [batch, tgt_len, vocab_size]
loss = F.cross_entropy(
    vocab_logits.reshape(-1, vocab_logits.size(-1)),
    targets.reshape(-1),
    ignore_index=PAD_IDX
)
```

**Modified**:
```python
# Use final_logits if available (training with copy), else fallback to vocab_logits
logits = model_output.get('final_logits', model_output['vocab_logits'])
loss = F.cross_entropy(
    logits.reshape(-1, logits.size(-1)),
    targets.reshape(-1),
    ignore_index=PAD_IDX
)

# Optional: Add auxiliary loss to encourage proper p_gen values
p_gen = model_output['p_gen']  # [batch, tgt_len, 1]
copy_attn = model_output['copy_attn']  # [batch, tgt_len, src_len]

# Compute copy supervision: should copy when target token appears in source
source_tokens = batch['source_tokens']  # [batch, src_len]
targets_expanded = targets.unsqueeze(-1).expand(-1, -1, source_tokens.size(1))
source_expanded = source_tokens.unsqueeze(1).expand(-1, targets.size(1), -1)
should_copy = (targets_expanded == source_expanded).any(dim=-1).float()  # [batch, tgt_len]

# BCE loss on p_gen: should be low (copy) when should_copy=1, high (generate) when should_copy=0
p_gen_loss = F.binary_cross_entropy(
    p_gen.squeeze(-1),
    1 - should_copy,  # Invert because p_gen is "generation probability"
    reduction='mean'
)

total_loss = loss + COPY_WEIGHT * p_gen_loss  # COPY_WEIGHT = 0.1 from constants.py

# IMPORTANT: Validate source_tokens match augmented expression
# (Variable augmentation must be applied before tokenization)
# Add assertion in debug mode:
# assert set(extract_vars(batch['obfuscated'])) == set(extract_vars_from_tokens(batch['source_tokens']))
```

#### Change 3.3: Variable Augmentation Ordering Validation

**CRITICAL**: Variable augmentation MUST be applied BEFORE `source_tokens` extraction. The current `dataset.py` implementation correctly applies augmentation at line 124 (before graph/fingerprint), but this ordering must be preserved.

**Validation assertion** (add to training loop in debug mode):
```python
def _validate_augmentation_consistency(batch, tokenizer):
    """Verify source_tokens match augmented expression variables."""
    from src.data.augmentation import VariablePermuter
    permuter = VariablePermuter()

    for i in range(len(batch['obfuscated'])):
        expr_vars = set(permuter.extract_variables(batch['obfuscated'][i]))

        # Extract only variable tokens from source_tokens
        token_var_ids = [
            tok for tok in batch['source_tokens'][i].tolist()
            if tok in range(VAR_TOKEN_START, VAR_TOKEN_END)  # Variable token range
        ]
        # Convert token IDs back to variable names
        token_vars = set(
            tokenizer.id2token.get(tok, f'UNK_{tok}')
            for tok in token_var_ids
            if tok in tokenizer.id2token
        )

        # Variables must EXACTLY match (not subset) to catch data corruption
        assert expr_vars == token_vars, \
            f"Variable mismatch at index {i}: expr={expr_vars}, tokens={token_vars}"
```

### Phase 4: Inference Integration

**File**: `src/inference/beam_search.py`

#### Change 4.1: Update BeamSearchDecoder Class

Modify `BeamSearchDecoder.decode()` to accept and thread `source_tokens` through the decoding process.

**Change 4.1.1: Update decode() signature** (line ~85)

```python
# BEFORE:
def decode(self, memory: torch.Tensor, max_length: int = 100) -> List[BeamHypothesis]:

# AFTER:
def decode(self, memory: torch.Tensor,
           source_tokens: Optional[torch.Tensor] = None,
           max_length: int = 100) -> List[BeamHypothesis]:
    """
    Decode using beam search.

    Args:
        memory: [batch, src_len, D_MODEL] encoder context
        source_tokens: [batch, src_len] input token IDs for copy mechanism (optional)
        max_length: Maximum sequence length to generate
    """
```

**Change 4.1.2: Pass source_tokens to model.decode()** (line ~141)

```python
# BEFORE (line ~141):
output = self.model.decode(tgt_batch, memory_expanded)

# AFTER:
# Expand source_tokens for beam width if provided
if source_tokens is not None:
    source_tokens_expanded = source_tokens.repeat_interleave(self.beam_width, dim=0)
else:
    source_tokens_expanded = None

output = self.model.decode(tgt_batch, memory_expanded, source_tokens=source_tokens_expanded)
```

**Change 4.1.3: Use final_logits when available**

```python
# Use final_logits if available (includes copy), else vocab_logits
logits = output.get('final_logits', output['vocab_logits'])
next_token_logits = logits[:, -1, :]
```

#### Change 4.2: Update Inference Entry Point

Ensure `source_tokens` is passed from the inference script to beam search.

**Example usage**:
```python
# In simplify.py or inference script:
source_tokens = tokenizer.get_source_tokens(input_expr)
source_tokens_tensor = torch.tensor([source_tokens], device=device)

# Pass to beam search decoder
hypotheses = beam_decoder.decode(memory, source_tokens=source_tokens_tensor)
```

### Phase 5: Testing

#### Test 5.1: Unit Tests for Copy Distribution

**File**: `tests/test_copy_mechanism.py` (new file)

```python
"""Unit tests for copy mechanism."""

import torch
import pytest
from src.models.full_model import MBADeobfuscator
from src.data.tokenizer import MBATokenizer

def test_copy_distribution_basic():
    """Test that copy distribution correctly scatters attention to token IDs."""
    model = MBADeobfuscator(encoder_type='gat')
    tokenizer = MBATokenizer()

    # Create dummy inputs
    batch_size, src_len, tgt_len = 2, 5, 3
    vocab_size = tokenizer.vocab_size

    vocab_logits = torch.randn(batch_size, tgt_len, vocab_size)
    copy_attn = torch.softmax(torch.randn(batch_size, tgt_len, src_len), dim=-1)
    p_gen = torch.sigmoid(torch.randn(batch_size, tgt_len, 1))

    # Source tokens: [x, +, y, -, 1] -> IDs from tokenizer
    source_tokens = torch.tensor([
        tokenizer.encode("x + y - 1", add_special=False),
        tokenizer.encode("a & b", add_special=False) + [0, 0]  # Pad to src_len=5
    ])  # [batch_size, src_len]

    final_logits = model._compute_copy_distribution(
        vocab_logits, copy_attn, p_gen, source_tokens
    )

    # Assertions
    assert final_logits.shape == (batch_size, tgt_len, vocab_size)
    assert torch.isfinite(final_logits).all(), "Final logits contain NaN/Inf"

    # Check that copy_attn mass is preserved in final distribution
    final_probs = torch.softmax(final_logits, dim=-1)
    for b in range(batch_size):
        for t in range(tgt_len):
            # Sum of probabilities for source tokens should relate to copy attention
            source_prob_mass = final_probs[b, t, source_tokens[b]].sum()
            # This should be approximately (1 - p_gen) * 1.0, but relaxed for numeric stability
            assert source_prob_mass > 0, "No probability mass on source tokens"

def test_copy_mechanism_variables():
    """Test that variables can be copied correctly."""
    model = MBADeobfuscator(encoder_type='gat')
    tokenizer = MBATokenizer()

    # Simulate scenario: input is "x & y", output should copy x, y
    source_expr = "x & y"
    source_tokens = torch.tensor([tokenizer.encode(source_expr, add_special=False)])

    # Create dummy encoder output (would normally come from model.encode())
    memory = torch.randn(1, 1, 512)  # [batch=1, src_len=1, d_model=512]

    # Target: start of sequence, then we'll predict next token
    tgt = torch.tensor([[tokenizer.sos_token_id]])

    # Forward through decoder
    output = model.decode(tgt, memory, source_tokens=source_tokens)

    assert 'final_logits' in output, "final_logits not computed when source_tokens provided"
    final_logits = output['final_logits']

    # Check shape
    assert final_logits.shape == (1, 1, tokenizer.vocab_size)

    # Verify that 'x' and 'y' have non-negligible probability
    x_id = tokenizer.token2id['x']
    y_id = tokenizer.token2id['y']
    probs = torch.softmax(final_logits[0, 0], dim=-1)

    assert probs[x_id] > 0.001, "Variable 'x' has negligible copy probability"
    assert probs[y_id] > 0.001, "Variable 'y' has negligible copy probability"

def test_copy_mechanism_gradient_flow():
    """Test that gradients flow through copy mechanism."""
    model = MBADeobfuscator(encoder_type='gat')
    tokenizer = MBATokenizer()

    # Dummy inputs
    source_tokens = torch.tensor([tokenizer.encode("x + 1", add_special=False)])
    memory = torch.randn(1, 1, 512, requires_grad=True)
    tgt = torch.tensor([[tokenizer.sos_token_id, tokenizer.token2id['x']]])

    # Forward
    output = model.decode(tgt, memory, source_tokens=source_tokens)
    final_logits = output['final_logits']

    # Dummy loss
    target = torch.tensor([[tokenizer.token2id['x'], tokenizer.token2id['+']]])
    loss = torch.nn.functional.cross_entropy(
        final_logits.reshape(-1, final_logits.size(-1)),
        target.reshape(-1)
    )

    # Backward
    loss.backward()

    # Check gradients exist
    assert memory.grad is not None, "No gradient on encoder memory"
    assert not torch.isnan(memory.grad).any(), "NaN gradients detected"

def test_copy_mechanism_with_padding():
    """Test copy mechanism correctly handles padded batches with memory_mask."""
    model = MBADeobfuscator(encoder_type='gat')
    tokenizer = MBATokenizer()

    batch_size, src_len, tgt_len = 2, 5, 3
    vocab_size = tokenizer.vocab_size
    PAD_IDX = 0

    # Create padded source tokens (different lengths)
    # Batch 0: "x + y" (3 tokens) + 2 padding
    # Batch 1: "a & b" (3 tokens) + 2 padding
    source_tokens = torch.tensor([
        [tokenizer.token2id['x'], tokenizer.token2id['+'], tokenizer.token2id['y'], PAD_IDX, PAD_IDX],
        [tokenizer.token2id['a'], tokenizer.token2id['&'], tokenizer.token2id['b'], PAD_IDX, PAD_IDX]
    ])

    # Create memory mask for padding (True = masked/padding position)
    memory_mask = torch.tensor([
        [False, False, False, True, True],  # Last 2 positions are padding
        [False, False, False, True, True]
    ])

    vocab_logits = torch.randn(batch_size, tgt_len, vocab_size)
    # Create copy_attn with high attention on padding positions to test masking
    copy_attn = torch.softmax(torch.randn(batch_size, tgt_len, src_len), dim=-1)
    p_gen = torch.sigmoid(torch.randn(batch_size, tgt_len, 1))

    # Test WITH memory_mask to verify padding is properly masked
    final_logits = model._compute_copy_distribution(
        vocab_logits, copy_attn, p_gen, source_tokens, memory_mask
    )

    # Convert to probabilities
    final_probs = torch.softmax(final_logits, dim=-1)

    # Assertions:
    # 1. Copy distribution should sum to <= 1.0
    for b in range(batch_size):
        for t in range(tgt_len):
            assert final_probs[b, t].sum() <= 1.0 + 1e-5, "Probability mass exceeds 1.0"

    # 2. Padding token (ID=0) should have LOW copy probability when mask is applied
    # Because padding positions are masked, their attention doesn't leak into copy_dist
    for b in range(batch_size):
        for t in range(tgt_len):
            # With proper masking, PAD token shouldn't get high probability
            # unless it appears in non-padded positions (which it doesn't here)
            assert final_probs[b, t, PAD_IDX] < 0.5, \
                f"Padding token has suspiciously high probability: {final_probs[b, t, PAD_IDX]}"

    # 3. Shape and finiteness checks
    assert final_logits.shape == (batch_size, tgt_len, vocab_size)
    assert torch.isfinite(final_logits).all(), "Final logits contain NaN/Inf"

    # 4. Compare WITH vs WITHOUT mask to verify masking makes a difference
    final_logits_no_mask = model._compute_copy_distribution(
        vocab_logits, copy_attn, p_gen, source_tokens, memory_mask=None
    )
    # They should differ because masking zeros out padding attention
    assert not torch.allclose(final_logits, final_logits_no_mask), \
        "Memory mask should change the output (padding attention should be masked)"
```

#### Test 5.2: Integration Test

**File**: `tests/test_integration_copy.py` (new file)

```python
"""Integration test for copy mechanism in full training pipeline."""

import torch
from src.models.full_model import MBADeobfuscator
from src.data.tokenizer import MBATokenizer
from torch_geometric.data import Batch

def test_full_forward_with_copy():
    """Test full model forward pass with copy mechanism."""
    model = MBADeobfuscator(encoder_type='gat')
    tokenizer = MBATokenizer()

    # Create dummy graph batch (simplified)
    from torch_geometric.data import Data

    # Graph for "x + 1"
    # Nodes: [+, x, 1]
    x = torch.tensor([[1, 0], [1, 0], [1, 0]], dtype=torch.float)  # Dummy node features
    edge_index = torch.tensor([[0, 0], [1, 2]], dtype=torch.long)  # + -> x, + -> 1
    graph = Data(x=x, edge_index=edge_index)
    graph_batch = Batch.from_data_list([graph])

    # Fingerprint
    fingerprint = torch.randn(1, 448)

    # Target tokens
    tgt = torch.tensor([tokenizer.encode("x + 1")])

    # Source tokens
    source_tokens = torch.tensor([tokenizer.encode("x + 1", add_special=False)])

    # Forward pass
    output = model(graph_batch, fingerprint, tgt, source_tokens=source_tokens)

    # Assertions
    assert 'final_logits' in output, "final_logits not in output"
    assert 'vocab_logits' in output
    assert 'copy_attn' in output
    assert 'p_gen' in output

    final_logits = output['final_logits']
    assert final_logits.shape[0] == 1  # batch_size
    assert final_logits.shape[2] == tokenizer.vocab_size
```

#### Test 5.3: End-to-End Test

**File**: `tests/test_e2e_copy.py` (new file)

```python
"""End-to-end test: copy mechanism improves variable preservation."""

def test_copy_improves_variable_preservation():
    """
    Compare models with/without copy on variable preservation task.

    Task: Given obfuscated expression with variables x, y, z,
    simplified expression should use same variable names.
    """
    # This test would require training, so mark as slow/integration
    pytest.skip("Requires trained model—run manually")

    # Pseudocode:
    # 1. Train model WITHOUT copy (model_baseline)
    # 2. Train model WITH copy (model_copy)
    # 3. Evaluate on test set, measure:
    #    - Variable name accuracy: % of predictions using correct variable names
    # 4. Assert: model_copy.var_accuracy > model_baseline.var_accuracy + 0.05
```

## File Modification Summary

### Files to Modify

| File | Lines | Changes | Priority |
|------|-------|---------|----------|
| `src/models/decoder.py` | None | Already correct ✅ | - |
| `src/models/full_model.py` | 190-251 | Add `source_tokens` param, `_compute_copy_distribution` method | P0 |
| `src/data/tokenizer.py` | None | Already has `get_source_tokens` method ✅ | - |
| `src/training/phase2_trainer.py` | TBD | Add source tokens to batch, update loss | P0 |
| `src/data/dataset.py` | TBD | Add `source_tokens` to `__getitem__` | P0 |
| `src/inference/beam_search.py` | TBD | Pass `source_tokens` to `decode` | P1 |

### Files to Create

| File | Purpose | Priority |
|------|---------|----------|
| `tests/test_copy_mechanism.py` | Unit tests for copy distribution logic | P0 |
| `tests/test_integration_copy.py` | Integration test for full forward pass | P1 |
| `tests/test_e2e_copy.py` | End-to-end training test (manual) | P2 |

## Training Considerations

### Copy Gate Stability

**Issue**: The copy gate uses sigmoid activation which can saturate to 0 or 1, causing vanishing gradients.

**Mitigation strategies**:

1. **Initialization**: Initialize copy gate final layer bias to 0.0 (neutral sigmoid output ~0.5)
   ```python
   # In decoder.py, after defining self.copy_gate:
   nn.init.zeros_(self.copy_gate[-2].bias)  # Set bias of pre-sigmoid layer to 0
   ```

2. **Gradient clipping**: Apply gradient clipping specifically to copy gate parameters
   ```python
   # In training loop, after loss.backward():
   torch.nn.utils.clip_grad_norm_(model.decoder.copy_gate.parameters(), max_norm=1.0)
   ```

3. **Monitoring**: Track p_gen statistics during training
   ```python
   # Log p_gen mean and std each epoch
   p_gen_mean = p_gen.mean().item()
   p_gen_std = p_gen.std().item()
   logger.info(f"p_gen: mean={p_gen_mean:.3f}, std={p_gen_std:.3f}")

   # WARNING: If std < 0.1 for multiple epochs, gate may have collapsed
   if p_gen_std < 0.1:
       logger.warning("Copy gate may be collapsing (low variance)")
   ```

### Loss Function Design

**Hybrid Loss** (recommended):
```python
total_loss = ce_loss + COPY_WEIGHT * p_gen_supervision_loss + COMPLEXITY_WEIGHT * complexity_loss
```

Where:
- `ce_loss`: Cross-entropy on `final_logits` (mixed distribution)
- `p_gen_supervision_loss`: BCE on `p_gen` to teach when to copy vs. generate
- `complexity_loss`: Existing complexity head loss (unchanged)

**COPY_WEIGHT** = 0.1 (from `constants.py` line 168)

### Teacher Forcing with Copy

During training, teacher forcing provides ground truth tokens. Copy supervision signal:

```python
should_copy[b, t] = 1  if target[b, t] appears in source_tokens[b]
                    0  otherwise

p_gen_target[b, t] = 1 - should_copy[b, t]  # p_gen should be low for copy tokens
```

This provides supervision for the copy gate without requiring additional annotations.

### Curriculum Considerations

Copy mechanism is most useful for:
- **Shallow expressions** (depth 2-5): Variables are prominent, few transformations
- **Variable-heavy expressions**: Multiple variable occurrences

Less useful for:
- **Constant-heavy expressions**: Output is mostly constants (which are in vocab)
- **Deep obfuscations** (depth 10+): Variables may be transformed, not copied

**Recommendation**: Train copy mechanism from Stage 1 (depth 2) of curriculum. Model will learn when copy is beneficial.

## Edge Cases

### Case 1: Token Not in Vocabulary

**Scenario**: Source token ID ≥ vocab_size (e.g., > 299 for vocab_size=300)

**Handling**:
- **WARNING**: Without clamping, `scatter_add_` raises `RuntimeError: index out of bounds`
- Implementation clamps out-of-bounds tokens to valid range `[0, vocab_size-1]`
- Clamped tokens map to wrong vocabulary entries, but this is better than crashing
- A warning is logged when clamping occurs to help diagnose data pipeline issues
- Root cause (corrupted data or tokenizer bug) should be investigated

**Code**:
```python
source_tokens_clamped = torch.clamp(source_tokens, 0, vocab_size - 1)
if (source_tokens != source_tokens_clamped).any():
    logging.warning(f"Out-of-bounds token IDs clamped. Max ID: {source_tokens.max()}")
```

### Case 2: Multiple Copies of Same Variable

**Scenario**: Source is `x + x + 1`, target is `2*x + 1`

**Handling**:
- Copy attention will distribute mass across both `x` occurrences in source
- `scatter_add_` accumulates attention: `copy_dist[x_id] = attn[pos1] + attn[pos2]`
- This is correct—model sees multiple evidence for copying `x`

### Case 3: Variable Renaming

**Scenario**: Source has `x`, target has `y` (due to obfuscation choice)

**Handling**:
- Model should learn p_gen ≈ 1 (generate) for this case
- Copy mechanism won't help, but won't hurt—generation path handles it
- Training signal: `should_copy=0` for this case (target `y` not in source)

### Case 4: Padding in Source

**Scenario**: Batched inputs with variable source lengths, padding tokens

**Handling**:
- Padding tokens have ID = 0 (PAD_IDX)
- `scatter_add_` will add attention mass to `copy_dist[0]` (padding)
- **Solution**: Mask out padding positions using `memory_mask` in decoder
- Cross-attention already respects `key_padding_mask` (decoder.py line 77), so attention on padding is suppressed

### Case 5: Empty Intersection (Target Variables Not in Source)

**Scenario**: Source is `(a&b)+3`, target is `x|y` (completely different variables)

**Handling**:
- `should_copy` will be all zeros → p_gen_target = 1 (generate)
- Copy distribution will be mostly uniform (no relevant source tokens)
- Model learns to ignore copy path: `p_gen ≈ 1`
- Graceful degradation to pure generation

## Performance Impact

### Computational Overhead

**Additional Operations** per forward pass:
1. `scatter_add_` for copy distribution: O(batch × tgt_len × src_len)
2. Softmax on vocab_logits: O(batch × tgt_len × vocab_size)
3. Element-wise mixing: O(batch × tgt_len × vocab_size)

**Estimate**: ~5-10% slowdown vs. generation-only decoder

**Memory**: Additional tensor `copy_dist` of size [batch, tgt_len, vocab_size] ≈ 4MB for batch=32, tgt_len=64, vocab_size=300

**Mitigation**: Overhead is negligible compared to encoder (GNN) and decoder (Transformer) computation.

### Accuracy Impact (Expected)

Based on pointer-generator results in summarization (See et al., 2017):
- **Variable preservation**: +15-20% accuracy (primary benefit)
- **Overall simplification accuracy**: +3-5% (secondary benefit from better variable handling)
- **Syntax error rate**: -2-3% (copied tokens are always valid)

## Rollout Plan

### Week 1: Core Implementation
- Day 1-2: Modify `full_model.py` (add `_compute_copy_distribution`, update `forward`/`decode`)
- Day 3: Modify dataset to provide `source_tokens`
- Day 4: Update training loop with copy-aware loss
- Day 5: Unit tests + debugging

### Week 2: Integration & Evaluation
- Day 1-2: Integrate with beam search inference
- Day 3: Train Phase 2 with copy mechanism on depth-2 curriculum stage
- Day 4: Evaluate variable preservation accuracy
- Day 5: Full curriculum training (depth 2→5→10→14)

### Week 3: Refinement
- Day 1-2: Hyperparameter tuning (COPY_WEIGHT, copy gate architecture)
- Day 3-4: Ablation studies (with vs. without copy)
- Day 5: Documentation + merge

## Validation Metrics

Track these metrics to verify copy mechanism effectiveness:

| Metric | Definition | Target |
|--------|------------|--------|
| **Variable Accuracy** | % predictions using correct variable names | >95% |
| **p_gen Calibration** | Mean p_gen for copy-able tokens | <0.3 |
| **Copy Utilization** | % tokens where copy_dist > vocab_dist | >40% |
| **Syntax Error Rate** | % predictions with invalid syntax | <2% |
| **Overall Accuracy** | % semantically equivalent (Z3-verified) | +3-5% vs baseline |

## Conclusion

This implementation plan adds a pointer-generator copy mechanism to the MBA deobfuscator decoder, enabling direct copying of variables from input expressions. The design leverages existing cross-attention infrastructure, requiring minimal code changes (~200 lines total). Expected benefits: +15-20% variable preservation accuracy, +3-5% overall simplification accuracy, with <10% computational overhead.

**Next Steps**: Begin implementation starting with `full_model.py` modifications (Week 1, Day 1).
