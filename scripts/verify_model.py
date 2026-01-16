#!/usr/bin/env python3
"""
Verify scaled MBA Deobfuscator model configuration.

Validates:
- Parameter count (~362M expected)
- Forward pass with dummy data
- Memory estimation
- Component breakdown
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from torch_geometric.data import Data, Batch


def count_parameters(model: torch.nn.Module) -> dict:
    """Count parameters by component."""
    counts = {}

    for name, param in model.named_parameters():
        component = name.split('.')[0]
        if component not in counts:
            counts[component] = 0
        counts[component] += param.numel()

    counts['total'] = sum(counts.values())
    return counts


def format_params(n: int) -> str:
    """Format parameter count for readability."""
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    elif n >= 1e6:
        return f"{n / 1e6:.2f}M"
    elif n >= 1e3:
        return f"{n / 1e3:.2f}K"
    return str(n)


def create_dummy_batch(batch_size: int = 2, num_nodes: int = 50, device: str = 'cpu'):
    """Create dummy PyG batch for testing."""
    graphs = []
    for _ in range(batch_size):
        # Random node types (0-9 for 10 node types)
        x = torch.randint(0, 10, (num_nodes,), dtype=torch.long)

        # Random edges
        num_edges = num_nodes * 3
        edge_index = torch.randint(0, num_nodes, (2, num_edges), dtype=torch.long)

        # Random edge types (0-7 for 8 edge types)
        edge_type = torch.randint(0, 8, (num_edges,), dtype=torch.long)

        graphs.append(Data(x=x, edge_index=edge_index, edge_type=edge_type))

    return Batch.from_data_list(graphs).to(device)


def verify_scaled_model():
    """Verify ScaledMBADeobfuscator configuration."""
    print("=" * 60)
    print("MBA Deobfuscator - Scaled Model Verification")
    print("=" * 60)

    # Import here to catch import errors
    try:
        from src.models.full_model import ScaledMBADeobfuscator
        from src.constants import (
            SCALED_HIDDEN_DIM, SCALED_D_MODEL, SCALED_D_FF,
            SCALED_NUM_DECODER_LAYERS, SCALED_NUM_DECODER_HEADS,
            FINGERPRINT_DIM
        )
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure all dependencies are installed.")
        return False

    # Create model
    print("\n[1/4] Creating ScaledMBADeobfuscator (HGT encoder)...")
    try:
        model = ScaledMBADeobfuscator(encoder_type='hgt')
        print("  - Model created successfully")
    except Exception as e:
        print(f"  - ERROR: {e}")
        return False

    # Count parameters
    print("\n[2/4] Parameter count by component:")
    counts = count_parameters(model)
    for name, count in sorted(counts.items()):
        if name != 'total':
            print(f"  - {name}: {format_params(count)}")

    total = counts['total']
    print(f"\n  TOTAL: {format_params(total)} ({total:,} parameters)")

    target = 362_000_000
    deviation = abs(total - target) / target * 100
    if total < 300_000_000:
        print(f"  WARNING: Below target (~360M). Deviation: {deviation:.1f}%")
    elif total > 400_000_000:
        print(f"  WARNING: Above target (~360M). Deviation: {deviation:.1f}%")
    else:
        print(f"  OK: Within expected range. Deviation: {deviation:.1f}%")

    # Forward pass test
    print("\n[3/4] Forward pass test...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  - Device: {device}")

    model = model.to(device)
    model.eval()

    batch_size = 2
    seq_len = 32

    try:
        graph_batch = create_dummy_batch(batch_size=batch_size, num_nodes=50, device=device)
        fingerprint = torch.randn(batch_size, FINGERPRINT_DIM, device=device)
        tgt = torch.randint(0, 300, (batch_size, seq_len), dtype=torch.long, device=device)
        boolean_domain = torch.zeros(batch_size, dtype=torch.long, device=device)

        with torch.no_grad():
            output = model(graph_batch, fingerprint, tgt, boolean_domain)

        print(f"  - Input graph batch: {batch_size} graphs, {graph_batch.x.shape[0]} total nodes")
        print(f"  - Fingerprint shape: {fingerprint.shape}")
        print(f"  - Target shape: {tgt.shape}")
        print(f"  - Output shapes:")
        for key, val in output.items():
            if isinstance(val, torch.Tensor):
                print(f"      {key}: {val.shape}")

        print("  - Forward pass: OK")
    except Exception as e:
        print(f"  - ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Memory estimation
    print("\n[4/4] Memory estimation:")
    # Rough activation memory estimate
    heads = SCALED_NUM_DECODER_HEADS
    layers = SCALED_NUM_DECODER_LAYERS
    max_seq = 2048
    effective_layers = layers // 3  # With gradient checkpointing

    act_memory_full = batch_size * heads * (max_seq ** 2) * 2 * layers
    act_memory_ckpt = batch_size * heads * (max_seq ** 2) * 2 * effective_layers

    print(f"  - Full training (no checkpointing):")
    print(f"      Activation memory: ~{act_memory_full / 1e9:.1f} GB (seq_len={max_seq})")
    print(f"  - With gradient checkpointing:")
    print(f"      Activation memory: ~{act_memory_ckpt / 1e9:.1f} GB (seq_len={max_seq})")

    model_memory = total * 4 / 1e9  # 4 bytes per float32 param
    print(f"  - Model parameters: ~{model_memory:.1f} GB (FP32)")
    print(f"  - Model parameters: ~{model_memory / 2:.1f} GB (FP16/BF16)")

    print("\n" + "=" * 60)
    print("Verification complete!")
    print("=" * 60)

    return True


def verify_rgcn_model():
    """Verify ScaledMBADeobfuscator with RGCN encoder."""
    print("\n" + "-" * 60)
    print("Verifying RGCN encoder variant...")
    print("-" * 60)

    try:
        from src.models.full_model import ScaledMBADeobfuscator
    except ImportError as e:
        print(f"Import error: {e}")
        return False

    try:
        model = ScaledMBADeobfuscator(encoder_type='rgcn')
        counts = count_parameters(model)
        print(f"  - RGCN model: {format_params(counts['total'])}")
        print("  - RGCN variant: OK")
        return True
    except Exception as e:
        print(f"  - ERROR: {e}")
        return False


if __name__ == '__main__':
    success = verify_scaled_model()

    if success:
        verify_rgcn_model()

    sys.exit(0 if success else 1)
